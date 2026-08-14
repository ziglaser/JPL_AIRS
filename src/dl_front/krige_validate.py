"""Kriging validation study: how well does gap-filling recover the truth?

The degraded-reanalysis design gives us something rare: EXACT ground truth.
We take a clean MERRA-2 reanalysis step, punch the real AIRS availability
mask into it (the same mask source ``krige_fill._gap_valid_frac`` uses for
the stage-B' training caches), reconstruct the masked-out pixels, and score
the reconstruction against the reanalysis values we hid.  Every error
number below is a true error, not a proxy.

Three questions, one study (user decision 2026-08-13):

1. **Is ordinary kriging worth its cost?**  Each requested variogram model
   is raced against two baselines -- nearest-observed-neighbour and the
   observed-mean constant -- on RMSE / MAE / bias plus a *gradient ratio*
   (mean |grad| of the fill over mean |grad| of the truth on the held-out
   pixels).  The gradient ratio matters because the downstream consumer is
   a front detector: a smoother-than-truth fill (ratio << 1) erases exactly
   the thermal gradients the CNN looks for.
2. **Where does kriging fail?**  Metrics are stratified by gap *cause*
   (cloud/retrieval gap inside the swath vs out-of-swath, via
   ``krige_fill._classify_step_gaps``: the date's climatological swath for
   real fullgrid masks, the drawn mask's own envelope for gap-bank draws --
   donor-date swaths sit at a different cycle position, review 2026-08-13)
   and by distance to the nearest observed pixel
   (1-2, 3-5, > 5 degrees).  Cloud gaps are small holes ringed by data;
   out-of-swath gaps are wide unconstrained voids -- we expect (and want to
   quantify) very different error regimes.
3. **Which swath-projection method wins?**  ``swath.compare_projections``
   races composite / shift / hull footprint predictors on the sampled dates
   that have real fullgrid files.

Fill and scoring follow the schema-v3 domain decision (user 2026-08-13):
reconstruction runs over the CROP domain (analysis box + receptive-field
halo, ``dataset.crop_domain()``, exactly what the cache builders fill),
but metrics are scored ONLY over held-out pixels of the ANALYSIS domain
(box ∩ land, ``dataset.analysis_domain()``) -- the 18/21/00Z analysis
product is what training scores, so errors in the halo (accepted kriging
drift) or beyond are irrelevant.  Distance/gap strata are likewise taken
within the analysis domain.

Outputs under ``--out`` (default results/dl_front/krige_validation):
``metrics.csv`` (tidy, one row per date/hour/channel/method/stratum),
``projection_methods.csv``, ``summary.md``, and ``panels/*.png`` 3-panel
maps (truth | kriged | bias) for seeded random cases.

CLI::

    python -m dl_front.krige_validate --years 2007-2015 --n-days 40 \
        [--hours 18,21,0] [--variograms linear,spherical,exponential] \
        [--panels 6] [--allow-small-bank] [--out DIR] [--seed 20260813]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import airs_fcst, config, krige_fill, swath
from .acquire_merra2_sfc import day_path
from .dataset import analysis_domain, crop_domain
from .krige_fill import parse_years

#: Default variogram sweep: pykrige's three standard non-nugget families.
DEFAULT_VARIOGRAMS = ("linear", "spherical", "exponential")
#: Physical units of the kriged channels (panel colorbars, summary tables).
CHANNEL_UNITS = {"T2M": "K", "QV2M": "kg/kg"}
#: Distance-to-nearest-observed strata, degrees (1-degree grid => Euclidean
#: pixel distance IS degrees).  Held-out pixels always have distance >= 1.
DISTANCE_BINS = (("1-2deg", 0.0, 2.0), ("3-5deg", 2.0, 5.0),
                 (">5deg", 5.0, np.inf))
#: gap_type flag -> stratum name in metrics.csv.
GAP_STRATA = {config.GAP_CLOUD: "cloud",
              config.GAP_OUT_OF_SWATH: "out_of_swath"}
#: metrics.csv column order (frozen contract 2026-08-13).
METRIC_COLUMNS = ("date", "hour", "channel", "method", "stratum_kind",
                  "stratum", "n_pixels", "rmse", "mae", "bias",
                  "gradient_ratio", "ms")


# --------------------------------------------------------------------------- #
# Metrics (pure numpy, unit-tested on hand-checkable arrays)
# --------------------------------------------------------------------------- #

def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    """|grad| of a 2-D field via np.gradient (grid units = degrees).

    NaNs (out-of-domain pixels) propagate into their neighbours' central
    differences; callers average with nanmean so a pixel hugging the domain
    edge simply drops out of the gradient statistic instead of poisoning it.
    """
    gy, gx = np.gradient(np.asarray(field, float))
    return np.hypot(gy, gx)


def reconstruction_metrics(filled: np.ndarray, truth: np.ndarray,
                           pick: np.ndarray) -> dict | None:
    """Error statistics of ``filled`` vs ``truth`` over the ``pick`` pixels.

    Returns {n_pixels, rmse, mae, bias, gradient_ratio} or None when the
    stratum is empty (the caller writes no row -- an all-NaN row would only
    clutter the tidy CSV).  ``bias`` is mean(filled - truth): positive =
    the fill runs warm/moist.  ``gradient_ratio`` < 1 means the fill is
    smoother than the truth (the failure mode that erases fronts).
    """
    pick = np.asarray(pick, bool)
    n = int(pick.sum())
    if n == 0:
        return None
    err = np.asarray(filled, float)[pick] - np.asarray(truth, float)[pick]
    truth_grad = np.nanmean(gradient_magnitude(truth)[pick])
    fill_grad = np.nanmean(gradient_magnitude(filled)[pick])
    return {"n_pixels": n,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "gradient_ratio": float(fill_grad / truth_grad)
                              if truth_grad > 0 else np.nan}


def distance_to_observed(observed: np.ndarray) -> np.ndarray:
    """Euclidean pixel distance to the nearest observed pixel (0 on them)."""
    return ndimage.distance_transform_edt(~np.asarray(observed, bool))


def distance_stratum(dist: np.ndarray) -> np.ndarray:
    """Distance grid -> stratum-name grid per :data:`DISTANCE_BINS`.

    Membership is ``lo < d <= hi`` so the diagonal neighbour (sqrt(2)) and
    the exact 2-degree pixel both land in '1-2deg'.  Observed pixels (d=0)
    get '' and are never scored.
    """
    out = np.full(dist.shape, "", dtype=object)
    for name, lo, hi in DISTANCE_BINS:
        out[(dist > lo) & (dist <= hi)] = name
    out[dist == 0] = ""
    return out


# --------------------------------------------------------------------------- #
# Reconstruction methods
# --------------------------------------------------------------------------- #

def fill_nearest(truth: np.ndarray, observed: np.ndarray,
                 target: np.ndarray) -> np.ndarray:
    """Nearest-observed-neighbour baseline (piecewise-constant Voronoi fill).

    distance_transform_edt(return_indices) gives, for every pixel, the
    coordinates of its nearest observed pixel; target pixels copy that
    value.  Zero smoothing -- the gradient-ratio foil to kriging.
    """
    field = np.where(observed, truth, np.nan)
    _, ind = ndimage.distance_transform_edt(~np.asarray(observed, bool),
                                            return_indices=True)
    nearest = np.asarray(truth, float)[tuple(ind)]
    out = field.copy()
    out[target] = nearest[target]
    return out


def fill_mean(truth: np.ndarray, observed: np.ndarray,
              target: np.ndarray) -> np.ndarray:
    """Observed-mean constant baseline: the no-spatial-information floor."""
    field = np.where(observed, truth, np.nan)
    out = field.copy()
    out[target] = np.asarray(truth, float)[np.asarray(observed, bool)].mean()
    return out


def evaluate_case(truth: np.ndarray, observed: np.ndarray,
                  crop: np.ndarray, score_domain: np.ndarray,
                  gap_type: np.ndarray,
                  variograms, date, hour: int, channel: str) -> list[dict]:
    """Score every method on one (date, hour, channel) masked-truth case.

    ``truth`` is the physical-units reanalysis field (NaN allowed outside
    ``crop``); ``observed`` must already be restricted to the crop.
    Reconstruction fills ``crop & ~observed`` (mirroring the cache
    builders, halo included), but metrics are scored ONLY over
    ``score_domain & ~observed`` -- the held-out ANALYSIS-domain pixels
    (user decision 2026-08-13).  Returns tidy metric rows (METRIC_COLUMNS
    order), stratified overall / by gap_type / by distance bin over those
    scored pixels.
    """
    observed = np.asarray(observed, bool)
    crop = np.asarray(crop, bool)
    score = np.asarray(score_domain, bool) & ~observed   # scored pixels
    target = crop & ~observed          # filled: in-crop, truth hidden
    # Gradients compare like with like: both fields NaN outside the crop.
    truth_dom = np.where(crop, np.asarray(truth, float), np.nan)
    dist_name = distance_stratum(distance_to_observed(observed))

    fills = {}
    for v in variograms:
        field = np.where(observed, truth_dom, np.nan)
        rng = krige_fill._step_rng(pd.Timestamp(date), hour, channel)
        t0 = time.perf_counter()
        filled = krige_fill.krige_fill(field, rng=rng, target=target,
                                       variogram=v)
        fills[f"krige_{v}"] = (filled, (time.perf_counter() - t0) * 1e3)
    for name, fn in (("nearest", fill_nearest), ("mean", fill_mean)):
        t0 = time.perf_counter()
        filled = fn(truth_dom, observed, target)
        fills[name] = (filled, (time.perf_counter() - t0) * 1e3)

    rows = []
    strata = [("overall", "all", score)]
    strata += [("gap_type", name, score & (gap_type == flag))
               for flag, name in GAP_STRATA.items()]
    strata += [("distance", name, score & (dist_name == name))
               for name, _, _ in DISTANCE_BINS]
    for method, (filled, ms) in fills.items():
        for kind, name, pick in strata:
            m = reconstruction_metrics(filled, truth_dom, pick)
            if m is None:
                continue
            rows.append({"date": f"{pd.Timestamp(date):%Y-%m-%d}",
                         "hour": hour, "channel": channel, "method": method,
                         "stratum_kind": kind, "stratum": name,
                         "ms": ms, **m})
    return rows


# --------------------------------------------------------------------------- #
# Day sampling
# --------------------------------------------------------------------------- #

def sample_study_days(years, n_days: int, hours, seed: int) -> list:
    """Seeded sample of overpass dates, spread across years and months.

    Candidates are every date in ``years`` whose sfc_daily reanalysis
    file(s) exist (the next day's too when hour 0 is requested -- its step
    lives at next-day 00 UTC).  Each candidate gets one seeded uniform
    draw; candidates are sorted by (rank of that draw within the
    candidate's (year, month) group, draw) so the first ``n_days`` take at
    most ceil(n / groups) from any single month -- a seeded draw that
    cannot pile onto one season the way a plain shuffle can.
    """
    cand = []
    for year in sorted(years):
        for date in pd.date_range(f"{year}-01-01", f"{year}-12-31"):
            if not day_path(date).exists():
                continue
            if 0 in hours and not day_path(date + pd.Timedelta(days=1)).exists():
                continue
            cand.append(date)
    if not cand:
        raise FileNotFoundError(
            f"no sfc_daily reanalysis files under {config.SFC_DIR} for "
            f"years {sorted(years)}; fetch with dl_front.acquire_merra2_sfc")
    rng = np.random.default_rng([seed, 0])
    draw = rng.random(len(cand))
    ranks = {}
    order = []
    for i in np.argsort(draw):                # rank of draw within its month
        key = (cand[i].year, cand[i].month)
        ranks[key] = ranks.get(key, -1) + 1
        order.append((ranks[key], draw[i], i))
    picked = [cand[i] for _, _, i in sorted(order)[:n_days]]
    return sorted(picked)


# --------------------------------------------------------------------------- #
# 3-panel maps
# --------------------------------------------------------------------------- #

def render_panel(truth: np.ndarray, kriged: np.ndarray,
                 observed: np.ndarray, envelope: np.ndarray,
                 domain: np.ndarray, channel: str, date, hour: int,
                 out_path: Path, crop: np.ndarray | None = None) -> Path:
    """PNG: [truth | kriged | bias] maps for one case (dataviz-rule styling).

    Sequential job (magnitude): truth and kriged share ONE viridis scale
    with identical vmin/vmax so the eye can difference them.  Polarity job:
    bias = kriged - truth on a diverging RdBu_r symmetric about 0 (neutral
    near-white midpoint).  ``domain`` is the ANALYSIS domain: pixels
    outside it are light gray in all three panels.  ``crop`` (the analysis
    box + halo, user decision 2026-08-13) sets the axis limits so the
    panels display only the crop window instead of losing the box in the
    old hemispheric frame; None keeps the full-grid frame.  The observed
    mask is drawn as a thin contour and the swath envelope as a dashed
    contour.  Text in neutral dark ink; labeled unit colorbars; Agg
    backend, dpi 150.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#333333"
    unit = CHANNEL_UNITS.get(channel, "")
    domain = np.asarray(domain, bool)
    truth_m = np.ma.masked_invalid(np.where(domain, truth, np.nan))
    kriged_m = np.ma.masked_invalid(np.where(domain, kriged, np.nan))
    bias_m = kriged_m - truth_m
    vmin = float(min(truth_m.min(), kriged_m.min()))
    vmax = float(max(truth_m.max(), kriged_m.max()))
    bmax = float(np.abs(bias_m).max()) or 1e-12
    lats, lons = np.asarray(config.LABEL_LATS), np.asarray(config.LABEL_LONS)
    # Cell-EDGE extent (review 2026-08-13): imshow's extent is the raster's
    # OUTER edges, and grid values are cell centers on integer degrees, so
    # the edges sit half a cell beyond the first/last centers.  A
    # center-based extent compressed the raster by 140/141 and shifted
    # features up to ~0.5 deg against the axis labels -- invisible in the
    # old hemispheric frame, perceptible in the zoomed crop window.
    # contour shares imshow's extent convention, so both stay aligned.
    extent = (lons[0] - 0.5, lons[-1] + 0.5, lats[0] - 0.5, lats[-1] + 0.5)
    window = None                       # (lon_lo, lon_hi, lat_lo, lat_hi)
    if crop is not None:
        rows, cols = np.nonzero(np.asarray(crop, bool))
        # the crop cells' outer edges: exact now that extent is edge-based
        window = (lons[cols.min()] - 0.5, lons[cols.max()] + 0.5,
                  lats[rows.min()] - 0.5, lats[rows.max()] + 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    panels = [("truth", truth_m, "viridis", vmin, vmax),
              ("kriged", kriged_m, "viridis", vmin, vmax),
              ("bias = kriged - truth", bias_m, "RdBu_r", -bmax, bmax)]
    for ax, (name, grid, cmap, lo, hi) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.88")                    # out-of-domain: light gray
        im = ax.imshow(grid, origin="lower", extent=extent, cmap=cm,
                       vmin=lo, vmax=hi, aspect="auto",
                       interpolation="nearest")
        ax.contour(np.asarray(observed, float), levels=[0.5], colors=[ink],
                   linewidths=0.4, extent=extent, origin="lower")
        ax.contour(np.asarray(envelope, float), levels=[0.5],
                   colors=["#888888"], linewidths=0.8, linestyles="dashed",
                   extent=extent, origin="lower")
        if window is not None:          # display the crop window only
            ax.set_xlim(window[0], window[1])
            ax.set_ylim(window[2], window[3])
        ax.set_title(name, color=ink, fontsize=10)
        ax.tick_params(colors=ink, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("0.6")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(f"{channel} [{unit}]" if name != panels[2][0]
                     else f"bias [{unit}]", color=ink, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=ink)
    fig.suptitle(f"{channel} {pd.Timestamp(date):%Y-%m-%d} {hour:02d}Z  "
                 f"(thin contour = observed, dashed = swath envelope)",
                 color=ink, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def _pooled_rmse(df: pd.DataFrame) -> float:
    """n-weighted pool of per-case RMSE rows: sqrt(sum n*rmse^2 / sum n)."""
    n = df["n_pixels"].to_numpy(float)
    return float(np.sqrt((n * df["rmse"].to_numpy(float) ** 2).sum() / n.sum()))


def _pool_table(df: pd.DataFrame, stratum_kind: str) -> pd.DataFrame:
    """(channel, method[, stratum]) pooled-RMSE table for one stratum kind."""
    sub = df[df["stratum_kind"] == stratum_kind]
    keys = ["channel", "method"] + ([] if stratum_kind == "overall"
                                    else ["stratum"])
    rows = [dict(zip(keys, key if isinstance(key, tuple) else (key,)),
                 n_pixels=int(g["n_pixels"].sum()),
                 pooled_rmse=_pooled_rmse(g),
                 mean_gradient_ratio=float(g["gradient_ratio"].mean()),
                 mean_ms=float(g["ms"].mean()))
            for key, g in sub.groupby(keys)]
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """A DataFrame as a GitHub-markdown table (no tabulate dependency)."""
    fmt = lambda v: (f"{v:{floatfmt}}" if isinstance(v, float) else str(v))
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def write_summary(metrics: pd.DataFrame, proj: pd.DataFrame, out_dir: Path,
                  args_note: str, bank_note: str) -> Path:
    """summary.md: pooled tables, cloud-vs-swath verdict, variogram winner."""
    lines = ["# Kriging validation summary", "", args_note, ""]

    overall = _pool_table(metrics, "overall")
    lines += ["## Pooled RMSE, overall (held-out in-domain pixels)", "",
              _md_table(overall), ""]

    by_gap = _pool_table(metrics, "gap_type")
    lines += ["## Pooled RMSE by gap type", "",
              _md_table(by_gap), ""]

    # The question the user asked: cloud gaps vs out-of-swath gaps.
    krige_rows = by_gap[by_gap["method"].str.startswith("krige_")]
    lines += ["## Cloud vs out-of-swath (the headline comparison)", ""]
    for channel, g in krige_rows.groupby("channel"):
        piv = g.pivot_table(index="method", columns="stratum",
                            values="pooled_rmse")
        if {"cloud", "out_of_swath"} <= set(piv.columns):
            ratio = piv["out_of_swath"] / piv["cloud"]
            lines.append(
                f"- **{channel}**: kriging RMSE in out-of-swath voids is "
                f"{ratio.min():.2f}-{ratio.max():.2f}x the cloud-gap RMSE "
                f"across variograms (cloud gaps are small holes ringed by "
                f"data; swath voids are wide and unconstrained).")
        else:
            lines.append(f"- **{channel}**: only strata "
                         f"{sorted(piv.columns)} present in this sample.")
    lines.append("")

    # Variogram winner via RMSE SKILL vs the nearest baseline, per channel.
    # Raw pooled RMSE must never be averaged across channels: T2M (~K) and
    # QV2M (~1e-3 kg/kg) are incommensurate, so a plain mean gives the
    # moisture channel -- the dryline signal carrier -- ~0.04 % of the vote
    # (review 2026-08-13).  skill = 1 - rmse_variogram / rmse_nearest is
    # unitless, so channels vote equally in the cross-channel mean.
    krige_overall = overall[overall["method"].str.startswith("krige_")]
    nearest_rmse = (overall[overall["method"] == "nearest"]
                    .set_index("channel")["pooled_rmse"])
    if len(krige_overall) and len(nearest_rmse):
        skill = krige_overall[["channel", "method", "pooled_rmse"]].copy()
        skill["skill_vs_nearest"] = (
            1.0 - skill["pooled_rmse"]
            / skill["channel"].map(nearest_rmse).to_numpy(float))
        per_channel = [
            f"- **{ch}**: best variogram "
            f"**{g.set_index('method')['skill_vs_nearest'].idxmax()}**"
            for ch, g in skill.groupby("channel")]
        winner = (skill.groupby("method")["skill_vs_nearest"]
                  .mean().idxmax())
        lines += ["## Variogram winner", "",
                  "Skill vs the nearest-observed baseline, per channel "
                  "(skill = 1 - RMSE_variogram / RMSE_nearest; unitless, "
                  "so K and kg/kg channels vote equally):", "",
                  _md_table(skill.sort_values(["channel", "method"])), "",
                  *per_channel, "",
                  f"Highest channel-mean skill: **{winner}**.  If this "
                  f"is not the configured model "
                  f"(`kriging.variogram_model: {config.KRIGE_VARIOGRAM}` in "
                  f"configs/dl_front.yaml), swap it there before building "
                  f"the kriged caches -- but check the per-channel table "
                  f"first: never trade QV2M skill for a marginal T2M win.",
                  ""]

    lines += ["## Swath-projection race (IoU vs actual envelope)", ""]
    if len(proj):
        tab = (proj.groupby("method")[["iou", "ms"]].mean()
               .reset_index())
        lines += [_md_table(tab, floatfmt=".3f"), ""]
        n_days = proj["date"].nunique()
        if n_days < 10:
            lines += [f"NOTE: only {n_days} sampled day(s) had real "
                      f"fullgrid files -- IoU ranking is anecdotal at this "
                      f"n; rerun on the JPL archive.", ""]
    else:
        lines += ["No sampled date had a real fullgrid file (local run "
                  "without the JPL archive) -- projection race skipped.", ""]

    lines += ["## Caveats", "",
              f"- Gap masks: {bank_note}.",
              f"- {metrics['date'].nunique()} day(s), hours "
              f"{sorted(int(h) for h in metrics['hour'].unique())} -- "
              f"pooled numbers "
              f"stabilise around 40 days.",
              "- Local runs lack the JPL fullgrid archive: gap geometry "
              "then comes from the harvested gap bank and the swath bank "
              "may be absent (per-day envelope fallback), so the "
              "cloud/out-of-swath split is approximate.", ""]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Study driver
# --------------------------------------------------------------------------- #

def run_study(years, n_days: int, hours=None, variograms=DEFAULT_VARIOGRAMS,
              panels: int = 6, allow_small_bank: bool = False,
              out_dir=None, seed: int = 20260813) -> Path:
    """The full validation study; returns the output directory."""
    hours = tuple(config.AIRS_HOURS if hours is None else hours)
    variograms = tuple(variograms)
    out_dir = Path(config.RESULTS_DIR / "krige_validation"
                   if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A rerun (different seed/years/n-days) overwrites metrics.csv and
    # summary.md but samples different panel cases, so stale PNGs from the
    # previous configuration would sit beside the new metrics as if they
    # belonged to it.  Clear ONLY the files this study itself writes
    # (panel_*.png) -- anything else a user parked in panels/ survives
    # (review 2026-08-13).
    for stale in sorted((out_dir / "panels").glob("panel_*.png")):
        stale.unlink()
    # Fill over the crop (box + halo, what the cache builders fill); score
    # only held-out analysis-domain pixels (user decision 2026-08-13).
    crop = crop_domain()
    analysis = analysis_domain()
    dates = sample_study_days(years, n_days, hours, seed)
    print(f"sampled {len(dates)} day(s): "
          f"{', '.join(f'{d:%Y-%m-%d}' for d in dates)}", flush=True)

    rows, cases = [], []
    n_steps, n_bank_steps = 0, 0
    for date in dates:
        for hour in hours:
            when = krige_fill._step_timestamp(date, hour)
            rea = krige_fill._reanalysis_step(when)
            if rea is None:
                print(f"{date:%Y-%m-%d} {hour:02d}Z: no reanalysis step, "
                      f"skipped", flush=True)
                continue
            valid_frac, note, used_bank = krige_fill._gap_valid_frac(
                date, hour, allow_small_bank)
            if note:
                print(f"{date:%Y-%m-%d} {note}", flush=True)
            # halo observations are USED, like the cache builders
            observed = swath.observed_mask(valid_frac) & crop
            if not observed.any():
                print(f"{date:%Y-%m-%d} {hour:02d}Z: zero observed pixels "
                      f"inside crop, skipped", flush=True)
                continue
            n_steps += 1
            n_bank_steps += int(used_bank)
            # bank draws are DONOR-date swaths: classify against their own
            # envelope, not this date's climatology (review 2026-08-13;
            # see krige_fill._classify_step_gaps)
            gap_type = krige_fill._classify_step_gaps(
                date, hour, valid_frac, crop, used_bank)
            for channel in config.KRIGED_CHANNELS:
                truth = rea[channel].values.astype(np.float64)
                case_rows = evaluate_case(truth, observed, crop, analysis,
                                          gap_type, variograms, date, hour,
                                          channel)
                rows.extend(case_rows)
                cases.append({"date": date, "hour": hour, "channel": channel,
                              "truth": truth, "observed": observed,
                              "valid_frac": valid_frac,
                              "used_bank": used_bank})
                solves = [r for r in case_rows
                          if r["stratum_kind"] == "overall"]
                print(f"{date:%Y-%m-%d} {hour:02d}Z {channel}: "
                      + ", ".join(f"{r['method']} rmse={r['rmse']:.3g} "
                                  f"({r['ms']:.0f} ms)" for r in solves),
                      flush=True)

    metrics = pd.DataFrame(rows, columns=list(METRIC_COLUMNS))
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    if not len(metrics):
        raise RuntimeError("no case produced metrics -- every sampled step "
                           "was missing or unobserved; widen --years")

    # Swath-projection race on the sampled dates with REAL fullgrid files
    # (compare_projections itself skips dates without one).
    proj = swath.compare_projections(dates)
    proj.to_csv(out_dir / "projection_methods.csv", index=False)

    bank = swath.load_swath_bank()
    bank_note = (f"real fullgrid masks on {n_steps - n_bank_steps} of "
                 f"{n_steps} scored step(s); gap-bank draws on "
                 f"{n_bank_steps} (donor-date swaths, classified against "
                 f"their own per-day envelope); swath bank "
                 + (str(config.SWATH_BANK_PATH) if bank is not None
                    else "ABSENT (per-day morphological envelope fallback)"))
    args_note = (f"Years {min(years)}-{max(years)}, --n-days {n_days} "
                 f"requested -> {len(dates)} sampled day(s), hours "
                 f"{list(hours)}, variograms "
                 f"{list(variograms)}, seed {seed}.  Truth = clean MERRA-2 "
                 f"reanalysis; fill over the box+halo crop, scores over "
                 f"held-out ANALYSIS-domain pixels only (user decision "
                 f"2026-08-13).")
    write_summary(metrics, proj, out_dir, args_note, bank_note)

    # 3-panel maps for seeded random cases, config variogram
    # (the model the caches are actually built with).
    rng = np.random.default_rng([seed, 1])
    for i in rng.choice(len(cases), size=min(panels, len(cases)),
                        replace=False):
        c = cases[i]
        target = crop & ~c["observed"]
        field = np.where(c["observed"], c["truth"], np.nan)
        kriged = krige_fill.krige_fill(
            field, rng=krige_fill._step_rng(c["date"], c["hour"],
                                            c["channel"]), target=target)
        # bank-draw cases show their own envelope (donor-date swath), same
        # rule as the metrics' gap classification (review 2026-08-13)
        envelope = (None if c["used_bank"]
                    else swath.expected_swath(c["date"], c["hour"]))
        if envelope is None:
            envelope = swath.swath_envelope(
                swath.observed_mask(c["valid_frac"]))
        out_png = (out_dir / "panels" /
                   f"panel_{c['date']:%Y-%m-%d}_{c['hour']:02d}Z_"
                   f"{c['channel']}.png")
        render_panel(c["truth"], kriged, c["observed"], envelope, analysis,
                     c["channel"], c["date"], c["hour"], out_png, crop=crop)
        print(f"wrote {out_png}", flush=True)
    print(f"wrote {out_dir}/metrics.csv, projection_methods.csv, summary.md",
          flush=True)
    return out_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", required=True, type=parse_years,
                    help="'2007-2015' or '2007,2010'")
    ap.add_argument("--n-days", required=True, type=int,
                    help="sampled overpass days (seeded, month-spread)")
    ap.add_argument("--hours", default=None,
                    help=f"comma UTC hours (default {config.AIRS_HOURS})")
    ap.add_argument("--variograms",
                    default=",".join(DEFAULT_VARIOGRAMS),
                    help="comma list of pykrige variogram models to race")
    ap.add_argument("--panels", type=int, default=6,
                    help="number of 3-panel case maps")
    ap.add_argument("--allow-small-bank", action="store_true",
                    help="accept a gap bank below MIN_REAL_BANK (local "
                         "smoke runs without the JPL archive)")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/dl_front/"
                         "krige_validation)")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args(argv)
    hours = (tuple(int(h) for h in args.hours.split(","))
             if args.hours else None)
    run_study(args.years, args.n_days, hours=hours,
              variograms=tuple(args.variograms.split(",")),
              panels=args.panels, allow_small_bank=args.allow_small_bank,
              out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
