"""dl_front.six_panel: truth / BK19 / checkpoint comparison maps.

For a random sample of AIRS-hour (config.AIRS_HOURS = 21Z / next-day 00Z)
timestamps in March-November 2016-2018, renders one 2-row x 3-col PNG per
sampled instant::

    row 0:  met-drawn truth      | D6A on kriged-AIRS  | D6C on kriged-AIRS
    row 1:  BK19 published pred  | D6A on reanalysis   | D6C on reanalysis

D6A-f<fold> is the reanalysis-training-complete checkpoint (stage A only,
before any AIRS-simulator fine-tuning); D6C-f<fold> is the full-curriculum
checkpoint (stage A -> B -> C).  Reading a column shows how each checkpoint's
own prediction changes between its native/clean input and the real-AIRS-
shaped input; reading a row shows curriculum progress at one input type.

Sample selection: only timestamps present in ALL THREE of reanalysis (at
AIRS_HOURS), the kriged-AIRS cache and the published BK19 archive qualify --
the same identical-sample-set discipline evaluate_test.compare() enforces,
so every panel in a figure depicts the exact same instant.  Requires the
cluster's full data (reanalysis sfc_daily, kriged-airs cache, BK19 archive)
and the checkpoints under $JPL_AIRS_RESULTS/dl_front/models -- run this on
the cluster (fronts-tf env), not against a local partial checkout.

Usage::

    PYTHONPATH=src python -m dl_front.six_panel --fold 0 --n-days 12 \\
        --out-dir results/dl_front/quicklook/six_panel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, dataset, evaluate_test, predict
from .quicklook import (INK, OUT_OF_DOMAIN_GRAY, _edge_extent, _style_axis,
                        _window)

MONTHS = tuple(range(3, 12))          # March .. November inclusive
YEARS = (2016, 2017, 2018)            # BK19-comparable test years

#: Meteorological convention (matches notebooks/13_codsus_vs_noaa_plots.py
#: TYPE_COLOR).  These five hues also coincide with slots 1/8/6/7/2 of the
#: dataviz skill's accessibility-validated categorical palette (blue/red/
#: green/violet/orange), so the domain-standard front-color convention and
#: the CVD-safe validated set agree here -- kept as-is rather than
#: substituted, per the skill's "identity never color-alone" rule the
#: legend below still carries text labels for every swatch.
FRONT_COLORS = {
    "cold": "#2a78d6", "warm": "#e34948", "stationary": "#008300",
    "occluded": "#4a3aa7", "dryline": "#eb6834",
}
NONE_COLOR = "#f2f1ec"                # near-white: recessive, not a hue


def _class_cmap(n_classes: int):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    names = dataset.class_names(n_classes)
    colors = [FRONT_COLORS[n] for n in names[:-1]] + [NONE_COLOR]
    cmap = ListedColormap(colors)
    cmap.set_bad(OUT_OF_DOMAIN_GRAY)
    norm = BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names))
    return cmap, norm, names


def _mask_outside(cls: np.ndarray) -> np.ndarray:
    """float copy of a class-index grid, NaN outside the analysis domain
    (nothing was trained/scored there; graying it avoids reading noise in
    the halo as a real prediction)."""
    out = cls.astype(np.float32)
    out[~dataset.analysis_domain()] = np.nan
    return out


def _load_sources(year: int, n_classes: int, stats: dict) -> dict:
    """One year's (x, y, times) for reanalysis (restricted to AIRS_HOURS),
    kriged-airs (already AIRS_HOURS-only by construction) and bk19."""
    x, y, t = evaluate_test.load_year(year, n_classes, stats, "reanalysis")
    x, y, t = dataset.filter_hours(x, y, t, config.AIRS_HOURS)
    xa, ya, ta = evaluate_test.load_year(year, n_classes, stats, "kriged-airs")
    xb, yb, tb = evaluate_test.bk19_year_arrays(year, n_classes)
    return {"reanalysis": (x, y, t), "kriged-airs": (xa, ya, ta),
            "bk19": (xb, yb, tb)}


def sample_pool(n_classes: int, stats: dict) -> tuple[pd.DataFrame, dict]:
    """Candidate (year, time) rows common to all three sources, Mar-Nov
    only, plus the loaded per-year source dict (reused for every draw)."""
    rows, year_sources = [], {}
    for year in tqdm(YEARS, desc="loading years", unit="yr"):
        src = _load_sources(year, n_classes, stats)
        year_sources[year] = src
        common = src["reanalysis"][2]
        for other in ("kriged-airs", "bk19"):
            common = common.intersection(src[other][2])
        common = common[common.month.isin(MONTHS)]
        rows.extend((year, t) for t in common)
    if not rows:
        raise RuntimeError(
            "no timestamps common to reanalysis/kriged-airs/bk19 in "
            f"{YEARS} months {MONTHS[0]}-{MONTHS[-1]}; check the caches "
            "and BK19 archive are all present for these years")
    pool = pd.DataFrame(rows, columns=["year", "time"])
    return pool, year_sources


def render(when: pd.Timestamp, truth: np.ndarray, bk19_pred: np.ndarray,
          a_air: np.ndarray, a_rea: np.ndarray,
          c_air: np.ndarray, c_rea: np.ndarray,
          n_classes: int, fold: int, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap, norm, names = _class_cmap(n_classes)
    # The panel arrays are FULL 68x141 label grids, so imshow must be given
    # the full-grid edge extent (quicklook._edge_extent); handing it the crop
    # window instead squeezed the whole -171..-31 E / 10..77 N grid into the
    # crop box, which shifted every front tens of degrees and shrank the
    # analysis domain to a staircase blob over the east coast.  Zoom with the
    # axis limits instead.
    extent = _edge_extent()
    crop_box = _window(dataset.crop_domain())
    domain_box = _window(dataset.analysis_domain())

    panels = [
        ("met-drawn truth", truth),
        (f"D6A-f{fold} on kriged-AIRS", a_air),
        (f"D6C-f{fold} on kriged-AIRS", c_air),
        ("BK19 published prediction", bk19_pred),
        (f"D6A-f{fold} on reanalysis", a_rea),
        (f"D6C-f{fold} on reanalysis", c_rea),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True)
    for ax, (title, cls) in zip(axes.flat, panels):
        ax.imshow(_mask_outside(cls), origin="lower", extent=extent,
                  cmap=cmap, norm=norm, interpolation="nearest")
        ax.plot([domain_box[0], domain_box[1], domain_box[1], domain_box[0],
                 domain_box[0]],
                [domain_box[2], domain_box[2], domain_box[3], domain_box[3],
                 domain_box[2]],
                color=INK, lw=0.6, ls="--")
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlim(crop_box[0], crop_box[1])
        ax.set_ylim(crop_box[2], crop_box[3])
        _style_axis(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FRONT_COLORS[n])
              for n in names[:-1]] + \
              [plt.Rectangle((0, 0), 1, 1, color=NONE_COLOR,
                             ec="0.6", lw=0.5)]
    fig.legend(handles, list(names), loc="lower center", ncol=len(names),
              frameon=False, fontsize=8, labelcolor=INK,
              bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{when:%Y-%m-%d %H:%MZ}  (fold {fold}, dashed = analysis "
                f"domain, gray = outside it)", color=INK, fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"six_panel_f{fold}_{when:%Y%m%d_%H%M}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--n-days", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260816,
                     help="RNG seed for the day sample (default reproduces "
                          "the same draw across reruns)")
    ap.add_argument("--out-dir", default=None,
                     help=f"default {config.RESULTS_DIR}/quicklook/six_panel")
    ap.add_argument("--channels", default=None,
                     help="comma-separated input channels BOTH checkpoints "
                          f"consume, a subset of {','.join(config.SFC_VARS)} "
                          "(default: adopt run_args.channels from the D6A "
                          "checkpoint's run_config.yaml, else the yaml "
                          "inputs.channels).  D6A and D6C are fed the SAME "
                          "loaded x arrays below, so they must agree on "
                          "channels; a reduced-channel ladder checkpoint or "
                          "a same-arity relabelling is refused here with an "
                          "actionable message instead of a raw Keras shape "
                          "error further down (review 2026-08-18, "
                          "consumers)")
    a = ap.parse_args(argv)

    out_dir = (Path(a.out_dir) if a.out_dir is not None
              else config.RESULTS_DIR / "quicklook/six_panel")
    models_dir = config.RESULTS_DIR / "models"
    ckpt_a = models_dir / f"D6A-f{a.fold}" / f"D6A-f{a.fold}.h5"
    ckpt_c = models_dir / f"D6C-f{a.fold}" / f"D6C-f{a.fold}.h5"
    # Channels first (evaluate_test.resolve_channels/check_model_channels,
    # reused rather than reimplemented, review 2026-08-18): install the
    # resolved channel tuple BEFORE sample_pool loads any x array below, then
    # verify BOTH checkpoints against it -- a mismatch (either checkpoint)
    # raises the checkpoint's own actionable fix instead of Keras's raw
    # "expected shape" error once .predict() is finally called.
    evaluate_test.resolve_channels(str(ckpt_a), a.channels)
    m_a = predict.load_model(ckpt_a)
    m_c = predict.load_model(ckpt_c)
    evaluate_test.check_model_channels(m_a, ckpt_a)
    evaluate_test.check_model_channels(m_c, ckpt_c)

    stats = dataset.load_norm_stats()
    pool, year_sources = sample_pool(a.classes, stats)
    n = min(a.n_days, len(pool))
    if n < a.n_days:
        print(f"only {n} common timestamps available (requested {a.n_days})",
              flush=True)
    rng = np.random.default_rng(a.seed)
    chosen = pool.iloc[rng.choice(len(pool), size=n, replace=False)] \
                .sort_values(["year", "time"])

    written = []
    steps = tqdm(list(zip(chosen["year"], chosen["time"])),
                desc="rendering", unit="fig")
    for year, when in steps:
        steps.set_postfix_str(f"{when:%Y-%m-%d %H:%MZ}")
        src = year_sources[year]
        xr_, yr_, tr_ = src["reanalysis"]
        xa_, ya_, ta_ = src["kriged-airs"]
        xb_, yb_, tb_ = src["bk19"]
        ir, ia, ib = tr_.get_loc(when), ta_.get_loc(when), tb_.get_loc(when)

        truth = yr_[ir]
        bk19_pred = np.rint(xb_[ib][..., 0]).astype(np.int64)
        a_air = m_a.predict(xa_[ia:ia + 1].astype(np.float32),
                           verbose=0)[0].argmax(-1)
        a_rea = m_a.predict(xr_[ir:ir + 1].astype(np.float32),
                           verbose=0)[0].argmax(-1)
        c_air = m_c.predict(xa_[ia:ia + 1].astype(np.float32),
                           verbose=0)[0].argmax(-1)
        c_rea = m_c.predict(xr_[ir:ir + 1].astype(np.float32),
                           verbose=0)[0].argmax(-1)

        path = render(when, truth, bk19_pred, a_air, a_rea, c_air, c_rea,
                     a.classes, a.fold, out_dir)
        written.append(path)
        steps.write(f"{when:%Y-%m-%d %H:%MZ}: wrote {path}")

    print(f"wrote {len(written)} six-panel figures to {out_dir}", flush=True)
    return written


if __name__ == "__main__":
    main()
