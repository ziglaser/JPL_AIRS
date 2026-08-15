"""dl_front.quicklook: spot-check PNGs for the three CPU-built products.

The kriged caches and the swath bank are consumed only as tensors by the
training/eval code, so a build mistake (wrong domain, empty swaths, a
mis-scaled channel, gap_type garbage) would otherwise surface as nothing
more than a bad loss curve.  This module renders a handful of human-checkable
maps per product; the chain script (scripts/dlfront_jpl_chain.sh) runs each
subcommand right after the corresponding build phase:

* ``swath-bank``     -> one PNG per period hour: the 16 cycle-day coverage-
  frequency maps from ``masks/swath_bank.npz`` with the footprint threshold
  (config.SWATH_MIN_FRACTION) drawn as a contour -- the "16-day cycle swath
  maps".
* ``kriged-degraded`` / ``kriged-airs`` -> one PNG per sampled cache step:
  all five kriged surface channels plus the gap_type decomposition, framed
  to the crop window with the analysis domain outlined.

Sampling is deterministic (evenly spaced over the caches' pooled time axis,
no RNG), so reruns overwrite the same filenames and two builds of the same
cache produce byte-comparable spot checks.

Styling follows krige_validate.render_panel (dataviz rules): magnitude
channels (T2M/QV2M/SLP) on one-hue viridis; signed channels (U10M/V10M) on
diverging RdBu_r symmetric about a neutral 0; gap_type as a FIXED-order
categorical palette (Paul Tol "bright" -- colorblind-safe by construction)
with a text-labeled legend so identity is never carried by color alone;
NaN/out-of-domain in light gray; text in neutral dark ink; Agg backend.

Usage::

    python -m dl_front.quicklook swath-bank      [--path P] [--out-dir D]
    python -m dl_front.quicklook kriged-degraded [--years 2007-2015] [--n 6]
    python -m dl_front.quicklook kriged-airs     [--years 2007-2021] [--n 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, swath
from .krige_fill import parse_years

INK = "0.2"                      # neutral dark text/contour ink
OUT_OF_DOMAIN_GRAY = "0.88"      # matches krige_validate.render_panel

#: Display units for the cached surface channels (MERRA-2 native units;
#: values are shown untransformed so a reader can cross-check the source
#: files directly -- SLP deliberately stays in Pa).
CHANNEL_UNITS = {"T2M": "K", "QV2M": "kg/kg", "SLP": "Pa",
                 "U10M": "m/s", "V10M": "m/s"}

#: gap_type categorical palette in FIXED value order -1/0/1/2
#: (config.GAP_OUT_OF_DOMAIN/_OBSERVED/_CLOUD/_OUT_OF_SWATH).  Paul Tol's
#: "bright" scheme (https://sronpersonalpages.nl/~pault/): designed to stay
#: distinct under deuteranopia/protanopia; the legend carries the labels.
GAP_COLORS = ("#DDDDDD",   # -1 out-of-domain (recessive gray, not a hue)
              "#4477AA",   # 0  observed (AIRS retrieval present)
              "#EE7733",   # 1  cloud gap (in-swath, kriged)
              "#AA3377")   # 2  out-of-swath (orbit geometry)
GAP_LABELS = ("out-of-domain", "observed", "cloud gap", "out-of-swath")


def _edge_extent() -> tuple[float, float, float, float]:
    """imshow/contour extent at cell EDGES (see krige_validate 2026-08-13:
    grid values are cell centers on integer degrees; a center-based extent
    shifts features up to ~0.5 deg against the axis labels)."""
    lats, lons = config.LABEL_LATS, config.LABEL_LONS
    return (lons[0] - 0.5, lons[-1] + 0.5, lats[0] - 0.5, lats[-1] + 0.5)


def _window(mask: np.ndarray) -> tuple[float, float, float, float]:
    """(lon_lo, lon_hi, lat_lo, lat_hi) outer-edge frame of a bool mask."""
    lats, lons = config.LABEL_LATS, config.LABEL_LONS
    rows, cols = np.nonzero(np.asarray(mask, bool))
    return (lons[cols.min()] - 0.5, lons[cols.max()] + 0.5,
            lats[rows.min()] - 0.5, lats[rows.max()] + 0.5)


def _style_axis(ax) -> None:
    ax.tick_params(colors=INK, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("0.6")


# --------------------------------------------------------------------------- #
# swath bank: 16 cycle-day coverage maps per hour
# --------------------------------------------------------------------------- #

def render_swath_bank(path: Path = None, out_dir: Path = None) -> list[Path]:
    """One PNG per composited hour: a 4x4 grid of the 16 cycle-day coverage
    frequencies, one shared 0..1 viridis scale, the expected-swath footprint
    (freq >= SWATH_MIN_FRACTION, the read-time threshold) as a solid contour
    and the crop window as a dashed box.  Panel titles carry the per-bin day
    count so an undersampled cycle day (< swath.MIN_DAYS_PER_CYCLE_DAY,
    where expected_swath falls back to the per-day envelope) is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = swath.load_swath_bank(path)
    if bank is None:
        raise FileNotFoundError(
            f"no swath bank at {Path(config.SWATH_BANK_PATH if path is None else path)}"
            " -- build it first: python -m dl_front.swath build-bank --years ...")
    out_dir = Path(config.RESULTS_DIR / "quicklook/swath_bank"
                   if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .dataset import crop_domain
    extent = _edge_extent()
    crop_box = _window(crop_domain())
    years = bank["years"].tolist()
    written = []
    for h_idx, hour in enumerate(bank["hours"].tolist()):
        fig, axes = plt.subplots(4, 4, figsize=(16, 8.5), dpi=150,
                                 sharex=True, sharey=True)
        for cyc, ax in enumerate(axes.ravel()):
            freq = bank["freq"][cyc, h_idx]
            n = int(bank["n_days"][cyc, h_idx])
            im = ax.imshow(freq, origin="lower", extent=extent,
                           cmap="viridis", vmin=0.0, vmax=1.0,
                           aspect="auto", interpolation="nearest")
            ax.contour(freq, levels=[config.SWATH_MIN_FRACTION],
                       colors=["#EE7733"], linewidths=0.8,
                       extent=extent, origin="lower")
            ax.plot([crop_box[0], crop_box[1], crop_box[1], crop_box[0],
                     crop_box[0]],
                    [crop_box[2], crop_box[2], crop_box[3], crop_box[3],
                     crop_box[2]],
                    color=INK, linewidth=0.6, linestyle="dashed")
            flag = ("" if n >= swath.MIN_DAYS_PER_CYCLE_DAY
                    else "  (UNDERSAMPLED)")
            ax.set_title(f"cycle day {cyc}  (n={n}){flag}",
                         color=INK, fontsize=8)
            _style_axis(ax)
        cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
        cb.set_label("coverage frequency", color=INK, fontsize=9)
        cb.ax.tick_params(colors=INK, labelsize=7)
        fig.suptitle(
            f"AIRS 16-day-cycle swath climatology, {hour:02d}Z "
            f"(years {years[0]}-{years[-1]}; footprint contour at "
            f"freq >= {config.SWATH_MIN_FRACTION}; dashed = crop window)",
            color=INK, fontsize=11)
        out_png = out_dir / f"swath_bank_{hour:02d}Z.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_png}", flush=True)
        written.append(out_png)
    return written


# --------------------------------------------------------------------------- #
# kriged caches: sampled per-step channel + gap_type maps
# --------------------------------------------------------------------------- #

def sample_steps(source: str, years=None,
                 n: int = 6, cache_dir: Path = None) -> list[tuple[Path, int]]:
    """Deterministic (file, time-index) sample: ``n`` steps evenly spaced
    over the pooled time axis of every requested year cache, so the sample
    spans years AND seasons without any RNG.  Empty (time=0) year caches --
    legitimate done-markers, see krige_fill._write_year_cache -- contribute
    nothing.
    """
    cache_dir = Path(config.KRIGED_SOURCE_DIRS[source]
                     if cache_dir is None else cache_dir)
    paths = sorted(cache_dir.glob("kriged_sfc_*.nc"))
    if years is not None:
        keep = {int(y) for y in years}
        paths = [p for p in paths if int(p.stem.rsplit("_", 1)[1]) in keep]
    if not paths:
        raise FileNotFoundError(
            f"no kriged_sfc_*.nc caches under {cache_dir}"
            f"{f' for years {sorted(keep)}' if years is not None else ''}")
    pool = []                       # (path, local index) over all steps
    for p in paths:
        with xr.open_dataset(p) as ds:
            pool.extend((p, t) for t in range(ds.sizes["time"]))
    if not pool:
        raise ValueError(f"every cache under {cache_dir} is empty (time=0)")
    picks = np.unique(np.linspace(0, len(pool) - 1,
                                  min(n, len(pool))).round().astype(int))
    return [pool[i] for i in picks]


def render_step(path: Path, t: int, source: str, out_dir: Path) -> Path:
    """One sampled step -> a 2x3 panel PNG: T2M / QV2M / SLP (sequential
    viridis, per-channel scale), U10M / V10M (diverging RdBu_r symmetric
    about 0), gap_type (fixed categorical palette + labeled legend).  Every
    panel outlines the analysis domain (solid ink) and is framed to the
    crop window; the field panels also contour the observed footprint
    (gap_type == GAP_OBSERVED) so the eye can compare real vs kriged
    texture directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from .dataset import analysis_domain, crop_domain

    with xr.open_dataset(path) as ds:
        step = ds.isel(time=t).load()
    when = pd.Timestamp(step["time"].values)
    gap = step["gap_type"].values
    observed = (gap == config.GAP_OBSERVED).astype(float)
    domain = analysis_domain().astype(float)
    extent = _edge_extent()
    window = _window(crop_domain())

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), dpi=150,
                             sharex=True, sharey=True)
    field_axes = axes.ravel()[:5]
    for ax, var in zip(field_axes, config.SFC_VARS):
        grid = np.ma.masked_invalid(step[var].values)
        if var in ("U10M", "V10M"):          # signed -> diverging about 0
            vmax = float(np.abs(grid).max()) or 1e-12
            cmap, vmin = "RdBu_r", -vmax
        else:                                # magnitude -> one-hue sequential
            cmap = "viridis"
            vmin, vmax = float(grid.min()), float(grid.max())
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(OUT_OF_DOMAIN_GRAY)       # NaN outside crop (schema v3)
        im = ax.imshow(grid, origin="lower", extent=extent, cmap=cm,
                       vmin=vmin, vmax=vmax, aspect="auto",
                       interpolation="nearest")
        ax.contour(observed, levels=[0.5], colors=[INK], linewidths=0.4,
                   extent=extent, origin="lower")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(f"{var} [{CHANNEL_UNITS[var]}]", color=INK, fontsize=8)
        cb.ax.tick_params(colors=INK, labelsize=7)
        ax.set_title(var, color=INK, fontsize=10)

    gax = axes.ravel()[5]
    bounds = (config.GAP_OUT_OF_DOMAIN - 0.5, config.GAP_OBSERVED - 0.5,
              config.GAP_CLOUD - 0.5, config.GAP_OUT_OF_SWATH - 0.5,
              config.GAP_OUT_OF_SWATH + 0.5)
    gax.imshow(gap, origin="lower", extent=extent,
               cmap=ListedColormap(GAP_COLORS),
               norm=BoundaryNorm(bounds, len(GAP_COLORS)),
               aspect="auto", interpolation="nearest")
    gax.legend(handles=[Patch(facecolor=c, label=l)
                        for c, l in zip(GAP_COLORS, GAP_LABELS)],
               loc="lower left", fontsize=7, framealpha=0.9)
    gax.set_title("gap_type", color=INK, fontsize=10)

    for ax in axes.ravel():
        ax.contour(domain, levels=[0.5], colors=[INK], linewidths=0.7,
                   extent=extent, origin="lower")
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        _style_axis(ax)
    fig.suptitle(f"{source} spot check: {when:%Y-%m-%d %H}Z   "
                 f"({path.name}; solid outline = analysis domain, "
                 f"thin contour = observed AIRS)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = out_dir / f"{source}_{when:%Y%m%d_%H}Z.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)
    return out_png


def render_cache(source: str, years=None, n: int = 6,
                 cache_dir: Path = None, out_dir: Path = None) -> list[Path]:
    """Sample ``n`` steps from one cache flavor and render each."""
    out_dir = Path(config.RESULTS_DIR / f"quicklook/{source}"
                   if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [render_step(p, t, source, out_dir)
            for p, t in sample_steps(source, years, n, cache_dir)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m dl_front.quicklook", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    bank = sub.add_parser("swath-bank",
                          help="render the 16-day-cycle swath coverage maps")
    bank.add_argument("--path", default=None,
                      help=f"bank npz (default {config.SWATH_BANK_PATH})")
    bank.add_argument("--out-dir", default=None)

    for source in config.KRIGED_SOURCE_DIRS:   # kriged-degraded, kriged-airs
        p = sub.add_parser(source,
                           help=f"render sampled {source} cache steps")
        p.add_argument("--years", default=None, type=parse_years,
                       help="restrict to these cache years, e.g. 2007-2015 "
                            "(default: every cache file present)")
        p.add_argument("--n", type=int, default=6,
                       help="steps to sample, evenly spaced in time")
        p.add_argument("--cache-dir", default=None,
                       help="override the canonical cache directory")
        p.add_argument("--out-dir", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "swath-bank":
        render_swath_bank(path=None if args.path is None else Path(args.path),
                          out_dir=None if args.out_dir is None
                          else Path(args.out_dir))
    else:
        render_cache(args.cmd, years=args.years, n=args.n,
                     cache_dir=None if args.cache_dir is None
                     else Path(args.cache_dir),
                     out_dir=None if args.out_dir is None
                     else Path(args.out_dir))


if __name__ == "__main__":
    main()
