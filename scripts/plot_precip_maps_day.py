#!/usr/bin/env python
"""Hourly MRMS precipitation maps for one day's forecast window.

Draws the 7 forecast slots of a given date -- the AIRS overpass hour (from the
daily ``qpe_overpass`` column; the row table itself carries slots 1-6) and the
fixed 21,22,23,00,01,02 UTC valid times (the 00-02 UTC panels are the early
hours of the NEXT calendar day) -- as CONUS panels of cell-mean QPE on a log
color scale. Reads the cached base table, so it works without the raw netCDFs.

Usage::

    python scripts/plot_precip_maps_day.py 2019-06-05
    python scripts/plot_precip_maps_day.py 2019-06-05 --point 39.42 -83.83
    python scripts/plot_precip_maps_day.py 2020-07-26 --out results/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from convection_skill import config  # noqa: E402
from convection_skill import plotting as cp  # noqa: E402
from convection_skill.config import AnalysisConfig  # noqa: E402

SLOT_TITLE = {0: "overpass (~18-20 UTC)", 1: "21 UTC", 2: "22 UTC",
              3: "23 UTC", 4: "00 UTC (+1 d)", 5: "01 UTC (+1 d)",
              6: "02 UTC (+1 d)"}
WET_MIN_MM = 0.01  # cells at/below this render transparent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("date", help="day to plot, e.g. 2019-06-05")
    ap.add_argument("--point", type=float, nargs=2, metavar=("LAT", "LON"),
                    help="mark this location with a star on every panel")
    ap.add_argument("--out", type=Path,
                    default=config.RESULTS_DIR / "figures")
    args = ap.parse_args(argv)
    day_ts = pd.Timestamp(args.date)

    # base table via the standard builder, stale-cache fallback included
    sys.path.insert(0, str(REPO / "scripts"))
    from extreme_cell_climatology import load_base_table
    cfg = AnalysisConfig(years=(day_ts.year,))
    base = load_base_table(cfg)
    day = base[base["date"] == day_ts]
    if day.empty:
        raise SystemExit(f"no rows for {args.date} in the base table")

    import cartopy.crs as ccrs
    lats = np.sort(day["lat"].unique())
    lons = np.sort(day["lon"].unique())
    lat_e = np.append(lats - 0.5, lats[-1] + 0.5)
    lon_e = np.append(lons - 0.5, lons[-1] + 0.5)

    def cell_grid(sub: pd.DataFrame, col: str) -> np.ndarray:
        grid = np.full((lats.size, lons.size), np.nan)
        grid[((sub["lat"] - lats[0]).round().astype(int),
              (sub["lon"] - lons[0]).round().astype(int))] = sub[col]
        return np.where(grid > WET_MIN_MM, grid, np.nan)

    norm = mcolors.LogNorm(
        vmin=WET_MIN_MM,
        vmax=max(float(np.nanmax(day[["qpe", "qpe_overpass"]])), 1.0))
    slots = sorted(day["slot"].unique())
    any_slot = day[day["slot"] == slots[0]]  # daily columns: one row per cell

    fig = plt.figure(figsize=(19, 5.8))
    panels = [(0, "qpe_overpass", any_slot)] + [
        (s, "qpe", day[day["slot"] == s]) for s in slots]
    for i, (slot, col, sub) in enumerate(panels):
        ax = cp.make_conus_axes(fig=fig, rect=(2, 4, i + 1))
        mesh = ax.pcolormesh(lon_e, lat_e, cell_grid(sub, col), cmap="turbo",
                             norm=norm, transform=ccrs.PlateCarree(), zorder=3)
        if args.point:
            ax.plot(args.point[1], args.point[0], marker="*", ms=14,
                    mec="k", mfc="cyan", transform=ccrs.PlateCarree(), zorder=5)
        ax.set_title(f"slot {slot}: {SLOT_TITLE[slot]}", fontsize=10)
    cax = fig.add_axes([0.76, 0.10, 0.012, 0.32])  # the empty 8th grid cell
    fig.colorbar(mesh, cax=cax, label="MRMS cell-mean QPE (mm/h)\n"
                                      f"(cells > {WET_MIN_MM:g} mm/h shown)")
    fig.subplots_adjust(wspace=0.08, hspace=0.18)
    star = (f" (star = {args.point[0]:g}N, {abs(args.point[1]):g}W)"
            if args.point else "")
    fig.suptitle(f"Hourly precipitation, {day_ts:%Y-%m-%d} forecast window"
                 + star, fontsize=14, y=1.02)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"precip_maps_{day_ts:%Y%m%d}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
