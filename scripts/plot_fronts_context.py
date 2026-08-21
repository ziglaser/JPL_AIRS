#!/usr/bin/env python
"""Frontal context around a point: analyst fronts over surface T / q / wind.

For each 3-hourly analysis time spanning a day's forecast window (15, 18, 21
UTC + 00, 03 UTC next day), draws a regional map pair:

- top row: MERRA-2 925-hPa temperature (shaded, deg C) + wind barbs;
- bottom row: MERRA-2 925-hPa specific humidity (shaded, g/kg) + wind barbs;
- both rows: WPC CODSUS analyst-drawn front cells (1-wide masks, regenerated
  from the IEM bulletin archive) as colored squares -- cold/warm/stationary/
  occluded -- and a star at the requested point.

The MERRA-2 compact corpus and the CODSUS masks live on the SAME integer-degree
grid and the SAME 3-hourly clock, so no regridding or time interpolation is
involved. 925 hPa (~750 m AGL over the Midwest) is used as the near-surface
level: the corpus carries no 2-m fields, and its 1000-hPa level is BELOW
ground (NaN) over most of the interior, where surface pressure is ~990 hPa.

Also prints the analysis cell's own front flags (the base table's 2x2 max-pool
convention: "a front touches this 1-deg cell", 1-wide and 3-wide) per analysis
time.

Usage::

    python scripts/plot_fronts_context.py 2019-06-05 --point 39.15 -89.33
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from convection_skill import config as cs_config  # noqa: E402
from trajectory_kernels import config as tk_config  # noqa: E402

MERRA2_DAILY = tk_config.DATA_DIR / "front_id" / "reanalysis" / "MERRA2" / "daily"
FRONTS_TPL = (tk_config.DATA_DIR / "front_id" / "met_drawn_fronts" /
              "WPC_CODSUS" / "WPC_1deg_gridded" / "{width}" /
              "codsus_masked_merra2-1deg_{width}_{year}.nc")

FRONT_COLORS = {"cold": "blue", "warm": "red",
                "stationary": "limegreen", "occluded": "purple"}
#: analysis hours drawn: window-bracketing 3-hourly times (last two are +1 day)
PANEL_HOURS = (15, 18, 21, 24, 27)


def open_year(tpl_year: int, width: str) -> xr.Dataset | None:
    path = Path(str(FRONTS_TPL).format(width=width, year=tpl_year))
    return xr.open_dataset(path) if path.exists() else None


def cell_front_flags(fronts: xr.Dataset, time: pd.Timestamp,
                     cell: tuple[float, float]) -> list[str]:
    """Front types touching the 1-deg analysis cell (2x2 max-pool, fronts.py)."""
    lat, lon = cell
    sub = fronts["fronts"].sel(time=time, method="nearest").sel(
        lat=slice(np.floor(lat), np.ceil(lat)),
        lon=slice(np.floor(lon), np.ceil(lon)))
    types = [str(t) for t in fronts["front_type"].values]
    return [t for i, t in enumerate(types)
            if t != "none" and bool(sub.isel(front=i).max() > 0)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("date", help="day whose forecast window to frame")
    ap.add_argument("--point", type=float, nargs=2, metavar=("LAT", "LON"),
                    required=True)
    ap.add_argument("--halfwidth", type=float, nargs=2, default=(8.0, 13.0),
                    metavar=("DLAT", "DLON"), help="map half-extents (deg)")
    ap.add_argument("--out", type=Path,
                    default=cs_config.RESULTS_DIR / "daily_reports")
    ap.add_argument("--single", type=float, default=None, metavar="HOURS",
                    help="ALSO write a standalone two-panel (T | q) figure "
                         "for the analysis time this many hours after the "
                         "date's 00Z (e.g. 24 = 00Z next day); colorbars and "
                         "lat/lon labels, no titles")
    args = ap.parse_args(argv)
    day = pd.Timestamp(args.date)
    plat, plon = args.point
    dlat, dlon = args.halfwidth
    box = dict(lat=slice(plat - dlat, plat + dlat),
               lon=slice(plon - dlon, plon + dlon))
    times = [day + pd.Timedelta(hours=h) for h in PANEL_HOURS]

    # MERRA-2 compact profiles for the day and the next (00/03 UTC panels)
    m2 = xr.concat(
        [xr.open_dataset(MERRA2_DAILY / f"{d:%Y}" / f"m2_{d:%Y%m%d}.nc")
         for d in (day, day + pd.Timedelta(days=1))
         if (MERRA2_DAILY / f"{d:%Y}" / f"m2_{d:%Y%m%d}.nc").exists()],
        dim="time").sel(lev=925.0, **box)
    # corrupt-transfer guard: some daily files carry QV identically 0 (bad
    # OPeNDAP bytes that pass the acquirer's bounds check); 0 is not a
    # physical 925-hPa humidity anywhere in this domain -> treat as missing
    n_zero = int((m2["QV"].values == 0).sum())
    if n_zero:
        print(f"WARNING: {n_zero} QV==0 samples masked as missing "
              "(corrupt daily file -- consider re-downloading that day)")
        m2["QV"] = m2["QV"].where(m2["QV"] > 0)
    # sea-level pressure from the surface corpus (contours on both rows)
    slp_files = [MERRA2_DAILY.parent / "sfc_daily" / f"{d:%Y}" /
                 f"m2sfc_{d:%Y%m%d}.nc"
                 for d in (day, day + pd.Timedelta(days=1))]
    slp = None
    if all(f.exists() for f in slp_files):
        slp = xr.concat([xr.open_dataset(f) for f in slp_files],
                        dim="time")["SLP"].sel(**box) / 100.0  # Pa -> hPa
    fronts = {w: open_year(day.year, w) for w in ("1wide", "3wide")}
    if fronts["1wide"] is None:
        raise SystemExit(f"no CODSUS front file for {day.year}")

    # the cell's own flag ladder, printed + saved with the figure
    cell = (np.floor(plat) + 0.5, np.floor(plon) + 0.5)
    flag_lines = [f"Front flags for cell {cell[0]:g}N {abs(cell[1]):g}W "
                  "(2x2 max-pool onto the analysis grid):"]
    for t in times:
        row = {w: cell_front_flags(f, t, cell) for w, f in fronts.items()
               if f is not None}
        flag_lines.append(
            f"  {t:%m-%d %H}Z  on-cell(1w): {row.get('1wide') or '-'}  "
            f"near-cell(3w): {row.get('3wide') or '-'}")
    print("\n".join(flag_lines))

    try:
        from convection_skill.plotting import make_conus_axes
        import cartopy.crs as ccrs
        have_cartopy = True
    except Exception:
        have_cartopy = False

    FIELDS = [("T", "RdYlBu_r", "925-hPa T (degC)", 1.0),
              ("QV", "YlGnBu", "925-hPa q (g/kg)", 1000.0)]
    extent = (plon - dlon, plon + dlon, plat - dlat, plat + dlat)
    tvals = m2["T"] - 273.15
    tmin, tmax = float(tvals.quantile(0.01)), float(tvals.quantile(0.99))

    def draw_panel(fig, rect, t, field, cmap, unit, scale):
        """One map panel: shaded field + barbs + SLP contours + fronts + star."""
        if have_cartopy:
            ax = make_conus_axes(fig=fig, rect=rect, extent=extent)
            tr = dict(transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(*rect)
            ax.set_xlim(extent[:2]), ax.set_ylim(extent[2:])
            tr = {}
        snap = m2.sel(time=t, method="nearest")
        fld = (snap[field] - 273.15) if field == "T" else snap[field] * scale
        mesh = ax.pcolormesh(
            snap["lon"], snap["lat"], fld, cmap=cmap, zorder=3,
            vmin=tmin if field == "T" else 4,
            vmax=tmax if field == "T" else 20, alpha=0.8, **tr)
        step = 2  # barb thinning
        blon, blat = np.meshgrid(snap["lon"].values[::step],
                                 snap["lat"].values[::step])
        ax.barbs(blon, blat, snap["U"].values[::step, ::step],
                 snap["V"].values[::step, ::step],
                 length=4.5, linewidth=0.6, zorder=4, **tr)
        if slp is not None:
            ps = slp.sel(time=t, method="nearest")
            cs = ax.contour(ps["lon"].values, ps["lat"].values, ps.values,
                            levels=np.arange(960, 1048, 2), colors="k",
                            linewidths=0.7, alpha=0.8, zorder=4.5, **tr)
            ax.clabel(cs, fontsize=7, fmt="%d")
        fr = fronts["1wide"]["fronts"].sel(time=t, method="nearest").sel(**box)
        for i, ftype in enumerate(
                str(x) for x in fronts["1wide"]["front_type"].values):
            if ftype not in FRONT_COLORS:
                continue
            yy, xx = np.where(fr.isel(front=i).values > 0)
            if yy.size:
                ax.plot(fr["lon"].values[xx], fr["lat"].values[yy], "s",
                        ms=7, mec="k", mew=0.4, color=FRONT_COLORS[ftype],
                        zorder=5, **tr)
        ax.plot(plon, plat, marker="*", ms=16, mec="k", mfc="cyan",
                zorder=6, **tr)
        return ax, mesh

    # ---- multi-time overview figure ----------------------------------------
    fig = plt.figure(figsize=(4.2 * len(times), 7.6))
    meshes = []
    for row, (field, cmap, unit, scale) in enumerate(FIELDS):
        for coli, t in enumerate(times):
            ax, mesh = draw_panel(fig, (2, len(times),
                                        row * len(times) + coli + 1),
                                  t, field, cmap, unit, scale)
            if row == 0:
                snap_t = pd.Timestamp(m2.sel(time=t, method="nearest")
                                      ["time"].item())
                ax.set_title(f"{snap_t:%H}Z {snap_t:%b-%d}", fontsize=11)
            if coli == len(times) - 1:
                meshes.append((mesh, unit))
    for k, (mesh, unit) in enumerate(meshes):
        cax = fig.add_axes([0.92, 0.55 - 0.45 * k, 0.011, 0.33])
        fig.colorbar(mesh, cax=cax, label=unit)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, mec="k",
                          label=n) for n, c in FRONT_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10)
    fig.suptitle(f"WPC analyst fronts + MERRA-2 925-hPa T / q / wind + SLP, "
                 f"{day:%Y-%m-%d} window (star = {plat:g}N {abs(plon):g}W)",
                 fontsize=14)
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.08,
                        wspace=0.06, hspace=0.08)
    out = args.out / f"{day:%Y%m%d}_{cell[0]:g}N_{abs(cell[1]):g}W"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fronts_context_{day:%Y%m%d}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    (out / "front_flags.txt").write_text("\n".join(flag_lines) + "\n")
    print(f"saved {path}")

    if args.single is not None:
        t = day + pd.Timedelta(hours=args.single)
        fig1 = plt.figure(figsize=(13, 5.2))
        for k, (field, cmap, unit, scale) in enumerate(FIELDS):
            ax, mesh = draw_panel(fig1, (1, 2, k + 1), t, field, cmap, unit,
                                  scale)
            fig1.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, label=unit)
        fig1.subplots_adjust(wspace=0.30)
        p1 = out / f"fronts_context_{day:%Y%m%d}_single_{t:%H}Z_{t:%d}.png"
        fig1.savefig(p1, dpi=170, bbox_inches="tight")
        print(f"saved {p1}")


if __name__ == "__main__":
    main()
