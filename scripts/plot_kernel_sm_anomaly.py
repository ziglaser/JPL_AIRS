#!/usr/bin/env python
"""Geographic influence kernels for Psi over a soil-moisture-anomaly map.

For one receptor cell and arrival step of the HYSPLIT demo day, draws the
per-lag kernel outlines (the established closed-staircase 90%-of-lag-mass
regions) on top of the day's SMAP L4 surface soil-moisture ANOMALY (pre-window
daily mean minus the 2016-2021 per-cell monthly baseline -- the cardinal Psi
anomaly form). Each lag's outline (and its member-parcel dots) is colored by
the **normalized per-hour Psi weight** ``w_k`` = that lag's ``lag_weight`` --
the physical kernel mass of the hour, i.e. available surface energy (a*DSWF)
x PBL-contact-gated residence x land, exactly the factor Psi applies to that
hour's soil moisture (UPWIND_INDEX_REVIEW.md 1.6). Filled dots = parcels in
PBL contact at that lag, hollow = aloft (no deposit).

The kernel is built fresh with the production physics: GriddedPBL (assessed
Guo PBLH for the day, climatology/analytic fallback) and
ClearSkyAvailableEnergy -- NOT read from the pre-fix demo kernel file, which
lacks ``lag_weight``.

Usage::

    python scripts/plot_kernel_sm_anomaly.py 2019-06-05 --point 39.15 -89.33
    python scripts/plot_kernel_sm_anomaly.py 2019-06-05 --point 39.15 -89.33 --arrival-step 3
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from convection_skill import config as cs_config  # noqa: E402
from convection_skill import data_loading as dl  # noqa: E402
from trajectory_kernels import config as tk_config  # noqa: E402
from trajectory_kernels import footprint as fp  # noqa: E402
from trajectory_kernels import plotting as tkplot  # noqa: E402
from trajectory_kernels.insolation import ClearSkyAvailableEnergy  # noqa: E402
from trajectory_kernels.pbl import GriddedPBL  # noqa: E402
from daily_cell_report import cell_center, load_trajectories  # noqa: E402

SM_BASELINE = (tk_config.DATA_DIR / "soil_moisture" /
               "SMAP_L4_smsfc_monthly_baseline_2016-2021.nc")
WEIGHT_CMAP = "plasma"


def sm_anomaly_map(day: pd.Timestamp, cell, halfwidth_deg: float
                   ) -> xr.DataArray:
    """Pre-window daily-mean SMAP L4 soil moisture minus the monthly baseline."""
    lat, lon = cell
    raw = dl.load_raw([day.year],
                      lat_range=(lat - halfwidth_deg, lat + halfwidth_deg),
                      lon_range=(lon - halfwidth_deg, lon + halfwidth_deg),
                      months=(day.month, day.month),
                      variables=[cs_config.SM_VAR])
    sm = raw[cs_config.SM_VAR].sel(date=day).isel(
        L4_nhours=list(cs_config.L4_PREWINDOW_SLOTS)).mean(
        "L4_nhours", skipna=True)
    with xr.open_dataset(SM_BASELINE) as base:
        clim_name = next(v for v in base.data_vars
                         if "smsfc" in v or "baseline" in v or "mean" in v)
        clim = base[clim_name].sel(month=day.month).reindex_like(
            sm, method="nearest", tolerance=0.01)
    anom = (sm - clim).rename("sm_anom")
    anom.attrs["long_name"] = ("SMAP L4 surface SM anomaly vs 2016-2021 "
                               f"month-{day.month} baseline")
    return anom


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("date", help="kernel day (demo: 2019-06-05)")
    ap.add_argument("--point", type=float, nargs=2, metavar=("LAT", "LON"),
                    required=True)
    ap.add_argument("--arrival-step", type=int, default=6,
                    help="forecast arrival step 1-6 (default 6 = 02 UTC, "
                         "deepest look-back)")
    ap.add_argument("--hdr-frac", type=float, default=0.9,
                    help="fraction of each lag's mass outlined (0.9)")
    ap.add_argument("--out", type=Path,
                    default=cs_config.RESULTS_DIR / "daily_reports")
    args = ap.parse_args(argv)
    day_ts = pd.Timestamp(args.date)
    cell = (cell_center(args.point[0]), cell_center(args.point[1]))
    out = args.out / f"{day_ts:%Y%m%d}_{cell[0]:g}N_{abs(cell[1]):g}W"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] trajectories ...")
    day = load_trajectories(day_ts)
    if day is None:
        raise SystemExit(f"no HYSPLIT trajectory files for {args.date}")

    print("[2/4] kernel (GriddedPBL + clear-sky energy) ...")
    pbl = GriddedPBL(date=str(day_ts.date()))
    ds = fp.build_footprint(day, cell[0], cell[1], args.arrival_step,
                            pbl_model=pbl, energy_fn=ClearSkyAvailableEnergy())
    if "lag_weight" not in ds:
        raise SystemExit("build_footprint returned no lag_weight -- "
                         "trajectory_kernels version too old")

    print("[3/4] soil-moisture anomaly underlay ...")
    anom = sm_anomaly_map(
        day_ts, cell, tk_config.SOURCE_WINDOW_HALFWIDTH_DEG + 1.0)

    print("[4/4] figure ...")
    from trajectory_kernels.insolation import clear_sky_dswf
    lags = ds["lag"].values
    weights = ds["lag_weight"].values.astype(float)
    populated = [(float(l), float(w)) for l, w in zip(lags, weights)
                 if np.isfinite(w) and w > 0
                 and np.nansum(ds["footprint"].sel(lag=l).values) > 0]
    if not populated:
        raise SystemExit("no populated lag hours for this receptor/step")
    wmax = max(w for _, w in populated)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(WEIGHT_CMAP)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    mesh = ax.pcolormesh(anom["lon"], anom["lat"], anom.values,
                         cmap="BrBG", vmin=-0.12, vmax=0.12, alpha=0.65,
                         zorder=0)
    tkplot._land_outline(ax)

    hist = tkplot._member_history(day, ds, max_trajectories=150)
    if hist is not None:
        for i in range(hist["lat"].shape[0]):
            ax.plot(hist["lon"][i], hist["lat"][i], "-", color="0.35",
                    lw=0.6, alpha=0.6, zorder=1)

    # per-lag table ingredients alongside the outlines
    anom_at = anom.sel(lat=ds["source_lat"], lon=ds["source_lon"],
                       method="nearest").values
    slat = ds["source_lat"].values
    cell_step = float(slat[1] - slat[0]) if slat.size > 1 else 0.25
    contact_fraction = float(ds.attrs.get("contact_fraction", 1.0))
    rows, row_colors = [], []
    for rank, (l, w) in enumerate(populated[::-1]):
        color = cmap(norm(w / wmax))
        k = np.nan_to_num(ds["kernel"].sel(lag=l).values.astype(float))
        tkplot._contour_lag_kernel(ax, ds, k, color, (args.hdr_frac,),
                                   lw=2.6, inset=0.035 * cell_step * rank)
        fin = np.isfinite(anom_at) & (k > 0)
        sm_mean = (float((k[fin] * anom_at[fin]).sum() / k[fin].sum())
                   if fin.any() else np.nan)
        plat, plon, in_pbl = tkplot._contact_at_lag(
            hist, l, pbl, contact_fraction)
        when = (hist["time_sec"][:, -1] - l * 3600.0).astype(
            "int64").astype("datetime64[s]")
        good = np.isfinite(plat) & np.isfinite(plon)
        depth = float(np.nanmean(np.asarray(
            pbl(plat[good], plon[good], when[good]), dtype=float)))
        sdown = float(np.nanmean(np.asarray(
            clear_sky_dswf(plat[good], plon[good], when[good]), dtype=float)))
        ax.scatter(plon[in_pbl], plat[in_pbl], s=20, color=color,
                   edgecolor="k", linewidth=0.3, zorder=4)
        aloft = good & ~in_pbl
        ax.scatter(plon[aloft], plat[aloft], s=20, facecolor="none",
                   edgecolor=color, linewidth=1.0, zorder=4)
        rows.append([f"{l:.0f}", f"{w / wmax:.2f}", f"{sm_mean:+.3f}",
                     f"{depth:.0f}", f"{100 * in_pbl.sum() / in_pbl.size:.0f}",
                     f"{sdown:.0f}"])
        row_colors.append(color)

    # red dots: member-parcel positions at the forecast (arrival) hour
    if hist is not None:
        alat, alon = tkplot._positions_at_lag(hist, 0.0)
        ax.scatter(alon, alat, s=26, color="red", edgecolor="k",
                   linewidth=0.4, zorder=5)
    tkplot._receptor_cell_box(ax, ds)
    tkplot._zoom_to_content(ax, ds, hist)
    # linear-in-distance: 1 deg lat appears 1/cos(lat) times longer than
    # 1 deg lon, so N-S and E-W kilometres render at the same scale
    ax.set_aspect(1.0 / np.cos(np.deg2rad(ds.attrs["target_lat"])))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02,
                 label="Soil Moisture Anomaly (m$^3$/m$^3$)")
    sm_w = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm_w, ax=ax, shrink=0.8, pad=0.06, label="$\\Psi$ Weight")
    ax.set_xlabel("Longitude"), ax.set_ylabel("Latitude")

    # color-coded per-lag table instead of a legend
    ax_t = fig.add_subplot(gs[0, 1])
    ax_t.set_axis_off()
    col_labels = ["Lag Hour", "$\\Psi$ Weight", "SM Anomaly\n(m$^3$/m$^3$)",
                  "PBL Depth\n(m)", "In PBL\n(%)", "Insolation\n(W/m$^2$)"]
    tbl = ax_t.table(cellText=rows, colLabels=col_labels,
                     cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    tbl.scale(1.15, 2.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_height(tbl[0, j].get_height() * 1.4)
        tbl[0, j].set_text_props(fontweight="bold")
    for i, color in enumerate(row_colors, start=1):
        cell0 = tbl[i, 0]
        cell0.set_facecolor(color)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        cell0.set_text_props(color="white" if lum < 0.5 else "black",
                             fontweight="bold")

    path = out / (f"kernel_sm_anomaly_{day_ts:%Y%m%d}_s{args.arrival_step}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
