#!/usr/bin/env python
"""One-day diagnostic figure: did the upwind soil-moisture index work?

Six panels for a chosen day and arrival slot (Zach's diagnosis layout,
2026-08-21), all on the 1-degree FCST_SMAP_MRMS grid:

    (a) SMAP L4 surface soil-moisture ANOMALY (vs the per-cell monthly
        baseline, m3/m3) at the analysis slot nearest the arrival -- the same
        field the index convolves, so (a) -> (c) -> (d) read as one
        comparison on one diverging scale;
    (b) prevailing low-level winds (FCST_u/v quiver over speed shading) --
        where the air came from;
    (c) the index Psi_anom (m3/m3 anomaly vs the monthly baseline) -- the
        energy-weighted soil anomaly along each cell's inflow;
    (d) Psi_anom minus the local endpoint anomaly -- the REMOTE content: what
        the Lagrangian accumulation says that reading the soil under the cell
        does not. Structure here should be flow-aligned (compare with (b));
        salt-and-pepper noise here means the index adds nothing;
    (e) Omega (J/kg) -- surviving surface energy per kg of arriving air;
    (f) assessed PBL height (m) at the arrival time -- the dilution/geometry
        field.

Anomaly panels use the diverging BrBG map centred on 0 (brown = dry, green =
wet -- the soil-moisture convention); single-quantity panels use one-hue
sequential maps (water = Blues, energy = Oranges, depth = Purples); NaN is
light gray; the land outline is drawn from the global lsm for orientation.

Usage:
  JPL_AIRS_DATA=/mnt/d/JPL_AIRS/data PYTHONPATH=src \\
    python scripts/plot_upwind_day.py --date 2019-06-05          # 00 UTC slot
  ... --arrival-step 2                                           # 22 UTC slot
Needs: the day's UPW_<YYYYMMDD>.nc (build_upwind_features.py), the year's
FCST_SMAP_MRMS file, and the 3-hourly assessed PBLH aggregate.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as Date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_kernels import config  # noqa: E402

#: Nominal UTC valid hour per FCST time index (hours after 00 UTC of `date`;
#: 24+ = the next calendar day). Same table as merge_upwind_features.py --
#: duplicated because scripts/ is not a package; keep them in agreement.
SLOT_HOURS_AFTER_DATE = {1: 21, 2: 22, 3: 23, 4: 24, 5: 25, 6: 26}

_GRAY = "0.85"  # NaN background, stated in the caption


def _panel(ax, lat, lon, field, land, title, cmap, units,
           diverging=False, vmax=None):
    """One map panel: pcolormesh + land outline, NaN in light gray."""
    ax.set_facecolor(_GRAY)
    if diverging:
        lim = vmax or float(np.nanmax(np.abs(field))) or 1.0
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=-lim, vmax=lim,
                             shading="nearest")
    else:
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="nearest",
                             vmax=vmax)
    ax.contour(land["lon"], land["lat"], land.values, levels=[0.5],
               colors="k", linewidths=0.5)
    ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
    ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.4)
    cb = plt.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(units, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=7)


def render_day(date_str: str, arrival_step: int = 4,
               daily_dir: Path | None = None, fcst_dir: Path | None = None,
               pblh_3hrly: Path | None = None, lsm: Path | None = None,
               baseline: Path | None = None, out: Path | None = None) -> Path:
    """Render the six-panel diagnosis for one day; importable (QA check 5)."""
    daily_dir = daily_dir or config.RESULTS_DIR / "upwind_features" / "daily"
    fcst_dir = fcst_dir or config.FCST_TABLE_DIR
    pblh_3hrly = pblh_3hrly or config.PBLH_3HRLY_PATH
    lsm = lsm or config.LSM_PATH
    baseline = baseline or config.SMAP_BASELINE_PATH

    day = Date.fromisoformat(date_str)
    stamp = day.strftime("%Y%m%d")
    step = arrival_step
    slot_dt = (np.datetime64(day.isoformat())
               + np.timedelta64(SLOT_HOURS_AFTER_DATE[step], "h"))
    slot_label = str(slot_dt)[11:16] + " UTC"

    daily_path = Path(daily_dir) / f"UPW_{stamp}.nc"
    if not daily_path.exists():
        raise FileNotFoundError(
            f"{daily_path} missing: run build_upwind_features.py first")
    upw = xr.open_dataset(daily_path).sel(arrival_step=step)
    lat, lon = upw["target_lat"].values, upw["target_lon"].values

    fcst_path = Path(fcst_dir) / f"FCST_SMAP_MRMS_{day.year}.nc"
    with xr.open_dataset(fcst_path) as ds:
        fc = ds.sel(date=day.isoformat()).isel(time=step)
        u, v = fc["FCST_u"].load(), fc["FCST_v"].load()
        # the analysis slot nearest THIS arrival slot (the daily builder used
        # the one nearest the mean arrival; sub-daily SM differences are noise)
        slots = ds["L4_nhours"].values.astype(float)
        chosen = float(slots[np.argmin(np.abs(slots - SLOT_HOURS_AFTER_DATE[step]))])
        sm_raw = (ds["SMAP_L4_smsfc_av"]
                  .sel(date=day.isoformat(), L4_nhours=chosen).load())

    # panel (a) shows the ANOMALY -- the very field the index convolves
    with xr.open_dataset(baseline) as base:
        ref = base["sm_baseline"].sel(month=day.month).load()
    sm_anom = (sm_raw.values
               - ref.sel(lat=lat, lon=lon, method="nearest").values)

    # PBLH panel degrades to all-NaN (gray, labelled) when the file is absent
    if Path(pblh_3hrly).exists():
        with xr.open_dataset(pblh_3hrly) as pb:
            idx = int(np.argmin(np.abs(pb["time"].values - slot_dt)))
            # nearest-cell reindex onto the feature grid: axis-order-proof
            # (the aggregate stores lat descending), grid-exact at X.5 centres
            pblh = pb["pblh"].isel(time=idx).sel(lat=lat, lon=lon,
                                                 method="nearest").load().values
            pblh_when = str(pb["time"].values[idx])[11:16] + " UTC"
    else:
        pblh = np.full((lat.size, lon.size), np.nan)
        pblh_when = "file absent"

    with xr.open_dataset(lsm) as lm:
        land = (lm["lsm"].sortby("lat")
                .sel(lat=slice(lat.min() - 1, lat.max() + 1),
                     lon=slice(lon.min() - 1, lon.max() + 1)).load())

    psi = upw["psi_anom"].values
    endpoint = upw["s_endpoint_anom"].values
    remote = psi - endpoint
    ok = np.isfinite(psi) & np.isfinite(endpoint)
    r = float(np.corrcoef(psi[ok], endpoint[ok])[0, 1]) if ok.sum() > 2 else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    fig.suptitle(f"Upwind soil-moisture index diagnosis: {day} arrival "
                 f"{slot_label} (arrival_step {step}) -- NaN shown gray",
                 fontsize=13)

    _panel(axes[0, 0], lat, lon, sm_anom, land,
           f"(a) SMAP L4 surface SM anomaly, {chosen:.1f} h slot",
           "BrBG", "m$^3$ m$^{-3}$", diverging=True, vmax=0.15)

    speed = np.hypot(u.values, v.values)
    _panel(axes[0, 1], lat, lon, speed, land,
           "(b) low-level parcel wind", "Greys", "m s$^{-1}$")
    q = 2  # quiver every other cell so arrows stay legible
    axes[0, 1].quiver(lon[::q], lat[::q], u.values[::q, ::q], v.values[::q, ::q],
                      color="#1f5fa8", width=0.0035, scale=250)

    _panel(axes[0, 2], lat, lon, psi, land,
           r"(c) index $\Psi_{anom}$: energy-weighted upwind soil anomaly",
           "BrBG", "m$^3$ m$^{-3}$", diverging=True, vmax=0.15)

    _panel(axes[1, 0], lat, lon, remote, land,
           rf"(d) remote content: $\Psi_{{anom}}$ - endpoint  (r={r:.2f})",
           "BrBG", "m$^3$ m$^{-3}$", diverging=True, vmax=0.08)

    _panel(axes[1, 1], lat, lon, upw["omega"].values, land,
           r"(e) $\Omega$: surviving surface energy per kg", "Oranges",
           "J kg$^{-1}$")

    _panel(axes[1, 2], lat, lon, pblh, land,
           f"(f) assessed PBL height, {pblh_when}", "Purples", "m")

    out = out or (config.RESULTS_DIR / "upwind_features" / "figures"
                  / f"upwind_day_{stamp}_s{step}.png")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--arrival-step", type=int, default=4,
                   help="FCST time index 1..6 (21..02 UTC); default 4 = 00 UTC")
    p.add_argument("--daily-dir", type=Path, default=None)
    p.add_argument("--fcst-dir", type=Path, default=None)
    p.add_argument("--pblh-3hrly", type=Path, default=None)
    p.add_argument("--lsm", type=Path, default=None)
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out = render_day(args.date, args.arrival_step, args.daily_dir,
                     args.fcst_dir, args.pblh_3hrly, args.lsm,
                     args.baseline, args.out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
