#!/usr/bin/env python
"""Daily report for one 1-degree cell: hour-by-hour table + hourly Skew-Ts.

Everything is read from the CURRENT raw data (no cached tables):

1. **Hour-by-hour table** from ``FCST_SMAP_MRMS_<year>.nc`` for the chosen
   date and cell: MRMS precip (cell-mean QPE = _av*_cnt/81, wet-area mean,
   sub-pixel max), AIRS-FCST thermodynamics (MU + MML CAPE/CIN/LCL/LFC/EL --
   the files carry no surface-based parcel), parcel near-surface q/T, the
   Guo et al. (2024) 3-hourly assessed PBL depth, and the SMAP L4 ladder
   (soil moisture, its sub-cell SD and W-E/S-N/|.| gradients, layer-1 q/T,
   wind speed, precip flux) on its own ~16:30/19:30/22:30/01:30/04:30 UTC
   observation clock.
2. **Skew-T log-P panels, one per forecast hour**, from two profile sources:
   - HYSPLIT demo parcels (the 2019-06-05 method-development day): at each
     trajectory step, every parcel currently located INSIDE the cell gives a
     (pres, T, Td-from-q) point -- the advected AIRS sounding over the cell.
     Skipped gracefully for dates without trajectory files.
   - MERRA-2 reanalysis (compact front-id corpus: 1000/925/850/700/500 hPa,
     3-hourly): profile + wind barbs at the grid point nearest the requested
     location, at the 3-hourly time nearest each panel hour.

Requires ``JPL_AIRS_DATA`` to point at the data root (dev:
``/mnt/d/JPL_AIRS/data``). Usage::

    python scripts/daily_cell_report.py 2019-06-05 --point 40.15 -89.33
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
from convection_skill import data_loading as dl  # noqa: E402
from trajectory_kernels import config as tk_config  # noqa: E402

MERRA2_DAILY = tk_config.DATA_DIR / "front_id" / "reanalysis" / "MERRA2" / "daily"
#: demo-day trajectory directory candidates (config.TRAJ_DIR predates the
#: HYSPLIT_demo/ subfolder on the current D: tree)
TRAJ_DIR_CANDIDATES = (
    tk_config.TRAJ_DIR,
    tk_config.DATA_DIR / "HYSPLIT_demo" / "wrf27km_20190605" / "wrf27km_20190605",
)

#: slot-level variables for the table: raw file name -> (row label, unit)
SLOT_VARS = {
    "qpe":           ("MRMS cell-mean QPE", "mm/h"),
    "qpe_wet":       ("MRMS wet-area-mean QPE", "mm/h"),
    "MRMS_GaugeCorrQPE01H_max": ("MRMS sub-pixel max QPE", "mm/h"),
    "FCST_MU_CAPE":  ("MU CAPE", "J/kg"),
    "FCST_MML_CAPE": ("MML CAPE", "J/kg"),
    "FCST_MU_CIN":   ("MU CIN", "J/kg"),
    "FCST_MML_CIN":  ("MML CIN", "J/kg"),
    "FCST_MU_LCL":   ("MU LCL", "m"),
    "FCST_MML_LCL":  ("MML LCL", "m"),
    "FCST_MU_LFC":   ("MU LFC", "m"),
    "FCST_MML_LFC":  ("MML LFC", "m"),
    "FCST_MU_EL":    ("MU EL", "m"),
    "FCST_q":        ("parcel near-sfc q", "g/kg"),
    "FCST_t":        ("parcel near-sfc T", "K"),
    "pblh":          ("PBL depth (Guo 3-hrly, nearest)", "m"),
}
#: SMAP L4 variables (their own 5-slot observation clock)
L4_VARS = {
    "SMAP_L4_smsfc_av":      ("soil moisture", "m3/m3"),
    "SMAP_L4_smsfc_sd":      ("SM sub-cell SD", "m3/m3"),
    "SMAP_L4_smsfc_wegrad":  ("SM W-E gradient", "m3/m3/deg"),
    "SMAP_L4_smsfc_sngrad":  ("SM S-N gradient", "m3/m3/deg"),
    "SMAP_L4_smsfc_absgrad": ("|grad SM|", "m3/m3/deg"),
    "SMAP_L4_qlay1_av":      ("layer-1 q", "g/kg"),
    "SMAP_L4_tlay1_av":      ("layer-1 T", "K"),
    "SMAP_L4_ulay1_av":      ("near-sfc wind speed", "m/s"),
    "SMAP_L4_pflux_av":      ("precip flux", "kg/m2/s"),
}
PBLH_TOL_H = 1.5  # same tolerance as trajectory_kernels.pbl.GriddedPBL


def cell_center(v: float) -> float:
    """Nearest half-degree cell center (the FCST/MRMS/SMAP 1-deg grid)."""
    return np.floor(v) + 0.5


# --------------------------------------------------------------------------- #
# Hour-by-hour table from the raw match-up file
# --------------------------------------------------------------------------- #
def slot_datetimes(day: pd.Timestamp, slots) -> list[pd.Timestamp]:
    """True datetimes of the forecast slots (00-02 UTC land on the next day)."""
    out = []
    for s in slots:
        h = dict(zip(cs_config.FORECAST_SLOTS, cs_config.FORECAST_HOURS_UTC)).get(s)
        if s == 0:
            out.append(day + pd.Timedelta(hours=19))  # nominal overpass ~19 UTC
        else:
            out.append(day + pd.Timedelta(hours=h if h >= 12 else h + 24))
    return out


def build_hourly_table(day: pd.Timestamp, cell: tuple[float, float]
                       ) -> tuple[pd.DataFrame, pd.DataFrame, xr.Dataset]:
    """(slot table, SMAP-L4 table, single-cell raw Dataset) for one cell-day."""
    lat, lon = cell
    variables = ([v for v in SLOT_VARS if v.startswith(("FCST", "MRMS"))]
                 + [cs_config.QPE_VAR, cs_config.QPE_CNT_VAR]
                 + list(L4_VARS) + ["SMAP_L4_hour"])
    raw = dl.load_raw([day.year], lat_range=(lat, lat), lon_range=(lon, lon),
                      months=(day.month, day.month), variables=variables)
    raw = raw.sel(date=day).squeeze(["lat", "lon"])

    slots = (0,) + tuple(cs_config.FORECAST_SLOTS)
    axis = next(d for d in dl.FORECAST_DIMS if d in raw[cs_config.QPE_VAR].dims)
    times = slot_datetimes(day, slots)

    rows = {}
    qpe_av = raw[cs_config.QPE_VAR]
    qpe_cnt = raw[cs_config.QPE_CNT_VAR]
    derived = {
        "qpe": (qpe_av * qpe_cnt / cs_config.WET_CELL_MAX_CNT),
        "qpe_wet": qpe_av,
    }
    for var, (label, unit) in SLOT_VARS.items():
        if var == "pblh":
            continue
        da = derived.get(var)
        if da is None:
            if var not in raw:
                continue
            da = raw[var]
        ax = next(d for d in dl.FORECAST_DIMS if d in da.dims)
        rows[f"{label} [{unit}]"] = [float(da.isel({ax: s})) for s in slots]

    # assessed PBL depth at each slot's true datetime (nearest 3-hourly sample)
    pblh_vals = [np.nan] * len(slots)
    if tk_config.PBLH_3HRLY_PATH.exists():
        with xr.open_dataset(tk_config.PBLH_3HRLY_PATH) as pb:
            col = pb["pblh"].sel(lat=lat, lon=lon, method="nearest")
            for i, t in enumerate(times):
                near = col.sel(time=t, method="nearest")
                if abs(pd.Timestamp(near["time"].item()) - t) <= pd.Timedelta(
                        hours=PBLH_TOL_H):
                    pblh_vals[i] = float(near)
    label, unit = SLOT_VARS["pblh"]
    rows[f"{label} [{unit}]"] = pblh_vals

    hour_labels = ["overpass (~18-20Z)" if s == 0 else f"{t:%H}Z {t:%b-%d}"
                   for s, t in zip(slots, times)]
    table = pd.DataFrame(rows, index=hour_labels).T

    # SMAP L4 ladder on its own clock
    l4_hours = raw["SMAP_L4_hour"].values
    l4_labels = [f"{h:04.1f} UTC" if np.isfinite(h) else f"slot {i}"
                 for i, h in enumerate(l4_hours)]
    l4 = pd.DataFrame(
        {f"{label} [{unit}]": raw[var].values
         for var, (label, unit) in L4_VARS.items() if var in raw},
        index=l4_labels).T
    return table, l4, raw


# --------------------------------------------------------------------------- #
# Profiles for the Skew-Ts
# --------------------------------------------------------------------------- #
def load_trajectories(day: pd.Timestamp):
    """The tidy (parcel, step) HYSPLIT dataset for the demo day, or None."""
    if day != pd.Timestamp("2019-06-05"):
        return None  # only the method-development day exists locally
    from trajectory_kernels import trajectories as tj
    for cand in TRAJ_DIR_CANDIDATES:
        if cand.exists():
            return tj.load_day_dir(cand)
    return None


def merra2_point(day: pd.Timestamp, lat: float, lon: float):
    """Concatenated MERRA-2 compact profiles for the day and the next (or None)."""
    parts = []
    for d in (day, day + pd.Timedelta(days=1)):
        path = MERRA2_DAILY / f"{d:%Y}" / f"m2_{d:%Y%m%d}.nc"
        if path.exists():
            with xr.open_dataset(path) as ds:
                parts.append(ds.sel(lat=lat, lon=lon, method="nearest").load())
    if not parts:
        return None
    m2 = xr.concat(parts, dim="time")
    # corrupt-transfer guard (QV identically 0 in some daily files): mask
    n_zero = int((m2["QV"].values == 0).sum())
    if n_zero:
        print(f"      WARNING: {n_zero} MERRA-2 QV==0 samples masked as "
              "missing (corrupt daily file); Td circles absent there")
        m2["QV"] = m2["QV"].where(m2["QV"] > 0)
    return m2


#: dataset level markers drawn on each Skew-T:
#: table row -> (color, linestyle, label x-position in axes fraction)
LEVEL_MARKERS = {
    "MU LCL [m]":  ("tab:orange", "-", 0.02),
    "MML LCL [m]": ("tab:orange", ":", 0.24),
    "MU LFC [m]":  ("tab:purple", "-", 0.48),
    "MML LFC [m]": ("tab:purple", ":", 0.72),
    "MU EL [m]":   ("tab:brown", "-", 0.02),
    "PBL depth (Guo 3-hrly, nearest) [m]": ("tab:blue", "-", 0.72),
}


def _add_level_markers(skew, col: pd.Series, p_of_z) -> None:
    """Horizontal lines at the dataset's LCL/LFC/EL/PBLH for one forecast hour.

    Heights are AGL (the AIRS-FCST datum was verified AGL, and the Guo PBLH is
    AGL by construction); ``p_of_z`` converts to pressure via the day's own
    parcel alt/pres relation (or a fallback profile when no parcels exist).
    Labels sit inside the axes at staggered x positions so the tight
    LCL/LFC/PBLH cluster stays legible.
    """
    for row, (color, ls, x0) in LEVEL_MARKERS.items():
        z = float(col.get(row, np.nan))
        if not np.isfinite(z):
            continue
        p = p_of_z(z)
        if not np.isfinite(p):
            continue
        skew.ax.axhline(p, color=color, ls=ls, lw=1.4, alpha=0.85)
        short = row.split(" [")[0].replace("PBL depth (Guo 3-hrly, nearest)",
                                           "PBLH")
        skew.ax.text(x0, p, f"{short} {z:.0f}m", color=color, fontsize=6.5,
                     va="bottom", ha="left", clip_on=True,
                     transform=skew.ax.get_yaxis_transform(),
                     bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))


def plot_skewts(out: Path, day: pd.Timestamp, point: tuple[float, float],
                cell: tuple[float, float], traj, m2, table: pd.DataFrame) -> Path:
    """One Skew-T per forecast hour: HYSPLIT parcels in-cell + MERRA-2 anchor."""
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    lat, lon = cell
    steps = list(range(7))
    fig = plt.figure(figsize=(24, 11))
    for i, step in enumerate(steps):
        skew = SkewT(fig, rotation=45, subplot=(2, 4, i + 1))
        title = None
        # height-AGL -> pressure fallback (scale height); replaced by the
        # parcels' own alt/pres relation when in-cell parcels exist
        p_of_z = lambda z: 1013.25 * np.exp(-z / 8400.0)  # noqa: E731

        if traj is not None:
            at = traj.isel(step=step)
            inside = ((np.abs(at["lat"] - lat) <= 0.5)
                      & (np.abs(at["lon"] - lon) <= 0.5)).values
            n_in = int(inside.sum())
            if n_in:
                p = at["pres"].values[inside]
                t = at["t"].values[inside] - 273.15
                q = at["q"].values[inside]
                z = at["alt"].values[inside]
                ok = (np.isfinite(p) & np.isfinite(t) & np.isfinite(q)
                      & np.isfinite(z))
                p, t, q, z = p[ok], t[ok], q[ok], z[ok]
                order = np.argsort(p)
                p, t, q, z = p[order], t[order], q[order], z[order]
                zo = np.argsort(z)
                p_of_z = lambda h, zz=z[zo], pp=p[zo]: (  # noqa: E731
                    float(np.interp(h, zz, pp)))
                td = mpcalc.dewpoint_from_specific_humidity(
                    p * units.hPa, (q / 1000.0) * units("kg/kg")).magnitude
                skew.plot(p, t, "r.", ms=3, alpha=0.35)
                skew.plot(p, td, "g.", ms=3, alpha=0.35)
                # median profile in 25-hPa bins (readable line over the cloud)
                bins = np.arange(100, 1050, 25)
                idx = np.digitize(p, bins)
                pm, tm, dm = [], [], []
                for b in np.unique(idx):
                    sel = idx == b
                    if sel.sum() >= 3:
                        pm.append(np.median(p[sel]))
                        tm.append(np.median(t[sel]))
                        dm.append(np.median(td[sel]))
                skew.plot(pm, tm, "r-", lw=2, label=f"HYSPLIT T (n={n_in})")
                skew.plot(pm, dm, "g-", lw=2, label="HYSPLIT Td")
                utc = pd.Timestamp(np.nanmin(at["time_utc"].values[inside]))
                title = f"step {step}: {utc:%H:%M}Z {utc:%b-%d}"

        if m2 is not None:
            panel_time = None
            if traj is not None:
                ts = pd.Series(pd.to_datetime(
                    traj["time_utc"].isel(step=step).values.ravel())).dropna()
                panel_time = ts.median() if len(ts) else None
            if panel_time is None or pd.isna(panel_time):
                panel_time = slot_datetimes(day, (0,) + tuple(
                    cs_config.FORECAST_SLOTS))[step]
            near = m2.sel(time=panel_time, method="nearest")
            pl = near["lev"].values
            tt = near["T"].values - 273.15
            td = mpcalc.dewpoint_from_specific_humidity(
                pl * units.hPa, near["QV"].values * units("kg/kg")).magnitude
            skew.plot(pl, tt, "ro", mfc="none", ms=9, mew=2,
                      label="MERRA-2 T")
            skew.plot(pl, td, "go", mfc="none", ms=9, mew=2,
                      label="MERRA-2 Td")
            skew.plot_barbs(pl, near["U"].values, near["V"].values)
            if title is None:
                title = (f"step {step}: "
                         f"{pd.Timestamp(near['time'].item()):%H:%M}Z (M2)")

        _add_level_markers(skew, table.iloc[:, i], p_of_z)
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 45)
        skew.plot_dry_adiabats(alpha=0.15)
        skew.plot_moist_adiabats(alpha=0.15)
        skew.plot_mixing_lines(alpha=0.15)
        skew.ax.set_title(title or f"step {step}", fontsize=11)
        if i == 0:
            skew.ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Skew-T log-P by forecast hour, {day:%Y-%m-%d} -- cell "
        f"{lat:g}N {abs(lon):g}W (requested {point[0]:g}, {point[1]:g}); "
        "dots = advected AIRS/HYSPLIT parcels in-cell, circles = MERRA-2 "
        "(5 levels, nearest 3-hourly time)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / f"skewt_{day:%Y%m%d}_{lat:g}N_{abs(lon):g}W.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("date", help="day to report, e.g. 2019-06-05")
    ap.add_argument("--point", type=float, nargs=2, metavar=("LAT", "LON"),
                    required=True)
    ap.add_argument("--out", type=Path,
                    default=cs_config.RESULTS_DIR / "daily_reports")
    args = ap.parse_args(argv)
    day = pd.Timestamp(args.date)
    cell = (cell_center(args.point[0]), cell_center(args.point[1]))
    out = args.out / f"{day:%Y%m%d}_{cell[0]:g}N_{abs(cell[1]):g}W"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] hourly table for cell {cell[0]:g}N {abs(cell[1]):g}W ...")
    table, l4, raw = build_hourly_table(day, cell)
    table.to_csv(out / "hourly_table.csv")
    l4.to_csv(out / "smap_l4_table.csv")

    print("[2/3] profiles (HYSPLIT demo + MERRA-2) ...")
    traj = load_trajectories(day)
    if traj is None:
        print("      no trajectory files for this date -> HYSPLIT panels skipped")
    m2 = merra2_point(day, args.point[0], args.point[1])
    if m2 is None:
        print("      no MERRA-2 daily files found -> MERRA-2 overlay skipped")

    print("[3/3] Skew-T panels + report ...")
    fig_path = None
    if traj is not None or m2 is not None:
        fig_path = plot_skewts(out, day, tuple(args.point), cell, traj, m2,
                               table)

    fmt = lambda v: "-" if not np.isfinite(v) else f"{v:.3g}"
    lines = [f"# Daily cell report: {day:%Y-%m-%d}, "
             f"{cell[0]:g}N {abs(cell[1]):g}W",
             "",
             f"Requested point {args.point[0]:g}, {args.point[1]:g} -> 1-deg "
             f"cell centered {cell[0]:g}, {cell[1]:g}. All values read from "
             "the raw current files (no cached tables). The AIRS-FCST files "
             "carry MU and MML parcels only (no surface-based CAPE/CIN).",
             "", "## Forecast-hour ladder", "",
             "| variable | " + " | ".join(table.columns) + " |",
             "|---" * (len(table.columns) + 1) + "|"]
    for name, r in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in r) + " |")
    lines += ["", "## SMAP L4 ladder (observation clock; 01:30/04:30 UTC are "
              "the NEXT calendar day)", "",
              "| variable | " + " | ".join(l4.columns) + " |",
              "|---" * (len(l4.columns) + 1) + "|"]
    for name, r in l4.iterrows():
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in r) + " |")
    if fig_path is not None:
        lines += ["", f"Skew-T panels: `{fig_path.name}`"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
