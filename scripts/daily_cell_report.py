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


def merra2_point(day: pd.Timestamp, lat: float, lon: float,
                 dense_dir: Path | None = None):
    """MERRA-2 point profiles for the day and the next (or None).

    Prefers the DENSE case-study files (``m2_dense_point_<YYYYMMDD>.nc4``,
    42-level M2I3NPASM subsets pulled from the cloud OPeNDAP endpoint) when
    present in ``dense_dir``; otherwise falls back to the compact front-id
    corpus (5 levels). Both are 3-hourly -- the densest cadence MERRA-2
    provides for profiles.
    """
    parts, dense = [], False
    for d in (day, day + pd.Timedelta(days=1)):
        dense_path = (dense_dir / f"m2_dense_point_{d:%Y%m%d}.nc4"
                      if dense_dir else None)
        if dense_path is not None and dense_path.exists():
            with xr.open_dataset(dense_path) as ds:
                parts.append(ds.squeeze(["lat", "lon"]).load())
            dense = True
            continue
        path = MERRA2_DAILY / f"{d:%Y}" / f"m2_{d:%Y%m%d}.nc"
        if path.exists():
            with xr.open_dataset(path) as ds:
                parts.append(ds.sel(lat=lat, lon=lon, method="nearest").load())
    if not parts:
        return None
    print(f"      MERRA-2 source: {'dense 42-level point subsets' if dense else 'compact 5-level corpus'}")
    m2 = xr.concat(parts, dim="time")
    # corrupt-transfer guard (QV identically 0 in some daily files): mask
    n_zero = int((m2["QV"].values == 0).sum())
    if n_zero:
        print(f"      WARNING: {n_zero} MERRA-2 QV==0 samples masked as "
              "missing (corrupt daily file); Td circles absent there")
        m2["QV"] = m2["QV"].where(m2["QV"] > 0)
    return m2


#: dataset level markers drawn on each Skew-T (MU parcel heights + PBLH only;
#: MML dropped per Zach 2026-08-20): row -> (color, linestyle, label x-frac)
LEVEL_MARKERS = {
    "MU LCL [m]":  ("tab:orange", "-", 0.02),
    "MU LFC [m]":  ("tab:purple", "-", 0.38),
    "MU EL [m]":   ("tab:brown", "-", 0.02),
    "PBL depth (Guo 3-hrly, nearest) [m]": ("tab:blue", "-", 0.70),
}


def _add_level_markers(skew, col: pd.Series, p_of_z, fontsize=7.5) -> None:
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
        skew.ax.text(x0, p, f"{short} {z:.0f}m", color=color,
                     fontsize=fontsize,
                     va="bottom", ha="left", clip_on=True,
                     transform=skew.ax.get_yaxis_transform(),
                     bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))


def _cell_parcels(traj, step: int, cell, mpcalc, units):
    """HYSPLIT parcels inside the cell at one step, sorted by pressure."""
    at = traj.isel(step=step)
    lat, lon = cell
    inside = ((np.abs(at["lat"] - lat) <= 0.5)
              & (np.abs(at["lon"] - lon) <= 0.5)).values
    if not inside.any():
        return None
    p = at["pres"].values[inside]
    t = at["t"].values[inside] - 273.15
    q = at["q"].values[inside]
    z = at["alt"].values[inside]
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(q) & np.isfinite(z)
    p, t, q, z = p[ok], t[ok], q[ok], z[ok]
    order = np.argsort(p)
    p, t, q, z = p[order], t[order], q[order], z[order]
    td = mpcalc.dewpoint_from_specific_humidity(
        p * units.hPa, (q / 1000.0) * units("kg/kg")).magnitude
    zo = np.argsort(z)
    time = pd.Timestamp(np.nanmin(at["time_utc"].values[inside]))
    return dict(p=p, t=t, td=td, z=z, n=int(inside.sum()), time=time,
                p_of_z=lambda h, zz=z[zo], pp=p[zo]: float(np.interp(h, zz, pp)))


def _parcel_analysis(p, t, td, mpcalc, units):
    """Most-unstable parcel path + CAPE/CIN integrated from a sounding.

    Input arrays are surface-first (descending pressure). Returns None when
    the profile is too short/broken for metpy's parcel routines.
    """
    try:
        P = np.asarray(p, dtype=float) * units.hPa
        T = np.asarray(t, dtype=float) * units.degC
        D = np.asarray(td, dtype=float) * units.degC
        cape, cin = mpcalc.most_unstable_cape_cin(P, T, D)
        _, _, _, idx = mpcalc.most_unstable_parcel(P, T, D)
        prof = mpcalc.parcel_profile(P[idx:], T[idx], D[idx]).to("degC")
        return dict(p=P[idx:], env_t=T[idx:], prof=prof,
                    cape=float(cape.magnitude), cin=float(cin.magnitude))
    except Exception as err:
        print(f"      parcel analysis skipped ({err})")
        return None


def _binned_median(p, x, bin_hpa: float):
    bins = np.arange(100, 1055, bin_hpa)
    idx = np.digitize(p, bins)
    pm, xm = [], []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= 3:
            pm.append(np.median(p[sel]))
            xm.append(np.median(x[sel]))
    return pm, xm


def _draw_panel(skew, parc, m2near, col, mpcalc, units,
                ylim, xlim, bin_hpa, label_fs, legend=False, pa=None):
    """One Skew-T panel: parcels, MERRA-2 profile, CAPE/CIN, level markers."""
    if pa is not None:  # MU parcel path + shaded CAPE (red) / CIN (blue)
        skew.plot(pa["p"], pa["prof"], "k-", lw=1.6, alpha=0.9,
                  label="MU parcel path")
        skew.shade_cape(pa["p"], pa["env_t"], pa["prof"], alpha=0.18)
        skew.shade_cin(pa["p"], pa["env_t"], pa["prof"], alpha=0.22)
    if parc is not None:
        skew.plot(parc["p"], parc["t"], "r.", ms=3.5, alpha=0.35)
        skew.plot(parc["p"], parc["td"], "g.", ms=3.5, alpha=0.35)
        pm, tm = _binned_median(parc["p"], parc["t"], bin_hpa)
        _, dm = _binned_median(parc["p"], parc["td"], bin_hpa)
        skew.plot(pm, tm, "r-", lw=2, label=f"HYSPLIT T (n={parc['n']})")
        skew.plot(pm, dm, "g-", lw=2, label="HYSPLIT Td")
    if m2near is not None:
        pl = m2near["lev"].values
        tt = m2near["T"].values - 273.15
        td = mpcalc.dewpoint_from_specific_humidity(
            pl * units.hPa, m2near["QV"].values * units("kg/kg")).magnitude
        fin = np.isfinite(tt)  # 1000 hPa is below ground here -> NaN
        if fin.sum() > 10:  # dense subset: profile lines
            skew.plot(pl[fin], tt[fin], color="darkred", ls="--", lw=1.6,
                      label="MERRA-2 T (42-lev)")
            fin_d = fin & np.isfinite(td)
            skew.plot(pl[fin_d], td[fin_d], color="darkgreen", ls="--",
                      lw=1.6, label="MERRA-2 Td (42-lev)")
        else:  # compact corpus: 5-level anchor circles
            skew.plot(pl, tt, "ro", mfc="none", ms=9, mew=2, label="MERRA-2 T")
            skew.plot(pl, td, "go", mfc="none", ms=9, mew=2, label="MERRA-2 Td")
        bsel = fin & (pl >= ylim[1]) & (pl <= ylim[0])
        skew.plot_barbs(pl[bsel], m2near["U"].values[bsel],
                        m2near["V"].values[bsel])
    p_of_z = (parc["p_of_z"] if parc is not None
              else (lambda z: 1013.25 * np.exp(-z / 8400.0)))
    _add_level_markers(skew, col, p_of_z, fontsize=label_fs)
    skew.ax.set_ylim(*ylim)
    skew.ax.set_xlim(*xlim)
    skew.plot_dry_adiabats(alpha=0.15)
    skew.plot_moist_adiabats(alpha=0.15)
    skew.plot_mixing_lines(alpha=0.15)
    if legend:
        skew.ax.legend(fontsize=8, loc="upper left")


def plot_skewts(out: Path, day: pd.Timestamp, point: tuple[float, float],
                cell: tuple[float, float], traj, m2,
                table: pd.DataFrame) -> list[Path]:
    """One figure PER forecast hour: full Skew-T (to 200 hPa) + BL zoom panel.

    The boundary layer cannot be "stretched" on a Skew-T without breaking the
    45-degree isotherm geometry, so the accepted practice is a companion
    zoomed panel: right subplot repeats the sounding over 1050-750 hPa with
    finer parcel binning, where the PBL/LCL/LFC structure is actually legible.
    """
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    lat, lon = cell
    paths = []
    for step in range(7):
        parc = (_cell_parcels(traj, step, cell, mpcalc, units)
                if traj is not None else None)
        panel_time = (parc["time"] if parc is not None else
                      slot_datetimes(day, (0,) + tuple(
                          cs_config.FORECAST_SLOTS))[step])
        m2near = (m2.sel(time=panel_time, method="nearest")
                  if m2 is not None else None)

        # BL-zoom x-range from the data actually below 750 hPa
        lo_t = [v for src in (
            (parc["t"][parc["p"] >= 750], parc["td"][parc["p"] >= 750])
            if parc is not None else ()) for v in src]
        if m2near is not None:
            pl = m2near["lev"].values
            lo_t += list(m2near["T"].values[(pl >= 750) & (pl <= 1000)] - 273.15)
        lo_t = np.asarray([v for v in lo_t if np.isfinite(v)])
        zoom_xlim = ((float(lo_t.min()) - 2, float(lo_t.max()) + 4)
                     if lo_t.size else (0, 35))

        # CAPE/CIN integrations: one from the HYSPLIT binned-median
        # sounding, one from the dense MERRA-2 profile (each shades its own
        # panel)
        pa_hy = None
        if parc is not None:
            pm, tm = _binned_median(parc["p"], parc["t"], 10)
            _, dm = _binned_median(parc["p"], parc["td"], 10)
            pa_hy = _parcel_analysis(pm[::-1], tm[::-1], dm[::-1], mpcalc,
                                     units)
        pa_m2 = None
        if m2near is not None:
            pl = m2near["lev"].values
            tt = m2near["T"].values - 273.15
            td = mpcalc.dewpoint_from_specific_humidity(
                pl * units.hPa, m2near["QV"].values * units("kg/kg")).magnitude
            fin = np.isfinite(tt) & np.isfinite(td) & (pl >= 100)
            pa_m2 = _parcel_analysis(pl[fin], tt[fin], td[fin], mpcalc, units)

        # layout: left column = HYSPLIT full skew-T over a half-height BL
        # zoom; right column (full height) = the MERRA-2 skew-T, side by side
        import matplotlib.gridspec as mgridspec
        fig = plt.figure(figsize=(15, 10))
        gs = mgridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1],
                                hspace=0.18, wspace=0.16)
        col = table.iloc[:, step]

        skew_full = SkewT(fig, rotation=45, subplot=gs[0, 0])
        _draw_panel(skew_full, parc, None, col, mpcalc, units,
                    ylim=(1050, 200), xlim=(-30, 45), bin_hpa=25,
                    label_fs=7.5, legend=True, pa=pa_hy)
        skew_full.ax.set_title("HYSPLIT/AIRS advected sounding (to 200 hPa)",
                               fontsize=11)
        note = [f"dataset MU CAPE {col.get('MU CAPE [J/kg]', np.nan):.0f} / "
                f"CIN {col.get('MU CIN [J/kg]', np.nan):.0f} J/kg"]
        if pa_hy is not None:
            note.insert(0, f"profile MU CAPE {pa_hy['cape']:.0f} / "
                           f"CIN {pa_hy['cin']:.0f} J/kg (shaded)")
        skew_full.ax.text(0.02, 0.02, "\n".join(note), fontsize=8.5,
                          transform=skew_full.ax.transAxes, va="bottom",
                          bbox=dict(fc="white", ec="0.6", alpha=0.85))

        # BL zoom drawn UNSKEWED (rotation=0, emagram-style): metpy's 45-deg
        # shear is an axes-space transform, so a stretched shallow layer
        # cannot keep the skew geometry; unskewed, temperature reads straight
        # off the x-axis and the fine BL structure fills the panel.
        skew_zoom = SkewT(fig, rotation=0, subplot=gs[1, 0], aspect="auto")
        _draw_panel(skew_zoom, parc, None, col, mpcalc, units,
                    ylim=(1050, 750), xlim=zoom_xlim, bin_hpa=10,
                    label_fs=9, pa=pa_hy)
        skew_zoom.ax.set_title("boundary layer zoom (1050-750 hPa, unskewed)",
                               fontsize=11)

        skew_m2 = SkewT(fig, rotation=45, subplot=gs[:, 1])
        _draw_panel(skew_m2, None, m2near, col, mpcalc, units,
                    ylim=(1050, 200), xlim=(-30, 45), bin_hpa=25,
                    label_fs=7.5, legend=True, pa=pa_m2)
        m2_when = (f"{pd.Timestamp(m2near['time'].item()):%H:%M}Z"
                   if m2near is not None else "n/a")
        skew_m2.ax.set_title(f"MERRA-2 sounding ({m2_when}, 42-lev)",
                             fontsize=11)
        if pa_m2 is not None:
            skew_m2.ax.text(0.02, 0.02,
                            f"MERRA-2 MU CAPE {pa_m2['cape']:.0f} / "
                            f"CIN {pa_m2['cin']:.0f} J/kg (shaded)",
                            fontsize=8.5, transform=skew_m2.ax.transAxes,
                            va="bottom",
                            bbox=dict(fc="white", ec="0.6", alpha=0.85))

        when = (f"{parc['time']:%H:%M}Z {parc['time']:%b-%d}" if parc is not None
                else f"{panel_time:%H:%M}Z {panel_time:%b-%d}")
        fig.suptitle(
            f"Skew-T log-P, step {step} ({when}) -- cell {lat:g}N "
            f"{abs(lon):g}W; left = HYSPLIT/AIRS parcels + BL zoom, right = MERRA-2;"
            " lines = dataset MU LCL/LFC/EL + Guo PBLH; shading = "
            "profile-integrated MU CAPE (red) / CIN (blue)", fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        tag = "ovp" if step == 0 else f"{panel_time:%H}Z"
        path = out / (f"skewt_{day:%Y%m%d}_{lat:g}N_{abs(lon):g}W_"
                      f"s{step}_{tag}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


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
    m2 = merra2_point(day, args.point[0], args.point[1], dense_dir=out)
    if m2 is None:
        print("      no MERRA-2 daily files found -> MERRA-2 overlay skipped")

    print("[3/3] Skew-T panels + report ...")
    fig_paths = []
    if traj is not None or m2 is not None:
        fig_paths = plot_skewts(out, day, tuple(args.point), cell, traj, m2,
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
    if fig_paths:
        lines += ["", "Skew-T figures (one per forecast hour, full column + "
                  "boundary-layer zoom):"]
        lines += [f"- `{p.name}`" for p in fig_paths]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
