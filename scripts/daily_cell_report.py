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
    "pblh":          ("PBL depth (Guo 3-hrly, interp)", "m"),
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

    # assessed PBL depth at each slot's true datetime: LINEAR interpolation
    # between the bracketing 3-hourly samples (both finite and <=3 h away);
    # falls back to the nearest sample within PBLH_TOL_H at coverage edges
    pblh_vals = [np.nan] * len(slots)
    if tk_config.PBLH_3HRLY_PATH.exists():
        with xr.open_dataset(tk_config.PBLH_3HRLY_PATH) as pb:
            col = pb["pblh"].sel(lat=lat, lon=lon, method="nearest").load()
            tt = pd.to_datetime(col["time"].values)
            vv = col.values
            fin = np.isfinite(vv)
            for i, t in enumerate(times):
                before = fin & (tt <= t) & (tt >= t - pd.Timedelta(hours=3))
                after = fin & (tt >= t) & (tt <= t + pd.Timedelta(hours=3))
                if before.any() and after.any():
                    t0, v0 = tt[before][-1], vv[before][-1]
                    t1, v1 = tt[after][0], vv[after][0]
                    if t1 == t0:
                        pblh_vals[i] = float(v0)
                    else:
                        w = (t - t0) / (t1 - t0)
                        pblh_vals[i] = float((1 - w) * v0 + w * v1)
                else:  # edge of coverage: previous nearest-sample behavior
                    near = col.sel(time=t, method="nearest")
                    if abs(pd.Timestamp(near["time"].item()) - t) <= (
                            pd.Timedelta(hours=PBLH_TOL_H)):
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


#: level-marker styling: short name -> (color, linestyle, FIXED label x-slot
#: in axes fraction, staggered so close-packed lines never collide). Values
#: are dataset-specific per panel (AIRS-FCST levels on the AIRS panel,
#: MERRA-2-derived levels on the MERRA-2 panel); only the Guo PBLH height is
#: shared, converted to pressure through each panel's own column.
#: (color, linestyle, label x-slot, label side): PBLH labels BELOW its line,
#: parcel levels ABOVE theirs, so near-coincident lines never collide
LEVEL_STYLE = {
    "MU LCL": ("tab:orange", "-", 0.26, "bottom"),
    "MU LFC": ("tab:purple", "-", 0.46, "bottom"),
    "PBLH":   ("tab:blue", "-", 0.62, "top"),
}


def _add_level_markers(skew, entries, fontsize=11) -> None:
    """Horizontal level lines from ``entries`` = [(name, pressure_hPa,
    height_m_AGL), ...]; labels at each name's fixed x-slot."""
    for name, p_hpa, z_m in entries:
        if not np.isfinite(p_hpa):
            continue
        color, ls, x0, side = LEVEL_STYLE[name]
        skew.ax.axhline(p_hpa, color=color, ls=ls, lw=1.4, alpha=0.85)
        z_txt = f" {z_m:.0f}m" if np.isfinite(z_m) else ""
        skew.ax.text(x0, p_hpa, f"{name}{z_txt}", color=color,
                     fontsize=fontsize, fontweight="bold",
                     va=side, ha="left", clip_on=True,
                     transform=skew.ax.get_yaxis_transform(),
                     bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


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
    # ONE canonical ENVIRONMENT sounding for the drawn lines and the CAPE/CIN
    # environment: 10-hPa bin medians + three Hann passes. The smoothing is
    # free of CAPE cost because the parcel is launched separately from the
    # raw max-theta-e parcel (lowest 300 hPa) -- the dataset's per-parcel MU
    # convention (scheme selection experiment 2026-08-21, see
    # _parcel_analysis docstring).
    pm, tm = _binned_median(p, t, 10)
    _, dm = _binned_median(p, td, 10)
    for _ in range(3):
        tm, dm = _hann3(tm), _hann3(dm)
    low = p >= (p.max() - 300.0)
    theta_e = mpcalc.equivalent_potential_temperature(
        p[low] * units.hPa, t[low] * units.degC,
        td[low] * units.degC).magnitude
    imu = int(np.argmax(theta_e))
    mu_launch = (float(p[low][imu]), float(t[low][imu]), float(td[low][imu]))
    return dict(p=p, t=t, td=td, z=z, n=int(inside.sum()), time=time,
                pm=pm, tm=tm, dm=dm, mu_launch=mu_launch,
                p_of_z=lambda h, zz=z[zo], pp=p[zo]: float(np.interp(h, zz, pp)))


def _parcel_analysis(p, t, td, mpcalc, units, launch=None):
    """Parcel path + CAPE/CIN from a sounding, on a fine 2-hPa grid.

    ``launch=(p0, t0, td0)`` launches from an explicit parcel (the raw
    max-theta-e parcel -- the dataset's per-parcel MU convention; chosen
    2026-08-21 by scoring against the dataset MU CAPE: raw-parcel launch MAE
    ~85 J/kg regardless of environment smoothing, median-profile launch
    degrades to 140-204 as smoothing strengthens). Without ``launch`` the MU
    parcel is taken from the profile itself (MERRA-2 panel). The fine grid
    matters for the SHADING: metpy fills only between supplied levels, so a
    coarse grid leaves gaps at the LFC/EL crossover tips.

    Input arrays are surface-first (descending pressure). Returns None when
    the profile is too short/broken for metpy's parcel routines.
    """
    try:
        P = np.asarray(p, dtype=float)
        T = np.asarray(t, dtype=float)
        D = np.asarray(td, dtype=float)
        if launch is None:
            _, _, _, idx = mpcalc.most_unstable_parcel(
                P * units.hPa, T * units.degC, D * units.degC)
            p0, t0, d0 = P[idx], T[idx], D[idx]
            P, T, D = P[idx:], T[idx:], D[idx:]
        else:
            p0, t0, d0 = launch
            keep = P < p0
            P = np.r_[p0, P[keep]]
            T = np.r_[t0, T[keep]]
            D = np.r_[d0, D[keep]]
        # fine 2-hPa grid from the launch level to the profile top
        pf = np.arange(P[0], P.min(), -2.0)
        lp = np.log(P[::-1])
        envf = np.interp(np.log(pf), lp, T[::-1])
        envdf = np.interp(np.log(pf), lp, D[::-1])
        prof = mpcalc.parcel_profile(pf * units.hPa, t0 * units.degC,
                                     d0 * units.degC).to("degC")
        cape, cin = mpcalc.cape_cin(pf * units.hPa, envf * units.degC,
                                    envdf * units.degC, prof)
        lcl_p, _ = mpcalc.lcl(pf[0] * units.hPa, t0 * units.degC,
                              d0 * units.degC)
        try:
            lfc_p, _ = mpcalc.lfc(pf * units.hPa, envf * units.degC,
                                  envdf * units.degC, prof)
            lfc_p = float(lfc_p.to("hPa").magnitude)
        except Exception:
            lfc_p = np.nan
        return dict(p=pf * units.hPa, env_t=envf * units.degC,
                    env_td=envdf * units.degC, prof=prof,
                    lcl_p=float(lcl_p.to("hPa").magnitude), lfc_p=lfc_p,
                    cape=float(cape.magnitude), cin=float(cin.magnitude))
    except Exception as err:
        print(f"      parcel analysis skipped ({err})")
        return None


def _hann3(x):
    """One 3-point Hann (0.25/0.5/0.25) smoothing pass, edge-preserving."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    pad = np.r_[x[0], x, x[-1]]
    return 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]


def _binned_median(p, x, bin_hpa: float, min_n: int = 1):
    """Median profile on a fixed pressure grid. ``min_n=1`` keeps every bin
    with data, so the line reaches the top/bottom of the parcel cloud and the
    profile has no interior gaps (a >=3 rule used to truncate/perforate it)."""
    bins = np.arange(100, 1055, bin_hpa)
    idx = np.digitize(p, bins)
    pm, xm = [], []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= min_n:
            pm.append(np.median(p[sel]))
            xm.append(np.median(x[sel]))
    return pm, xm


def _draw_panel(skew, parc, m2near, col, mpcalc, units,
                ylim, xlim, bin_hpa, label_fs, legend=False, pa=None,
                ground_hpa=None, level_entries=()):
    """One Skew-T panel: parcels, MERRA-2 profile, CAPE/CIN, level markers."""
    if pa is not None:  # MU parcel path + shaded CAPE (red) / CIN (blue)
        skew.plot(pa["p"], pa["prof"], "k-", lw=1.6, alpha=0.9,
                  label="MU parcel path")
        skew.shade_cape(pa["p"], pa["env_t"], pa["prof"], alpha=0.18)
        # dewpoint arg limits the CIN shading to below the LFC (without it
        # metpy also shades the parcel-colder-than-environment area above EL)
        skew.shade_cin(pa["p"], pa["env_t"], pa["prof"], pa["env_td"],
                       alpha=0.22)
    if parc is not None:
        skew.plot(parc["p"], parc["t"], "r.", ms=3.5, alpha=0.35)
        skew.plot(parc["p"], parc["td"], "g.", ms=3.5, alpha=0.35)
        skew.plot(parc["pm"], parc["tm"], "r-", lw=2,
                  label="Environmental Temp")
        skew.plot(parc["pm"], parc["dm"], "g-", lw=2, label="Dew Point")
    if m2near is not None:
        pl = m2near["lev"].values
        tt = m2near["T"].values - 273.15
        td = mpcalc.dewpoint_from_specific_humidity(
            pl * units.hPa, m2near["QV"].values * units("kg/kg")).magnitude
        fin = np.isfinite(tt)  # 1000 hPa is below ground here -> NaN
        if fin.sum() > 10:  # dense subset: profile lines
            skew.plot(pl[fin], tt[fin], color="darkred", ls="--", lw=1.6,
                      label="Environmental Temp")
            fin_d = fin & np.isfinite(td)
            skew.plot(pl[fin_d], td[fin_d], color="darkgreen", ls="--",
                      lw=1.6, label="Dew Point")
        else:  # compact corpus: 5-level anchor circles
            skew.plot(pl, tt, "ro", mfc="none", ms=9, mew=2, label="MERRA-2 T")
            skew.plot(pl, td, "go", mfc="none", ms=9, mew=2, label="MERRA-2 Td")
        bsel = fin & (pl >= ylim[1]) & (pl <= ylim[0])
        skew.plot_barbs(pl[bsel], m2near["U"].values[bsel],
                        m2near["V"].values[bsel])
    if ground_hpa is not None and np.isfinite(ground_hpa):
        # the ground: everything below the MERRA-2 surface pressure
        skew.ax.axhspan(ground_hpa, ylim[0] + 20, facecolor="tan",
                        edgecolor="saddlebrown", hatch="///", lw=0,
                        alpha=0.5, zorder=1.5)
    _add_level_markers(skew, level_entries, fontsize=label_fs)
    skew.ax.set_ylim(*ylim)
    skew.ax.set_xlim(*xlim)
    skew.ax.set_xlabel("Temperature ($^\\circ$C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.plot_dry_adiabats(alpha=0.15)
    skew.plot_moist_adiabats(alpha=0.15)
    skew.plot_mixing_lines(alpha=0.15)
    if legend:
        skew.ax.legend(fontsize=8, loc="upper left")


def _tag_cape_cin(skew, pa, ds_cape=None, ds_cin=None) -> None:
    """Arrow tags pointing at the shaded CAPE/CIN areas, labeled with values.

    Replaces the corner value box: each tag anchors inside its shaded region
    (largest parcel-environment temperature difference of the right sign).
    """
    if pa is None:
        return
    prof = pa["prof"].magnitude
    env = pa["env_t"].magnitude
    pres = pa["p"].magnitude
    diff = prof - env
    if pa["cape"] > 0 and (diff > 0).any():
        i = int(np.argmax(diff))
        shown = (ds_cape if ds_cape is not None and np.isfinite(ds_cape)
                 else pa["cape"])
        label = f"CAPE {shown:.0f} J/kg"
        skew.ax.annotate(label, xy=(env[i] + 0.5 * diff[i], pres[i]),
                         xytext=(-60, -8), textcoords="offset points",
                         fontsize=13, fontweight="bold", color="darkred",
                         ha="right", zorder=7,
                         bbox=dict(fc="white", ec="darkred", alpha=1.0),
                         arrowprops=dict(arrowstyle="->", color="darkred",
                                         lw=1.2))
    # CIN tag: only the sub-LFC negative area (indices before the first REAL
    # positive crossing -- 0.5-K tolerance skips the zero at the parcel's own
    # start level), not the parcel-colder-than-environment layer aloft
    pos = np.where(diff > 0.5)[0]
    ipos = int(pos[0]) if pos.size else diff.size
    if pa["cin"] < -1 and ipos > 0:
        low = diff[:ipos]
        j = (int(np.argmin(low)) if (low < -0.1).any()
             else max(0, ipos // 2))  # smoothed profile: anchor mid-layer
        shown = (ds_cin if ds_cin is not None and np.isfinite(ds_cin)
                 else pa["cin"])
        label = f"CIN {shown:.0f} J/kg"
        skew.ax.annotate(label, xy=(env[j] + 0.5 * diff[j], pres[j]),
                         xytext=(-70, 22), textcoords="offset points",
                         fontsize=13, fontweight="bold", color="tab:blue",
                         ha="right", zorder=7,
                         bbox=dict(fc="white", ec="tab:blue", alpha=1.0),
                         arrowprops=dict(arrowstyle="->", color="tab:blue",
                                         lw=1.2))


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
            pa_hy = _parcel_analysis(np.asarray(parc["pm"])[::-1],
                                     np.asarray(parc["tm"])[::-1],
                                     np.asarray(parc["dm"])[::-1],
                                     mpcalc, units, launch=parc["mu_launch"])
        pa_m2 = None
        if m2near is not None:
            pl = m2near["lev"].values
            tt = m2near["T"].values - 273.15
            td = mpcalc.dewpoint_from_specific_humidity(
                pl * units.hPa, m2near["QV"].values * units("kg/kg")).magnitude
            fin = np.isfinite(tt) & np.isfinite(td) & (pl >= 100)
            pa_m2 = _parcel_analysis(pl[fin], tt[fin], td[fin], mpcalc, units)

        # layout: 2x2 -- top row full skew-Ts (AIRS-FCST | MERRA-2), bottom
        # row half-height boundary-layer zooms under each
        import matplotlib.gridspec as mgridspec
        fig = plt.figure(figsize=(10.5, 7.6))
        gs = mgridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1],
                                hspace=0.18, wspace=0.16)
        col = table.iloc[:, step]
        ds_cape = float(col.get("MU CAPE [J/kg]", np.nan))
        ds_cin = float(col.get("MU CIN [J/kg]", np.nan))

        ps_hpa = (float(m2near["PS"]) / 100.0 if m2near is not None
                  and "PS" in m2near else np.nan)
        pblh_z = float(col.get("PBL depth (Guo 3-hrly, interp) [m]", np.nan))

        # AIRS-FCST levels: dataset heights -> pressure via the parcels' own
        # alt/pres relation
        p_of_z = (parc["p_of_z"] if parc is not None
                  else (lambda z: 1013.25 * np.exp(-z / 8400.0)))
        airs_entries = []
        for name, row in (("MU LCL", "MU LCL [m]"), ("MU LFC", "MU LFC [m]"),
                          ("PBLH", None)):
            z = pblh_z if row is None else float(col.get(row, np.nan))
            airs_entries.append(
                (name, p_of_z(z) if np.isfinite(z) else np.nan, z))

        # MERRA-2 levels: derived from ITS OWN profile (metpy LCL/LFC of the
        # MU parcel analysis); heights AGL via the file's geopotential H(p)
        # referenced to the surface pressure. Only the Guo PBLH is shared.
        merra_entries = []
        if m2near is not None and pa_m2 is not None and "H" in m2near:
            pl = m2near["lev"].values
            hh = m2near["H"].values
            fin = np.isfinite(hh)
            lp = np.log(pl[fin])
            oo = np.argsort(lp)
            lp, hh = lp[oo], hh[fin][oo]
            h_sfc = np.interp(np.log(ps_hpa), lp, hh)
            h_of_p = lambda q: float(np.interp(np.log(q), lp, hh) - h_sfc)  # noqa: E731
            hagl = hh - h_sfc
            ho = np.argsort(hagl)
            p_of_h = lambda z: float(np.exp(np.interp(z, hagl[ho], lp[ho])))  # noqa: E731
            for name, q in (("MU LCL", pa_m2["lcl_p"]),
                            ("MU LFC", pa_m2["lfc_p"])):
                merra_entries.append(
                    (name, q, h_of_p(q) if np.isfinite(q) else np.nan))
            if np.isfinite(pblh_z):
                merra_entries.append(("PBLH", p_of_h(pblh_z), pblh_z))
        skew_full = SkewT(fig, rotation=45, subplot=gs[0, 0])
        _draw_panel(skew_full, parc, None, col, mpcalc, units,
                    ylim=(1050, 200), xlim=(-30, 45), bin_hpa=25,
                    label_fs=10, legend=True, pa=pa_hy,
                    ground_hpa=ps_hpa)
        skew_full.ax.set_title("AIRS-FCST", fontsize=12)
        _tag_cape_cin(skew_full, pa_hy, ds_cape=ds_cape, ds_cin=ds_cin)

        # BL zooms drawn UNSKEWED (rotation=0, emagram-style): metpy's 45-deg
        # shear is an axes-space transform, so a stretched shallow layer
        # cannot keep the skew geometry; unskewed, temperature reads straight
        # off the x-axis and the fine BL structure fills the panel.
        skew_zoom = SkewT(fig, rotation=0, subplot=gs[1, 0], aspect="auto")
        _draw_panel(skew_zoom, parc, None, col, mpcalc, units,
                    ylim=(1050, 750), xlim=(-30, 45), bin_hpa=10,
                    label_fs=11, pa=pa_hy,
                    ground_hpa=ps_hpa, level_entries=airs_entries)


        skew_m2 = SkewT(fig, rotation=45, subplot=gs[0, 1])
        _draw_panel(skew_m2, None, m2near, col, mpcalc, units,
                    ylim=(1050, 200), xlim=(-30, 45), bin_hpa=25,
                    label_fs=10, legend=True, pa=pa_m2,
                    ground_hpa=ps_hpa)
        skew_m2.ax.set_title("MERRA-2", fontsize=12)
        _tag_cape_cin(skew_m2, pa_m2)

        skew_m2z = SkewT(fig, rotation=0, subplot=gs[1, 1], aspect="auto")
        _draw_panel(skew_m2z, None, m2near, col, mpcalc, units,
                    ylim=(1050, 750), xlim=(-30, 45), bin_hpa=10,
                    label_fs=11, pa=pa_m2,
                    ground_hpa=ps_hpa, level_entries=merra_entries)


        fig.tight_layout()
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
