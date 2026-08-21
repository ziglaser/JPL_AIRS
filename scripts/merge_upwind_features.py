#!/usr/bin/env python
"""Merge per-day upwind kernel features into a yearly companion file, and add
the trajectory-free features (Gamma_gap, assessed PBLH, PBLH anomaly).

Produces ``UPWIND_FEATURES_<YYYY>.nc`` -- a COMPANION to
``FCST_SMAP_MRMS_<YYYY>.nc`` on the same ``(date, time, lat, lon)`` axes, every
variable prefixed ``UPW_``. The source match-up file is never modified.

Two ingredient streams (UPWIND_INDEX_REVIEW.md section 4.2, merge step):

1. **Kernel-borne features** from the per-day slurm sweep
   (``scripts/build_upwind_features.py``): one small netCDF per day,
   ``<daily-dir>/UPW_<YYYYMMDD>.nc``, variables on
   ``(arrival_step 1..6, target_lat, target_lon)``. Arrival step ``s`` is the
   FCST ``time`` index ``s``: the match-up ``time`` axis is 0 = the AIRS
   overpass itself (per-cell clock, two swaths ~18:5x and ~20:3x UTC) and
   1..6 = the fixed forecast valid times 21, 22, 23, 00, 01, 02 UTC (verified
   against ``FCST_parceltime``, 2019-06-05). Dates without a daily file stay
   NaN -- honest gaps, counted in the global attrs, never interpolated.
   Daily-file variables on other dims (``lag_weight``, which carries an extra
   ``lag`` axis) are skipped and their names recorded in the companion attr
   ``daily_vars_skipped``; the QA battery reads them from the daily dir.

2. **Trajectory-free features**, computed here because they need no kernel
   (review 1.8, F2): the PBL-top/free-convection race
   ``Gamma_gap = z_LFC - z_i`` per (date, slot, cell), plus the assessed PBLH
   itself and its anomaly against the monthly-diurnal climatology.

Datum note (Gamma_gap validity). Both terms are heights above ground: the Guo
et al. (2024) PBLH is AGL by construction, and the AIRS forecast LFC datum was
verified AGL empirically (2026-08-19 check: ``FCST_alt`` ~200 m in both the
high plains and the southeast, and the plains-vs-southeast LCL contrast matches
the physical AGL contrast, not the ~1.3 km terrain offset an ASL datum would
impose). So the subtraction needs no terrain correction.

Timekeeping. Every PBLH lookup happens at the TRUE datetime of the slot, taken
from ``FCST_parceltime`` where present (per-cell overpass clock at slot 0) and
otherwise rebuilt as date + nominal slot hour -- on a real datetime axis, so
the 00-02 UTC slots land on the NEXT calendar day automatically. Nearest
3-hourly sample within ``config.PBLH_TIME_TOLERANCE_H`` (1.5 h), via the very
same lookup the production PBL model (:class:`trajectory_kernels.pbl.GriddedPBL`)
uses, so this merge and the kernel sweep can never disagree about what "the
assessed PBLH" was.

NaN policy (never fabricate). Where the 3-hourly assessed file has no coverage
(all of 2016, the Oct-2021 hole, a missing file entirely), ``UPW_pblh``,
``UPW_pblh_anom`` and both ``UPW_gamma_gap_*`` are NaN and counted in attrs --
NEVER filled from climatology: an anomaly of climatology from climatology is
identically 0, which would fabricate certainty exactly where there is none.

Why ``UPW_pblh_anom`` exists at all (Zach, 19 Aug): it is the LFC-free carrier
of the geometric pathway in case the AIRS LFC is bad. By linearity of the
pooling, the 1-degree assessed value minus the 1-degree climatology equals the
1-degree aggregate of the native 0.25-degree anomaly -- no information is lost
to the order of aggregation and differencing.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_kernels import config  # noqa: E402
from trajectory_kernels.pbl import GriddedPBL  # noqa: E402

#: Nominal UTC valid hour per FCST ``time`` index, as hours after 00 UTC of
#: ``date`` -- slots 4..6 are 24..26, i.e. 00-02 UTC of the NEXT calendar day.
#: Slot 0 (the overpass) has no nominal clock: its true time is per-cell and
#: spans ~1.7 h between the two Aqua swaths, so where ``FCST_parceltime`` is
#: missing there, the time (and every PBLH-derived feature) stays NaN rather
#: than being invented. Verified against FCST_parceltime on 2019-06-05.
SLOT_HOURS_AFTER_DATE: dict[int, int] = {1: 21, 2: 22, 3: 23, 4: 24, 5: 25, 6: 26}

#: feature_tier fallback for daily-file variables that predate the attr, and
#: the authoritative tiers for the trajectory-free variables (review 1.9).
FEATURE_TIERS: dict[str, str] = {
    "psi_anom": "core", "omega": "core",
    "phi": "ablation", "m_star": "ablation",
    "psi_raw": "ablation", "s_endpoint_raw": "ablation",
    "coverage": "honesty", "n_parcels": "honesty",
    "s_endpoint_anom": "honesty", "psi_meso_anom": "honesty",
    "containment_applied": "honesty",
    "upwind_dlat": "honesty", "upwind_dlon": "honesty",
    "upwind_km": "honesty", "mean_lag_hours": "honesty",
    "psi_land": "honesty",
    "gamma_gap_mml": "core", "gamma_gap_mu": "core",
    "pblh_anom": "core", "pblh": "honesty",
    "omega_anom": "ablation",
}


# --------------------------------------------------------------------------- #
# Stream 1: the per-day kernel feature files
# --------------------------------------------------------------------------- #
def discover_daily_files(daily_dir: Path | None, dates: np.ndarray) -> dict:
    """Map each date of the year to its ``UPW_<YYYYMMDD>.nc`` file, if present.

    Missing files are the expected state early in the sweep (and permanently
    for days with no HYSPLIT run); they become NaN slabs, not errors.
    """
    found: dict[np.datetime64, Path] = {}
    if daily_dir is None:
        return found
    daily_dir = Path(daily_dir)
    if not daily_dir.is_dir():
        warnings.warn(f"daily dir {daily_dir} does not exist; "
                      "all kernel-borne features will be NaN")
        return found
    for d in dates:
        stamp = str(np.datetime_as_string(d, unit="D")).replace("-", "")
        path = daily_dir / f"UPW_{stamp}.nc"
        if path.exists():
            found[d] = path
    return found


def load_daily_features(daily_files: dict, dates: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray,
                        n_slots: int) -> tuple[dict, dict, set]:
    """Assemble kernel-borne variables onto (date, time, lat, lon), NaN-padded.

    The daily contract: variables on ``(arrival_step, target_lat, target_lon)``
    with ``arrival_step`` values in 1..6, ``target_lat``/``target_lon`` equal to
    the match-up grid. EVERY variable on those dims is carried (the merge takes
    the daily file's word for what a feature is); the variable union across
    files is taken, so a day built without a PBL model (no omega/phi/m_star)
    simply leaves those slabs NaN for that date. Arrival step ``s`` is inserted
    at ``time=s``; slot 0 (the overpass -- no forward path, no kernel) is NaN
    by construction.

    Variables on OTHER dims are skipped gracefully and their names recorded:
    ``lag_weight`` (on (arrival_step, target_lat, target_lon, lag)) is an
    expected daily-file resident with no home on the (date, time, lat, lon)
    axes -- the QA battery reads it from the daily dir directly. The skip is
    silent (it happens for every file of the sweep) but never invisible: the
    names land in the companion attr ``daily_vars_skipped``.

    Returns
    -------
    arrays : dict of str -> float32 ndarray (date, time, lat, lon)
    var_attrs : dict of str -> dict, attrs propagated from the daily files.
    skipped : set of str, variable names left out for having other dims.
    """
    arrays: dict[str, np.ndarray] = {}
    var_attrs: dict[str, dict] = {}
    skipped: set[str] = set()
    date_index = {d: i for i, d in enumerate(dates)}
    shape = (dates.size, n_slots, lat.size, lon.size)

    for d, path in sorted(daily_files.items()):
        with xr.open_dataset(path) as ds:
            if not (np.allclose(ds["target_lat"].values, lat)
                    and np.allclose(ds["target_lon"].values, lon)):
                raise ValueError(f"{path}: target grid does not match the "
                                 "FCST_SMAP_MRMS lat/lon axes")
            steps = ds["arrival_step"].values.astype(int)
            if steps.min() < 1 or steps.max() >= n_slots:
                raise ValueError(f"{path}: arrival_step {steps} outside the "
                                 f"FCST time axis 1..{n_slots - 1}")
            for name, da in ds.data_vars.items():
                if set(da.dims) != {"arrival_step", "target_lat", "target_lon"}:
                    skipped.add(name)
                    continue
                if name not in arrays:
                    arrays[name] = np.full(shape, np.nan, dtype=np.float32)
                    var_attrs[name] = dict(da.attrs)
                vals = da.transpose("arrival_step", "target_lat",
                                    "target_lon").values.astype(np.float32)
                arrays[name][date_index[d], steps] = vals
    return arrays, var_attrs, skipped


# --------------------------------------------------------------------------- #
# Stream 2: trajectory-free features (Gamma_gap, PBLH, PBLH anomaly)
# --------------------------------------------------------------------------- #
def slot_datetimes(fcst: xr.Dataset) -> np.ndarray:
    """True datetime of every (date, time, lat, lon) sample, datetime64[ns].

    ``FCST_parceltime`` where present (it carries the per-cell overpass clock
    at slot 0 and the exact forecast valid times at slots 1..6); rebuilt as
    date + nominal slot hour where missing, on a real datetime axis so the
    00-02 UTC slots roll into the next calendar day -- never (date, hour)
    pairs. Slot 0 without a parceltime stays NaT (see SLOT_HOURS_AFTER_DATE).
    """
    times = fcst["FCST_parceltime"].values.astype("datetime64[ns]").copy()
    dates = fcst["date"].values.astype("datetime64[ns]")
    for s, hours in SLOT_HOURS_AFTER_DATE.items():
        nominal = dates + np.timedelta64(hours, "h").astype("timedelta64[ns]")
        slab = times[:, s]
        gap = np.isnat(slab)
        slab[gap] = np.broadcast_to(nominal[:, None, None], slab.shape)[gap]
    return times


def trajectory_free_features(fcst: xr.Dataset, times: np.ndarray,
                             pblh_3hrly: Path, pblh_clim: Path
                             ) -> tuple[dict, dict, dict]:
    """Gamma_gap (MML and MU), assessed PBLH, and its climatological anomaly.

    Uses :class:`GriddedPBL`'s own nearest-cell / nearest-time-within-1.5 h
    lookups (its assessed and climatology layers directly, NOT its fallback
    chain): the fallback chain exists so a kernel integral can always evaluate,
    whereas a merged FEATURE must never silently substitute climatology for
    the day's state (review F2; the NaN policy in the module docstring).

    Returns ``(arrays, var_attrs, notes)``; all-NaN arrays with a warning when
    the assessed 3-hourly file is absent, so the companion file is always
    buildable and always honest about what it does not know.
    """
    shape = times.shape
    pblh = np.full(shape, np.nan, dtype=np.float32)
    pblh_anom = np.full(shape, np.nan, dtype=np.float32)
    notes: dict[str, str] = {}

    gp = GriddedPBL(three_hourly_path=pblh_3hrly, clim_path=pblh_clim)
    if gp.available["assessed"]:
        gp._assessed = gp._assessed.load()
        lat2d, lon2d = np.meshgrid(fcst["lat"].values.astype(float),
                                   fcst["lon"].values.astype(float),
                                   indexing="ij")
        lat4d = np.broadcast_to(lat2d, shape)
        lon4d = np.broadcast_to(lon2d, shape)
        valid = ~np.isnat(times)
        t_ns = times[valid].astype(np.int64)
        pblh[valid] = gp.assessed_lookup(lat4d[valid], lon4d[valid], t_ns)

        if gp.available["climatology"]:
            have = valid.copy()
            have[valid] = np.isfinite(pblh[valid])
            clim = gp.climatology_lookup(lat4d[have], lon4d[have],
                                          times[have].astype(np.int64))
            pblh_anom[have] = pblh[have] - clim
        else:
            notes["pblh_anom"] = (f"climatology file {pblh_clim} absent; "
                                  "UPW_pblh_anom is all NaN")
            warnings.warn(notes["pblh_anom"])
    else:
        notes["pblh"] = (
            f"assessed 3-hourly PBLH file {pblh_3hrly} absent or empty; "
            "UPW_pblh, UPW_pblh_anom and UPW_gamma_gap_* are all NaN "
            "(never filled from climatology)")
        warnings.warn(notes["pblh"])

    arrays = {
        "pblh": pblh,
        "pblh_anom": pblh_anom,
        "gamma_gap_mml": fcst["FCST_MML_LFC"].values.astype(np.float32) - pblh,
        "gamma_gap_mu": fcst["FCST_MU_LFC"].values.astype(np.float32) - pblh,
    }
    var_attrs = {
        "pblh": {
            "units": "m",
            "long_name": "assessed PBL depth (Guo et al. 2024, 1 deg 3-hourly) "
                         "at the true slot datetime, AGL",
        },
        "pblh_anom": {
            "units": "m",
            "long_name": "assessed PBLH minus monthly-diurnal climatology "
                         "(month, UTC 3 h slot, cell); the LFC-free carrier "
                         "of the geometric pathway",
        },
        "gamma_gap_mml": {
            "units": "m",
            "long_name": "Gamma_gap = FCST_MML_LFC - assessed PBLH; both AGL "
                         "(LFC datum verified empirically 2026-08-19), "
                         "no terrain correction",
        },
        "gamma_gap_mu": {
            "units": "m",
            "long_name": "Gamma_gap = FCST_MU_LFC - assessed PBLH; both AGL "
                         "(LFC datum verified empirically 2026-08-19), "
                         "no terrain correction",
        },
    }
    return arrays, var_attrs, notes


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_companion(fcst: xr.Dataset, daily_files: dict,
                    pblh_3hrly: Path, pblh_clim: Path) -> xr.Dataset:
    """The full companion dataset: both streams, tiered, on the FCST axes."""
    dates = fcst["date"].values
    lat = fcst["lat"].values
    lon = fcst["lon"].values
    n_slots = fcst.sizes["time"]

    kernel_arrays, kernel_attrs, skipped = load_daily_features(
        daily_files, dates, lat, lon, n_slots)
    times = slot_datetimes(fcst)
    free_arrays, free_attrs, notes = trajectory_free_features(
        fcst, times, pblh_3hrly, pblh_clim)

    out = xr.Dataset(coords={"date": fcst["date"], "time": fcst["time"],
                             "lat": fcst["lat"], "lon": fcst["lon"]})
    for name, vals in {**kernel_arrays, **free_arrays}.items():
        attrs = {**kernel_attrs, **free_attrs}.get(name, {})
        attrs.setdefault("feature_tier", FEATURE_TIERS.get(name, "honesty"))
        out["UPW_" + name] = xr.DataArray(
            vals, dims=("date", "time", "lat", "lon"), attrs=attrs)

    n_days = int(dates.size)
    out.attrs.update({
        "title": "Upwind soil-moisture features: companion to FCST_SMAP_MRMS "
                 "(UPWIND_INDEX_REVIEW.md section 4.2 merge step)",
        "source_fcst": str(fcst.encoding.get("source", "unknown")),
        "pblh_3hrly": f"{pblh_3hrly} ({'present' if Path(pblh_3hrly).exists() else 'ABSENT'})",
        "pblh_clim": f"{pblh_clim} ({'present' if Path(pblh_clim).exists() else 'ABSENT'})",
        "n_dates": n_days,
        "n_dates_with_daily_file": len(daily_files),
        "daily_coverage_fraction": round(len(daily_files) / n_days, 4),
        "daily_vars_skipped": ", ".join(sorted(skipped)) if skipped else "(none)",
        "nan_policy": "gaps are NaN and counted here, never interpolated or "
                      "filled from climatology (an anomaly of climatology from "
                      "climatology is identically 0 -- fabricated certainty)",
        "feature_tiers": "core = distinct physical axis; ablation = kept to "
                         "verify ~0 importance or run the psi_raw decision "
                         "rule; honesty = sampling/validity columns",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    for key, msg in notes.items():
        out.attrs[f"note_{key}"] = msg
    return out


# --------------------------------------------------------------------------- #
# Pass 2: the multi-year Omega climatology and UPW_omega_anom
# --------------------------------------------------------------------------- #
def build_omega_climatology(out_dir: Path, min_samples: int = 10,
                            out_path: Path | None = None) -> Path:
    """Pool UPW_omega from every merged year into a (month, slot, cell) mean.

    Omega is extensive in time, so its raw distribution shifts with arrival
    slot and season; the anomaly against this climatology is the
    slot-comparable variant (ablation tier -- the forest-native treatment is
    conditioning on the slot feature, review discussion 2026-08-19).

    MULTI-YEAR ONLY, enforced: a single year's climatology would subtract that
    specific month-of-that-year's mean, silently erasing the interannual
    signal (a uniformly dry June would read as neutral). Zach's requirement,
    19 Aug 2026. Cells/months pooling fewer than ``min_samples`` finite days
    are NaN, never thinly estimated.
    """
    files = sorted(Path(out_dir).glob("UPWIND_FEATURES_*.nc"))
    years = [f.stem.rsplit("_", 1)[-1] for f in files]
    if len(files) < 2:
        raise SystemExit(
            f"omega climatology needs >= 2 merged years, found {len(files)} in "
            f"{out_dir} ({years or 'none'}): merge more years first (the "
            "anomaly is only meaningful against a multi-year base).")
    omega = xr.concat(
        [xr.open_dataset(f)["UPW_omega"].load() for f in files], dim="date")

    month = omega["date"].dt.month
    mean = omega.groupby(month).mean("date")
    n = omega.notnull().groupby(month).sum("date").astype("int32")
    clim = mean.where(n >= min_samples)

    ds = xr.Dataset({
        "omega_clim": clim.assign_attrs(
            units="J kg-1",
            long_name="multi-year mean UPW_omega per (month, slot, cell)"),
        "omega_clim_n": n.assign_attrs(
            long_name="finite days pooled per (month, slot, cell)"),
    })
    ds.attrs.update({
        "source_files": ", ".join(f.name for f in files),
        "years": f"{years[0]}-{years[-1]}",
        "min_samples": int(min_samples),
        "purpose": "reference for UPW_omega_anom = UPW_omega - omega_clim; "
                   "rebuild after merging additional years",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    dest = out_path or Path(out_dir) / f"omega_clim_{years[0]}-{years[-1]}.nc"
    atomic_to_netcdf(ds, dest, encoding={v: {"zlib": True, "complevel": 4}
                                         for v in ds.data_vars})
    return dest


def omega_anomaly(omega: xr.DataArray, clim_ds: xr.Dataset,
                  clim_path: Path) -> xr.DataArray:
    """UPW_omega minus the multi-year (month, slot, cell) climatology, J/kg."""
    anom = omega.groupby("date.month") - clim_ds["omega_clim"]
    anom = anom.drop_vars("month", errors="ignore")
    anom.attrs = {
        "units": "J kg-1",
        "long_name": ("Omega anomaly vs the multi-year (month, slot, cell) "
                      "climatology: slot-comparable day-to-day departure"),
        "feature_tier": "ablation",
        "omega_clim": f"{clim_path} (years {clim_ds.attrs.get('years', '?')})",
    }
    return anom


def add_omega_anom(companion_path: Path, clim_path: Path) -> Path:
    """Amend an existing yearly companion with UPW_omega_anom (pass 2).

    Cheaper than re-merging: needs neither the FCST file nor the daily files,
    only the companion itself and the pooled climatology.
    """
    if not companion_path.exists():
        raise SystemExit(f"{companion_path} not found: merge that year first")
    with xr.open_dataset(companion_path) as src:
        out = src.load()
    with xr.open_dataset(clim_path) as clim_ds:
        out["UPW_omega_anom"] = omega_anomaly(out["UPW_omega"],
                                              clim_ds.load(), clim_path)
    out.attrs["omega_clim"] = str(clim_path)
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    atomic_to_netcdf(out, companion_path, encoding=encoding)
    return companion_path


def _newest_omega_clim(out_dir: Path) -> Path:
    hits = sorted(Path(out_dir).glob("omega_clim_*.nc"))
    if not hits:
        raise SystemExit(
            f"no omega_clim_*.nc in {out_dir}: run --build-omega-clim first "
            "(after merging >= 2 years), or pass --omega-clim explicitly.")
    return hits[-1]


def atomic_to_netcdf(ds: xr.Dataset, out_path: Path, encoding: dict | None = None) -> None:
    """Write ``ds`` to ``out_path`` atomically: tmp file in the same dir, then rename.

    A preemption mid-``to_netcdf`` leaves a truncated file at the final path,
    which the exists-check resume logic would then treat as complete forever.
    Writing to ``<name>.nc.tmp`` and renaming with ``os.replace`` (atomic on
    POSIX within one filesystem, which "same directory" guarantees) makes the
    final path appear only once the file is whole. Mirrors
    ``build_upwind_features.atomic_to_netcdf`` (scripts/ is not a package, so
    the ~10 lines are duplicated rather than imported).
    """
    tmp = out_path.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)  # survives only if the write/rename failed


def summarize(ds: xr.Dataset) -> str:
    """Per-variable finite fraction -- the one-look QA table."""
    lines = [f"{'variable':<28} {'tier':<10} finite"]
    for name in sorted(ds.data_vars):
        da = ds[name]
        frac = float(np.isfinite(da.values).mean())
        lines.append(f"{name:<28} {da.attrs.get('feature_tier', '?'):<10} {frac:6.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge per-day upwind kernel features and the "
                    "trajectory-free features into UPWIND_FEATURES_<YYYY>.nc, "
                    "a companion to FCST_SMAP_MRMS_<YYYY>.nc.")
    p.add_argument("--year", type=int, default=None,
                   help="year to merge/amend (required except with "
                        "--build-omega-clim)")
    p.add_argument("--daily-dir", type=Path,
                   default=config.RESULTS_DIR / "upwind_features" / "daily",
                   help="directory of per-day UPW_<YYYYMMDD>.nc files; the "
                        "default is exactly where build_upwind_features.py "
                        "writes them, so the two scripts cannot silently "
                        "disagree about the handoff location")
    p.add_argument("--no-daily", action="store_true",
                   help="intentionally trajectory-free build: skip the per-day "
                        "kernel features entirely (no daily dir read, no "
                        "missing-files warning)")
    p.add_argument("--fcst-dir", type=Path, default=config.FCST_TABLE_DIR)
    p.add_argument("--pblh-3hrly", type=Path, default=config.PBLH_3HRLY_PATH)
    p.add_argument("--pblh-clim", type=Path, default=config.PBLH_CLIM_PATH)
    p.add_argument("--out-dir", type=Path,
                   default=config.RESULTS_DIR / "upwind_features")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    # Pass 2 (after ALL years are merged): the multi-year Omega climatology
    # and the slot-comparable anomaly. Multi-year only, by design.
    p.add_argument("--build-omega-clim", action="store_true",
                   help="pool UPW_omega across every UPWIND_FEATURES_*.nc in "
                        "--out-dir into omega_clim_<y0>-<y1>.nc, then exit "
                        "(requires >= 2 merged years)")
    p.add_argument("--add-omega-anom", action="store_true",
                   help="amend the existing companion for --year with "
                        "UPW_omega_anom against --omega-clim (defaults to the "
                        "newest omega_clim_*.nc in --out-dir)")
    p.add_argument("--omega-clim", type=Path, default=None,
                   help="pooled multi-year climatology file; with a normal "
                        "merge, adds UPW_omega_anom in one pass (useful when "
                        "re-merging after the record was extended)")
    p.add_argument("--omega-clim-out", type=Path, default=None,
                   help="output path for --build-omega-clim (default: "
                        "<out-dir>/omega_clim_<y0>-<y1>.nc)")
    p.add_argument("--omega-min-samples", type=int, default=10,
                   help="minimum finite days per (month, slot, cell) pool; "
                        "thinner pools are NaN in the climatology")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)

    if args.build_omega_clim:
        dest = build_omega_climatology(args.out_dir, args.omega_min_samples,
                                       args.omega_clim_out)
        print(f"wrote {dest}")
        return dest
    if args.year is None:
        raise SystemExit("--year is required (except with --build-omega-clim)")
    out_path = args.out_dir / f"UPWIND_FEATURES_{args.year}.nc"
    if args.add_omega_anom:
        clim_path = args.omega_clim or _newest_omega_clim(args.out_dir)
        add_omega_anom(out_path, clim_path)
        print(f"amended {out_path} with UPW_omega_anom (clim: {clim_path})")
        return out_path
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")

    fcst_path = args.fcst_dir / f"FCST_SMAP_MRMS_{args.year}.nc"
    with xr.open_dataset(fcst_path) as fcst:
        fcst = fcst[["FCST_parceltime", "FCST_MML_LFC", "FCST_MU_LFC"]].load()
    daily_dir = None if args.no_daily else args.daily_dir
    daily_files = discover_daily_files(daily_dir, fcst["date"].values)
    if daily_dir is not None and not daily_files:
        # A daily dir was EXPECTED here; an all-NaN kernel tier must arrive
        # with a shout, not in silence (the trajectory-free mode is --no-daily).
        warnings.warn(
            f"expected per-day UPW_*.nc files for {args.year} in {daily_dir} "
            "but found none: every kernel-borne feature will be NaN. Run "
            "scripts/build_upwind_features.py first, or pass --no-daily for "
            "an intentionally trajectory-free build.")
    out = build_companion(fcst, daily_files, args.pblh_3hrly, args.pblh_clim)
    if args.omega_clim is not None:
        with xr.open_dataset(args.omega_clim) as clim_ds:
            out["UPW_omega_anom"] = omega_anomaly(out["UPW_omega"],
                                                  clim_ds.load(), args.omega_clim)
        out.attrs["omega_clim"] = str(args.omega_clim)
    out.attrs["daily_files_found"] = len(daily_files)
    out.attrs["daily_dir"] = "(none: --no-daily)" if daily_dir is None else str(daily_dir)
    out.attrs["command"] = "scripts/merge_upwind_features.py " + " ".join(
        argv if argv is not None else sys.argv[1:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    atomic_to_netcdf(out, out_path, encoding=encoding)
    print(f"wrote {out_path}")
    print(f"daily files: {len(daily_files)}/{out.attrs['n_dates']} "
          f"({out.attrs['daily_coverage_fraction']:.1%})")
    print(summarize(out))
    return out_path


if __name__ == "__main__":
    main()
