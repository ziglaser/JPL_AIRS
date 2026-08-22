#!/usr/bin/env python
"""Compile the ONE big RF-ready netCDF: FCST_SMAP_MRMS + every derived metric.

Produces ``RF_dataset_<Y0>-<Y1>.nc`` (default under the gpfs scratch root,
the PARENT of $JPL_AIRS_DATA on gattaca2) holding, on the match-up file's own
``(date, time, lat, lon)`` axes:

1. **Every variable of FCST_SMAP_MRMS_<year>.nc, unchanged** -- the compile
   never re-derives or re-encodes the primary data, it concatenates the year
   files and APPENDS the derived variables, so any consumer can drop the
   separate year files entirely.
2. ``sm_anom`` -- the suite's timing-guarded surface soil-moisture index:
   pre-window L4 slots (16:30/19:30 UTC) mean of SMAP_L4_smsfc_av, 2-harmonic
   per-cell seasonal anomaly, z-scored per cell.  Computed by the SAME
   ``convection_skill.dataset`` helpers the hypothesis battery uses (per-year
   harmonic fit, exactly as in ``build_base_table``), never re-implemented.
3. ``UPW_psi_anom`` / ``UPW_omega`` -- the kernel-borne upwind predictors,
   read straight from the per-day ``UPW_<YYYYMMDD>.nc`` sweep output
   (``scripts/build_upwind_features.py``) via the merge script's own loader,
   so this file and any UPWIND_FEATURES companion can never disagree.  Days
   without a daily file are NaN (honest gaps, counted in attrs).
4. ``UPW_pblh`` / ``UPW_pblh_anom`` / ``UPW_gamma_gap_mu`` -- **INTERPOLATED**
   (user decision 2026-08-22): the assessed 3-hourly Guo-2024 PBLH is
   LINEARLY interpolated in time to each sample's true datetime
   (``FCST_parceltime`` where present, nominal slot hour otherwise), never
   nearest-neighbour, and never across gaps wider than one 3-hourly step
   (the Oct-2021 hole stays NaN -- fabricating it would erase exactly the
   anomaly signal these features carry).  The anomaly subtracts the
   monthly-diurnal climatology interpolated the same way over its diurnal
   axis.  Gamma_gap = FCST_MU_LFC - interpolated PBLH (both AGL; MU ONLY per
   the user's thermodynamics decision -- no MML twin here).
5. ``met_front_{type}_3w`` -- ANALYST fronts, NOAA-XML source (the only one
   with DRYLINES), 3-wide, via ``convection_skill.fronts.file_front_flags``
   (the exact alignment the base table and add_front_flags.py use).
6. ``pred_front_{type}_3w`` -- OUR MODEL's fronts from the kriged-AIRS D6C
   softmax-ensemble archive (``dlfront_D6C-ens3_kriged-airs``, the
   AIRS-driven product the full-sequence chain exports), same reader.

Front types: cold, warm, stationary, occluded, dryline, any.  3-wide only:
every model we trained saw 3wide labels, so 3wide is the only width with a
symmetric met-vs-pred comparison.

Idempotent: if the output exists the script prints the categorised variable
list and exits 0 (``--force`` rebuilds), so the SLURM chain can always run
"compile then train" unconditionally.  The variable list is also written to
``<output>.VARIABLES.md``.

Usage (cluster)::

    python scripts/compile_rf_dataset.py                 # 2016-2021 default
    python scripts/compile_rf_dataset.py --years 2019 --out /tmp/rf_smoke.nc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from convection_skill import config as cs_config                # noqa: E402
from convection_skill import dataset as cs_dataset              # noqa: E402
from convection_skill import fronts as cs_fronts                # noqa: E402
from trajectory_kernels import config as tk_config              # noqa: E402
import merge_upwind_features as muf                             # noqa: E402

FRONT_TYPES = ("cold", "warm", "stationary", "occluded", "dryline")
#: kernel-borne daily-file variables carried into the big file (the user's
#: ask: psi and omega; everything else in the daily files stays there).
KERNEL_VARS = ("psi_anom", "omega")
DEFAULT_PRED_TAG = "dlfront_D6C-ens3_kriged-airs"
#: interpolation refuses to bridge more than one 3-hourly step (+ slack for
#: the file's own timestamp jitter); wider gaps (Oct-2021) stay NaN.
MAX_INTERP_GAP_H = 3.05


def default_out_path(years: tuple[int, ...]) -> Path:
    """<gpfs scratch root>/RF_dataset_<Y0>-<Y1>.nc.

    On gattaca2 JPL_AIRS_DATA=<root>/AIRS_SMAP_Front_data, and the user asked
    for the file to live in <root> (/gpfs/scratch/smap-convection) itself; on
    any other layout fall back to DATA_DIR so a dev run stays self-contained.
    """
    data = cs_config.DATA_DIR
    root = data.parent if data.name == "AIRS_SMAP_Front_data" else data
    return root / f"RF_dataset_{years[0]}-{years[-1]}.nc"


# --------------------------------------------------------------------------- #
# Interpolated PBLH (the "(interpolated)" in the user's spec, 2026-08-22)
# --------------------------------------------------------------------------- #
def _gather_time_series(field: np.ndarray, t_idx: np.ndarray,
                        cell_col: np.ndarray) -> np.ndarray:
    """field (T, nlat*nlon) values at paired (time index, flat cell) samples."""
    return field.reshape(field.shape[0], -1)[t_idx, cell_col]


def interp_pblh_to_times(times: np.ndarray, lats: np.ndarray,
                         lons: np.ndarray, three_hourly_path: Path,
                         max_gap_h: float = MAX_INTERP_GAP_H) -> np.ndarray:
    """Assessed PBLH linearly interpolated in time at every (d,s,lat,lon).

    ``times`` is the (date, time, lat, lon) true-datetime array from
    ``merge_upwind_features.slot_datetimes`` (NaT where undefined).  For each
    sample the two bracketing 3-hourly assessed slices are combined with the
    exact linear weight; a sample is NaN when either bracket is missing/NaN
    or the brackets are further apart than ``max_gap_h`` (never fabricate
    across the product's holes).  Cells are matched nearest on the assessed
    file's own 1-degree grid (both grids are X.5-centred, so nearest = exact).
    """
    out = np.full(times.shape, np.nan, dtype=np.float32)
    if not Path(three_hourly_path).exists():
        warnings.warn(f"assessed PBLH file {three_hourly_path} absent; "
                      "interpolated PBL variables are all NaN")
        return out
    with xr.open_dataset(three_hourly_path) as f:
        p = f["pblh"].sel(lat=lats, lon=lons, method="nearest").load()
    t_axis = p["time"].values.astype("datetime64[ns]").astype(np.int64)
    field = p.values.astype(np.float64)  # (T, nlat, nlon)

    valid = ~np.isnat(times)
    t = times[valid].astype("datetime64[ns]").astype(np.int64)
    # flat cell index of every valid sample: times is (date, slot, lat, lon)
    ii, jj = np.meshgrid(np.arange(lats.size), np.arange(lons.size),
                         indexing="ij")
    cell = np.broadcast_to(ii * lons.size + jj, times.shape)[valid]

    hi = np.searchsorted(t_axis, t, side="right")
    lo = hi - 1
    ok = (lo >= 0) & (hi < t_axis.size)
    lo_c, hi_c = np.clip(lo, 0, t_axis.size - 1), np.clip(hi, 0,
                                                          t_axis.size - 1)
    gap = t_axis[hi_c] - t_axis[lo_c]
    ok &= gap <= int(max_gap_h * 3.6e12)  # ns
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gap > 0, (t - t_axis[lo_c]) / np.where(gap > 0, gap, 1),
                     0.0)
    v_lo = _gather_time_series(field, lo_c, cell)
    v_hi = _gather_time_series(field, hi_c, cell)
    vals = (1.0 - w) * v_lo + w * v_hi
    vals[~ok] = np.nan
    out[valid] = vals.astype(np.float32)
    return out


def interp_clim_to_times(times: np.ndarray, lats: np.ndarray,
                         lons: np.ndarray, clim_path: Path) -> np.ndarray:
    """Monthly-diurnal PBLH climatology, linearly interpolated over its
    diurnal (UTC 0,3,..,21 h) axis at each sample's clock hour, with 21->24(0)
    wrap; the month is a step function (same convention as the nearest-lookup
    version -- the climatology has no sub-monthly axis to interpolate)."""
    out = np.full(times.shape, np.nan, dtype=np.float32)
    if not Path(clim_path).exists():
        warnings.warn(f"PBLH climatology {clim_path} absent; "
                      "UPW_pblh_anom is all NaN")
        return out
    with xr.open_dataset(clim_path) as f:
        c = f["pblh_mean"].sel(lat=lats, lon=lons, method="nearest").load()
    c = c.transpose("month", "hour", "lat", "lon")
    hours = c["hour"].values.astype(float)          # 0, 3, ..., 21
    step = float(np.diff(hours).mean())             # 3 h
    # append the wrapped hour-24 (= hour-0) slice so 21-24 h interpolates
    field = np.concatenate([c.values, c.values[:, :1]], axis=1)  # (12, 9, ...)

    valid = ~np.isnat(times)
    t = times[valid]
    month = t.astype("datetime64[M]").astype(int) % 12          # 0..11
    frac_h = ((t - t.astype("datetime64[D]"))
              / np.timedelta64(1, "h")).astype(float)           # [0, 24)
    lo = np.floor(frac_h / step).astype(int)                     # 0..7
    w = frac_h / step - lo
    ii, jj = np.meshgrid(np.arange(lats.size), np.arange(lons.size),
                         indexing="ij")
    cell = np.broadcast_to(ii * lons.size + jj, times.shape)[valid]
    flat = field.reshape(field.shape[0], field.shape[1], -1)
    v_lo = flat[month, lo, cell]
    v_hi = flat[month, lo + 1, cell]
    out[valid] = ((1.0 - w) * v_lo + w * v_hi).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Per-year assembly
# --------------------------------------------------------------------------- #
def sm_anomaly(fcst: xr.Dataset) -> xr.DataArray:
    """The suite's sm_anom, from its OWN helpers (per-year harmonic fit,
    exactly as ``dataset.build_base_table`` computes it year by year)."""
    sm = cs_dataset._prewindow_daily(fcst, cs_config.SM_VAR).rename("sm_raw")
    return cs_dataset._anom(sm, "sm_anom")


def front_flag_vars(year: int, fcst: xr.Dataset,
                    pred_dir: Path) -> dict[str, xr.DataArray]:
    """met_front_*_3w (NOAA analyst, incl. drylines) + pred_front_*_3w
    (kriged-AIRS D6C ensemble), both through the ONE alignment implementation
    (``fronts.file_front_flags``: 2x2 grid max-pool, concurrent slot->bulletin
    mapping, NaN where the governing analysis is unavailable)."""
    dates = fcst["date"].values
    slots = tuple(range(fcst.sizes["time"]))
    lats, lons = fcst["lat"].values, fcst["lon"].values
    dims = ("date", "time", "lat", "lon")
    out: dict[str, xr.DataArray] = {}

    met_path = cs_fronts.label_path("noaa", 3, year)
    if met_path is None:
        raise FileNotFoundError(
            f"no NOAA 3wide front file for {year} under "
            f"{cs_fronts.NOAA_FRONTS_DIR} -- the met-drawn front variables "
            f"are a hard requirement of this compile")
    for name, arr in cs_fronts.file_front_flags(
            met_path, FRONT_TYPES, dates, slots, lats, lons).items():
        out[f"met_front_{name}_3w"] = xr.DataArray(
            arr, dims=dims, attrs={
                "source": "NOAA-XML analyst fronts (met-drawn), 3wide",
                "file": str(met_path)})

    pred_path = (Path(pred_dir) / "1deg_3wide" / "3hr"
                 / f"merra2_merra2-1deg_3wide_3hr_{year}.nc")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predicted-front archive {pred_path} does not exist -- export it "
            f"first (dl_front.export_predictions, D6C 3-fold ensemble, "
            f"--source kriged-airs; the full-sequence chain's step 6)")
    for name, arr in cs_fronts.file_front_flags(
            pred_path, FRONT_TYPES, dates, slots, lats, lons).items():
        out[f"pred_front_{name}_3w"] = xr.DataArray(
            arr, dims=dims, attrs={
                "source": "D6C 3-fold softmax-ensemble prediction "
                          "(kriged-AIRS inputs), 3wide",
                "file": str(pred_path)})
    return out


def compile_year(year: int, pred_dir: Path, daily_dir: Path,
                 pblh_3hrly: Path, pblh_clim: Path) -> xr.Dataset:
    """One year's slab: the full primary file + every derived variable."""
    path = cs_config.DATA_DIR / cs_config.YEAR_FILE_TEMPLATE.format(year=year)
    fcst = xr.open_dataset(path).load()
    dims = ("date", "time", "lat", "lon")

    # 2. soil-moisture index
    fcst["sm_anom"] = sm_anomaly(fcst).assign_attrs(
        long_name="pre-window SMAP_L4_smsfc 2-harmonic seasonal anomaly, "
                  "z-scored per cell (suite predictor, per-year fit)",
        category="derived soil moisture")

    # 3. kernel-borne upwind predictors (psi, omega)
    daily = muf.discover_daily_files(daily_dir, fcst["date"].values)
    arrays, var_attrs, _ = muf.load_daily_features(
        daily, fcst["date"].values, fcst["lat"].values, fcst["lon"].values,
        n_slots=fcst.sizes["time"])
    for name in KERNEL_VARS:
        vals = arrays.get(name)
        if vals is None:
            warnings.warn(f"{year}: no daily file carries '{name}'; "
                          f"UPW_{name} is all NaN this year")
            vals = np.full(tuple(fcst.sizes[d] for d in dims), np.nan,
                           dtype=np.float32)
        fcst[f"UPW_{name}"] = xr.DataArray(
            vals, dims=dims,
            attrs={**var_attrs.get(name, {}), "category": "upwind kernel",
                   "n_days_with_kernel": len(daily)})

    # 4. interpolated PBL family (linear in time; MU-only gamma gap)
    times = muf.slot_datetimes(fcst)
    pblh = interp_pblh_to_times(times, fcst["lat"].values,
                                fcst["lon"].values, pblh_3hrly)
    clim = interp_clim_to_times(times, fcst["lat"].values,
                                fcst["lon"].values, pblh_clim)
    fcst["UPW_pblh"] = xr.DataArray(pblh, dims=dims, attrs={
        "units": "m", "category": "PBL (interpolated)",
        "long_name": "assessed PBLH (Guo 2024) LINEARLY INTERPOLATED in time "
                     "to the sample's true datetime; NaN across gaps > "
                     f"{MAX_INTERP_GAP_H} h (never fabricated)"})
    fcst["UPW_pblh_anom"] = xr.DataArray(pblh - clim, dims=dims, attrs={
        "units": "m", "category": "PBL (interpolated)",
        "long_name": "interpolated PBLH minus diurnally-interpolated "
                     "monthly climatology"})
    fcst["UPW_gamma_gap_mu"] = xr.DataArray(
        fcst["FCST_MU_LFC"].values.astype(np.float32) - pblh, dims=dims,
        attrs={"units": "m", "category": "PBL (interpolated)",
               "long_name": "Gamma_gap = FCST_MU_LFC - interpolated PBLH, "
                            "both AGL (MU only per user decision)"})

    # 5.-6. front flags
    for name, da in front_flag_vars(year, fcst, pred_dir).items():
        da.attrs["category"] = ("fronts (met-drawn)" if name.startswith("met")
                                else "fronts (predicted)")
        fcst[name] = da
    return fcst


# --------------------------------------------------------------------------- #
# The categorised variable listing (printed AND written next to the file)
# --------------------------------------------------------------------------- #
#: (category title, predicate) in print order; first match wins.
CATEGORY_RULES: tuple[tuple[str, callable], ...] = (
    ("AIRS-FCST thermodynamics -- MU parcel (the RF thermo block)",
     lambda v: v.startswith("FCST_MU_")),
    ("AIRS-FCST thermodynamics -- MML parcel (carried, NOT used by the RF)",
     lambda v: v.startswith("FCST_MML_")),
    ("AIRS-FCST parcel state / bookkeeping",
     lambda v: v.startswith("FCST_")),
    ("MRMS precipitation (target side)",
     lambda v: v.startswith("MRMS_")),
    ("SMAP L4 land-surface fields",
     lambda v: v.startswith("SMAP_L4_")),
    ("SMAP legacy 7-slot fields (all-NaN in every year, carried verbatim)",
     lambda v: v.startswith("SMAP_")),
    ("Derived soil moisture", lambda v: v == "sm_anom"),
    ("Upwind kernel predictors (psi, omega)",
     lambda v: v in ("UPW_psi_anom", "UPW_omega")),
    ("PBL depth family (time-INTERPOLATED)",
     lambda v: v.startswith("UPW_")),
    ("Surface fronts -- met-drawn (NOAA analyst, 3wide, incl. drylines)",
     lambda v: v.startswith("met_front_")),
    ("Surface fronts -- predicted (D6C kriged-AIRS ensemble, 3wide)",
     lambda v: v.startswith("pred_front_")),
)


def categorised_listing(ds: xr.Dataset) -> str:
    groups: dict[str, list[str]] = {title: [] for title, _ in CATEGORY_RULES}
    groups["Other / uncategorised"] = []
    for v in ds.data_vars:
        for title, match in CATEGORY_RULES:
            if match(v):
                groups[title].append(v)
                break
        else:
            groups["Other / uncategorised"].append(v)
    lines = [f"# RF dataset variables ({len(ds.data_vars)} total)", ""]
    for title, names in groups.items():
        if not names:
            continue
        lines.append(f"## {title} ({len(names)})")
        for v in sorted(names):
            long = ds[v].attrs.get("long_name", "")
            lines.append(f"- `{v}`" + (f" -- {long}" if long else ""))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def parse_years(spec: str) -> tuple[int, ...]:
    years: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            years += list(range(int(lo), int(hi) + 1))
        else:
            years.append(int(part))
    return tuple(sorted(set(years)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", default="2016-2021")
    ap.add_argument("--out", default=None,
                    help="output netCDF (default: <gpfs scratch root>/"
                         "RF_dataset_<Y0>-<Y1>.nc)")
    ap.add_argument("--pred-dir", default=None,
                    help="bk19-schema predicted-front tree (default "
                         f"$JPL_AIRS_DATA/front_id/predicted_fronts/"
                         f"{DEFAULT_PRED_TAG})")
    ap.add_argument("--daily-dir", type=Path,
                    default=tk_config.RESULTS_DIR / "upwind_features" / "daily",
                    help="per-day UPW_<YYYYMMDD>.nc kernel sweep output")
    ap.add_argument("--pblh-3hrly", type=Path,
                    default=tk_config.PBLH_3HRLY_PATH)
    ap.add_argument("--pblh-clim", type=Path, default=tk_config.PBLH_CLIM_PATH)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    years = parse_years(args.years)
    out = Path(args.out) if args.out else default_out_path(years)
    pred_dir = Path(args.pred_dir) if args.pred_dir else (
        cs_config.DATA_DIR / "front_id" / "predicted_fronts"
        / DEFAULT_PRED_TAG)

    if out.exists() and not args.force:
        print(f"{out} already exists -- printing its variable list and "
              f"exiting 0 (the chain proceeds to training); --force rebuilds")
        with xr.open_dataset(out) as ds:
            print(categorised_listing(ds))
        return 0

    slabs = [compile_year(y, pred_dir, args.daily_dir, args.pblh_3hrly,
                          args.pblh_clim) for y in years]
    big = xr.concat(slabs, dim="date", data_vars="all", combine_attrs="override")
    big.attrs.update({
        "title": f"RF-ready compiled dataset {years[0]}-{years[-1]}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_years": json.dumps(list(years)),
        "pred_front_tree": str(pred_dir),
        "history": "scripts/compile_rf_dataset.py "
                   + " ".join(argv or sys.argv[1:]),
    })

    listing = categorised_listing(big)
    print(listing)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4}
           for v in big.data_vars if big[v].dtype.kind == "f"}
    tmp = out.with_name(out.name + ".tmp")
    big.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, out)
    out.with_suffix(out.suffix + ".VARIABLES.md").write_text(listing)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) "
          f"+ {out.name}.VARIABLES.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
