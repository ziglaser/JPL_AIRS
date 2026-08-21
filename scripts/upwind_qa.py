#!/usr/bin/env python
"""Physical-realism QA battery for the upwind soil-moisture features.

Seven checks (agreed with Zach, 2026-08-19), each asking one question the
physics answers unambiguously, so a failure localizes a bug rather than a
vibe. Inputs are the PRODUCTS of the pipeline -- the yearly companions
``UPWIND_FEATURES_<YYYY>.nc`` (scripts/merge_upwind_features.py), the per-day
``UPW_<YYYYMMDD>.nc`` files (scripts/build_upwind_features.py, for the
lag-weight profiles the merge cannot carry), and the ``FCST_SMAP_MRMS_<YYYY>.nc``
match-up files (winds and MRMS labels) -- never the intermediate kernels, so
the battery audits exactly what the forest will see.

    1. land_closure        Psi[land fraction] ~ 1 for interior receptors
    2. upwind_alignment    displacement anti-parallel to the wind, |d| ~ |v| t
    3. diurnal_lag_profile lag-weight centroids march back with arrival slot
    4. magnitude_realism   Omega in physical range; Psi decorrelates from the
                           endpoint with horizon; Gamma_gap sign census
    5. event_case_study    eyeball maps on the top-3 rain days per year
    6. sample_size_leakage parcel count must not leak into Psi or Omega
    7. label_monotonicity  exploratory smoke test against the MRMS label

Each check is a plain function taking preloaded data and returning
``{"name", "status": PASS|WARN|FAIL|SKIP, "metrics", "detail", "figures"}``;
``main`` prints the aligned verdict table, writes ``qa_report.json`` and the
figures into ``--out-dir``. Missing inputs (unmounted data drive, a year with
all-NaN kernel features like 2016, no daily dir, no MRMS) SKIP the affected
checks with the reason -- the battery never fabricates a verdict it cannot
support, and never fails just because an optional input is absent.

Multi-year behavior: checks 2, 4, 6, 7 pool the samples across every
requested year (per-year gaps drop out as NaN); checks 3 and 5 produce
per-year figures. Memory stays bounded: companions are opened lazily and only
the needed variables are pulled; MRMS is loaded one year at a time and
reduced before the next year is touched.

Usage (all defaults follow the pipeline's own output locations):
  PYTHONPATH=src python scripts/upwind_qa.py --years 2018 2019
  PYTHONPATH=src python scripts/upwind_qa.py           # every merged year
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; QA runs on login nodes and in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_kernels import config  # noqa: E402

# --------------------------------------------------------------------------- #
# Thresholds (each with its provenance; change here, never inline)
# --------------------------------------------------------------------------- #
#: Check 1: interior receptors' source regions are all-land, so the kernel's
#: land-weighted mean of the land fraction must be ~1; 3% slack for fuzz tails
#: brushing lakes/coast (review 4.3, the label-free land-fraction closure).
PSI_LAND_INTERIOR_MIN: float = 0.97
#: Check 1: a receptor is "interior" if no all-NaN (ocean / never-populated)
#: cell lies within this Chebyshev radius -- 2 cells ~ 220 km, beyond the p99
#: parcel displacement (2.6 deg, config SOURCE_WINDOW note) minus fuzz.
INTERIOR_CLEARANCE_CELLS: int = 2
#: Check 2: air comes FROM upwind, so the kernel-centroid displacement must be
#: anti-parallel to the arriving wind; median cosine <= -0.5 (within 60 deg of
#: dead-upwind) tolerates fuzz + energy weighting bending the effective path.
ALIGN_MEDIAN_COS_MAX: float = -0.5
#: Check 2: fraction of samples with cosine > 0 (displacement DOWNWIND -- a
#: geometric sign error); 15% allows weak/veering-wind noise (Stohl 1998
#: position errors ~20% of distance) but catches a flipped sign instantly.
ALIGN_FRAC_DOWNWIND_MAX: float = 0.15
#: Check 2: minimum centroid fetch (km) for the verdict statistics -- half a
#: 1-deg cell. Below this the centroid's SIGN is sub-grid noise: measured on
#: 2019-06-05, fetch >= 50 km gives median cos -0.96 / 3% downwind while
#: sub-cell fetches alone give -0.67 / 28% (short early-slot look-backs).
ALIGN_MIN_FETCH_KM: float = 50.0
#: Check 3: slot 6 arrives 02 UTC -- after sunset domain-wide -- so clear-sky
#: available energy at lags 0-1 h is ~0 and the energy weighting must have
#: killed those hours; 10% allows western-edge twilight. This bound is what
#: makes the check sensitive to a cancelled/uniform energy weight.
LATE_SLOT_EARLY_MASS_MAX: float = 0.10
LATE_SLOT_EARLY_LAGS_H: float = 1.5  # "lags 0-1": lag bins < 1.5 h
#: Check 4a: Omega per-slot medians; worked example is 2250 J/kg for 3 h of
#: clear-sky afternoon contact through an 1800 m layer (review 1.5), so an
#: order of magnitude either side brackets plausible horizons and PBL depths.
OMEGA_MEDIAN_RANGE_J_KG: tuple[float, float] = (500.0, 5000.0)
#: Check 4a: the p99 Omega, read as pure warming Omega/cp, must stay below
#: 10 K -- more than that delivered to one air column in one afternoon exceeds
#: any observed CONUS mixed-layer heating (cp = 1005 J/kg/K, dry air).
CP_J_KG_K: float = 1005.0
OMEGA_P99_WARMING_MAX_K: float = 10.0
#: Check 4b: the accumulation must decorrelate from the endpoint as the path
#: lengthens (review 7.8's falsifiable prediction); require r(slot 6) below
#: r(slot 1) by at least this margin (beyond sampling noise for ~10^3 cells).
R_SLOT_DROP_MIN: float = 0.02
#: Check 4b: if r(psi, endpoint) > 0.98 at EVERY slot, the accumulation adds
#: nothing over the point predictor -- the review 7.8 failure verdict.
R_SATURATION: float = 0.98
#: Check 4c: Gamma_gap < 0 (PBL top above the LFC) over half the domain-hours
#: would mean near-universal triggering; and identically 0 means a dead field.
GAMMA_NEG_FRAC_MAX: float = 0.5
#: Check 6: containment switches off below CONTAINMENT_MIN_PARCELS (= 20),
#: exactly between the 10-19 and 20-49 bins; a variance step > 2x across that
#: seam means the containment rule, not the atmosphere, sets Psi's spread.
VAR_STEP_RATIO_MAX: float = 2.0
NPARCEL_BIN_EDGES: tuple = (1, 5, 10, 20, 50)
NPARCEL_BIN_LABELS: tuple = ("1-4", "5-9", "10-19", "20-49", "50+")
#: Check 6: the ensemble-mean fix made Phi (hence Omega) parcel-count
#: invariant (review 1.5 note); any |r| >= 0.1 is that leakage reviving.
#: Check 6: raw r(omega, n_parcels) is reported but NOT gated -- via
#: RECEPTOR_BAND_M the parcel count proxies PBL depth, so omega and n covary
#: physically through m* (measured 2019-06-05: r(m*,n)=+0.36; m*-partialing
#: halves r(omega,n) from -0.27 to -0.13). The gate is on the m*-PARTIALED
#: residual correlation: >= WARN means mechanical dependence beyond the PBL
#: pathway is creeping back; >= FAIL means the phi ensemble-mean fix regressed.
OMEGA_NPARCELS_RPARTIAL_WARN: float = 0.2
OMEGA_NPARCELS_RPARTIAL_FAIL: float = 0.4
#: Check 6: minimum finite cells per n_parcels bin for the variance-seam gate
#: (10 correlated neighbours on one day produced a spurious 21.5x "seam").
SEAM_MIN_BIN_CELLS: int = 100
#: Check 7a: smoke-test bar for the exploratory precip-rate monotonicity --
#: |Spearman rho| > 0.5 over 10 deciles, correct (negative) sign.
LABEL_RHO_MIN: float = 0.5
#: Geometry/units: km per degree of latitude (2 pi R_earth / 360), and m/s ->
#: km/h for the |wind| x lag distance comparison.
KM_PER_DEG: float = 2.0 * np.pi * config.EARTH_RADIUS_KM / 360.0
MS_TO_KMH: float = 3.6

#: FCST match-up variable names (convection_skill.config conventions).
MRMS_AV_VAR: str = "MRMS_GaugeCorrQPE01H_av"
MRMS_MAX_VAR: str = "MRMS_GaugeCorrQPE01H_max"
WIND_U_VAR: str = "FCST_u"
WIND_V_VAR: str = "FCST_v"

#: Nominal UTC hour per arrival slot (merge_upwind_features.SLOT_HOURS_AFTER_DATE).
SLOT_UTC: dict[int, str] = {1: "21", 2: "22", 3: "23", 4: "00", 5: "01", 6: "02"}

DPI: int = 150
NAN_CAPTION: str = "NaN shown light gray"


# --------------------------------------------------------------------------- #
# Figure conventions (repo dataviz rules): matplotlib only, one axis per
# panel, sequential = single-hue Blues, diverging = BrBG centered on 0 for
# every soil-moisture anomaly, recessive grids, NaN as light gray + caption.
# --------------------------------------------------------------------------- #
def _seq_cmap():
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("lightgray")
    return cmap


def _div_cmap():
    cmap = plt.get_cmap("BrBG").copy()
    cmap.set_bad("lightgray")
    return cmap


def _style(ax):
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def _sym_limit(arr) -> float:
    """Symmetric color limit so a diverging map is truly centered on 0."""
    finite = np.asarray(arr)[np.isfinite(arr)]
    return float(np.abs(finite).max()) if finite.size else 1.0


def _finish(fig, n: int, name: str, status: str, key_metric: str,
            out_path: Path) -> str:
    fig.suptitle(f"QA {n}: {name} - {status}  ({key_metric})")
    fig.text(0.99, 0.01, NAN_CAPTION, ha="right", va="bottom",
             fontsize=7, color="gray")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class YearBundle:
    """Paths + validity flags for one year's inputs."""
    year: int
    companion: Path
    fcst: Path | None            # FCST_SMAP_MRMS_<year>.nc, if present
    kernel_valid: bool = True    # any finite UPW_psi_anom? (2016: no)
    reason: str = ""


@dataclass
class Pool:
    """Flattened multi-year samples (slots 1..6 only) plus the pooled maps.

    ``cols`` maps column name -> 1-D float32 array, all the same length, one
    entry per (date, slot, cell) sample across every kernel-valid year; a
    variable absent from some year contributes NaN there. ``psi_land_map`` is
    the slot- and date-pooled mean UPW_psi_land on (lat, lon).
    """
    cols: dict = field(default_factory=dict)
    psi_land_map: np.ndarray | None = None
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    n_kernel_years: int = 0
    notes: list = field(default_factory=list)

    def finite(self, *names) -> tuple:
        """The named columns restricted to rows finite in ALL of them."""
        arrs = [self.cols[n] for n in names]
        ok = np.ones(arrs[0].shape, dtype=bool)
        for a in arrs:
            ok &= np.isfinite(a)
        return tuple(a[ok] for a in arrs)


#: Companion variables pulled into the pool (contract of the daily files after
#: the upwind-geometry change, merged with the UPW_ prefix) + FCST winds.
POOL_UPW_VARS: tuple = (
    "psi_anom", "s_endpoint_anom", "psi_meso_anom", "omega", "n_parcels",
    "gamma_gap_mml", "upwind_dlat", "upwind_dlon", "upwind_km",
    "mean_lag_hours", "m_star",
)


def _slot_field(ds: xr.Dataset, name: str) -> xr.DataArray | None:
    """A (date, time, lat, lon) view of an FCST variable, or None if absent.

    Any extra dimension (e.g. a release-level axis on the winds) is averaged
    out -- the QA wants the cell's bulk low-level wind, not the ladder.
    """
    if name not in ds:
        return None
    da = ds[name]
    # The MRMS variables carry the 7 forecast slots under the dim name
    # ``nhours`` (slot-aligned with ``time``; verified FCST_SMAP_MRMS_2019,
    # whose nhours coord is degenerate zeros -- drop it, adopt the time axis).
    if "nhours" in da.dims and da.sizes["nhours"] == ds.sizes.get("time"):
        da = (da.drop_vars("nhours", errors="ignore")
                .rename({"nhours": "time"})
                .assign_coords(time=ds["time"].values))
    extra = [d for d in da.dims if d not in ("date", "time", "lat", "lon")]
    if extra:
        da = da.mean(extra, skipna=True)
    return da.transpose("date", "time", "lat", "lon")


def discover_years(companion_dir: Path, years: list[int] | None) -> list[Path]:
    files = sorted(companion_dir.glob("UPWIND_FEATURES_*.nc"))
    if years:
        wanted = {int(y) for y in years}
        files = [f for f in files
                 if f.stem.rsplit("_", 1)[-1].isdigit()
                 and int(f.stem.rsplit("_", 1)[-1]) in wanted]
    return files


def make_bundles(companion_files: list[Path], fcst_dir: Path) -> list[YearBundle]:
    """Attach the FCST file and probe kernel validity for each year.

    The probe is cheap (one variable): a year whose UPW_psi_anom is absent or
    all-NaN (2016 -- no assessed PBLH, or an unmerged daily sweep) carries no
    kernel features, so checks 2-6 must not read it as evidence.
    """
    bundles = []
    for path in companion_files:
        year = int(path.stem.rsplit("_", 1)[-1])
        fcst = fcst_dir / f"FCST_SMAP_MRMS_{year}.nc"
        b = YearBundle(year=year, companion=path,
                       fcst=fcst if fcst.exists() else None)
        with xr.open_dataset(path) as ds:
            if "UPW_psi_anom" not in ds:
                b.kernel_valid, b.reason = False, "no UPW_psi_anom variable"
            elif not bool(np.isfinite(ds["UPW_psi_anom"].values).any()):
                b.kernel_valid, b.reason = False, "UPW_psi_anom all NaN"
        bundles.append(b)
    return bundles


def build_pool(bundles: list[YearBundle]) -> Pool:
    """Pool the per-sample columns and the psi_land map across kernel-valid years.

    One year in memory at a time; each contributes float32 flattened slabs
    (~10 MB/variable/year), which is the bounded-memory contract.
    """
    pool = Pool()
    parts: dict[str, list] = {name: [] for name in
                              POOL_UPW_VARS + ("slot", "lat_deg", "u", "v", "event")}
    land_sum = land_cnt = None

    for b in bundles:
        if not b.kernel_valid:
            pool.notes.append(f"{b.year}: excluded from pooled kernel checks "
                              f"({b.reason})")
            continue
        pool.n_kernel_years += 1
        with xr.open_dataset(b.companion) as ds:
            n_slots = ds.sizes["time"]
            lat = ds["lat"].values.astype(float)
            lon = ds["lon"].values.astype(float)
            if pool.lat is None:
                pool.lat, pool.lon = lat, lon
            shape = (ds.sizes["date"], n_slots - 1, lat.size, lon.size)

            def _flat(da: xr.DataArray | None) -> np.ndarray:
                if da is None:
                    return np.full(shape, np.nan, np.float32).ravel()
                vals = da.transpose("date", "time", "lat", "lon").values
                return vals[:, 1:].astype(np.float32).ravel()  # slots 1..6

            for name in POOL_UPW_VARS:
                da = ds.get("UPW_" + name)
                parts[name].append(_flat(da if da is not None else None))
            slot_idx = np.broadcast_to(
                np.arange(1, n_slots)[None, :, None, None], shape)
            parts["slot"].append(slot_idx.astype(np.float32).ravel())
            lat4d = np.broadcast_to(lat[None, None, :, None], shape)
            parts["lat_deg"].append(lat4d.astype(np.float32).ravel())

            if "UPW_psi_land" in ds:
                pl = ds["UPW_psi_land"].transpose("date", "time", "lat", "lon").values
                fin = np.isfinite(pl)
                if land_sum is None:
                    land_sum = np.zeros(pl.shape[2:])
                    land_cnt = np.zeros(pl.shape[2:])
                land_sum += np.where(fin, pl, 0.0).sum(axis=(0, 1))
                land_cnt += fin.sum(axis=(0, 1))

        # Winds + the precip label, from the year's FCST file (one at a time).
        u = v = ev = None
        if b.fcst is not None:
            with xr.open_dataset(b.fcst) as fc:
                u_da, v_da = _slot_field(fc, WIND_U_VAR), _slot_field(fc, WIND_V_VAR)
                u = _flat(u_da) if u_da is not None else None
                v = _flat(v_da) if v_da is not None else None
                mx = _slot_field(fc, MRMS_MAX_VAR)
                if mx is not None:
                    ev = _event_flags(mx.values).astype(np.float32).ravel()
        empty = np.full(np.prod(shape), np.nan, np.float32)
        parts["u"].append(u if u is not None else empty)
        parts["v"].append(v if v is not None else empty)
        parts["event"].append(ev if ev is not None else empty)

    if pool.n_kernel_years:
        pool.cols = {name: np.concatenate(chunks)
                     for name, chunks in parts.items()}
    if land_cnt is not None and land_cnt.max() > 0:
        with np.errstate(invalid="ignore"):
            pool.psi_land_map = np.where(land_cnt > 0, land_sum / land_cnt, np.nan)
    return pool


def _event_flags(mrms_max: np.ndarray, thresh: float = 1.0) -> np.ndarray:
    """Precip occurrence per (date, slot 1..6, cell): the matching OR the
    following slot exceeds the threshold (the label window of check 7); the
    last slot has no follower, so it stands alone. NaN where both slots are
    NaN (unknown, not dry)."""
    m = mrms_max  # (date, time, lat, lon), slot 0 = overpass
    out = np.full((m.shape[0], m.shape[1] - 1) + m.shape[2:], np.nan)
    for s in range(1, m.shape[1]):
        pair = m[:, s:min(s + 2, m.shape[1])]
        known = np.isfinite(pair).any(axis=1)
        hit = (np.where(np.isfinite(pair), pair, -np.inf) > thresh).any(axis=1)
        out[:, s - 1] = np.where(known, hit.astype(float), np.nan)
    return out


# --------------------------------------------------------------------------- #
# The seven checks
# --------------------------------------------------------------------------- #
def _skip(name: str, reason: str) -> dict:
    return {"name": name, "status": "SKIP", "metrics": {},
            "detail": reason, "figures": []}


def check_land_closure(pool: Pool, out_dir: Path) -> dict:
    """QA 1 -- Psi[land fraction] must close to ~1 away from the coast.

    The kernel weights every source point by the continuous land fraction, so
    contracting it against the land-fraction field itself is a label-free
    closure test (review 4.3): an interior receptor, whose whole source region
    is land, must return ~1; anything materially below means kernel mass is
    escaping to ocean/NaN cells or the land weighting broke. Coastal
    receptors legitimately dip below 1 and are reported, never failed.
    """
    name = "land_closure"
    if pool.psi_land_map is None:
        return _skip(name, "UPW_psi_land absent from every kernel-valid year "
                           "(rebuild the daily files with the geometry vars)")
    m = pool.psi_land_map
    nan_mask = ~np.isfinite(m)
    # Interior = land cells with no all-NaN (ocean/never-populated) cell
    # within INTERIOR_CLEARANCE_CELLS (Chebyshev). Plain sliding-window
    # dilation; the grid is 28x43, cost is nil.
    r = INTERIOR_CLEARANCE_CELLS
    padded = np.pad(nan_mask, r, constant_values=False)
    near_nan = np.zeros_like(nan_mask)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            near_nan |= padded[r + di:r + di + m.shape[0],
                               r + dj:r + dj + m.shape[1]]
    interior = ~nan_mask & ~near_nan
    coastal = ~nan_mask & near_nan

    med_int = float(np.median(m[interior])) if interior.any() else float("nan")
    med_coast = float(np.median(m[coastal])) if coastal.any() else float("nan")
    ok = interior.any() and PSI_LAND_INTERIOR_MIN <= med_int <= 1.0 + 1e-9
    status = "PASS" if ok else ("SKIP" if not interior.any() else "FAIL")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pc = ax.pcolormesh(pool.lon, pool.lat, m, cmap=_seq_cmap(), vmin=0.9, vmax=1.0)
    fig.colorbar(pc, ax=ax, label="pooled mean UPW_psi_land (land fraction, 0-1)")
    ii, jj = np.where(coastal)
    ax.plot(pool.lon[jj], pool.lat[ii], ".", ms=2, color="#555555",
            label=f"coastal (reported, not failed): n={int(coastal.sum())}")
    ax.set_xlabel("lon (deg)"); ax.set_ylabel("lat (deg)")
    ax.legend(loc="lower left", fontsize=7)
    _style(ax)
    figp = _finish(fig, 1, name, status,
                   f"interior median {med_int:.3f}", out_dir / "qa1_land_closure.png")
    return {"name": name, "status": status,
            "metrics": {"interior_median": med_int, "coastal_median": med_coast,
                        "n_interior": int(interior.sum()),
                        "n_coastal": int(coastal.sum()),
                        "threshold": PSI_LAND_INTERIOR_MIN},
            "detail": (f"interior median psi_land {med_int:.3f} "
                       f"(need >= {PSI_LAND_INTERIOR_MIN}); coastal median "
                       f"{med_coast:.3f} over {int(coastal.sum())} cells"),
            "figures": [figp]}


def check_upwind_alignment(pool: Pool, out_dir: Path) -> dict:
    """QA 2 -- the air comes FROM upwind: displacement anti-parallel to wind.

    The kernel-centroid displacement (UPW_upwind_dlon, UPW_upwind_dlat) points
    from the receptor toward where the influencing soil was; the arriving wind
    (FCST_u, FCST_v) points where the air is GOING. So their cosine must sit
    near -1 -- a median above ALIGN_MEDIAN_COS_MAX or more than
    ALIGN_FRAC_DOWNWIND_MAX of samples on the wrong side means a sign or
    axis-order bug in the geometry, the first thing this battery must catch.
    Second panel: |displacement| against |wind| x mean lag -- advection says
    slope ~1, with spread from fuzz and the energy weighting bending the
    effective path toward the sunlit hours.
    """
    name = "upwind_alignment"
    if not pool.cols:
        return _skip(name, "no kernel-valid years")
    dlat, dlon, latd, u, v, km_a = pool.finite(
        "upwind_dlat", "upwind_dlon", "lat_deg", "u", "v", "upwind_km")
    if dlat.size == 0:
        return _skip(name, "no finite (upwind_dlat/dlon, FCST_u/v) samples "
                           "(geometry vars or winds missing)")
    dx = dlon * np.cos(np.deg2rad(latd))  # equal-km east component
    dy = dlat
    wind = np.hypot(u, v)
    disp = np.hypot(dx, dy)
    ok = (wind > 0) & (disp > 0)
    cos = (dx[ok] * u[ok] + dy[ok] * v[ok]) / (disp[ok] * wind[ok])
    med_all = float(np.median(cos))
    # The verdict statistics use only well-resolved fetches: below half a
    # 1-deg cell (ALIGN_MIN_FETCH_KM) the centroid sign is sub-grid noise --
    # measured on 2019-06-05: fetch >= 50 km gives median cos -0.96 / 3%
    # downwind, while sub-cell fetches alone read -0.67 / 28%.
    far = km_a[ok] >= ALIGN_MIN_FETCH_KM
    if far.any():
        med = float(np.median(cos[far]))
        frac_down = float((cos[far] > 0).mean())
        fetch_note = f"fetch>={ALIGN_MIN_FETCH_KM:.0f} km n={int(far.sum())}"
    else:  # all sub-cell (very short look-backs only): report, do not gate
        med, frac_down = med_all, float((cos > 0).mean())
        fetch_note = "no well-resolved fetches; all-sample stats (weak test)"
    status = ("PASS" if med <= ALIGN_MEDIAN_COS_MAX
              and frac_down <= ALIGN_FRAC_DOWNWIND_MAX else "FAIL")

    km, lag, u2, v2 = pool.finite("upwind_km", "mean_lag_hours", "u", "v")
    pred_km = np.hypot(u2, v2) * lag * MS_TO_KMH
    slope = (float((pred_km * km).sum() / (pred_km ** 2).sum())
             if pred_km.size and (pred_km ** 2).sum() > 0 else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(cos, bins=40, color=plt.get_cmap("Blues")(0.7))
    axes[0].axvline(ALIGN_MEDIAN_COS_MAX, color="#555555", ls="--", lw=1,
                    label=f"median bound {ALIGN_MEDIAN_COS_MAX}")
    axes[0].set_xlabel("cos(displacement, wind)  [-1 = dead upwind]")
    axes[0].set_ylabel("samples"); axes[0].legend(fontsize=7)
    if km.size > 5000:
        hb = axes[1].hexbin(pred_km, km, gridsize=40, cmap=_seq_cmap(), mincnt=1)
        fig.colorbar(hb, ax=axes[1], label="samples")
    else:
        axes[1].plot(pred_km, km, ".", ms=3, color=plt.get_cmap("Blues")(0.7),
                     alpha=0.5)
    lim = max(float(np.nanmax(pred_km, initial=1.0)),
              float(np.nanmax(km, initial=1.0)))
    axes[1].plot([0, lim], [0, lim], color="#555555", lw=1, label="1:1")
    axes[1].set_xlabel("|wind| x mean lag (km)")
    axes[1].set_ylabel("|upwind_km| (km)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        _style(ax)
    figp = _finish(fig, 2, name, status, f"median cos {med:.2f}",
                   out_dir / "qa2_upwind_alignment.png")
    return {"name": name, "status": status,
            "metrics": {"median_cosine": med, "frac_downwind": frac_down,
                        "median_cosine_all_fetches": med_all,
                        "distance_slope": slope, "n": int(cos.size)},
            "detail": (f"median cos {med:.2f} (need <= {ALIGN_MEDIAN_COS_MAX}; "
                       f"{fetch_note}), "
                       f"frac downwind {frac_down:.3f} "
                       f"(need <= {ALIGN_FRAC_DOWNWIND_MAX}), "
                       f"distance slope {slope:.2f} vs 1"),
            "figures": [figp]}


def check_diurnal_lag_profile(bundles: list[YearBundle], daily_dir: Path,
                              out_dir: Path) -> dict:
    """QA 3 -- lag-weight centroids must march back with arrival slot.

    From the DAILY files (the yearly merge cannot carry lag_weight -- wrong
    dims): the mean normalized lag-mass profile per arrival slot. Physics: the
    available-energy weight peaks near local noon (~18-19 UTC), so a 21 UTC
    arrival finds its mass at short lags and an 02 UTC arrival must reach 5+
    hours back -- the weighted centroid lag increases strictly with slot, and
    slot 6 (02 UTC, dark) must hold < 10% of its mass at lags 0-1 because the
    energy weighting has killed those hours. THIS is the check that would have
    caught the cancelled-energy-weight bug instantly: under a uniform hour
    weight every slot's profile flattens, the centroids collapse toward each
    other, and slot 6 keeps ~2/7 of its mass in the dark early lags.
    """
    name = "diurnal_lag_profile"
    if not daily_dir.is_dir():
        return _skip(name, f"daily dir absent: {daily_dir}")
    per_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for b in bundles:
        if not b.kernel_valid:
            continue
        files = sorted(p for p in daily_dir.glob(f"UPW_{b.year}*.nc")
                       if "uniform_energy" not in p.name)
        sums: np.ndarray | None = None
        cnts: np.ndarray | None = None
        lags: np.ndarray | None = None
        for f in files:
            with xr.open_dataset(f) as ds:
                if "lag_weight" not in ds:
                    continue
                lw = ds["lag_weight"].transpose(
                    "arrival_step", "target_lat", "target_lon", "lag").values
                lag_ax = ds["lag"].values.astype(float)
            tot = np.nansum(lw, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                norm = np.where(tot > 0, lw / tot, np.nan)
            prof = np.nanmean(norm, axis=(1, 2))          # (step, lag)
            fin = np.isfinite(prof)
            if sums is None:
                sums = np.zeros_like(prof); cnts = np.zeros_like(prof); lags = lag_ax
            n = min(sums.shape[1], prof.shape[1])
            sums[:, :n] += np.where(fin, prof, 0.0)[:, :n]
            cnts[:, :n] += fin[:, :n]
        if sums is not None and cnts.max() > 0:
            with np.errstate(invalid="ignore"):
                per_year[b.year] = (np.where(cnts > 0, sums / cnts, np.nan), lags)
    if not per_year:
        return _skip(name, f"no daily UPW_*.nc files with lag_weight under "
                           f"{daily_dir} for the kernel-valid years")

    # Pooled (across years) profile drives the verdict; figures are per-year.
    max_l = max(p.shape[1] for p, _ in per_year.values())
    pooled = np.full((len(per_year), 6, max_l), np.nan)
    lag_ax = max((l for _, l in per_year.values()), key=len)
    for i, (prof, _) in enumerate(per_year.values()):
        pooled[i, :prof.shape[0], :prof.shape[1]] = prof
    prof = np.nanmean(pooled, axis=0)
    prof = prof / np.nansum(prof, axis=1, keepdims=True)
    centroids = np.nansum(prof * lag_ax[None, :], axis=1)
    early = lag_ax < LATE_SLOT_EARLY_LAGS_H
    late_early_mass = float(np.nansum(prof[-1, early]))
    increasing = bool(np.all(np.diff(centroids) > 0))
    status = ("PASS" if increasing and late_early_mass < LATE_SLOT_EARLY_MASS_MAX
              else "FAIL")

    figures = []
    for year, (yprof, ylags) in per_year.items():
        yprof = yprof / np.nansum(yprof, axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        pc = ax.pcolormesh(ylags, np.arange(1, yprof.shape[0] + 1), yprof,
                           cmap=_seq_cmap())
        cen = np.nansum(yprof * ylags[None, :], axis=1)
        ax.plot(cen, np.arange(1, yprof.shape[0] + 1), "o-", ms=4,
                color="#333333", lw=1, label="centroid lag")
        fig.colorbar(pc, ax=ax, label="normalized lag mass (per slot, sums to 1)")
        ax.set_xlabel("lag (hours before arrival)")
        ax.set_ylabel("arrival slot (1=21 UTC .. 6=02 UTC)")
        ax.legend(fontsize=7); _style(ax)
        figures.append(_finish(fig, 3, f"{name} {year}", status,
                               f"slot-6 early mass {late_early_mass:.2f}",
                               out_dir / f"qa3_lag_profile_{year}.png"))
    return {"name": name, "status": status,
            "metrics": {"centroid_lag_hours_by_slot": [round(float(c), 3)
                                                       for c in centroids],
                        "centroids_strictly_increasing": increasing,
                        "slot6_mass_at_lags_0_1": late_early_mass,
                        "years": sorted(per_year)},
            "detail": (f"centroids {np.array2string(centroids, precision=2)} h "
                       f"({'increasing' if increasing else 'NOT increasing'}); "
                       f"slot-6 mass at lags 0-1 = {late_early_mass:.3f} "
                       f"(need < {LATE_SLOT_EARLY_MASS_MAX})"),
            "figures": figures}


def check_magnitude_realism(pool: Pool, out_dir: Path) -> dict:
    """QA 4 -- are the numbers physically sized, and does the path add anything?

    (a) Omega per slot must sit in the worked-example range (review 1.5:
        2250 J/kg reference) and the p99, read as pure warming Omega/cp, under
        10 K -- an afternoon cannot heat one column more than that.
    (b) r(psi_anom, s_endpoint_anom) per slot must decay with horizon: longer
        paths read soil farther from the receptor (review 7.8's falsifiable
        prediction). r > 0.98 everywhere = the accumulation adds nothing over
        the point predictor -> FAIL, and that verdict is a publishable result,
        not a bug.
    (c) Gamma_gap < 0 fraction by slot: reported, expected to peak late
        afternoon (deepest PBL vs falling LFC); WARN only if the field is dead
        (0 everywhere) or implies near-universal triggering (> 0.5 anywhere).
    """
    name = "magnitude_realism"
    if not pool.cols:
        return _skip(name, "no kernel-valid years")
    slots = np.arange(1, 7)
    slot_col = pool.cols["slot"]

    def per_slot(col, fn):
        out = []
        for s in slots:
            vals = pool.cols[col][(slot_col == s) & np.isfinite(pool.cols[col])]
            out.append(fn(vals) if vals.size else float("nan"))
        return np.array(out, dtype=float)

    # (a) Omega
    om_all = pool.cols["omega"][np.isfinite(pool.cols["omega"])]
    if om_all.size == 0:
        return _skip(name, "UPW_omega all NaN in every kernel-valid year")
    om_med = per_slot("omega", np.median)
    p99_k = float(np.percentile(om_all, 99)) / CP_J_KG_K
    lo, hi = OMEGA_MEDIAN_RANGE_J_KG
    a_ok = bool(np.all((om_med >= lo) & (om_med <= hi))
                and p99_k <= OMEGA_P99_WARMING_MAX_K)

    # (b) r(psi, endpoint) per slot
    from scipy.stats import spearmanr
    r_by_slot = []
    for s in slots:
        m = slot_col == s
        a = pool.cols["psi_anom"][m]; e = pool.cols["s_endpoint_anom"][m]
        ok = np.isfinite(a) & np.isfinite(e)
        r_by_slot.append(float(np.corrcoef(a[ok], e[ok])[0, 1])
                         if ok.sum() >= 3 else float("nan"))
    r_by_slot = np.array(r_by_slot)
    fin_r = np.isfinite(r_by_slot)
    saturated = bool(fin_r.any() and np.all(r_by_slot[fin_r] > R_SATURATION))
    if fin_r.sum() >= 3:
        trend = float(spearmanr(slots[fin_r], r_by_slot[fin_r]).statistic)
    else:
        trend = float("nan")
    b_ok = bool(np.isfinite(r_by_slot[0]) and np.isfinite(r_by_slot[-1])
                and trend <= 0
                and r_by_slot[-1] < r_by_slot[0] - R_SLOT_DROP_MIN)

    # (c) Gamma_gap sign census
    g_frac = per_slot("gamma_gap_mml", lambda v: float((v < 0).mean()))
    g_fin = g_frac[np.isfinite(g_frac)]
    c_warn = bool(g_fin.size and (np.all(g_fin == 0.0)
                                  or np.any(g_fin > GAMMA_NEG_FRAC_MAX)))

    if saturated:
        status = "FAIL"
        verdict = (f"r(psi, endpoint) > {R_SATURATION} at every slot: the "
                   "accumulation adds nothing over the point predictor "
                   "(review 7.8)")
    elif a_ok and b_ok and not c_warn:
        status, verdict = "PASS", "all magnitude and decorrelation criteria met"
    else:
        status = "WARN"
        verdict = (f"a_ok={a_ok} (medians in [{lo:.0f},{hi:.0f}] J/kg, "
                   f"p99 warming {p99_k:.1f} K), b_ok={b_ok} "
                   f"(r trend {trend:+.2f}, drop "
                   f"{r_by_slot[0] - r_by_slot[-1]:+.3f}), gamma_warn={c_warn}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    om_sets = [pool.cols["omega"][(slot_col == s) & np.isfinite(pool.cols["omega"])]
               for s in slots]
    om_sets = [v if v.size else np.array([np.nan]) for v in om_sets]
    vp = axes[0].violinplot(om_sets, positions=slots, showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor(plt.get_cmap("Blues")(0.6)); body.set_alpha(0.8)
    for part in ("cmedians", "cbars", "cmins", "cmaxes"):
        vp[part].set_color("#333333"); vp[part].set_linewidth(1)
    for y in OMEGA_MEDIAN_RANGE_J_KG:
        axes[0].axhline(y, color="#555555", ls="--", lw=1)
    axes[0].set_xlabel("arrival slot"); axes[0].set_ylabel("Omega (J/kg)")
    axes[0].set_title("(a) Omega by slot; dashes = median bounds", fontsize=9)
    axes[1].plot(slots, r_by_slot, "o-", color=plt.get_cmap("Blues")(0.8))
    axes[1].axhline(R_SATURATION, color="#555555", ls="--", lw=1,
                    label=f"saturation {R_SATURATION}")
    axes[1].set_xlabel("arrival slot"); axes[1].set_ylabel("r(psi_anom, endpoint)")
    axes[1].set_ylim(min(0.0, np.nanmin(r_by_slot) - 0.05), 1.02)
    axes[1].legend(fontsize=7)
    axes[1].set_title("(b) accumulation vs point predictor", fontsize=9)
    axes[2].plot(slots, g_frac, "o-", color=plt.get_cmap("Blues")(0.8))
    axes[2].axhline(GAMMA_NEG_FRAC_MAX, color="#555555", ls="--", lw=1,
                    label=f"warn above {GAMMA_NEG_FRAC_MAX}")
    axes[2].set_xlabel("arrival slot")
    axes[2].set_ylabel("fraction Gamma_gap < 0")
    axes[2].legend(fontsize=7)
    axes[2].set_title("(c) PBL top above LFC", fontsize=9)
    for ax in axes:
        _style(ax)
    figp = _finish(fig, 4, name, status,
                   f"median Omega {np.nanmedian(om_all):.0f} J/kg",
                   out_dir / "qa4_magnitude_realism.png")
    return {"name": name, "status": status,
            "metrics": {"omega_median_by_slot": [round(float(x), 1) for x in om_med],
                        "omega_p99_warming_K": p99_k,
                        "r_psi_endpoint_by_slot": [round(float(x), 4)
                                                   for x in r_by_slot],
                        "r_trend_spearman": trend,
                        "r_saturated_everywhere": saturated,
                        "gamma_neg_frac_by_slot": [round(float(x), 3)
                                                   for x in g_frac]},
            "detail": verdict, "figures": [figp]}


def _render_day_panel(dstr: str, fcst_dir: Path, daily_dir: Path,
                      out_dir: Path) -> tuple[Path | None, str | None]:
    """The six-panel one-day diagnosis (scripts/plot_upwind_day.py) for one
    case date, if its daily file exists. Returns (figure_path, skip_reason).

    Imported lazily by path (scripts/ is not a package) with sys.modules
    registration; any failure skips THIS panel with a reason, never the check
    (the 2x2 case figure above is the load-bearing artifact).
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plot_upwind_day", Path(__file__).parent / "plot_upwind_day.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plot_upwind_day"] = mod
        spec.loader.exec_module(mod)
        out = mod.render_day(
            dstr, arrival_step=4, daily_dir=daily_dir, fcst_dir=fcst_dir,
            out=out_dir / f"qa5_day_{dstr.replace('-', '')}_s4.png")
        return out, None
    except FileNotFoundError as err:
        return None, f"{dstr}: six-panel skipped ({err})"
    except Exception as err:  # noqa: BLE001 - a plot must never sink the check
        return None, f"{dstr}: six-panel failed ({type(err).__name__}: {err})"


def check_event_case_study(bundles: list[YearBundle], out_dir: Path,
                           daily_dir: Path | None = None) -> dict:
    """QA 5 -- eyeball the machinery on the days that matter.

    For the top-3 domain-total MRMS days per year: previous-day rain, the
    day's endpoint anomaly, the path-accumulated psi_anom, and their
    difference (what the path adds) under the wind arrows. A visual artifact
    -- always PASS -- but the fastest way to see a lon-flip, a units slip, or
    a kernel reading the wrong day. Soil anomalies diverge (BrBG, centered 0);
    rain is sequential Blues.

    When the case date has a per-day UPW file, the six-panel one-day
    diagnosis (plot_upwind_day.py: SM anomaly, winds, Psi, remote content,
    Omega, assessed PBLH -- Zach's diagnosis layout, 2026-08-21) is rendered
    alongside; absent inputs skip that panel with a note, never the check.
    """
    name = "event_case_study"
    day_notes: list[str] = []
    usable = [b for b in bundles if b.kernel_valid and b.fcst is not None]
    if not usable:
        return _skip(name, "needs a kernel-valid year with an FCST file "
                           "(MRMS + winds); none available")
    figures, chosen = [], []
    for b in usable:
        with xr.open_dataset(b.fcst) as fc:
            if MRMS_AV_VAR not in fc:
                continue
            av = _slot_field(fc, MRMS_AV_VAR).values  # (date, time, lat, lon)
            u4 = _slot_field(fc, WIND_U_VAR)
            v4 = _slot_field(fc, WIND_V_VAR)
            u4 = u4.values if u4 is not None else None
            v4 = v4.values if v4 is not None else None
            dates = fc["date"].values
            lat, lon = fc["lat"].values, fc["lon"].values
        totals = np.nansum(np.nan_to_num(av), axis=(1, 2, 3))
        top = np.argsort(totals)[::-1][:3]
        with xr.open_dataset(b.companion) as comp:
            for di in top:
                d = dates[di]
                dstr = str(np.datetime_as_string(d, unit="D"))
                chosen.append(dstr)
                day_sum = np.nansum(np.nan_to_num(av[di]), axis=0)
                prev = (np.nansum(np.nan_to_num(av[di - 1]), axis=0)
                        if di > 0 else np.full_like(day_sum, np.nan))
                sel = dict(date=d)
                s_end = comp["UPW_s_endpoint_anom"].sel(**sel).mean(
                    "time", skipna=True).values
                psi = comp["UPW_psi_anom"].sel(**sel).mean(
                    "time", skipna=True).values
                diff = psi - s_end

                fig, axes = plt.subplots(2, 2, figsize=(11, 7))
                pc = axes[0, 0].pcolormesh(lon, lat, prev, cmap=_seq_cmap())
                fig.colorbar(pc, ax=axes[0, 0], label="prev-day MRMS total (mm)")
                axes[0, 0].set_title("previous-day MRMS", fontsize=9)
                lim = _sym_limit(np.concatenate(
                    [s_end.ravel(), psi.ravel(), diff.ravel()]))
                for ax, fld, title in ((axes[0, 1], s_end, "UPW_s_endpoint_anom"),
                                       (axes[1, 0], psi, "UPW_psi_anom"),
                                       (axes[1, 1], diff, "psi_anom - s_endpoint_anom")):
                    pc = ax.pcolormesh(lon, lat, fld, cmap=_div_cmap(),
                                       vmin=-lim, vmax=lim)
                    fig.colorbar(pc, ax=ax, label="soil anomaly (m3/m3)")
                    ax.set_title(title, fontsize=9)
                if u4 is not None and v4 is not None:
                    uu = np.nanmean(u4[di], axis=0); vv = np.nanmean(v4[di], axis=0)
                    axes[1, 1].quiver(lon, lat, uu, vv, color="#333333",
                                      scale=250, width=0.003)
                for ax in axes.ravel():
                    _style(ax)
                figures.append(_finish(
                    fig, 5, f"{name} {dstr}", "PASS",
                    f"domain MRMS total rank <= 3 in {b.year}",
                    out_dir / f"qa5_case_{b.year}_{dstr.replace('-', '')}.png"))
                if daily_dir is not None:
                    panel, note = _render_day_panel(
                        dstr, Path(b.fcst).parent, daily_dir, out_dir)
                    if panel is not None:
                        figures.append(str(panel))
                    else:
                        day_notes.append(note)
    if not figures:
        return _skip(name, f"{MRMS_AV_VAR} absent from every FCST file")
    detail = "visual artifact (always PASS); cases: " + ", ".join(chosen)
    if day_notes:
        detail += "; " + "; ".join(day_notes)
    return {"name": name, "status": "PASS",
            "metrics": {"case_dates": chosen},
            "detail": detail,
            "figures": figures}


def check_sample_size_leakage(pool: Pool, out_dir: Path) -> dict:
    """QA 6 -- parcel count is sampling, not physics; it must not move features.

    Psi is a weighted mean, so its variance should decay smoothly (~1/n Monte
    Carlo noise on top of real variance) across n_parcels bins -- a step
    change > 2x exactly across the containment threshold (20 parcels, between
    the 10-19 and 20-49 bins) means the containment rule sets the spread
    (review F6). And Omega, after the ensemble-mean fix (review 1.5 note), is
    parcel-count invariant by construction: any |r(omega, n_parcels)| >= 0.1
    is that leakage reviving -> FAIL.
    """
    name = "sample_size_leakage"
    if not pool.cols:
        return _skip(name, "no kernel-valid years")
    npar, psi = pool.finite("n_parcels", "psi_anom")
    if npar.size == 0:
        return _skip(name, "no finite (n_parcels, psi_anom) samples")
    edges = np.array(NPARCEL_BIN_EDGES, dtype=float)
    idx = np.digitize(npar, edges) - 1  # 0..4; <1 parcel -> -1 (dropped)
    variances = np.full(len(NPARCEL_BIN_LABELS), np.nan)
    counts = np.zeros(len(NPARCEL_BIN_LABELS), dtype=int)
    for i in range(len(NPARCEL_BIN_LABELS)):
        vals = psi[idx == i]
        counts[i] = vals.size
        if vals.size >= 2:
            variances[i] = float(np.var(vals))
    # The seam gate needs POPULATED bins: on the demo day the 20-49 bin held
    # 10 cells (a handful of granule-dense neighbours over uniform wet soil),
    # whose "variance" produced a spurious 21.5x seam. Bins thinner than
    # SEAM_MIN_BIN_CELLS are excluded from the gate, reported as such.
    v_mid, v_hi = variances[2], variances[3]  # 10-19 vs 20-49 (the seam)
    seam_populated = (counts[2] >= SEAM_MIN_BIN_CELLS
                      and counts[3] >= SEAM_MIN_BIN_CELLS)
    step = (float(max(v_mid, v_hi) / min(v_mid, v_hi))
            if seam_populated and np.isfinite(v_mid) and np.isfinite(v_hi)
            and min(v_mid, v_hi) > 0 else float("nan"))
    step_warn = bool(np.isfinite(step) and step > VAR_STEP_RATIO_MAX)

    # r(omega, n_parcels) is NOT expected to be zero: through RECEPTOR_BAND_M
    # the parcel count proxies PBL depth (deep well-mixed layers keep more
    # parcels inside the 0-1000 m band), so omega and n covary PHYSICALLY via
    # m* -- measured 2026-08-21 on the demo day: r(m*, n) = +0.36 and
    # partialing m* out of omega halves r(omega, n) from -0.27 to -0.13. The
    # leakage-specific statistic is therefore the m*-PARTIALED residual
    # correlation: mechanical sample-size dependence beyond the PBL pathway.
    n2, om = pool.finite("n_parcels", "omega")
    r = (float(np.corrcoef(n2, om)[0, 1]) if n2.size >= 3 else float("nan"))
    n3, om3, ms = pool.finite("n_parcels", "omega", "m_star")
    if n3.size >= 3 and np.ptp(ms) > 0:
        resid = om3 - np.polyval(np.polyfit(ms, om3, 1), ms)
        r_partial = float(np.corrcoef(n3, resid)[0, 1])
    else:  # m_star absent (pre-geometry companion): fall back to the raw r
        r_partial = r
    leak_fail = bool(np.isfinite(r_partial)
                     and abs(r_partial) >= OMEGA_NPARCELS_RPARTIAL_FAIL)
    leak_warn = bool(np.isfinite(r_partial)
                     and abs(r_partial) >= OMEGA_NPARCELS_RPARTIAL_WARN)
    status = ("FAIL" if leak_fail
              else "WARN" if (leak_warn or step_warn) else "PASS")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(NPARCEL_BIN_LABELS))
    ax.plot(x, variances, "o-", color=plt.get_cmap("Blues")(0.8))
    ax.set_xticks(x, NPARCEL_BIN_LABELS)
    ax.set_xlabel("n_parcels bin"); ax.set_ylabel("Var(UPW_psi_anom) ((m3/m3)^2)")
    ax.axvspan(2.5, 3.5, color="lightgray", alpha=0.4, lw=0,
               label="containment threshold seam (20 parcels)")
    ax.annotate(f"r(omega,n) = {r:+.3f} raw (ungated: n proxies PBL depth); "
                f"m*-partialed {r_partial:+.3f} (warn >= "
                f"{OMEGA_NPARCELS_RPARTIAL_WARN})",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8)
    ax.legend(fontsize=7); _style(ax)
    figp = _finish(fig, 6, name, status, f"m*-partial r {r_partial:+.3f}",
                   out_dir / "qa6_sample_size_leakage.png")
    return {"name": name, "status": status,
            "metrics": {"psi_var_by_bin": [None if not np.isfinite(v)
                                           else round(float(v), 8)
                                           for v in variances],
                        "bin_counts": counts.tolist(),
                        "seam_step_ratio": step,
                        "seam_gated": bool(seam_populated),
                        "r_omega_nparcels": r,
                        "r_omega_nparcels_mstar_partial": r_partial},
            "detail": (f"r(omega,n) raw {r:+.3f} (ungated: n proxies PBL "
                       f"depth via the receptor band), m*-partialed "
                       f"{r_partial:+.3f} (warn >= "
                       f"{OMEGA_NPARCELS_RPARTIAL_WARN}, fail >= "
                       f"{OMEGA_NPARCELS_RPARTIAL_FAIL}); variance seam "
                       f"10-19 vs 20-49 = "
                       + (f"{step:.2f} (warn > {VAR_STEP_RATIO_MAX})"
                          if np.isfinite(step) else
                          f"ungated (bins under {SEAM_MIN_BIN_CELLS} cells)")),
            "figures": [figp]}


def check_label_monotonicity(pool: Pool, out_dir: Path,
                             precip_thresh: float) -> dict:
    """QA 7 -- exploratory smoke test against the MRMS label (NOT a skill claim).

    (a) Precip-occurrence rate (any MRMS max > threshold in the matching or
        following slot) by UPW_gamma_gap_mml decile: the geometric pathway
        says rain favors negative gaps (PBL top past the LFC), so the rate
        should fall monotonically across ascending-gap deciles -- Spearman rho
        of rate vs decile < 0 with |rho| > 0.5 to PASS, WARN otherwise.
    (b) The Taylor et al. (2012) sign: mean UPW_psi_meso_anom on precip minus
        no-precip cell-slots -- negative means rain fell over locally dry soil
        (mesoscale-circulation pathway); WARN if positive. No error bars:
        cell-slots within a date are not independent (the naive count would
        overstate certainty), and blocking by date is beyond a smoke test --
        the figure is labeled exploratory instead.
    """
    name = "label_monotonicity"
    if not pool.cols:
        return _skip(name, "no kernel-valid years")
    gam, ev = pool.finite("gamma_gap_mml", "event")
    if gam.size < 100:
        return _skip(name, "too few finite (gamma_gap, MRMS event) samples "
                           f"({gam.size}); MRMS or PBLH likely absent")
    from scipy.stats import spearmanr
    qs = np.quantile(gam, np.linspace(0, 1, 11))
    qs[0], qs[-1] = -np.inf, np.inf
    dec = np.digitize(gam, qs[1:-1])  # 0..9, ascending gap
    rate = np.array([ev[dec == i].mean() if (dec == i).any() else np.nan
                     for i in range(10)])
    fin = np.isfinite(rate)
    rho = (float(spearmanr(np.arange(10)[fin], rate[fin]).statistic)
           if fin.sum() >= 3 else float("nan"))
    a_pass = bool(np.isfinite(rho) and rho < 0 and abs(rho) > LABEL_RHO_MIN)

    meso, ev2 = pool.finite("psi_meso_anom", "event")
    if meso.size:
        wet = float(meso[ev2 > 0.5].mean()) if (ev2 > 0.5).any() else float("nan")
        dry = float(meso[ev2 < 0.5].mean()) if (ev2 < 0.5).any() else float("nan")
        taylor_diff = wet - dry
    else:
        wet = dry = taylor_diff = float("nan")
    b_warn = bool(np.isfinite(taylor_diff) and taylor_diff > 0)
    status = "PASS" if a_pass and not b_warn else "WARN"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(10), rate, color=plt.get_cmap("Blues")(0.6),
                edgecolor="white", linewidth=2)
    axes[0].set_xlabel("UPW_gamma_gap_mml decile (0 = most negative gap)")
    axes[0].set_ylabel(f"precip rate (MRMS max > {precip_thresh} mm/h)")
    axes[0].annotate(f"Spearman rho = {rho:+.2f}", xy=(0.55, 0.9),
                     xycoords="axes fraction", fontsize=9)
    axes[1].bar([0, 1], [dry, wet], color=plt.get_cmap("Blues")(0.6),
                edgecolor="white", linewidth=2)
    axes[1].set_xticks([0, 1], ["no precip", "precip"])
    axes[1].set_ylabel("mean UPW_psi_meso_anom (m3/m3)")
    axes[1].axhline(0, color="#555555", lw=1)
    axes[1].annotate(f"precip - no-precip = {taylor_diff:+.2e}\n"
                     "(negative = rain over locally dry;\nTaylor et al. 2012)",
                     xy=(0.02, 0.82), xycoords="axes fraction", fontsize=8)
    for ax in axes:
        _style(ax)
    fig.text(0.5, 0.005, "EXPLORATORY smoke test: cell-slots within a date are "
             "not independent; no error bars shown.", ha="center", fontsize=8,
             style="italic")
    figp = _finish(fig, 7, name, status, f"rho {rho:+.2f}",
                   out_dir / "qa7_label_monotonicity.png")
    return {"name": name, "status": status,
            "metrics": {"precip_rate_by_gamma_decile": [None if not np.isfinite(x)
                                                        else round(float(x), 4)
                                                        for x in rate],
                        "spearman_rho": rho,
                        "taylor_diff_precip_minus_dry": taylor_diff,
                        "mean_meso_precip": wet, "mean_meso_no_precip": dry,
                        "precip_thresh_mm": precip_thresh},
            "detail": (f"rate-vs-gap-decile rho {rho:+.2f} (pass: < 0, |rho| > "
                       f"{LABEL_RHO_MIN}); Taylor sign "
                       f"{taylor_diff:+.2e} m3/m3 "
                       f"({'matches' if not b_warn else 'OPPOSES'} rain-over-dry)"),
            "figures": [figp]}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physical-realism QA battery for the upwind features "
                    "(7 checks; see module docstring).")
    p.add_argument("--years", type=int, nargs="+", default=None,
                   help="years to audit (default: every UPWIND_FEATURES_*.nc "
                        "found in --companion-dir). ONE representative year "
                        "is sufficient -- this is a physical-realism audit, "
                        "not a per-year production step; pass several only "
                        "if you want a pooled aggregate (Zach, 2026-08-19)")
    p.add_argument("--companion-dir", type=Path,
                   default=config.RESULTS_DIR / "upwind_features",
                   help="directory of UPWIND_FEATURES_<YYYY>.nc companions")
    p.add_argument("--fcst-dir", type=Path, default=config.FCST_TABLE_DIR,
                   help="directory of FCST_SMAP_MRMS_<YYYY>.nc match-up files")
    p.add_argument("--daily-dir", type=Path, default=None,
                   help="per-day UPW_<YYYYMMDD>.nc dir (default: "
                        "<companion-dir>/daily; needed only by check 3, which "
                        "reads lag_weight -- the merge cannot carry it)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="destination for qa_report.json + figures "
                        "(default: <companion-dir>/qa)")
    p.add_argument("--precip-thresh-mm", type=float, default=1.0,
                   help="MRMS max threshold defining a precip event (check 7)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    daily_dir = args.daily_dir or args.companion_dir / "daily"
    out_dir = args.out_dir or args.companion_dir / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    companion_files = discover_years(args.companion_dir, args.years)
    if not companion_files:
        raise SystemExit(f"no UPWIND_FEATURES_*.nc in {args.companion_dir} "
                         f"matching years={args.years}: merge some years first "
                         "(scripts/merge_upwind_features.py)")
    bundles = make_bundles(companion_files, args.fcst_dir)
    pool = build_pool(bundles)
    # The precip-event column uses the CLI threshold, so rebuild it here if it
    # differs from the default the pool builder used.
    if pool.cols and args.precip_thresh_mm != 1.0:
        ev_parts = []
        for b in bundles:
            if not b.kernel_valid:
                continue
            with xr.open_dataset(b.companion) as ds:
                shape = (ds.sizes["date"], ds.sizes["time"] - 1,
                         ds.sizes["lat"], ds.sizes["lon"])
            if b.fcst is not None:
                with xr.open_dataset(b.fcst) as fc:
                    mx = _slot_field(fc, MRMS_MAX_VAR)
                    if mx is not None:
                        ev_parts.append(_event_flags(
                            mx.values, args.precip_thresh_mm
                        ).astype(np.float32).ravel())
                        continue
            ev_parts.append(np.full(int(np.prod(shape)), np.nan, np.float32))
        pool.cols["event"] = np.concatenate(ev_parts)

    results = [
        check_land_closure(pool, out_dir),
        check_upwind_alignment(pool, out_dir),
        check_diurnal_lag_profile(bundles, daily_dir, out_dir),
        check_magnitude_realism(pool, out_dir),
        check_event_case_study(bundles, out_dir, daily_dir=daily_dir),
        check_sample_size_leakage(pool, out_dir),
        check_label_monotonicity(pool, out_dir, args.precip_thresh_mm),
    ]

    width = max(len(r["name"]) for r in results)
    print(f"\n{'#':>2} {'check':<{width}} {'status':<6} detail")
    print("-" * (11 + width + 60))
    for i, r in enumerate(results, 1):
        print(f"{i:>2} {r['name']:<{width}} {r['status']:<6} {r['detail']}")
    worst = {"PASS": 0, "SKIP": 1, "WARN": 2, "FAIL": 3}
    overall = max((r["status"] for r in results), key=worst.get)
    print(f"\noverall: {overall}  "
          f"({sum(r['status'] == 'PASS' for r in results)} PASS, "
          f"{sum(r['status'] == 'WARN' for r in results)} WARN, "
          f"{sum(r['status'] == 'FAIL' for r in results)} FAIL, "
          f"{sum(r['status'] == 'SKIP' for r in results)} SKIP)")
    for note in pool.notes:
        print(f"note: {note}")

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "companion_dir": str(args.companion_dir),
        "daily_dir": str(daily_dir),
        "fcst_dir": str(args.fcst_dir),
        "years": [b.year for b in bundles],
        "kernel_valid_years": [b.year for b in bundles if b.kernel_valid],
        "excluded_years": {str(b.year): b.reason for b in bundles
                           if not b.kernel_valid},
        "precip_thresh_mm": args.precip_thresh_mm,
        "overall": overall,
        "checks": results,
        "command": "scripts/upwind_qa.py " + " ".join(
            argv if argv is not None else sys.argv[1:]),
    }
    report_path = out_dir / "qa_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {report_path} and "
          f"{sum(len(r['figures']) for r in results)} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
