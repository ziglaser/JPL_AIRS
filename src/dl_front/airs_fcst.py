"""Read HYSPLIT AIRS-FCST ``fullgrid`` files into DL-FRONT surface channels.

Source schema (audit 2026-08-04, sample day 2019-06-05):
``fullgrid_wrf27km_GOOD_1p00deg_YYYYMMDD_HHMM-HHMM[.nc]`` with dims
(time=7, level=33, lat=28, lon=43): slot 0 = the AIRS overpass (per-pixel
``parceltime``, ~18:53-20:39 UTC spread), slots 1-6 = uniform hourly
forecasts at 21,22,23,00,01,02 UTC; level = 30-hPa bin centers 115..1075
hPa; lat 25.5..52.5, lon -106.5..-64.5 (HALF-degree cell centers, offset
0.5 deg from the integer-degree label grid); fill -9999; ``q`` is MIXING
RATIO in g/kg (MERRA-2's QV2M is specific humidity in kg/kg); ``N`` =
parcels per box (100 % non-fill; N == 0 means unobserved).  There is no
surface-pressure/SLP variable.

The conversion pipeline mirrors the proven ``front_finder.ingest_hysplit``
(fill -> NaN, unit fixes, observed indicator BEFORE any interpolation,
vertical then bilinear horizontal interpolation, embed into the 68 x 141
label grid) but reduces each column to a SINGLE terrain-following surface
value and emits the DL-FRONT channel names T2M/QV2M/U10M/V10M.

Vertical extraction is TERRAIN-FOLLOWING (user decision 2026-08-16): each
column's surface value comes from its DEEPEST retrieval bin (level axis =
30-hPa bin centers 115..1075 hPa) at or below config.AIRS_SURFACE_SCAN_
FLOOR_HPA, accepted only when that bin sits within config.AIRS_SURFACE_
MAX_AGL_M of the local ground (static hypsometric terrain map,
scripts/build_surface_elevation.py); T is then extrapolated to the ground
at config.AIRS_SURFACE_LAPSE_K_PER_KM, q held constant, u/v taken from the
bin unchanged.  The previous fixed-985-hPa target had ZERO coverage over
all elevated terrain -- the high plains (1-2 km ASL, surface pressure
~800-850 hPa) never have 985-hPa air, so the entire dryline region looked
permanently out-of-swath while per-level checks showed it observed BETTER
than the east at 715-865 hPa (post-mortem 2026-08-16; cross-validated
against the nogrid granule parcels and FCST_SMAP_MRMS coverage).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config

FILL = -9999.0
#: A label point is "observed" if the bilinearly interpolated observed
#: indicator reaches this, i.e. at least ~half the surrounding half-degree
#: cells saw parcels.  Aliased from the ``degradation.observed_min_fraction``
#: tunable (configs/dl_front.yaml) so a JPL_DLFRONT_CONFIG override changes
#: the on-the-fly stage-B degradation AND every kriged-cache build together;
#: the tracked default (0.5) equals front_finder.ingest_hysplit's constant
#: (asserted in tests/test_dlfront_model.py).
OBSERVED_MIN_FRACTION = config.OBSERVED_MIN_FRACTION
#: fullgrid variable -> DL-FRONT surface channel it stands in for.
CHANNEL_MAP = {"t": "T2M", "q": "QV2M", "u": "U10M", "v": "V10M"}

_ELEV_CACHE: dict = {}

#: The fitted layer: all finite bins from the column's deepest retrieval up
#: to LAPSE_MAX_DZ_M above it enter the least-squares fit, and the fit is
#: trusted only when they span at least LAPSE_MIN_DZ_M of depth.  Within an
#: afternoon boundary layer T is close to linear in z (a well-mixed
#: convective layer is the MOST linear case, ~dry-adiabatic), so a fit over
#: the lowest ~2 km is the stable estimator: k bins cut retrieval-noise
#: variance ~1/k, where any 2-point difference over adjacent 30-hPa bins
#: (~250-300 m) would carry ~7-10 K/km of noise from ~1.5-2 K T errors.
LAPSE_MIN_DZ_M = 400.0
LAPSE_MAX_DZ_M = 2000.0


def _column_lapse(one: "xr.Dataset", sub: np.ndarray, idx: np.ndarray,
                  at_deepest) -> np.ndarray:
    """Per-column lapse rate (K/m, positive = cooling with height).

    Least-squares slope of T against altitude over ALL finite bins within
    LAPSE_MAX_DZ_M above the deepest bin (user decision 2026-08-16, refined
    same day from a 2-point difference to the multi-bin fit for noise
    stability), clipped to config.AIRS_SURFACE_LAPSE_CLIP_K_PER_KM.
    Columns whose usable bins span < LAPSE_MIN_DZ_M (or < 2 bins) fall back
    to config.AIRS_SURFACE_LAPSE_K_PER_KM.  Disable via
    ``airs.surface_lapse_derived: false`` (fallback everywhere).
    """
    default = float(config.AIRS_SURFACE_LAPSE_K_PER_KM) / 1000.0
    if not config.AIRS_SURFACE_LAPSE_DERIVED:
        return np.full(idx.shape, default)

    lev = one["level"].values
    scan = lev >= float(config.AIRS_SURFACE_SCAN_FLOOR_HPA)
    alt3 = one["alt"].values[scan]
    t3 = one["t"].values[scan]
    alt0 = at_deepest("alt")

    li = np.arange(sub.shape[0])[:, None, None]
    dz = alt3 - alt0[None]
    w = sub & (li <= idx[None]) & (dz >= 0) & (dz <= LAPSE_MAX_DZ_M)

    zf = np.where(w, alt3, 0.0)
    tf_ = np.where(w, t3, 0.0)
    n = w.sum(axis=0)
    sz, st = zf.sum(axis=0), tf_.sum(axis=0)
    szz = (zf * zf).sum(axis=0)
    szt = (zf * tf_).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * szt - sz * st) / (n * szz - sz * sz)
    depth = np.where(w, dz, 0.0).max(axis=0)
    valid = (n >= 2) & (depth >= LAPSE_MIN_DZ_M) & np.isfinite(slope)
    lapse = np.where(valid, -slope, default)     # positive = cooling with z
    lo, hi = (float(v) / 1000.0
              for v in config.AIRS_SURFACE_LAPSE_CLIP_K_PER_KM)
    return np.clip(lapse, lo, hi)


def _surface_elev_native(one: "xr.Dataset") -> np.ndarray:
    """Ground elevation (m ASL) on ``one``'s native half-degree grid.

    Bilinear from the static label-grid map (config.SURFACE_ELEV_PATH,
    built by scripts/build_surface_elevation.py); cached per native grid.
    """
    key = (float(one["lat"][0]), float(one["lon"][0]),
           one.sizes["lat"], one.sizes["lon"])
    if key not in _ELEV_CACHE:
        if not config.SURFACE_ELEV_PATH.exists():
            # deliberately NOT FileNotFoundError: that is an OSError, which
            # the per-step skip-with-note handlers (_FULLGRID_ERRORS, the
            # swath sweep) would swallow -- a missing terrain map would then
            # silently produce EMPTY banks/caches that act as done-markers
            raise RuntimeError(
                f"{config.SURFACE_ELEV_PATH} does not exist -- the "
                f"terrain-following surface extraction needs it; build it "
                f"once with 'PYTHONPATH=src python "
                f"scripts/build_surface_elevation.py'")
        with xr.open_dataset(config.SURFACE_ELEV_PATH) as elev_ds:
            elev = elev_ds["elev_m"].load()
        _ELEV_CACHE[key] = elev.interp(
            lat=one["lat"].values, lon=one["lon"].values,
            method="linear").values.astype(np.float64)
    return _ELEV_CACHE[key]


# --------------------------------------------------------------------------- #
# File discovery & loading
# --------------------------------------------------------------------------- #

#: One-time-per-process recursive archive scans: {root: {YYYYMMDD: path}}.
#: An AIRS archive is sparse by design, so per-date recursive globbing would
#: walk the whole (NFS) tree once per uncovered date -- thousands of times in
#: a multi-year cache build; the index costs exactly one walk per root.
_FULLGRID_INDEX: dict[Path, dict[str, Path]] = {}


def _archive_index(root: Path) -> dict[str, Path]:
    """{YYYYMMDD: fullgrid path} from ONE recursive scan of ``root``."""
    idx = _FULLGRID_INDEX.get(root)
    if idx is None:
        idx = {}
        for p in sorted(root.rglob("fullgrid_*")):
            m = re.search(r"(\d{8})_\d{4}-\d{4}", p.name)
            if m:
                idx.setdefault(m.group(1), p)      # first hit in sorted order
        _FULLGRID_INDEX[root] = idx
    return idx


def find_fullgrid(date, root=None) -> Path | None:
    """Locate the fullgrid file for one overpass day, or None.

    Tries the JPL archive layout ``root/YYYY/wrf27km_YYYYMMDD/fullgrid_*``
    first, then falls back to a recursive index of the whole tree (built
    once per process; the local demo tree nests
    ``wrf27km_20190605/wrf27km_20190605/``).  Files with or without the
    ``.nc`` suffix are accepted.
    """
    root = Path(root) if root is not None else config.AIRS_FCST_ROOT
    day = f"{pd.Timestamp(date):%Y%m%d}"
    subdir = root / day[:4] / f"wrf27km_{day}"
    hits = sorted(subdir.glob(f"fullgrid_*{day}*")) if subdir.is_dir() else []
    if hits:
        return hits[0]
    return _archive_index(root).get(day)


def load_fullgrid(path) -> xr.Dataset:
    """Open one fullgrid file: drop dup dim, fills -> NaN, q -> kg/kg.

    Same fixes as front_finder.ingest_hysplit.load_fullgrid: the duplicate
    'levels' dim is dropped, -9999 becomes NaN, and q (mixing ratio, g/kg)
    becomes specific humidity in kg/kg to match MERRA-2 QV2M.
    """
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    ds = ds.drop_dims("levels", errors="ignore")   # duplicate coord dim
    ds = ds.where(ds != FILL)
    r = ds["q"] / 1000.0                           # mixing ratio g/kg -> kg/kg
    ds["q"] = r / (1.0 + r)                        # -> specific humidity
    ds["q"].attrs.update(units="kg/kg", long_name="specific_humidity")
    ds.load()
    return ds


# --------------------------------------------------------------------------- #
# Period (time-slot) selection
# --------------------------------------------------------------------------- #

def overpass_midpoint(path) -> pd.Timestamp:
    """Midpoint of the ``..._YYYYMMDD_HHMM-HHMM`` filename window."""
    stem = Path(path).name.replace(".nc", "")
    datepart, window = stem.split("_")[-2], stem.split("_")[-1]
    t0 = pd.Timestamp(f"{datepart} {window.split('-')[0]}")
    t1 = pd.Timestamp(f"{datepart} {window.split('-')[1]}")
    return t0 + (t1 - t0) / 2


def slot_timestamp(ds: xr.Dataset, slot: int) -> pd.Timestamp:
    """Actual time of one slot: nan-median of per-pixel ``parceltime``.

    Slots 1-6 are uniform (every pixel shares the timestamp), so the median
    rounded to the hour IS the forecast time; for slot 0 it lands mid-swath.
    """
    med = float(np.nanmedian(ds["parceltime"].isel(time=slot).values))
    return pd.Timestamp(med, unit="s").round("h")


def _select_slot(ds: xr.Dataset, path, hour: int) -> tuple[int, pd.Timestamp]:
    """Hour in config.AIRS_HOURS -> (time slot, period timestamp).

    FORECAST SLOTS ONLY (user decision 2026-08-15): the returned slot is
    the uniform forecast slot whose parceltime matches ``hour`` (0 means
    next-day 00 UTC).  Slot 0 -- the overpass itself -- is never selected:
    its per-pixel obs times span ~2.6 h across the domain (Aqua 13:30 LT
    ascending), so no single 3-hourly label hour is correct for it, and
    the earlier conventions (always-18Z clamp, review 2026-08-13; then a
    nearest-label 19:30 rule) both left either misregistration or a mixed
    retrieval/forecast input distribution.  Forecast slots are per-pixel
    time-uniform, so every step is exactly label-aligned.  ValueError when
    the file has no forecast slot at ``hour`` (callers skip-with-note).
    """
    for slot in range(1, ds.sizes["time"]):
        t = slot_timestamp(ds, slot)
        if t.hour == hour:
            return slot, t
    raise ValueError(f"{Path(path).name}: no forecast slot at hour {hour}")
def period_fields(path, hour: int, ds: xr.Dataset | None = None) -> xr.Dataset:
    """One AIRS period -> surface channels on the full 68 x 141 label grid.

    Returns T2M, QV2M, U10M, V10M (float32, physical units, NaN where
    unobserved or off-swath), ``valid_frac`` (float32 in [0, 1], the
    bilinearly interpolated binarized observed indicator = the fraction of
    neighbor weight that saw a retrieval, 0 off-swath) and a scalar ``time``
    coordinate holding the period timestamp.

    ``ds``: an already-:func:`load_fullgrid`-ed dataset for ``path``, so
    multi-hour callers (swath.build_swath_bank reads all three periods of
    every archive day) load each file once instead of once per hour
    (review 2026-08-13); default: load ``path`` here.
    """
    if ds is None:
        ds = load_fullgrid(path)
    slot, when = _select_slot(ds, path, hour)
    one = ds.isel(time=slot)

    # TERRAIN-FOLLOWING vertical extraction (user decision 2026-08-16;
    # replaces the fixed 985-hPa target, which had zero coverage over all
    # elevated terrain -- the high plains never have 985-hPa air, so the
    # dryline region looked permanently out-of-swath).  Per column: take
    # the DEEPEST retrieval bin that carries every channel + alt, require
    # it to sit within AIRS_SURFACE_MAX_AGL_M of the local ground (rejects
    # cloud-blocked columns whose lowest retrieval is mid-tropospheric),
    # then extrapolate T down to the ground at the standard lapse rate,
    # hold q constant, and take u/v from the bin unchanged (proper MOST
    # needs surface fluxes/stability AIRS cannot provide).
    lev = one["level"].values
    scan = lev >= float(config.AIRS_SURFACE_SCAN_FLOOR_HPA)
    need = list(CHANNEL_MAP) + ["alt"]
    obs3d = (one["N"].fillna(0) > 0).values
    for var in need:
        obs3d = obs3d & np.isfinite(one[var].values)
    sub = obs3d[scan]                       # (nlev, nlat, nlon), level asc.
    rev = sub[::-1]                         # deepest (highest pressure) first
    has = rev.any(axis=0)
    idx = sub.shape[0] - 1 - rev.argmax(axis=0)

    def _at_deepest(var: str) -> np.ndarray:
        vals = one[var].values[scan]
        return np.take_along_axis(vals, idx[None], 0)[0]

    zs = _surface_elev_native(one)          # ground elevation, native grid
    agl = _at_deepest("alt") - zs
    ok = has & np.isfinite(zs) & (agl <= float(config.AIRS_SURFACE_MAX_AGL_M))
    # negative AGL (retrieval "below" the climatological ground: hypsometric
    # noise / sub-grid valleys) is fine as an observation; just never
    # extrapolate upward
    dz = np.clip(np.where(ok, agl, np.nan), 0.0, None)

    lapse = _column_lapse(one, sub, idx, _at_deepest)
    surf = {"t": _at_deepest("t") + lapse * dz,
            "q": np.where(ok, _at_deepest("q"), np.nan),
            "u": np.where(ok, _at_deepest("u"), np.nan),
            "v": np.where(ok, _at_deepest("v"), np.nan)}
    fields = xr.Dataset(
        {k: (("lat", "lon"), v.astype(np.float64)) for k, v in surf.items()},
        coords={"lat": one["lat"].values, "lon": one["lon"].values})
    ind = xr.DataArray(ok.astype("float64"),
                       dims=("lat", "lon"),
                       coords={"lat": one["lat"].values,
                               "lon": one["lon"].values})

    # horizontal: MASKED bilinear from half-degree centers to integer label
    # points.  A plain bilinear NaN-poisons every label point with even one
    # unobserved neighbor, which would leave NaN fields at pixels whose
    # valid_frac clears the threshold (0.5 = half the neighbor weight);
    # instead the observed neighbors are averaged with their bilinear
    # weights renormalized: value = interp(ind * f) / interp(ind).  Where
    # all neighbors are observed this is the exact bilinear value, and a
    # field value exists wherever valid_frac > 0 by construction.
    lat_t = np.arange(np.ceil(float(one.lat[0])), np.floor(float(one.lat[-1])) + 0.1)
    lon_t = np.arange(np.ceil(float(one.lon[0])), np.floor(float(one.lon[-1])) + 0.1)
    den = ind.interp(lat=lat_t, lon=lon_t, method="linear")
    num = fields.where(ind > 0).fillna(0.0).interp(lat=lat_t, lon=lon_t,
                                                   method="linear")
    fields = num / den.where(den > 0)
    obs = den

    # embed into the full label grid; off-swath = unobserved
    lat_full = np.asarray(config.LABEL_LATS)
    lon_full = np.asarray(config.LABEL_LONS)
    fields = fields.reindex(lat=lat_full, lon=lon_full)
    obs = obs.reindex(lat=lat_full, lon=lon_full).fillna(0.0)

    out = fields.rename(CHANNEL_MAP)
    for var in out.data_vars:
        out[var] = out[var].where(obs >= OBSERVED_MIN_FRACTION).astype("float32")
    out["valid_frac"] = obs.astype("float32")
    out = out.drop_vars("level", errors="ignore")
    return out.assign_coords(time=when)
