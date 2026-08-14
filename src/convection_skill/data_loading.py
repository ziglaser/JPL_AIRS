"""Low-level loading of the ``data/FCST_SMAP_MRMS_{year}.nc`` cubes.

Two stages, each usable on its own:

1. :func:`load_raw`       -- the yearly cubes as one queryable ``xr.Dataset``,
   sliced to years / lat-lon / months. Nothing is derived or renamed; variables
   keep their file names and native time axes.
2. :func:`make_uniform`   -- everything onto one (date, slot, lat, lon)
   forecast-hour grid: forecast-axis variables subset to the requested slots,
   the overpass baselines replicated across the hours, SMAP L4 fields placed on
   the hours by a time policy.

The row-table assembly (variable renames, QPE reconstruction, daily SM
predictors, config-driven screening) lives in ONE place:
:mod:`convection_skill.dataset` (``build_dataset`` / ``prepare``).

Grid facts (verified in notebooks/01_data_audit):
- ``FCST_*`` and overpass-window ``SMAP_*`` variables live on the 7-slot ``time``
  axis, MRMS variables on the equally sized ``nhours`` axis: slot 0 = AIRS
  overpass, slots 1-6 = 21,22,23,00,01,02 UTC.
- SMAP L4 variables live on their own 5-slot ``L4_nhours`` axis, observed at
  ~16,19,22,01,04 UTC (``SMAP_L4_hour``).
- The 1x1 grid (lat 25.5-52.5N, lon -106.5..-64.5) lines up exactly with the
  global land-sea mask in ``data/lsm.nc``.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import config

FORECAST_DIMS: tuple[str, ...] = ("time", "nhours")
L4_DIM: str = "L4_nhours"
SMAP_L4_HOUR_VAR: str = "SMAP_L4_hour"
#: Sub-pixel counts per MRMS precipitation-type category, on an extra ``nflags``
#: axis; unstacked into one uniform variable per category by :func:`make_uniform`.
PRECIP_FLAG_VAR: str = "MRMS_PrecipFlag_cnt"

SMAP_TIME_POLICIES: tuple[str, ...] = ("interp", "previous_valid", "last_before_window")
DEFAULT_SMAP_TIME_POLICY: str = "interp"
_DAY_BOUNDARY_UTC: float = 12.0

_SLOT_TO_HOUR: dict[int, int] = dict(
    zip(config.FORECAST_SLOTS, config.FORECAST_HOURS_UTC)
)


# --------------------------------------------------------------------------- #
# SMAP L4 -> forecast-hour time mapping
# --------------------------------------------------------------------------- #
def _linearize_hours(hours: np.ndarray) -> np.ndarray:
    hours = np.asarray(hours, dtype="float64")
    return np.where(hours < _DAY_BOUNDARY_UTC, hours + 24.0, hours)


def _l4_interp_weights(
    obs_hour: np.ndarray, forecast_hours: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-in-time interpolation weights from the L4 slots to each forecast hour.

    For each forecast hour, returns the two bracketing L4 slot indices ``(lo, hi)``
    and a weight ``w`` in [0, 1] toward ``hi``, so the interpolated field is
    ``(1 - w) * cube[lo] + w * cube[hi]``. A forecast hour that coincides with an
    observation gets ``w == 0``, reproducing that slot exactly. Forecast hours
    outside the observed range -- or with no finite observation times -- return
    ``lo == hi == -1`` (the caller fills NaN); we do not extrapolate. Observation
    times need not be pre-sorted.
    """
    obs_lin = _linearize_hours(obs_hour)
    fc_lin = _linearize_hours(forecast_hours)

    n_fc = fc_lin.shape[0]
    lo = np.full(n_fc, -1, dtype=int)
    hi = np.full(n_fc, -1, dtype=int)
    weight = np.zeros(n_fc, dtype="float64")

    finite = np.where(np.isfinite(obs_lin))[0]
    if finite.size == 0:
        return lo, hi, weight
    order = finite[np.argsort(obs_lin[finite])]  # original slot indices, ascending time
    times = obs_lin[order]

    for i, fh in enumerate(fc_lin):
        if not np.isfinite(fh) or fh < times[0] or fh > times[-1]:
            continue  # undefined or out of observed range -> NaN, no extrapolation
        j = int(np.clip(np.searchsorted(times, fh, side="right"), 1, times.size - 1))
        t0, t1 = times[j - 1], times[j]
        lo[i], hi[i] = order[j - 1], order[j]
        weight[i] = 0.0 if t1 == t0 else (fh - t0) / (t1 - t0)
    return lo, hi, weight


def _slots_at_or_before(obs_hour: np.ndarray, target_hour: float) -> np.ndarray:
    """L4 slot indices observed at/just-before ``target_hour``, nearest-past first.

    Returns the original slot indices with observation time ``<= target_hour``
    (on the :func:`_linearize_hours` clock), most recent first -- the candidate
    order a "previous valid" lookup walks until it finds a non-NaN cell.
    """
    obs_lin = _linearize_hours(obs_hour)
    target = float(_linearize_hours(np.array([target_hour]))[0])
    finite = np.where(np.isfinite(obs_lin))[0]
    at_or_before = finite[obs_lin[finite] <= target]
    return at_or_before[np.argsort(obs_lin[at_or_before])[::-1]]  # most recent first


def _previous_valid_order(
    obs_hour: np.ndarray, forecast_hours: np.ndarray
) -> list[np.ndarray]:
    """Per forecast hour, the L4 slots to try (nearest-past first) for that hour.

    A forecast hour with no prior observation (or undefined hour) gets an empty
    candidate list -> NaN.
    """
    orders = []
    for fh in np.asarray(forecast_hours, dtype="float64"):
        if not np.isfinite(fh):
            orders.append(np.empty(0, dtype=int))
        else:
            orders.append(_slots_at_or_before(obs_hour, fh))
    return orders


def _last_before_window_order(
    obs_hour: np.ndarray, forecast_hours: np.ndarray
) -> np.ndarray:
    """The L4 slots (nearest-past first) observed strictly before the window opens.

    A single antecedent snapshot taken just before the earliest forecast hour,
    held constant across the window. Strict inequality: an observation exactly at
    the window start is excluded.
    """
    fc_lin = _linearize_hours(forecast_hours)
    if not np.isfinite(fc_lin).any():
        return np.empty(0, dtype=int)
    window_start = float(np.nanmin(fc_lin))
    order = _slots_at_or_before(obs_hour, window_start)
    obs_lin = _linearize_hours(obs_hour)
    return order[obs_lin[order] < window_start]  # drop any obs exactly at the start


def _coalesce_slots(cube: np.ndarray, order: np.ndarray) -> np.ndarray:
    """First valid (non-NaN) L4 sample per cell, walking ``order``.

    ``cube`` is (date, L4_slots, lat, lon); ``order`` a slot-index sequence,
    nearest first. Returns (date, lat, lon). This is what makes "previous_valid"
    and "last_before_window" fall back through missing observations per cell.
    """
    n_date, _, n_lat, n_lon = cube.shape
    out = np.full((n_date, n_lat, n_lon), np.nan, dtype="float32")
    for slot in order:
        still_missing = np.isnan(out)
        if not still_missing.any():
            break
        out[still_missing] = cube[:, slot][still_missing]
    return out


def _make_l4_time_mapper(
    obs_hour: np.ndarray,
    forecast_hours: np.ndarray,
    policy: str,
    out_shape: tuple[int, int, int, int],
):
    """Return ``apply(cube) -> (D, S, LA, LO)`` for the chosen L4 time policy.

    The time logic (bracket weights or candidate slot orders) is derived once here
    from the observation and forecast hours; the returned closure applies it to
    each L4 field's ``(date, L4_slots, lat, lon)`` cube. See
    :func:`make_uniform` for what each policy means. Unknown policies raise.
    """
    if policy not in SMAP_TIME_POLICIES:
        raise ValueError(
            f"unknown smap_time_policy {policy!r}; choose one of {SMAP_TIME_POLICIES}"
        )

    if policy == "interp":
        lo_slot, hi_slot, weight = _l4_interp_weights(obs_hour, forecast_hours)

        def apply(cube: np.ndarray) -> np.ndarray:
            out = np.full(out_shape, np.nan, dtype="float32")
            for s, (lo, hi, w) in enumerate(zip(lo_slot, hi_slot, weight)):
                if lo >= 0:
                    out[:, s] = (1.0 - w) * cube[:, lo] + w * cube[:, hi]
            return out
        return apply

    if policy == "previous_valid":
        orders = _previous_valid_order(obs_hour, forecast_hours)  # one per forecast hour

        def apply(cube: np.ndarray) -> np.ndarray:
            out = np.full(out_shape, np.nan, dtype="float32")
            for s, order in enumerate(orders):
                if order.size:
                    out[:, s] = _coalesce_slots(cube, order)
            return out
        return apply

    # policy == "last_before_window": one antecedent snapshot held across every hour.
    order = _last_before_window_order(obs_hour, forecast_hours)

    def apply(cube: np.ndarray) -> np.ndarray:
        out = np.full(out_shape, np.nan, dtype="float32")
        if order.size:
            snapshot = _coalesce_slots(cube, order)  # (D, LA, LO)
            out[:] = snapshot[:, None, :, :]  # broadcast the single snapshot across hours
        return out
    return apply


# --------------------------------------------------------------------------- #
# The three pipeline stages
# --------------------------------------------------------------------------- #
def open_year(year: int, drop_parceltime: bool = True) -> xr.Dataset:
    year_path = config.DATA_DIR / config.YEAR_FILE_TEMPLATE.format(year=year)
    drop = ["FCST_parceltime"] if drop_parceltime else None
    return xr.open_dataset(year_path, drop_variables=drop)


def load_land_fraction_grid(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    # output shape (len(lats), len(lons))
    with xr.open_dataset(config.LSM_PATH) as lsm:
        sub = lsm["lsm"].sel(lat=lats, lon=lons, method="nearest")
        return np.asarray(sub.values, dtype=float)


def load_raw(
    years: Iterable[int],
    lat_range: Optional[tuple[float, float]] = None,
    lon_range: Optional[tuple[float, float]] = None,
    months: Optional[tuple[int, int]] = None,
    variables: Optional[Iterable[str]] = None,
) -> xr.Dataset:
    """Stage 1 of 3: the raw yearly cubes as one queryable dataset.

    Nothing is derived or renamed -- variables keep their file names and native
    time axes (see the module docstring). A ``land_frac`` (lat, lon) variable
    from the land-sea mask is attached so it can be queried alongside.

    Parameters
    ----------
    years
        Years to load, concatenated along ``date``.
    lat_range, lon_range
        Optional inclusive (min, max) slices in degrees, e.g. ``config.DOMAIN_LAT``.
    months
        Optional inclusive (first, last) month range, e.g. ``config.ANALYSIS_MONTHS``.
    variables
        Optional subset of file variables to load. Everything loaded is held in
        memory, so pass this (and/or the slices) when pulling many years.
    """
    parts = []
    for year in years:
        ds = open_year(year, drop_parceltime=False)
        if variables is not None:
            ds = ds[list(variables)]
        if lat_range is not None:
            ds = ds.sel(lat=slice(*lat_range))
        if lon_range is not None:
            ds = ds.sel(lon=slice(*lon_range))
        if months is not None:
            month = ds["date"].dt.month.values
            ds = ds.isel(date=np.where((month >= months[0]) & (month <= months[1]))[0])
        parts.append(ds.load())
        ds.close()

    raw = xr.concat(parts, dim="date")
    raw["land_frac"] = (
        ("lat", "lon"),
        load_land_fraction_grid(raw["lat"].values, raw["lon"].values).astype("float32"),
    )
    return raw


def make_uniform(
    ds: xr.Dataset,
    slots: tuple[int, ...] = config.FORECAST_SLOTS,
    smap_time_policy: Optional[str] = DEFAULT_SMAP_TIME_POLICY,
) -> xr.Dataset:
    """Stage 2 of 3: everything onto one (date, slot, lat, lon) forecast-hour grid.

    - Variables on a 7-slot forecast axis are subset to ``slots``; the axis
      becomes the common ``slot`` dimension with an ``hour_utc`` coordinate.
    - The overpass-time predictor (slot 0) is replicated across the forecast
      hours as ``FCST_MU_CAPE_overpass`` -- the paper's proximity-sounding
      baseline ("the values calculated at overpass time are replicated for each
      of the 21-02 UTC forecast timesteps").
    - SMAP L4 fields (native 5-slot axis, ~16,19,22,01,04 UTC) are placed on the
      forecast hours according to ``smap_time_policy``:

      - ``"interp"``: linear-in-time blend of the two bracketing observations
        (no extrapolation);
      - ``"previous_valid"``: newest observation at or before each forecast
        hour, falling back per cell through missing samples;
      - ``"last_before_window"``: one antecedent snapshot from just before the
        first forecast hour, held constant across the window;
      - ``None``: leave the SMAP L4 fields out entirely.

    - ``MRMS_PrecipFlag_cnt`` (extra per-category ``nflags`` axis) is unstacked
      into one variable per category, e.g. ``MRMS_PrecipFlag_cnt_Convection``.
    - Static (lat, lon) fields (``land_frac``) pass through; anything that fits
      none of these shapes is dropped.
    """
    slots = list(slots)
    n_date, n_lat, n_lon = ds.sizes["date"], ds.sizes["lat"], ds.sizes["lon"]

    out = xr.Dataset(
        coords={
            "date": ds["date"].values,
            "slot": np.array(slots, dtype="int8"),
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
            "hour_utc": (
                "slot",
                np.array([_SLOT_TO_HOUR.get(int(s), -1) for s in slots], dtype="int8"),
            ),
        }
    )

    # Assign via explicit dims + values: the files carry junk non-dimension
    # coordinates (e.g. a zero-filled ``nhours`` on the ``time`` axis) that would
    # otherwise ride along into the uniform dataset.
    for name, da in ds.data_vars.items():
        axis = next((d for d in FORECAST_DIMS if d in da.dims), None)
        if axis is not None and set(da.dims) == {"date", axis, "lat", "lon"}:
            sub = da.isel({axis: slots}).transpose("date", axis, "lat", "lon")
            out[name] = (("date", "slot", "lat", "lon"), sub.values)
        elif axis is not None and set(da.dims) == {"date", "nflags", axis, "lat", "lon"}:
            sub = da.isel({axis: slots}).transpose("nflags", "date", axis, "lat", "lon")
            for k, cat in enumerate(ds["nflags"].values):
                cat = cat.decode() if isinstance(cat, bytes) else str(cat)
                out[f"{name}_{cat}"] = (("date", "slot", "lat", "lon"), sub.values[k])
        elif set(da.dims) <= {"lat", "lon"}:
            out[name] = (da.dims, da.values)

    for var in config.OVERPASS_REPLICATED_VARS:
        if var in ds:
            overpass = ds[var].isel(time=config.OVERPASS_SLOT).values
            out[f"{var}_overpass"] = (
                ("date", "slot", "lat", "lon"),
                np.broadcast_to(overpass[:, None], (n_date, len(slots), n_lat, n_lon)),
            )

    l4_fields = [v for v in ds.data_vars if L4_DIM in ds[v].dims and ds[v].ndim == 4]
    if smap_time_policy is not None and l4_fields:
        # Observation hours are fixed 3-hourly in this data cut; the median over
        # days gives the one set of slot hours the mapping needs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN days
            obs_hour = np.nanmedian(ds[SMAP_L4_HOUR_VAR].values, axis=0)
        forecast_hours = np.array(
            [_SLOT_TO_HOUR.get(int(s), np.nan) for s in slots], dtype="float64"
        )
        apply_policy = _make_l4_time_mapper(
            obs_hour, forecast_hours, smap_time_policy,
            (n_date, len(slots), n_lat, n_lon),
        )
        for name in l4_fields:
            out[name] = (
                ("date", "slot", "lat", "lon"),
                apply_policy(np.asarray(ds[name].values, dtype="float32")),
            )

    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    raw = load_raw([2019], months=(6, 6), variables=[config.QPE_VAR, config.QPE_CNT_VAR])
    uniform = make_uniform(raw, config.FORECAST_SLOTS, smap_time_policy=None)
    print(f"June 2019 uniform grid: {dict(uniform.sizes)}")
