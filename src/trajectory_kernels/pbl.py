"""Boundary-layer depth models: pluggable ``PBLModel`` callables mapping
``(lat, lon, time_utc) -> depth_m`` (broadcasting over array inputs).

:class:`ConstantPBL` is a fixed depth; :class:`ClimatologicalPBL` is a diurnal
curve with a west-east daytime-depth gradient from the summer-CONUS climatology
(McGrath-Spangler & Denning 2012; Seidel+2012); :class:`GriddedPBL` reads the
assessed Guo et al. (2024) PBLH product with a climatology + analytic fallback
chain, so ``m_star`` and the contact gate carry the actual day's boundary-layer
state (UPWIND_INDEX_REVIEW.md F2) while every lookup stays answerable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def _utc_fractional_hour(time_utc) -> np.ndarray:
    """UTC hour-of-day in [0, 24) as a float array, from datetime64 input."""
    t = np.asarray(time_utc, dtype="datetime64[ns]")
    day = t.astype("datetime64[D]")
    seconds = (t - day) / np.timedelta64(1, "s")
    return (seconds / 3600.0).astype(float)


def local_solar_hour(lon_deg: np.ndarray, time_utc) -> np.ndarray:
    """Approximate local solar hour = UTC hour + lon/15, wrapped to [0, 24).

    A 15 deg-per-hour longitude offset; good enough to place the diurnal PBL
    cycle, and far below the trajectory uncertainty the fuzz kernel represents.
    """
    utc_h = _utc_fractional_hour(time_utc)
    return np.mod(utc_h + np.asarray(lon_deg, dtype=float) / 15.0, 24.0)


class PBLModel:
    """Base class: subclasses implement ``depth(lat, lon, time_utc) -> metres``."""

    def depth(self, lat, lon, time_utc) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, lat, lon, time_utc) -> np.ndarray:
        return self.depth(lat, lon, time_utc)


class ConstantPBL(PBLModel):
    """A spatially/temporally uniform PBL depth (metres)."""

    def __init__(self, depth_m: float = config.PBL_FIXED_DEFAULT_M):
        self.depth_m = float(depth_m)

    def depth(self, lat, lon, time_utc) -> np.ndarray:
        hour = _utc_fractional_hour(time_utc)
        shape = np.broadcast(np.asarray(lat), np.asarray(lon), hour).shape
        return np.full(shape, self.depth_m, dtype=float)


class ClimatologicalPBL(PBLModel):
    """Seasonal-climatology diurnal PBL: nocturnal floor ramping to a daytime max.

    The daytime maximum deepens westward (Great Plains/Rockies vs the East); a
    smooth cosine rises from sunrise to a mid-afternoon peak and falls to sunset,
    so the evening PBL collapse emerges naturally. Parameters live in :mod:`config`.
    """

    def __init__(
        self,
        nocturnal_m: float = config.PBL_NOCTURNAL_M,
        daytime_m: float = config.PBL_DAYTIME_M,
        peak_hour: float = config.PBL_PEAK_HOUR_LOCAL,
        sunrise_hour: float = config.PBL_SUNRISE_HOUR_LOCAL,
        sunset_hour: float = config.PBL_SUNSET_HOUR_LOCAL,
        west_deepening_m_per_deg: float = config.PBL_WEST_DEEPENING_M_PER_DEG,
        reference_lon: float = config.PBL_REFERENCE_LON,
    ):
        self.nocturnal_m = float(nocturnal_m)
        self.daytime_m = float(daytime_m)
        self.peak_hour = float(peak_hour)
        self.sunrise_hour = float(sunrise_hour)
        self.sunset_hour = float(sunset_hour)
        self.west_deepening_m_per_deg = float(west_deepening_m_per_deg)
        self.reference_lon = float(reference_lon)

    def _diurnal_shape(self, h: np.ndarray) -> np.ndarray:
        """0 at night / sunrise / sunset, 1 at the afternoon peak; smooth between."""
        h = np.asarray(h, dtype=float)
        shape = np.zeros_like(h)
        rising = (h > self.sunrise_hour) & (h <= self.peak_hour)
        falling = (h > self.peak_hour) & (h < self.sunset_hour)
        shape[rising] = 0.5 * (
            1.0 - np.cos(np.pi * (h[rising] - self.sunrise_hour) / (self.peak_hour - self.sunrise_hour))
        )
        shape[falling] = 0.5 * (
            1.0 + np.cos(np.pi * (h[falling] - self.peak_hour) / (self.sunset_hour - self.peak_hour))
        )
        return shape

    def depth(self, lat, lon, time_utc) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        h = local_solar_hour(lon, time_utc)
        daytime_max = self.daytime_m + self.west_deepening_m_per_deg * (self.reference_lon - lon)
        daytime_max = np.maximum(daytime_max, self.nocturnal_m)
        depth = self.nocturnal_m + (daytime_max - self.nocturnal_m) * self._diurnal_shape(h)
        # broadcast to include lat if lat carries the shape
        out_shape = np.broadcast(np.asarray(lat), depth).shape
        return np.broadcast_to(depth, out_shape).astype(float)


def _nearest_cell_index(centres: np.ndarray, query: np.ndarray):
    """Grid-exact nearest-cell index on a regular 1-D axis, ascending or descending.

    Index arithmetic on the constant step (no search), so a query is assigned to
    the cell whose centre it is nearest -- the same "nearest, NaN only where the
    cell is genuinely missing" philosophy as :func:`apply.lookup_from_dataarray`.
    Returns ``(index, in_range)``: indices are clipped to the axis edges so they
    are always safe to gather with, and ``in_range`` is False for queries beyond
    the outermost cell *edges* (centre +- step/2), which callers must map to NaN.
    """
    centres = np.asarray(centres, dtype=float)
    step = centres[1] - centres[0] if centres.size > 1 else 1.0
    fractional = (np.asarray(query, dtype=float) - centres[0]) / step
    in_range = (fractional >= -0.5) & (fractional <= centres.size - 0.5)
    index = np.clip(np.rint(fractional).astype(np.intp), 0, centres.size - 1)
    return index, in_range


def _nearest_time_index(axis_ns: np.ndarray, query_ns: np.ndarray, tolerance_ns: int):
    """Nearest sample on a sorted datetime axis (int64 ns), within a tolerance.

    Binary search rather than index arithmetic because the assessed record has
    gaps (missing source times; the Oct-2021 hole), so the time axis -- unlike
    the lat/lon grid -- is not guaranteed regular. Returns ``(index, in_range)``
    with the same clip-and-mask contract as :func:`_nearest_cell_index`.
    """
    if axis_ns.size == 1:
        index = np.zeros(np.shape(query_ns), dtype=np.intp)
    else:
        right = np.clip(np.searchsorted(axis_ns, query_ns), 1, axis_ns.size - 1)
        left = right - 1
        pick_right = np.abs(axis_ns[right] - query_ns) < np.abs(axis_ns[left] - query_ns)
        index = np.where(pick_right, right, left)
    in_range = np.abs(axis_ns[index] - query_ns) <= tolerance_ns
    return index, in_range


class GriddedPBL(PBLModel):
    """Assessed PBL depth with a three-layer fallback chain, so every lookup answers.

    Layer 1 reads the day's actual boundary-layer state from the 1-degree
    3-hourly aggregate of the Guo et al. (2024) assessed (ML-merged,
    radiosonde/lidar-constrained) PBLH product -- the input that makes ``m_star``
    and the contact gate carry the day's soil-driven PBL signal instead of a
    smooth geography proxy (UPWIND_INDEX_REVIEW.md F2). Layer 2 fills its holes
    (off-grid, 2016 and other pre-2017 dates, the Oct-2021 gap, ocean/no-retrieval
    cells) from the monthly-diurnal climatology of the same product. Layer 3
    answers whatever remains (off the CONUS climatology domain, over water) with
    an analytic :class:`PBLModel`, default :class:`ClimatologicalPBL`.

    Parameters
    ----------
    three_hourly_path : path or None
        Per-timestamp 1-degree aggregate: variable ``pblh`` (metres) on
        ``(time, lat, lon)`` with a true datetime64 3-hourly time axis
        (built by ``scripts/build_pblh_3hrly_1deg.py``). ``None`` or a missing
        file skips layer 1.
    clim_path : path or None
        Monthly-diurnal climatology: ``pblh_mean`` (metres) on
        ``(month, hour, lat, lon)``, hour labelled by UTC slot 0, 3, ..., 21
        (built by ``scripts/build_pbl_climatology.py``). ``None`` or a missing
        file skips layer 2.
    date : datetime64-like or None
        When given (the per-day production job), only time slices within
        ``[date - 1 day, date + 2 days]`` of the 3-hourly file are loaded into
        memory -- ample for a <=12 h look-back from any arrival slot -- and the
        rest never leaves disk. ``None`` lazy-opens the full record.
    analytic_fallback : PBLModel, optional
        Layer 3. Default ``ClimatologicalPBL()``.

    Attributes
    ----------
    available : dict of bool
        Which layers can answer: ``{"assessed", "climatology", "analytic"}``.
    last_source_fractions : dict of float
        Fraction of the most recent :meth:`depth` call's points answered by each
        layer -- cheap honesty for the at-scale QA logs (review section 4.3).

    Examples
    --------
    Pure-analytic operation (both files absent) still functions:

    >>> pbl = GriddedPBL(three_hourly_path=None, clim_path=None)
    >>> pbl.available
    {'assessed': False, 'climatology': False, 'analytic': True}
    >>> z = pbl.depth([40.5, 35.5], -90.5, np.datetime64("2019-06-05T19:00"))
    >>> z.shape, bool(np.isfinite(z).all())
    ((2,), True)
    >>> pbl.last_source_fractions
    {'assessed': 0.0, 'climatology': 0.0, 'analytic': 1.0}

    With the real files, a 2016 query (no assessed coverage) falls through to
    the climatology and an off-CONUS ocean point to the analytic layer::

        pbl = GriddedPBL(date="2019-06-05")
        pbl.depth(40.5, -90.5, np.datetime64("2019-06-05T21:00"))  # assessed
        pbl.last_source_fractions   # {'assessed': 1.0, ...}
    """

    def __init__(
        self,
        three_hourly_path=config.PBLH_3HRLY_PATH,
        clim_path=config.PBLH_CLIM_PATH,
        date=None,
        analytic_fallback: PBLModel | None = None,
    ):
        import xarray as xr

        self.analytic = analytic_fallback if analytic_fallback is not None else ClimatologicalPBL()
        self._assessed = None  # DataArray "pblh" (time, lat, lon), loaded or lazy
        self._assessed_time_ns = np.empty(0, dtype=np.int64)
        self._clim_vals = None  # ndarray (month 1..12, hour 0..21 by 3, lat, lon)

        if three_hourly_path is not None and Path(three_hourly_path).exists():
            ds = xr.open_dataset(three_hourly_path)
            if date is not None:
                day = np.datetime64(date, "D")
                ds = ds.sel(time=slice(day - np.timedelta64(1, "D"),
                                       day + np.timedelta64(2, "D"))).load()
            self._assessed = ds["pblh"]
            self._assessed_time_ns = (
                ds["time"].values.astype("datetime64[ns]").astype(np.int64))

        if clim_path is not None and Path(clim_path).exists():
            with xr.open_dataset(clim_path) as clim:
                mean = clim["pblh_mean"].sortby("month").sortby("hour")
                self._clim_lat = mean["lat"].values.astype(float)
                self._clim_lon = mean["lon"].values.astype(float)
                self._clim_vals = mean.values.astype(float)

        self.available = {
            "assessed": self._assessed is not None and self._assessed_time_ns.size > 0,
            "climatology": self._clim_vals is not None,
            "analytic": True,
        }
        self.last_source_fractions = {"assessed": 0.0, "climatology": 0.0, "analytic": 0.0}

    def assessed_lookup(self, lat, lon, t_ns) -> np.ndarray:
        """Layer 1: nearest-time (within tolerance), nearest-cell assessed PBLH."""
        import xarray as xr

        out = np.full(lat.shape, np.nan)
        tol_ns = int(config.PBLH_TIME_TOLERANCE_H * 3600 * 1e9)
        it, ok_t = _nearest_time_index(self._assessed_time_ns, t_ns, tol_ns)
        iy, ok_y = _nearest_cell_index(self._assessed["lat"].values, lat)
        ix, ok_x = _nearest_cell_index(self._assessed["lon"].values, lon)
        hit = ok_t & ok_y & ok_x
        if hit.any():
            point = lambda idx: xr.DataArray(idx[hit], dims="pt")  # noqa: E731
            out[hit] = self._assessed.isel(
                time=point(it), lat=point(iy), lon=point(ix)).values.astype(float)
        return out

    def climatology_lookup(self, lat, lon, t_ns) -> np.ndarray:
        """Layer 2: (month, nearest 3 h UTC slot, nearest cell) climatology."""
        out = np.full(lat.shape, np.nan)
        t = t_ns.astype("datetime64[ns]")
        month_idx = t.astype("datetime64[M]").astype(int) % 12
        slot_idx = np.rint(_utc_fractional_hour(t) / 3.0).astype(np.intp) % 8
        iy, ok_y = _nearest_cell_index(self._clim_lat, lat)
        ix, ok_x = _nearest_cell_index(self._clim_lon, lon)
        hit = ok_y & ok_x
        out[hit] = self._clim_vals[month_idx[hit], slot_idx[hit], iy[hit], ix[hit]]
        return out

    def depth(self, lat, lon, time_utc) -> np.ndarray:
        t = np.asarray(time_utc, dtype="datetime64[ns]")
        lat_b, lon_b, t_b = np.broadcast_arrays(
            np.asarray(lat, dtype=float), np.asarray(lon, dtype=float), t)
        shape = lat_b.shape
        lat_f, lon_f = lat_b.ravel(), lon_b.ravel()
        t_ns = t_b.ravel().astype(np.int64)

        out = np.full(lat_f.shape, np.nan)
        n = max(out.size, 1)

        if self.available["assessed"]:
            out = self.assessed_lookup(lat_f, lon_f, t_ns)
        n_assessed = int(np.isfinite(out).sum())

        gap = ~np.isfinite(out)
        if self.available["climatology"] and gap.any():
            out[gap] = self.climatology_lookup(lat_f[gap], lon_f[gap], t_ns[gap])
        n_clim = int(np.isfinite(out).sum()) - n_assessed

        gap = ~np.isfinite(out)
        if gap.any():
            out[gap] = np.asarray(
                self.analytic.depth(lat_f[gap], lon_f[gap], t_ns[gap].astype("datetime64[ns]")),
                dtype=float)

        self.last_source_fractions = {
            "assessed": n_assessed / n,
            "climatology": n_clim / n,
            "analytic": (out.size - n_assessed - n_clim) / n,
        }
        return out.reshape(shape)
