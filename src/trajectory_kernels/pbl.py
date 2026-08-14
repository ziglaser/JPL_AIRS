"""Boundary-layer depth models: pluggable ``PBLModel`` callables mapping
``(lat, lon, time_utc) -> depth_m`` (broadcasting over array inputs).

:class:`ConstantPBL` is a fixed depth; :class:`ClimatologicalPBL` is a diurnal
curve with a west-east daytime-depth gradient from the summer-CONUS climatology
(McGrath-Spangler & Denning 2012; Seidel+2012).
"""

from __future__ import annotations

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
