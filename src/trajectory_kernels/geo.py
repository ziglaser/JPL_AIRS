"""Small, pure geodesy helpers.

Distances use the spherical-earth approximation (``config.EARTH_RADIUS_KM``),
accurate to ~0.5% over the ~hundreds-of-km scales here.
"""

from __future__ import annotations

import numpy as np

from . import config

_DEG2RAD = np.pi / 180.0


def km_per_deg_lon(lat_deg: np.ndarray | float) -> np.ndarray | float:
    """Kilometres per degree of longitude at ``lat_deg``: ``(pi/180) * R * cos(lat)``."""
    return _DEG2RAD * config.EARTH_RADIUS_KM * np.cos(np.asarray(lat_deg) * _DEG2RAD)


def km_per_deg_lat() -> float:
    """Kilometres per degree of latitude (constant on a sphere)."""
    return _DEG2RAD * config.EARTH_RADIUS_KM


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance (km) between two lat/lon points (degrees)."""
    lat1, lon1, lat2, lon2 = (np.asarray(a, dtype=float) for a in (lat1, lon1, lat2, lon2))
    dlat = (lat2 - lat1) * _DEG2RAD
    dlon = (lon2 - lon1) * _DEG2RAD
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1 * _DEG2RAD) * np.cos(lat2 * _DEG2RAD) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def cumulative_path_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Along-track cumulative distance (km) from the first point of a trajectory.

    Same length as the input, with ``[0] == 0``. NaN segments propagate as NaN
    from that point on (the caller truncates there).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    seg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])
