"""Sub-hourly trajectory resampling.

A fast parcel can cross more than one source grid cell per stored hourly step,
so ``(lat, lon, alt)`` are linearly interpolated in time onto a fine grid
(``config.RESAMPLE_STEP_MIN``) before the residence-time deposit.
"""

from __future__ import annotations

import numpy as np

from . import config


def resample_series(t_hours: np.ndarray, y: np.ndarray, dt_hours: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample ``y(t)`` onto a uniform grid of spacing ``dt_hours``.

    ``t_hours`` must be increasing. The fine grid spans ``[t[0], t[-1]]`` and
    always includes the endpoint, so resampled endpoints equal the inputs.
    """
    t_hours = np.asarray(t_hours, dtype=float)
    y = np.asarray(y, dtype=float)
    n = max(int(np.ceil((t_hours[-1] - t_hours[0]) / dt_hours)), 1)
    fine_t = t_hours[0] + dt_hours * np.arange(n + 1)
    fine_t[-1] = t_hours[-1]  # exact endpoint
    return fine_t, np.interp(fine_t, t_hours, y)


def resample_trajectory(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    t_hours: np.ndarray,
    step_min: float = config.RESAMPLE_STEP_MIN,
) -> dict[str, np.ndarray]:
    """Resample one parcel's ``(lat, lon, alt)`` onto a fine, uniform time grid.

    Only the leading contiguous finite run is used: a trajectory that goes NaN
    mid-run is truncated at its last finite step. Returns a dict with
    ``t_hours``, ``lat``, ``lon``, ``alt``.
    """
    t_hours = np.asarray(t_hours, dtype=float)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)

    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt) & np.isfinite(t_hours)
    bad = np.where(~finite)[0]
    n_keep = int(bad[0]) if bad.size else finite.size
    lead = slice(0, n_keep)

    if n_keep < 2:  # nothing to interpolate; pass the (possibly empty) run through
        return {"t_hours": t_hours[lead], "lat": lat[lead],
                "lon": lon[lead], "alt": alt[lead]}

    dt_hours = step_min / 60.0
    ft, flat = resample_series(t_hours[lead], lat[lead], dt_hours)
    _, flon = resample_series(t_hours[lead], lon[lead], dt_hours)
    _, falt = resample_series(t_hours[lead], alt[lead], dt_hours)
    return {"t_hours": ft, "lat": flat, "lon": flon, "alt": falt}
