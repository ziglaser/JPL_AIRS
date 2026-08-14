"""Predictor engineering shared by every hypothesis: seasonal-cycle removal,
antecedent lags, and the Guillod-style local/nonlocal decomposition.

All functions operate on ``xr.DataArray`` with a ``date`` dimension (plus any
spatial dims) and are pure -- the table builder applies them and merges results
onto the tidy row table. Seasonality is removed *at the predictor* (day-of-year
harmonic climatology, pooled across years, per cell) so that hypothesis tests
compare like-with-like anomalies rather than riding the annual cycle; season
remains available as an explicit stratum.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

DAYS_PER_YEAR = 365.25


def _harmonic_design(doy: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Design matrix [1, sin(k w doy), cos(k w doy)] for k=1..n_harmonics."""
    w = 2.0 * np.pi / DAYS_PER_YEAR
    cols = [np.ones_like(doy, dtype=float)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * w * doy))
        cols.append(np.cos(k * w * doy))
    return np.stack(cols, axis=1)  # (n_dates, 1 + 2*n_harmonics)


def harmonic_climatology(da: xr.DataArray, n_harmonics: int = 2) -> xr.DataArray:
    """Per-cell day-of-year climatology: least-squares harmonic fit over all years.

    Annual + semiannual (default) harmonics capture the smooth seasonal cycle
    without chasing weather (a 31-day rolling climatology with ~6 years of data
    still retains synoptic noise; two harmonics is the standard smooth
    alternative). NaNs are handled per cell: the fit uses only finite samples;
    cells with fewer than ``3 * (1 + 2 n_harmonics)`` finite dates return NaN.
    """
    doy = da["date"].dt.dayofyear.values.astype(float)
    X = _harmonic_design(doy, n_harmonics)  # (D, P)
    vals = da.values.reshape(da.sizes["date"], -1)  # (D, C)
    n_par = X.shape[1]

    clim = np.full_like(vals, np.nan, dtype=float)
    finite_mask = np.isfinite(vals)
    # Group cells by their finite-date pattern? Simpler: full-column fast path +
    # per-cell fallback. Most cells are either all-finite over valid days or share
    # the same missing days, so the fast path dominates.
    all_finite = finite_mask.all(axis=0)
    if all_finite.any():
        beta, *_ = np.linalg.lstsq(X, vals[:, all_finite], rcond=None)
        clim[:, all_finite] = X @ beta
    for c in np.where(~all_finite)[0]:
        ok = finite_mask[:, c]
        if ok.sum() < 3 * n_par:
            continue
        beta, *_ = np.linalg.lstsq(X[ok], vals[ok, c], rcond=None)
        clim[:, c] = X @ beta

    out = xr.DataArray(clim.reshape(da.shape), dims=da.dims, coords=da.coords)
    out.name = f"{da.name}_clim" if da.name else "clim"
    return out


def deseasonalize(da: xr.DataArray, n_harmonics: int = 2) -> xr.DataArray:
    """Anomaly = value - per-cell harmonic day-of-year climatology."""
    anom = da - harmonic_climatology(da, n_harmonics)
    anom.name = f"{da.name}_anom" if da.name else "anom"
    return anom


def zscore_by_cell(anom: xr.DataArray) -> xr.DataArray:
    """Standardize an anomaly by its per-cell std (comparable across the domain).

    Wet-region and dry-region SM anomalies have very different natural ranges;
    per-cell z-scores put one 'unusually wet/dry for *this* place' scale on the
    whole domain (the Koster/GLACE-style convention). Cells with zero/NaN std
    return NaN.
    """
    sd = anom.std(dim="date", skipna=True)
    out = anom / xr.where(sd > 0, sd, np.nan)
    out.name = f"{anom.name}_z" if anom.name else "z"
    return out


def antecedent_mean(da: xr.DataArray, lag_start: int, lag_end: int) -> xr.DataArray:
    """Mean of the field over dates [t - lag_end, t - lag_start] (days, inclusive).

    ``antecedent_mean(sm, 1, 7)`` is 'last week's soil moisture' aligned to each
    date -- the antecedent predictor for T4 and the antecedent-precip control for
    the Tuttle & Salvucci confounding guard. Requires a contiguous daily ``date``
    axis (asserted); edges with insufficient history are NaN.
    """
    if lag_start < 0 or lag_end < lag_start:
        raise ValueError("need 0 <= lag_start <= lag_end")
    step = np.unique(np.diff(da["date"].values).astype("timedelta64[D]").astype(int))
    if step.size != 1 or step[0] != 1:
        raise ValueError("antecedent_mean requires a contiguous daily date axis")
    stack = [da.shift(date=k) for k in range(lag_start, lag_end + 1)]
    out = xr.concat(stack, dim="__lag__").mean("__lag__", skipna=False)
    out.name = f"{da.name}_ante{lag_start}_{lag_end}" if da.name else "antecedent"
    return out


def neighborhood_mean(da: xr.DataArray, halfwidth: int = 2,
                      min_valid: int = 6, exclude_center: bool = True) -> xr.DataArray:
    """NaN-aware (2h+1)x(2h+1) neighborhood mean on the (lat, lon) grid.

    ``halfwidth=1`` -> a 3x3-cell (~300 km) box, the Guillod et al. (2015)
    event-surroundings analog at 1-deg resolution. ``exclude_center`` leaves the
    target cell out so 'local vs surroundings' is a clean contrast. Fewer than
    ``min_valid`` finite neighbors -> NaN.
    """
    from scipy.ndimage import uniform_filter

    vals = da.values
    finite = np.isfinite(vals)
    filled = np.where(finite, vals, 0.0)

    # box SUM over (lat, lon) only: uniform_filter gives the box mean (zeros
    # outside the edge with mode="constant"), so multiply back by the box size
    k = 2 * halfwidth + 1
    size = tuple(k if d in ("lat", "lon") else 1 for d in da.dims)
    total = uniform_filter(filled, size=size, mode="constant") * k * k
    count = uniform_filter(finite.astype(float), size=size, mode="constant") * k * k

    if exclude_center:
        total = total - filled
        count = count - finite.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count >= min_valid - 0.5, total / count, np.nan)

    out = xr.DataArray(mean, dims=da.dims, coords=da.coords)
    out.name = f"{da.name}_nbhd" if da.name else "nbhd"
    return out


def local_nonlocal_decomposition(
    anom: xr.DataArray, halfwidth: int = 2
) -> tuple[xr.DataArray, xr.DataArray]:
    """Guillod-style split of an anomaly field into local and nonlocal parts.

    - **local**   = cell anomaly - neighborhood-mean anomaly (is this cell
      wetter/drier than its immediate surroundings today?)  [spatial signal]
    - **nonlocal** = neighborhood-mean anomaly (is the whole area anomalously
      wet/dry today?)                                        [temporal signal]

    S4 predicts these enter heavy-precip prediction with opposite signs
    (rain favors locally-dry patches on regionally-wet days).
    """
    nbhd = neighborhood_mean(anom, halfwidth=halfwidth)
    local = anom - nbhd
    local.name = f"{anom.name}_local" if anom.name else "local"
    nbhd.name = f"{anom.name}_nonlocal" if anom.name else "nonlocal"
    return local, nbhd
