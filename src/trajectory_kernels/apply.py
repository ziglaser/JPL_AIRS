"""Apply a kernel to a surface field: the predictor-agnostic convolution

    influence(x_r, t_r) = sum over (x_s, tau) of K(x_s, tau) * S(x_s)

All soil-moisture physics enters here: the kernel is purely geometric, ``S`` is
any (lat, lon) surface field (SMAP soil moisture, land fraction, ...). v1
treats ``S`` as static over the few-hour trajectory window.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def lookup_from_dataarray(
    surface: xr.DataArray, fill_value: float = np.nan, method: str = "nearest"
) -> Callable:
    """Wrap a static ``(lat, lon)`` surface field as ``f(lat, lon)``.

    Default ``method="nearest"`` because linear interpolation blends NaN nodes
    into their neighbours (0*NaN poisons the neighbourhood); nearest returns
    NaN only where the nearest cell is genuinely missing. Use ``"linear"`` for
    smooth gap-free fields (e.g. the land mask).
    """
    lat = surface["lat"].values.astype(float)
    lon = surface["lon"].values.astype(float)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    vals = surface.values[np.ix_(lat_order, lon_order)].astype(float)
    interp = RegularGridInterpolator(
        (lat[lat_order], lon[lon_order]), vals,
        method=method, bounds_error=False, fill_value=fill_value,
    )

    def lookup(qlat, qlon):
        qlat = np.asarray(qlat, dtype=float)
        qlon = np.asarray(qlon, dtype=float)
        points = np.stack([qlat.ravel(), qlon.ravel()], axis=-1)
        return interp(points).reshape(qlat.shape)

    return lookup


def apply_kernel(
    kernel_ds: xr.Dataset,
    surface,
    which: str = "kernel",
    empty_to_nan: bool = True,
    min_coverage: float = 0.0,
    return_coverage: bool = False,
):
    """Convolve the relative-window kernels with a static surface field.

    ``surface`` is an ``xr.DataArray`` on (lat, lon) or a callable
    ``f(lat, lon)``. Returns ``influence(arrival_step, target_lat, target_lon)``.

    NaN handling (SMAP is ~half NaN here): source cells without surface data
    are dropped and, for ``which="kernel"``, the result is renormalized by the
    retained kernel weight -- so it stays a true weighted average (a uniform
    field returns that constant even with gaps). ``which="footprint"`` skips the
    renormalization (residence-time integral, units hours x field).

    ``min_coverage``: receptors whose retained kernel-weight fraction falls
    below this become NaN. ``return_coverage=True`` also returns that fraction.
    Empty receptors (``n_parcels == 0``) are NaN when ``empty_to_nan``, else 0.

    NOTE the kernel is normalized per lag hour (each populated lag slice sums
    to 1), so with ``which="kernel"`` every populated hour contributes EQUAL
    weight to the average until the separate hour-to-hour weighting step;
    ``which="footprint"`` still weights hours by physical contact time.
    """
    surface_fn = surface if callable(surface) else lookup_from_dataarray(surface)

    tlat = kernel_ds["target_lat"].values
    tlon = kernel_ds["target_lon"].values
    dlat = kernel_ds["dlat"].values
    dlon = kernel_ds["dlon"].values
    n_step = kernel_ds.sizes["arrival_step"]
    kernel = kernel_ds[which].values  # (step, tlat, tlon, lag, dlat, dlon)

    influence = np.full((n_step, tlat.size, tlon.size), np.nan)
    coverage = np.full((n_step, tlat.size, tlon.size), np.nan)
    for i, target_lat in enumerate(tlat):
        source_lats = target_lat + dlat
        for j, target_lon in enumerate(tlon):
            block = np.nan_to_num(kernel[:, i, j], nan=0.0)  # (step, lag, dlat, dlon)
            total_weight = block.sum(axis=(1, 2, 3))
            if not np.any(total_weight > 0):
                continue

            source_lons = target_lon + dlon
            grid_lat, grid_lon = np.meshgrid(source_lats, source_lons, indexing="ij")
            surf = surface_fn(grid_lat, grid_lon)          # (dlat, dlon)
            has_data = np.isfinite(surf)
            surf_filled = np.where(has_data, surf, 0.0)

            # sum over (lag, dlat, dlon); weight only counts where data exists
            weighted_sum = (block * surf_filled).sum(axis=(1, 2, 3))
            retained_weight = (block * has_data).sum(axis=(1, 2, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(total_weight > 0, retained_weight / total_weight, np.nan)
                if which == "kernel":
                    value = weighted_sum / retained_weight
                else:
                    value = weighted_sum
                value = np.where(retained_weight > 0, value, np.nan)
                value = np.where(frac >= min_coverage, value, np.nan)
            influence[:, i, j] = value
            coverage[:, i, j] = frac

    if empty_to_nan:
        populated = kernel_ds["n_parcels"].values > 0
        influence = np.where(populated, influence, np.nan)
    else:
        influence = np.nan_to_num(influence, nan=0.0)

    coords = {"arrival_step": kernel_ds["arrival_step"],
              "target_lat": tlat, "target_lon": tlon}
    out = xr.DataArray(
        influence, dims=("arrival_step", "target_lat", "target_lon"), coords=coords,
        name=f"influence_{which}",
        attrs={"long_name": f"kernel-convolved surface influence ({which})"},
    )
    if return_coverage:
        cov = xr.DataArray(coverage, dims=out.dims, coords=coords,
                           name="surface_coverage",
                           attrs={"long_name": "fraction of kernel mass over valid surface cells"})
        return out, cov
    return out
