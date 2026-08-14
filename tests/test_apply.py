"""Tests for the apply step -- the two analytic convolution identities."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from trajectory_kernels import apply as A


def _toy_kernels():
    """One populated receptor at (31.5, -94.5): half its influence on its own
    cell (lag 0) and half one cell west (lag 1). Kernel already normalized."""
    shape = (1, 3, 3, 2, 3, 3)  # step, tlat, tlon, lag, dlat, dlon
    kernel = np.full(shape, np.nan, dtype="float32")
    kernel[:] = np.nan
    k = np.zeros((2, 3, 3), dtype="float32")
    k[0, 1, 1] = 0.5   # own cell, lag 0
    k[1, 1, 0] = 0.5   # one cell west (dlon=-1), lag 1
    kernel[0, 1, 1] = k
    counts = np.zeros((1, 3, 3), dtype="int32")
    counts[0, 1, 1] = 5
    return xr.Dataset(
        {
            "kernel": (("arrival_step", "target_lat", "target_lon", "lag", "dlat", "dlon"), kernel),
            "n_parcels": (("arrival_step", "target_lat", "target_lon"), counts),
        },
        coords={
            "arrival_step": [3], "target_lat": [30.5, 31.5, 32.5],
            "target_lon": [-95.5, -94.5, -93.5], "lag": [0.0, 1.0],
            "dlat": [-1.0, 0.0, 1.0], "dlon": [-1.0, 0.0, 1.0],
        },
    )


def _surface(const=None, spike_at=None):
    lat = np.array([29.5, 30.5, 31.5, 32.5, 33.5])
    lon = np.array([-96.5, -95.5, -94.5, -93.5, -92.5])
    vals = np.zeros((lat.size, lon.size))
    if const is not None:
        vals[:] = const
    if spike_at is not None:
        i = np.argmin(np.abs(lat - spike_at[0]))
        j = np.argmin(np.abs(lon - spike_at[1]))
        vals[i, j] = 1.0
    return xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_uniform_field_returns_the_constant():
    """Normalized kernel sums to 1 -> convolution with a uniform field == that value."""
    ds = _toy_kernels()
    infl = A.apply_kernel(ds, _surface(const=0.3))
    assert float(infl.isel(arrival_step=0).sel(target_lat=31.5, target_lon=-94.5)) == pytest.approx(0.3)


def test_delta_field_returns_kernel_weight():
    """A spike at the receptor's own cell returns the lag-0 weight (0.5)."""
    ds = _toy_kernels()
    infl = A.apply_kernel(ds, _surface(spike_at=(31.5, -94.5)))
    assert float(infl.sel(arrival_step=3, target_lat=31.5, target_lon=-94.5)) == pytest.approx(0.5)


def test_delta_upwind_cell_returns_lag1_weight():
    """A spike one cell west returns the lag-1 weight (0.5)."""
    ds = _toy_kernels()
    infl = A.apply_kernel(ds, _surface(spike_at=(31.5, -95.5)))
    assert float(infl.sel(arrival_step=3, target_lat=31.5, target_lon=-94.5)) == pytest.approx(0.5)


def test_empty_receptor_is_nan():
    ds = _toy_kernels()
    infl = A.apply_kernel(ds, _surface(const=0.3))
    assert np.isnan(float(infl.sel(arrival_step=3, target_lat=30.5, target_lon=-95.5)))


def test_callable_surface_works():
    ds = _toy_kernels()
    infl = A.apply_kernel(ds, lambda la, lo: np.full_like(np.asarray(la, float), 2.0))
    assert float(infl.sel(arrival_step=3, target_lat=31.5, target_lon=-94.5)) == pytest.approx(2.0)


def test_uniform_field_with_nan_still_returns_constant():
    """Renormalization: a uniform field of 3.0 with NaN over the lag-1 cell still
    returns 3.0 (weighted average over the cells that have data), not a low-biased
    value -- the critical apply.py fix."""
    ds = _toy_kernels()
    s = _surface(const=3.0)
    s.values[np.argmin(np.abs(s["lat"].values - 31.5)),
             np.argmin(np.abs(s["lon"].values - -95.5))] = np.nan  # kill the upwind cell
    infl = A.apply_kernel(ds, s)
    assert float(infl.sel(arrival_step=3, target_lat=31.5, target_lon=-94.5)) == pytest.approx(3.0)


def test_nearest_lookup_does_not_propagate_nan():
    """A NaN node must not poison its neighbours (the 0*NaN linear-interp bug)."""
    s = _surface(const=1.0)
    s.values[0, 0] = np.nan
    fn = A.lookup_from_dataarray(s)  # default nearest
    # a valid node next to the NaN still returns its own finite value
    assert np.isfinite(fn(s["lat"].values[0], s["lon"].values[1]))
    assert fn(s["lat"].values[0], s["lon"].values[1]) == pytest.approx(1.0)


def test_min_coverage_blanks_low_data_receptors():
    """With half the kernel mass over NaN, min_coverage=0.6 blanks the receptor."""
    ds = _toy_kernels()
    s = _surface(const=3.0)
    s.values[np.argmin(np.abs(s["lat"].values - 31.5)),
             np.argmin(np.abs(s["lon"].values - -95.5))] = np.nan  # 0.5 of the mass gone
    infl, cov = A.apply_kernel(ds, s, min_coverage=0.6, return_coverage=True)
    assert np.isnan(float(infl.sel(arrival_step=3, target_lat=31.5, target_lon=-94.5)))
    assert float(cov.sel(arrival_step=3, target_lat=31.5, target_lon=-94.5)) == pytest.approx(0.5)
