"""Tests for the fuzz kernel -- mass conservation, delta limit, monotone width."""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import config
from trajectory_kernels.fuzz import EmpiricalFuzz, StohlFuzz, deposit_gaussian


def _grid():
    lat = np.arange(30.0, 45.0 + 0.001, 0.5)
    lon = np.arange(-100.0, -80.0 + 0.001, 0.5)
    return lat, lon


def test_sigma_grows_with_distance():
    f = StohlFuzz()
    assert f.sigma_km(0.0) == pytest.approx(config.FUZZ_SIGMA0_KM)
    assert f.sigma_km(1000.0) > f.sigma_km(100.0)
    # 20%-of-distance growth
    assert f.sigma_km(1000.0) - f.sigma_km(0.0) == pytest.approx(0.2 * 1000.0, rel=1e-6)


def test_fuzziness_scales_growth():
    sharp = StohlFuzz(fuzziness=0.5).sigma_km(1000.0)
    blurry = StohlFuzz(fuzziness=2.0).sigma_km(1000.0)
    assert blurry > sharp


def test_deposit_conserves_mass():
    lat, lon = _grid()
    for sig in (60.0, 150.0, 400.0):
        acc = np.zeros((lat.size, lon.size))
        deposit_gaussian(lat, lon, 37.5, -90.0, sig, weight=1.0, accumulator=acc)
        assert acc.sum() == pytest.approx(1.0, abs=1e-6)


def test_deposit_weight_scales():
    lat, lon = _grid()
    acc = np.zeros((lat.size, lon.size))
    deposit_gaussian(lat, lon, 37.5, -90.0, 200.0, weight=3.5, accumulator=acc)
    assert acc.sum() == pytest.approx(3.5, abs=1e-6)


def test_zero_sigma_is_delta_at_nearest_cell():
    lat, lon = _grid()
    acc = np.zeros((lat.size, lon.size))
    deposit_gaussian(lat, lon, 37.6, -90.1, 0.0, weight=1.0, accumulator=acc)
    assert acc.sum() == pytest.approx(1.0)
    i, j = np.unravel_index(np.argmax(acc), acc.shape)
    assert lat[i] == pytest.approx(37.5)  # nearest cell centre
    assert lon[j] == pytest.approx(-90.0)
    assert np.count_nonzero(acc) == 1


def test_deposit_is_centered_and_symmetric():
    lat = np.arange(30.0, 45.0 + 0.001, 0.5)
    lon = np.arange(-100.0, -80.0 + 0.001, 0.5)
    acc = np.zeros((lat.size, lon.size))
    # centre exactly on a cell so the blob is symmetric
    deposit_gaussian(lat, lon, 37.5, -90.0, 200.0, weight=1.0, accumulator=acc)
    i0 = int(np.argmin(np.abs(lat - 37.5)))
    j0 = int(np.argmin(np.abs(lon - -90.0)))
    assert acc[i0 - 1, j0] == pytest.approx(acc[i0 + 1, j0], rel=1e-6)
    assert acc[i0, j0 - 1] == pytest.approx(acc[i0, j0 + 1], rel=1e-6)
    assert acc[i0, j0] == acc.max()


def test_nonfinite_center_is_noop():
    lat, lon = _grid()
    acc = np.zeros((lat.size, lon.size))
    deposit_gaussian(lat, lon, np.nan, -90.0, 200.0, weight=1.0, accumulator=acc)
    assert acc.sum() == 0.0


def _synthetic_fullgrid(tmp_path):
    """A tiny fullgrid-shaped file with known winds: speed 5 m/s (u=3, v=4) and
    sub-box spread 1 m/s (u_std=0.6, v_std=0.8) -> alpha = 1/5 = 0.2 exactly."""
    import xarray as xr
    shape = (2, 3, 2, 2)  # time, level, lat, lon
    ds = xr.Dataset(
        {
            "u": (("time", "level", "lat", "lon"), np.full(shape, 3.0)),
            "v": (("time", "level", "lat", "lon"), np.full(shape, 4.0)),
            "u_std": (("time", "level", "lat", "lon"), np.full(shape, 0.6)),
            "v_std": (("time", "level", "lat", "lon"), np.full(shape, 0.8)),
            "N": (("time", "level", "lat", "lon"), np.full(shape, 10.0)),
        },
        coords={"time": [0, 1], "level": [800.0, 900.0, 1000.0],
                "lat": [40.5, 41.5], "lon": [-90.5, -89.5]},
    )
    p = tmp_path / "fullgrid.nc"
    ds.to_netcdf(p)
    return p


def test_empirical_fuzz_measures_alpha(tmp_path):
    f = EmpiricalFuzz.from_fullgrid(_synthetic_fullgrid(tmp_path))
    assert f.alpha == pytest.approx(0.2, rel=1e-6)
    # inherits the linear growth law with the measured alpha
    assert f.sigma_km(1000.0) - f.sigma_km(0.0) == pytest.approx(200.0, rel=1e-6)


def test_empirical_fuzz_respects_min_parcels(tmp_path):
    import xarray as xr
    p = _synthetic_fullgrid(tmp_path)
    with xr.open_dataset(p) as ds:
        ds = ds.load()
    ds["N"][:] = 1.0  # every box under-populated
    p2 = tmp_path / "sparse_fullgrid.nc"
    ds.to_netcdf(p2)
    with pytest.raises(ValueError, match="no populated"):
        EmpiricalFuzz.from_fullgrid(p2, min_parcels=5)
