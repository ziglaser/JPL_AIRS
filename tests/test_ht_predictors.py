"""Tests for hypothesis_tests.predictors -- synthetic fields with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import predictors as P


def _daily_field(n_days=730, nlat=4, nlon=5, seed=0, amp=2.0, noise=0.3):
    """Two years of daily data: per-cell mean + planted annual sinusoid + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values
    cell_mean = rng.uniform(5, 10, size=(nlat, nlon))
    seasonal = amp * np.sin(2 * np.pi * doy / 365.25)[:, None, None]
    vals = cell_mean[None] + seasonal + rng.normal(0, noise, (n_days, nlat, nlon))
    return xr.DataArray(
        vals, dims=("date", "lat", "lon"),
        coords={"date": dates, "lat": np.arange(nlat) + 30.5,
                "lon": np.arange(nlon) - 100.5},
        name="sm",
    )


def test_deseasonalize_removes_planted_sinusoid():
    da = _daily_field()
    anom = P.deseasonalize(da)
    # anomaly has ~no seasonal amplitude left: correlation with the sinusoid ~ 0
    doy = da["date"].dt.dayofyear.values
    sinus = np.sin(2 * np.pi * doy / 365.25)
    a = anom.values.reshape(anom.sizes["date"], -1).mean(axis=1)
    r = np.corrcoef(a, sinus)[0, 1]
    assert abs(r) < 0.05
    assert float(np.abs(anom.mean())) < 0.05      # centered
    assert float(anom.std()) < 0.5                # only the noise remains


def test_climatology_handles_nan_cells():
    da = _daily_field()
    vals = da.values.copy()
    vals[:, 0, 0] = np.nan                        # a fully missing cell
    vals[::7, 1, 1] = np.nan                      # a partially missing cell
    da2 = da.copy(data=vals)
    clim = P.harmonic_climatology(da2)
    assert np.isnan(clim.values[:, 0, 0]).all()
    assert np.isfinite(clim.values[:, 1, 1]).all()  # fit from remaining days


def test_zscore_by_cell_unit_variance():
    da = _daily_field()
    z = P.zscore_by_cell(P.deseasonalize(da))
    stds = z.std(dim="date").values
    assert np.allclose(stds, 1.0, atol=0.05)


def test_antecedent_mean_alignment():
    """antecedent_mean(x, 1, 3) at date t == mean(x[t-3:t-1]) exactly."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    da = xr.DataArray(np.arange(10.0)[:, None, None], dims=("date", "lat", "lon"),
                      coords={"date": dates, "lat": [30.5], "lon": [-100.5]})
    ante = P.antecedent_mean(da, 1, 3)
    # at index 5 (value 5): mean of values at indices 2,3,4 = 3
    assert float(ante.isel(date=5)) == pytest.approx(3.0)
    assert np.isnan(ante.isel(date=1))            # not enough history


def test_antecedent_requires_daily_axis():
    dates = pd.date_range("2020-01-01", periods=5, freq="2D")
    da = xr.DataArray(np.zeros((5, 1, 1)), dims=("date", "lat", "lon"),
                      coords={"date": dates, "lat": [30.5], "lon": [-100.5]})
    with pytest.raises(ValueError, match="daily"):
        P.antecedent_mean(da, 1, 2)


def test_neighborhood_mean_flat_field_is_identity():
    """A field constant in space: neighborhood mean == the constant."""
    da = _daily_field(n_days=3, noise=0.0, amp=0.0)
    da = da.copy(data=np.full_like(da.values, 7.0))  # truly flat in space
    nb = P.neighborhood_mean(da, halfwidth=1, min_valid=3)
    assert np.allclose(nb.values, 7.0, atol=1e-9)


def test_local_nonlocal_split_recovers_planted_structure():
    """Plant a domain-wide day anomaly + a single-cell local anomaly; the split
    must put each in the right component."""
    da = _daily_field(n_days=40, nlat=7, nlon=7, noise=0.0, amp=0.0)
    vals = da.values.copy()
    vals[10] += 3.0                                # domain-wide anomaly on day 10
    vals[20, 3, 3] += 3.0                          # single-cell anomaly on day 20
    anom = da.copy(data=vals - vals.mean(axis=0))  # crude anomaly, fine for test
    local, nonlocal_ = P.local_nonlocal_decomposition(anom, halfwidth=1)
    # day 10 domain event: nonlocal sees it, local ~ 0 at the center cell
    assert float(nonlocal_.values[10, 3, 3]) > 1.0
    assert abs(float(local.values[10, 3, 3])) < 0.5
    # day 20 single-cell event: local sees it, nonlocal barely moves
    assert float(local.values[20, 3, 3]) > 2.0
    assert float(nonlocal_.values[20, 3, 3]) < 1.0
