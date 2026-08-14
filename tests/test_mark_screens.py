"""Parity: the dataset screen columns must reproduce Mark's make_master_mask.

Mark's code (mark_screens.py, verbatim from his 2026-07-21 email) is the gold
standard; the pipeline applies the same constraints as cached columns so config
toggles can switch each one independently. These tests prove the two paths are
bit-identical, including the NaN semantics (NaN fails every constraint).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import mark_screens as mk


@pytest.fixture()
def raw():
    """A tiny raw-style cube with NaNs sprinkled into every screened variable."""
    rng = np.random.default_rng(7)
    n_date, n_time, n_lat, n_lon = 4, 7, 3, 3
    dims = ("date", "time", "lat", "lon")
    coords = {
        "date": pd.date_range("2019-06-01", periods=n_date),
        "lat": [32.5, 33.5, 34.5],
        "lon": [-100.5, -99.5, -98.5],
    }

    def cube(scale, nan_frac=0.1, offset=0.0):
        vals = rng.random((n_date, n_time, n_lat, n_lon)) * scale + offset
        vals[rng.random(vals.shape) < nan_frac] = np.nan
        return xr.DataArray(vals.astype("float32"), dims=dims, coords=coords)

    ds = xr.Dataset(
        {
            "FCST_MU_CAPE": cube(3000),
            "FCST_MU_CIN": cube(-300),
            "FCST_alt": cube(2500),
            # nhours axis on purpose: our files put MRMS there
            "MRMS_GaugeCorrQPE01H_av": cube(5).rename("av").swap_dims(
                {"time": "nhours"}),
            "MRMS_GaugeCorrQPE01H_cnt": (cube(81, nan_frac=0.0)).swap_dims(
                {"time": "nhours"}),
        }
    )
    # make some cells genuinely dry at hours 0-1 so the dry-start screen passes
    av = ds["MRMS_GaugeCorrQPE01H_av"].values
    av[:, :2][rng.random(av[:, :2].shape) < 0.5] = 0.0
    return ds


@pytest.fixture()
def land_mask(raw):
    vals = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=float)
    return xr.DataArray(vals, dims=("lat", "lon"),
                        coords={"lat": raw["lat"], "lon": raw["lon"]})


def test_components_reproduce_master_mask(raw, land_mask):
    ds = mk.add_gridav(raw)
    expected = mk.make_master_mask(ds, land_mask)  # Mark, verbatim

    comp = mk.master_mask_components(raw)
    land = land_mask.reindex_like(raw, method="nearest") > 0
    ours = (
        land
        & (comp["alt_max"] < mk.Z_THRESH_M)
        & (comp["dry_start_qpe"] <= mk.QPE_DRY_THRESH)
        & comp["valid7"]
    )
    xr.testing.assert_equal(expected, ours)


def test_nan_fails_every_screen(raw):
    comp = mk.master_mask_components(raw)
    alt_has_nan = raw["FCST_alt"].isnull().any(dim="time")
    # wherever any hour's altitude is NaN, the screen must fail regardless of cut
    fails = ~(comp["alt_max"] < np.inf)
    assert bool((fails == alt_has_nan).all())

    qpe_nan = raw[mk.GRIDAV_VAR].isel(time=[0, 1]).isnull().any(dim="time")
    dry_fails = ~(comp["dry_start_qpe"] <= np.inf)
    assert bool((dry_fails == qpe_nan).all())


def test_gridav_reconstruction_on_nhours_axis(raw):
    ds = mk.add_gridav(raw)
    assert ds[mk.GRIDAV_VAR].dims == ("date", "time", "lat", "lon")
    manual = (raw["MRMS_GaugeCorrQPE01H_av"].values
              * raw["MRMS_GaugeCorrQPE01H_cnt"].values / 81.0)
    np.testing.assert_allclose(ds[mk.GRIDAV_VAR].values, manual, rtol=1e-6)
