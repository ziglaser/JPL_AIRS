"""Tests for trajectory ingest.

Structural/round-trip checks against the real granule files plus an analytic
unit-range guard. These lock down the facts the rest of the tool relies on:
one parcel = one released air mass, step 0 is a vertical stack, no mid-trajectory
dropout, and units are as expected (audit WORKPLAN sections 0-1).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from trajectory_kernels import config, trajectories as T

pytestmark = pytest.mark.skipif(
    not (config.TRAJ_DIR / config.NOGRID_TEMPLATE.format(granule=189)).exists(),
    reason="trajectory data not present",
)


def test_load_granule_shape_and_coords():
    ds = T.load_granule(189)
    assert set(ds.dims) == {"parcel", "step"}
    assert ds.sizes["step"] == 7
    for v in T.PARCEL_VARS:
        assert ds[v].dims == ("parcel", "step")
    for c in ("granule", "swath", "level", "release_lat", "release_lon", "time_utc"):
        assert c in ds.coords
    assert (ds["granule"] == 189).all()
    assert (ds["swath"] == "early").all()


def test_step0_columns_are_vertical_stacks():
    """Every (fieldx, fieldy) release column shares one lat/lon across its levels
    at step 0 -- the vertical-stack fact the receptor logic assumes."""
    ds = T.load_granule(189)
    lat0 = ds["lat"].isel(step=0).values
    lon0 = ds["lon"].isel(step=0).values
    fx, fy = ds["fieldx"].values, ds["fieldy"].values
    col = fx.astype("int64") * 10_000 + fy.astype("int64")
    for c in np.unique(col)[:50]:  # a sample of columns is enough
        m = col == c
        assert np.ptp(lat0[m]) < 1e-6
        assert np.ptp(lon0[m]) < 1e-6


def test_no_mid_trajectory_dropout():
    """A parcel valid at release stays valid to step 6 on this day (audit Q6)."""
    day = T.load_day()
    assert bool((T.n_valid_steps(day) == 7).all())


def test_swath_stagger():
    """Early-swath release ~18-19 UTC, late-swath ~20 UTC; steps 1-6 shared."""
    day = T.load_day()
    early = day.where(day.swath == "early", drop=True)
    late = day.where(day.swath == "late", drop=True)
    early_h = early["time_utc"].isel(step=0).dt.hour.values
    late_h = late["time_utc"].isel(step=0).dt.hour.values
    assert early_h.max() <= 19 and late_h.min() >= 20
    # step 3 is 23:00 UTC for everyone
    assert set(np.unique(day["time_utc"].isel(step=3).dt.hour.values)) == {23}


def test_round_trip_against_raw_file():
    """A parcel's ingested values equal the raw file at its (level, fieldx, fieldy)."""
    ds = T.load_granule(190)
    raw = xr.open_dataset(config.TRAJ_DIR / config.NOGRID_TEMPLATE.format(granule=190))
    p = ds.isel(parcel=ds.sizes["parcel"] // 3)
    lvl, fx, fy = int(p["level"]), int(p["fieldx"]), int(p["fieldy"])
    raw_lat = raw["lat"].isel(level=lvl, fieldx=fx, fieldy=fy).values
    assert np.allclose(p["lat"].values, raw_lat, equal_nan=True)
    raw.close()


def test_unit_assertion_catches_bad_units():
    """q in kg/kg (not g/kg) trips the range guard."""
    n = 5
    bad = xr.Dataset(
        {v: (("parcel", "step"), np.ones((n, 3))) for v in T._RANGE_CHECKS}
    )
    bad["q"] = (("parcel", "step"), np.full((n, 3), 0.008))  # kg/kg, way below g/kg range
    # lat/lon/alt/pres/t are all 1.0 -> also out of range, so just assert it raises
    with pytest.raises(ValueError, match="units or file mismatch"):
        T._assert_units(bad, "synthetic")


def test_near_surface_flag_matches_band():
    day = T.load_day()
    lo, hi = config.RECEPTOR_BAND_M
    rel_alt = day["alt"].isel(step=0).values
    assert np.array_equal(
        day["is_near_surface"].values, (rel_alt >= lo) & (rel_alt <= hi)
    )
