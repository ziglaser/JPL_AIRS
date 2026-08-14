"""CODSUS front flags (convection_skill.fronts) + Mark's RF adapters (models).

Synthetic front files exercise the two alignment steps end to end: the 2x2
max-pool onto half-degree centers and the most-recent-bulletin slot mapping
(slots 1-3 -> same-day 21 UTC, slots 4-6 -> next-day 00 UTC).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import config
from convection_skill import fronts as fr
from convection_skill import models


# --------------------------------------------------------------------------- #
# Slot -> bulletin mapping
# --------------------------------------------------------------------------- #
def test_analysis_offsets_window_slots():
    offsets = fr._analysis_offsets(config.FORECAST_SLOTS)
    assert offsets == {1: 21, 2: 21, 3: 21, 4: 24, 5: 24, 6: 24}


def test_analysis_offsets_skips_unmapped_slots():
    # the overpass slot has no forecast-hour mapping -> omitted (stays NaN)
    assert 0 not in fr._analysis_offsets((0, 1))


# --------------------------------------------------------------------------- #
# 2x2 max-pool regrid
# --------------------------------------------------------------------------- #
def test_pool_takes_max_over_overlapping_cells():
    front_lats = np.array([30.0, 31.0, 32.0])
    front_lons = np.array([-100.0, -99.0, -98.0])
    vals = np.zeros((3, 3), dtype=np.float32)
    vals[1, 1] = 1.0  # front at (31, -99) only
    da = xr.DataArray(vals, coords={"lat": front_lats, "lon": front_lons})

    lats, lons = np.array([30.5, 31.5]), np.array([-99.5, -98.5])
    pooled = fr._pool_to_half_degree(da, lats, lons)
    # every half-degree cell overlapping (31, -99) must be flagged
    assert pooled.values.tolist() == [[1.0, 1.0], [1.0, 1.0]]


# --------------------------------------------------------------------------- #
# year_front_flags on a synthetic bulletin file
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_fronts_dir(tmp_path, monkeypatch):
    """One year of 3-hourly bulletins over a tiny grid, 1wide + 3wide files.

    The cold-front channel is ON everywhere at 21 UTC on day 1 and at 00 UTC
    on day 2, OFF at every other bulletin -- so slots 1-3 of day 1 and slots
    4-6 of day 1 (next-day 00 UTC) should both light up, while every slot of
    day 2 (21 UTC day 2 / 00 UTC day 3) stays dark.
    """
    times = pd.date_range("2016-06-01", "2016-06-03 21:00", freq="3h")
    lats, lons = np.array([30.0, 31.0, 32.0]), np.array([-100.0, -99.0, -98.0])
    types = np.array(list(fr.FRONT_TYPES) + ["none"], dtype="<U10")
    vals = np.zeros((len(times), len(types), len(lats), len(lons)), np.float32)
    on = [np.datetime64("2016-06-01T21:00"), np.datetime64("2016-06-02T00:00")]
    for t in on:
        vals[list(times).index(t), 0] = 1.0  # cold channel everywhere

    ds = xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), vals)},
        coords={"time": times, "lat": lats, "lon": lons,
                "front_type": ("front", types)})
    for width in fr.FRONT_WIDTHS:
        ds.to_netcdf(tmp_path / fr.FRONT_FILE_TEMPLATE.format(
            width=width, year=2016))
    monkeypatch.setattr(fr, "FRONTS_DIR", tmp_path)
    monkeypatch.setattr(fr, "REGEN_FRONTS_DIR", tmp_path)  # no real fallback
    return tmp_path


def test_year_front_flags_time_mapping(synthetic_fronts_dir):
    dates = np.array(["2016-06-01", "2016-06-02"], dtype="datetime64[ns]")
    lats, lons = np.array([30.5, 31.5]), np.array([-99.5, -98.5])
    out = fr.year_front_flags(2016, dates, config.FORECAST_SLOTS, lats, lons)

    day1 = out["front_cold_1w"].sel(date=dates[0])
    day2 = out["front_cold_1w"].sel(date=dates[1])
    assert (day1.values == 1.0).all()          # 21 UTC AND next-day 00 UTC on
    assert (day2.values == 0.0).all()          # both its bulletins off
    # only the cold channel fired; 'any' mirrors it, others stay off
    assert (out["front_any_3w"].sel(date=dates[0]).values == 1.0).all()
    assert (out["front_warm_1w"].values == 0.0).all()


def test_year_front_flags_missing_year_is_nan(synthetic_fronts_dir):
    dates = np.array(["2019-06-01"], dtype="datetime64[ns]")
    out = fr.year_front_flags(2019, dates, config.FORECAST_SLOTS,
                              np.array([30.5]), np.array([-99.5]))
    for name in fr.front_columns():
        assert np.isnan(out[name].values).all()


def test_missing_bulletin_is_nan(synthetic_fronts_dir):
    # Dec 31-style edge: the last date's 00 UTC bulletin is outside the file
    dates = np.array(["2016-06-03"], dtype="datetime64[ns]")
    lats, lons = np.array([30.5]), np.array([-99.5])
    out = fr.year_front_flags(2016, dates, config.FORECAST_SLOTS, lats, lons)
    da = out["front_cold_1w"].sel(date=dates[0])
    assert np.isfinite(da.sel(slot=[1, 2, 3]).values).all()   # 21 UTC exists
    assert np.isnan(da.sel(slot=[4, 5, 6]).values).all()      # 00 UTC missing


# --------------------------------------------------------------------------- #
# Mark's RF adapters
# --------------------------------------------------------------------------- #
@pytest.fixture()
def cell_days():
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({
        "day": pd.date_range("2016-06-01", periods=n),
        "lat": 30.5, "lon": -99.5,
        "sm_anom": rng.normal(size=n),
        "front_any_1w": rng.integers(0, 2, n).astype(float),
    })
    for s in (1, 2, 3):
        df[f"mu_cape_h{s}"] = rng.random(n) * 1000
        df[f"qpe_h{s}"] = rng.random(n)
    df.loc[:9, "front_any_1w"] = np.nan  # a no-front-data stretch
    return df


def test_samples_from_cell_days_round_trip(cell_days):
    ds = models.samples_from_cell_days(cell_days)
    assert ds["mu_cape"].dims == ("time", "sample")
    assert list(ds["time"].values) == [1, 2, 3]
    np.testing.assert_array_equal(
        ds["mu_cape"].sel(time=2).values, cell_days["mu_cape_h2"].to_numpy())
    np.testing.assert_array_equal(
        ds["sm_anom"].values, cell_days["sm_anom"].to_numpy())


def test_finite_samples_drops_nan_front_rows(cell_days):
    ds = models.samples_from_cell_days(cell_days)
    sub = models.finite_samples(ds, ["mu_cape", "front_any_1w", "qpe"])
    assert sub.sizes["sample"] == len(cell_days) - 10


def test_compare_with_fronts_runs_and_names_features(cell_days):
    out = models.compare_with_fronts(
        cell_days, base_features=("mu_cape", "sm_anom"),
        front_features=("front_any_1w",), target="qpe",
        rfr_kwargs={"max_depth": 3, "n_estimators": 5})
    assert out["n_samples"] == len(cell_days) - 10
    base_imp, front_imp = out["base"]["importances"], out["fronts"]["importances"]
    assert set(base_imp.index) == {"mu_cape_h1", "mu_cape_h2", "mu_cape_h3",
                                   "sm_anom"}
    assert set(front_imp.index) == set(base_imp.index) | {"front_any_1w"}
    assert np.isclose(base_imp.sum(), 1.0) and np.isclose(front_imp.sum(), 1.0)
