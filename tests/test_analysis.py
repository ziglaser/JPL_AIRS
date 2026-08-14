"""Tests for the shared analysis helpers on a synthetic table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from convection_skill import analysis, config


def _synthetic_table(n_per_hour=20_000, seed=0):
    """A table where QPE is driven by CAPE, with a modest hour-by-hour trend."""
    rng = np.random.default_rng(seed)
    frames = []
    for step, hour in enumerate(config.FORECAST_HOURS_UTC):
        cape = rng.gamma(2.0, 500.0, n_per_hour)  # J/kg-ish, many small values
        # QPE probability rises with CAPE and slightly with forecast step.
        p_event = (cape / cape.max()) ** 2 * (0.02 + 0.004 * step)
        qpe = np.where(rng.random(n_per_hour) < p_event,
                       rng.gamma(3.0, 3.0, n_per_hour), 0.0)
        frames.append(pd.DataFrame(dict(
            hour_utc=hour, step=step, mu_cape=cape,
            mu_cape_overpass=cape + rng.normal(0, 300, n_per_hour), qpe=qpe,
        )))
    return pd.concat(frames, ignore_index=True)


def test_gini_by_percentile_is_informative_and_stable():
    """The predictor is informative at every rarity and skill does not collapse.

    (The real-data property that Gini *rises* with rarity is a physics result
    validated in Phase 3, not something guaranteed for arbitrary data, so we do
    not assert strict monotonicity on synthetic input.)
    """
    table = _synthetic_table()
    res = analysis.gini_by_percentile(table, ["mu_cape"], percentiles=[95.0, 99.0, 99.9])
    ginis = res["mu_cape"].to_numpy()
    assert ginis.min() > 0.3               # informative at all thresholds
    assert np.all(np.diff(ginis) > -0.05)  # no skill collapse at rarer events


def test_gini_by_percentile_matches_valid_sample():
    """With match_valid, both predictors are scored on the same rows."""
    table = _synthetic_table()
    table.loc[:100, "mu_cape_overpass"] = np.nan
    res = analysis.gini_by_percentile(table, ["mu_cape", "mu_cape_overpass"],
                                      percentiles=[99.0])
    assert set(res.columns) == {"mu_cape", "mu_cape_overpass"}
    assert res.notna().all().all()


def test_hourly_gini_returns_all_hours():
    table = _synthetic_table()
    hourly = analysis.hourly_gini(table, "mu_cape", percentile=99.0)
    assert len(hourly) == len(config.FORECAST_HOURS_UTC)
    assert list(hourly["hour_utc"]) == list(config.FORECAST_HOURS_UTC)


def test_stratified_gini_splits_by_label():
    """stratified_gini returns one row per stratum, each with its own Gini."""
    table = _synthetic_table(n_per_hour=20_000)
    # Stratify by whether overpass-CAPE proxy is above/below its median.
    med = table["mu_cape_overpass"].median()
    strata = np.where(table["mu_cape_overpass"] > med, "high", "low")
    res = analysis.stratified_gini(table, "mu_cape", strata, percentile=99.0)
    assert set(res["stratum"]) == {"high", "low"}
    assert (res["n_events"] > 0).all()
    assert res["gini"].between(-1, 1).all()


def test_stratified_gini_handles_stratum_without_events():
    """A stratum with no events yields NaN Gini, not a crash."""
    n = 6000
    table = pd.DataFrame(dict(
        hour_utc=config.FORECAST_HOURS_UTC[0], mu_cape=np.arange(n, dtype=float),
        qpe=np.zeros(n),
    ))
    table.loc[table.index[-3:], "qpe"] = 100.0  # events only in the top rows
    strata = np.where(np.arange(n) < n // 2, "lower_half", "upper_half")
    res = analysis.stratified_gini(table, "mu_cape", strata, percentile=99.0)
    lower = res.set_index("stratum").loc["lower_half"]
    assert lower["n_events"] == 0 and np.isnan(lower["gini"])


def test_hourly_significance_detects_positive_trend():
    """The synthetic table has a built-in upward skill trend with forecast step."""
    table = _synthetic_table(n_per_hour=40_000)
    hourly, boot, trend = analysis.hourly_significance(table, "mu_cape", percentile=99.0)
    assert "se" in hourly.columns and boot.se > 0
    assert trend.slope > 0  # skill increases with step by construction
