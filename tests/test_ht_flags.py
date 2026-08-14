"""Tests for the precip-mode fractions and the convective-event toggle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import config
from convection_skill.dataset import (
    PRECIP_TARGETS, _add_precip_mode_fractions, build_flags,
)


def _thresholds(df) -> dict[str, float]:
    """Base-sample-style absolute thresholds computed from the toy table
    itself (the unified builder derives these once from the base superset)."""
    qpe = df["qpe"].to_numpy()
    wet = qpe > config.ANY_PRECIP_MM
    return {
        "heavy": float(np.nanpercentile(qpe, config.HEAVY_PERCENTILE)),
        "heavy_sens": float(np.nanpercentile(qpe, config.HEAVY_PERCENTILE_SENS)),
        "any": config.ANY_PRECIP_MM,
        "max_extreme": float(np.nanpercentile(df["qpe_max"].to_numpy(),
                                              config.MAX_PERCENTILE)),
        "sk_high": float(np.nanpercentile(df["qpe_sk"].to_numpy()[wet],
                                          config.SK_PERCENTILE)),
        "high_mml_cape": float(np.nanpercentile(df["mml_cape"].to_numpy(), 90.0)),
        "low_mml_lcl": float(np.nanpercentile(df["mml_lcl"].to_numpy(), 10.0)),
        "inhibited": float(np.nanpercentile(df["mu_cin"].to_numpy(), 10.0)),
    }


def _uniform_with_flags():
    """A tiny make_uniform-shaped dataset with hand-set flag counts."""
    shape = (1, 1, 1, 4)  # date, slot, lat, lon: four cells
    counts = {
        # cell 0: 60 conv + 20 strat + 20 dry
        # cell 1: all dry; cell 2: 10 strat + 10 HAIL; cell 3: 10 strat + 10 snow
        "Convection": [60.0, 0.0, 0.0, 0.0],
        "TropicalConvectiveRain": [0.0, 0.0, 0.0, 0.0],
        "WarmStratiformRain": [20.0, 0.0, 10.0, 10.0],
        "CoolStratiformRain": [0.0, 0.0, 0.0, 0.0],
        "TropicalStratiformRain": [0.0, 0.0, 0.0, 0.0],
        "Snow": [0.0, 0.0, 0.0, 10.0],
        "Hail": [0.0, 0.0, 10.0, 0.0],
        "NoPrecipitation": [20.0, 100.0, 80.0, 80.0],
        "CountsNaN": [0.0, 0.0, 0.0, 0.0],
        "NoCoverage": [0.0, 0.0, 0.0, 0.0],
        "TotalCountsAll": [100.0, 100.0, 100.0, 100.0],
    }
    dims = ("date", "slot", "lat", "lon")
    uniform = xr.Dataset(
        {f"MRMS_PrecipFlag_cnt_{cat}": (dims, np.array(v).reshape(shape))
         for cat, v in counts.items()},
        coords={"date": [np.datetime64("2019-06-05")], "slot": [1],
                "lat": [40.5], "lon": [-95.5, -94.5, -93.5, -92.5]},
    )
    return uniform


def test_mode_fractions_accounting():
    uniform = _uniform_with_flags()
    ds = xr.Dataset(coords=uniform.coords)
    _add_precip_mode_fractions(uniform, ds)

    conv = ds["convective_frac"].values.ravel()
    strat = ds["stratiform_frac"].values.ravel()
    conv_share = ds["convective_share"].values.ravel()
    strat_share = ds["stratiform_share"].values.ravel()
    # cell 0: coverage 0.6 conv / 0.2 strat; shares 60/80 and 20/80
    assert conv[0] == pytest.approx(0.6)
    assert strat[0] == pytest.approx(0.2)
    assert conv_share[0] == pytest.approx(0.75)
    assert strat_share[0] == pytest.approx(0.25)
    # cell 1: no precipitating pixels -> shares NaN, coverages 0
    assert conv[1] == pytest.approx(0.0)
    assert np.isnan(conv_share[1])
    # cell 2: 10 strat + 10 hail -> HAIL COUNTS AS CONVECTIVE (2026-07-23):
    # convective coverage 0.1, conv/strat shares split the precipitating pixels
    assert conv[2] == pytest.approx(0.1)
    assert conv_share[2] == pytest.approx(0.5)
    assert strat_share[2] == pytest.approx(0.5)
    # cell 3: 10 strat + 10 snow -> snow stays a NON-convective residual that
    # dilutes the stratiform share to 0.5
    assert strat_share[3] == pytest.approx(0.5)
    assert conv_share[3] == pytest.approx(0.0)


def _toy_table(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    qpe = rng.exponential(0.2, n)
    return pd.DataFrame({
        "qpe": qpe,
        "qpe_max": qpe * rng.uniform(1, 20, n),
        "qpe_sk": rng.uniform(0, 3, n),
        "mml_cape": rng.exponential(500, n),
        "mml_lcl": rng.uniform(200, 3000, n),
        "mu_cin": -rng.exponential(30, n),
        "convective_share": np.where(rng.random(n) < 0.1, np.nan, rng.random(n)),
    })


def test_toggle_filters_precip_targets_only():
    df = _toy_table()
    plain = build_flags(df, _thresholds(df))
    filtered = build_flags(df, _thresholds(df), convective_min=0.5)
    for target in PRECIP_TARGETS:
        # filtered events are a subset of unfiltered ones
        assert not np.any(filtered[target] & ~plain[target])
        assert filtered[target].sum() < plain[target].sum()
    for target in ("high_mml_cape", "low_mml_lcl", "inhibited"):
        assert np.array_equal(filtered[target], plain[target])


def test_toggle_nan_share_is_not_convective():
    df = _toy_table()
    df["convective_share"] = np.nan
    filtered = build_flags(df, _thresholds(df), convective_min=0.0)
    for target in PRECIP_TARGETS:
        assert filtered[target].sum() == 0


def test_toggle_threshold_monotone():
    df = _toy_table()
    n_loose = build_flags(df, _thresholds(df), convective_min=0.2)["heavy"].sum()
    n_strict = build_flags(df, _thresholds(df), convective_min=0.8)["heavy"].sum()
    assert n_strict <= n_loose


def test_toggle_alternative_column():
    df = _toy_table()
    df["convective_frac"] = df["convective_share"] * 0.3
    by_share = build_flags(df, _thresholds(df), convective_min=0.2)
    by_frac = build_flags(df, _thresholds(df), convective_min=0.2, convective_col="convective_frac")
    assert by_frac["heavy"].sum() <= by_share["heavy"].sum()
