"""Tests for convective_id on hand-built grids with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from convective_id.features import FEATURES, add_structure_features
from convective_id.methods import (
    cluster_classify, forest_classify, object_classify, threshold_classify,
)
from convective_id.validate import agreement_matrix, cape_auc, validate_with_cape


def _grid_df(n_days=2, n_slots=3, n_lat=6, n_lon=8, seed=0):
    """A complete (date, slot, lat, lon) grid of quiet drizzle."""
    rng = np.random.default_rng(seed)
    dates = np.array(["2019-06-05", "2019-06-06"], dtype="datetime64[ns]")[:n_days]
    idx = pd.MultiIndex.from_product(
        [dates, np.arange(1, n_slots + 1), 30.5 + np.arange(n_lat),
         -100.5 + np.arange(n_lon)], names=["date", "slot", "lat", "lon"])
    df = pd.DataFrame(index=idx).reset_index()
    n = len(df)
    df["qpe_wet"] = rng.uniform(0.5, 1.0, n)          # light rain everywhere
    df["wet_frac_true"] = rng.uniform(0.4, 0.9, n)    # keeps qpe above 0.1 mm/h
    df["qpe"] = df["qpe_wet"] * df["wet_frac_true"]
    df["qpe_max"] = df["qpe_wet"] * rng.uniform(1.2, 2.0, n)  # flat, no cores
    df["qpe_sk"] = rng.uniform(0.0, 0.5, n)
    df["convective_share"] = 0.0
    df["land_frac"] = 1.0
    df["mu_cape"] = rng.uniform(0, 50, n)             # stable environments
    return df.drop(columns="wet_frac_true")


def _plant_storm(df, lat, lon, cape=2000.0):
    """A convective core at (lat, lon): intense concentrated sub-pixel peak."""
    at = (df["lat"] == lat) & (df["lon"] == lon)
    df.loc[at, ["qpe_wet", "qpe_max", "qpe_sk"]] = [4.0, 40.0, 3.0]
    df.loc[at, "qpe"] = 2.0
    df.loc[at, "convective_share"] = 0.4
    df.loc[at, "mu_cape"] = cape
    return df


def test_features_and_incomplete_grid_guard():
    df = add_structure_features(_plant_storm(_grid_df(), 32.5, -97.5))
    assert df["is_precip"].all()          # everything drizzles in the fixture
    at = (df["lat"] == 32.5) & (df["lon"] == -97.5)
    nxt = (df["lat"] == 33.5) & (df["lon"] == -97.5)
    # the neighbor sees the core's peak; wet fraction is bounded
    assert (np.expm1(df.loc[nxt, "nbr_qpe_max_log"]) >= 39.9).all()
    assert df["wet_frac"].between(0, 1).all()
    assert (df.loc[at, "core_confident"]).all()
    with pytest.raises(ValueError, match="complete grid"):
        add_structure_features(df.iloc[:-3])


def test_threshold_flags_planted_core_only():
    df = add_structure_features(_plant_storm(_grid_df(), 32.5, -97.5))
    res = threshold_classify(df)
    at = ((df["lat"] == 32.5) & (df["lon"] == -97.5)).to_numpy()
    assert res.loc[at, "label"].all()
    assert not res.loc[~at, "label"].any()


def test_object_spreads_to_attached_cells_within_reach():
    """Core at (32.5,-97.5): its 8-neighbors are 'anvil' (attached, weakly
    rained, zero flag share) -> convective; a cell 4 steps away is not."""
    df = add_structure_features(_plant_storm(_grid_df(), 32.5, -97.5))
    res = object_classify(df, max_distance_cells=1)
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    neighbor = (np.abs(lat - 32.5) <= 1) & (np.abs(lon + 97.5) <= 1)
    far = (np.abs(lat - 32.5) + np.abs(lon + 97.5)) >= 4
    assert res.loc[neighbor, "label"].all()   # anvil inherits the storm label
    assert not res.loc[far, "label"].any()    # but it cannot run away


def test_forest_recovers_unflagged_core_structure():
    """Plant many flagged cores + one structurally identical cell with ZERO
    flag share (the anvil-obscured case): the forest must recover it."""
    df = _grid_df(n_days=2, n_slots=3, n_lat=10, n_lon=12, seed=1)
    rng = np.random.default_rng(2)
    cells = [(30.5 + i, -100.5 + j) for i in range(10) for j in range(12)]
    for lat, lon in rng.permutation(cells)[:25]:
        _plant_storm(df, lat, lon)
    # the hidden core: same structure, no flags
    hidden = (df["lat"] == 39.5) & (df["lon"] == -89.5)
    df.loc[hidden, ["qpe_wet", "qpe_max", "qpe_sk"]] = [4.0, 40.0, 3.0]
    df.loc[hidden, "qpe"] = 2.0
    df.loc[hidden, "convective_share"] = 0.0

    feat = add_structure_features(df)
    res = forest_classify(feat, n_estimators=50, min_samples_leaf=5)
    assert res.loc[hidden.to_numpy(), "label"].all()
    assert "feature_importances" in res.attrs


def test_cluster_isolates_intense_component():
    df = _grid_df(n_days=2, n_slots=3, n_lat=10, n_lon=12, seed=3)
    rng = np.random.default_rng(4)
    cells = [(30.5 + i, -100.5 + j) for i in range(10) for j in range(12)]
    planted = rng.permutation(cells)[:30]
    for lat, lon in planted:
        _plant_storm(df, lat, lon)
    feat = add_structure_features(df)
    res = cluster_classify(feat, n_components=2)
    is_planted = np.zeros(len(df), dtype=bool)
    for lat, lon in planted:
        is_planted |= ((df["lat"] == lat) & (df["lon"] == lon)).to_numpy()
    # the intense component captures the planted storms far above base rate
    assert res.loc[is_planted, "label"].mean() > 0.9
    assert res.loc[~is_planted, "label"].mean() < 0.35


def test_validation_prefers_cape_separating_labels():
    df = add_structure_features(_plant_storm(_grid_df(), 32.5, -97.5))
    results = {"threshold": threshold_classify(df)}
    summary = validate_with_cape(df, results)
    row = summary[summary["method"] == "threshold"].iloc[0]
    assert row["cape_median_conv"] > 1000 > row["cape_median_non"]
    assert row["cape_auc_conv_vs_non"] > 0.9


def test_cape_auc_and_agreement_basics():
    assert cape_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert np.isnan(cape_auc(np.array([np.nan]), np.array([1.0])))
    df = add_structure_features(_plant_storm(_grid_df(), 32.5, -97.5))
    res = threshold_classify(df)
    agree = agreement_matrix({"a": res, "b": res}, df["is_precip"].to_numpy())
    assert agree.loc["a", "b"] == 1.0


def test_features_are_flag_free():
    """The contract: no PrecipFlag-derived column may be a model feature."""
    forbidden = {"convective_share", "convective_frac", "stratiform_share",
                 "stratiform_frac", "no_precip_frac", "core_any",
                 "core_confident", "core_in_neighborhood"}
    assert not forbidden & set(FEATURES)
