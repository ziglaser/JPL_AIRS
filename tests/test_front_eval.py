"""Analytic-answer tests for ``front_finder.evaluate``.

- A hand-computable vertical-line-vs-vertical-line case checks
  ``contingency_by_day``/``scores_from_counts`` against exact numbers,
  including the cos(lat) area weighting.
- ``_metrics``' CSI (paper Eq. 4) is checked against the classic
  tp/(tp+fp+fn) identity in the degenerate case where both symmetric
  directions collapse to a single confusion matrix.
- A ``valid`` mask is checked to zero out a pixel exactly as if it were
  absent from the grid.
- ``block_bootstrap`` is checked against the degenerate case where every day
  has identical counts (every replicate must reproduce the point estimate
  exactly), and its result frames are checked to share the point estimate's
  row index (regression guard for a fixed CI-row misalignment bug).
- ``contingency_by_day`` is checked to pool multiple timesteps of one
  calendar day into a single row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from front_finder.evaluate import (
    _metrics,
    best_csi,
    block_bootstrap,
    contingency_by_day,
    scores_from_counts,
    threshold_sweep,
)

METRIC_COLS = ["csi", "pod", "far", "fb"]


def _line_dataarrays(lat, lon_col, n_lat=6, n_lon=6, time=None):
    """A single vertical line at ``lon_col`` for every lat row, one timestep."""
    if time is None:
        time = pd.date_range("2018-01-01", periods=1, freq="3h")
    arr = np.zeros((1, 1, n_lat, n_lon), dtype=bool)
    arr[0, 0, :, lon_col] = True
    return xr.DataArray(
        arr, dims=("time", "front", "lat", "lon"),
        coords={"time": time, "front": ["cold"], "lat": lat, "lon": np.arange(n_lon, dtype=float)},
    )


# --------------------------------------------------------------------------- #
# Hand-computable contingency / scores
# --------------------------------------------------------------------------- #

def test_offset_vertical_lines_exact_counts_and_scores():
    """truth = column 2, pred = column 3 (offset by one pixel), 6x6 grid.

    At dilation 0 the lines never touch: nothing is a hit in either
    direction. At dilation 1 the 8-connected dilation of a full-height
    column absorbs its immediate column neighbors, so every predicted and
    every truth pixel becomes a hit in its respective direction
    (POD=1, FAR=0, CSI=1). Because both lines have exactly one pixel per
    row, the (undilated) weighted pixel counts are equal, so FB=1 at every
    dilation -- FB depends only on total pred/truth weight, not on
    dilation.
    """
    lat = np.arange(6, dtype=float)  # cos(lat) varies row to row
    truth = _line_dataarrays(lat, lon_col=2)
    pred = _line_dataarrays(lat, lon_col=3)

    counts = contingency_by_day(pred, truth, dilations=(0, 1))
    w = np.cos(np.deg2rad(lat)).sum()  # total weight of one full-height line

    n0 = counts[counts["dilation"] == 0].iloc[0]
    assert n0["pred_hit"] == pytest.approx(0.0)
    assert n0["pred_miss"] == pytest.approx(w)
    assert n0["truth_hit"] == pytest.approx(0.0)
    assert n0["truth_miss"] == pytest.approx(w)

    n1 = counts[counts["dilation"] == 1].iloc[0]
    assert n1["pred_hit"] == pytest.approx(w)
    assert n1["pred_miss"] == pytest.approx(0.0)
    assert n1["truth_hit"] == pytest.approx(w)
    assert n1["truth_miss"] == pytest.approx(0.0)

    scores = scores_from_counts(counts)
    s0 = scores.loc[("cold", 0)]
    assert s0["pod"] == pytest.approx(0.0)
    assert s0["far"] == pytest.approx(1.0)
    assert s0["csi"] == pytest.approx(0.0)
    assert s0["fb"] == pytest.approx(1.0)

    s1 = scores.loc[("cold", 1)]
    assert s1["pod"] == pytest.approx(1.0)
    assert s1["far"] == pytest.approx(0.0)
    assert s1["csi"] == pytest.approx(1.0)
    assert s1["fb"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# _metrics identity (paper Eq. 4 vs classic CSI)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tp,fp,fn", [(4, 1, 2), (10, 3, 5), (1, 0, 0), (7, 7, 7)])
def test_metrics_csi_matches_classic_identity(tp, fp, fn):
    """When both symmetric directions coincide on one confusion matrix
    (pred_hit == truth_hit == tp, pred_miss == fp, truth_miss == fn),
    the paper's Eq. (4) construction ``1/(1/pod + 1/(1-far) - 1)`` reduces
    exactly to the classic CSI = tp / (tp + fp + fn).
    """
    csi, pod, far, fb = _metrics(
        ph=np.array([float(tp)]), pm=np.array([float(fp)]),
        th=np.array([float(tp)]), tm=np.array([float(fn)]),
    )
    classic = tp / (tp + fp + fn)
    assert csi[0] == pytest.approx(classic)
    assert pod[0] == pytest.approx(tp / (tp + fn))
    assert far[0] == pytest.approx(fp / (tp + fp))
    assert fb[0] == pytest.approx((tp + fp) / (tp + fn))


# --------------------------------------------------------------------------- #
# valid mask
# --------------------------------------------------------------------------- #

def test_valid_mask_excludes_pixel_as_if_absent():
    lat = np.arange(4, dtype=float)
    lon = np.arange(4, dtype=float)
    time = pd.date_range("2018-01-01", periods=1, freq="3h")
    front = ["cold"]

    truth = np.zeros((1, 1, 4, 4), dtype=bool)
    truth[0, 0, 1, 1] = True
    pred = np.zeros((1, 1, 4, 4), dtype=bool)
    pred[0, 0, 2, 2] = True  # disagrees with truth, far away

    coords = {"time": time, "front": front, "lat": lat, "lon": lon}
    truth_da = xr.DataArray(truth, dims=("time", "front", "lat", "lon"), coords=coords)
    pred_da = xr.DataArray(pred, dims=("time", "front", "lat", "lon"), coords=coords)

    valid = xr.DataArray(np.ones((1, 4, 4), dtype=bool), dims=("time", "lat", "lon"),
                          coords={"time": time, "lat": lat, "lon": lon})
    valid[0, 2, 2] = False  # invalidate the disagreeing pred pixel

    masked = contingency_by_day(pred_da, truth_da, valid=valid, dilations=(0,))

    pred_removed = pred_da.copy()
    pred_removed[0, 0, 2, 2] = False
    removed = contingency_by_day(pred_removed, truth_da, dilations=(0,))

    pd.testing.assert_frame_equal(
        masked.reset_index(drop=True), removed.reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# block_bootstrap
# --------------------------------------------------------------------------- #

def _constant_day_counts(n_days=10):
    """Multiple (front, dilation) groups, each with identical counts on
    every day -- any block resample must reproduce the same pooled ratios.
    """
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    per_day = {
        ("cold", 0): dict(pred_hit=5, pred_miss=2, truth_hit=4, truth_miss=1),
        ("warm", 1): dict(pred_hit=3, pred_miss=1, truth_hit=6, truth_miss=2),
    }
    rows = []
    for d in dates:
        for (front, dilation), vals in per_day.items():
            rows.append({"date": d, "front": front, "dilation": dilation, **vals})
    return pd.DataFrame(rows)


def test_block_bootstrap_constant_counts_give_zero_width_ci():
    counts = _constant_day_counts()
    result = block_bootstrap(counts, block_days=3, n_reps=10, seed=42)

    scores = result.scores[METRIC_COLS]
    lo = result.lo[METRIC_COLS]
    hi = result.hi[METRIC_COLS]

    np.testing.assert_allclose(lo.values, scores.values, atol=1e-9)
    np.testing.assert_allclose(hi.values, scores.values, atol=1e-9)


def test_block_bootstrap_result_frames_share_point_estimate_index():
    """Regression guard: CI frames must be reindexed to match the point
    estimate's row order (groupby in scores_from_counts sorts alphabetically
    by front, which need not match insertion order).
    """
    counts = _constant_day_counts()
    result = block_bootstrap(counts, block_days=3, n_reps=10, seed=42)

    assert result.scores.index.equals(result.lo.index)
    assert result.scores.index.equals(result.hi.index)
    # sanity: index is exactly the two (front, dilation) groups we built
    assert set(result.scores.index) == {("cold", 0), ("warm", 1)}


# --------------------------------------------------------------------------- #
# contingency_by_day: multi-timestep pooling within one calendar day
# --------------------------------------------------------------------------- #

def test_contingency_by_day_pools_multiple_timesteps_per_day():
    lat = np.arange(3, dtype=float)
    lon = np.arange(3, dtype=float)
    time = pd.to_datetime(["2018-01-01T00:00", "2018-01-01T12:00", "2018-01-02T00:00"])
    front = ["cold"]

    truth = np.zeros((3, 1, 3, 3), dtype=bool)
    truth[0, 0, 0, 0] = True
    truth[1, 0, 1, 1] = True
    truth[2, 0, 2, 2] = True
    pred = truth.copy()  # perfect agreement everywhere

    coords = {"time": time, "front": front, "lat": lat, "lon": lon}
    truth_da = xr.DataArray(truth, dims=("time", "front", "lat", "lon"), coords=coords)
    pred_da = xr.DataArray(pred, dims=("time", "front", "lat", "lon"), coords=coords)

    counts = contingency_by_day(pred_da, truth_da, dilations=(0,))

    # one row per (date, front, dilation), not per timestep
    assert len(counts) == 2
    assert set(counts["date"]) == {pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02")}

    day1 = counts[counts["date"] == pd.Timestamp("2018-01-01")].iloc[0]
    w0 = np.cos(np.deg2rad(lat[0]))
    w1 = np.cos(np.deg2rad(lat[1]))
    assert day1["pred_hit"] == pytest.approx(w0 + w1)
    assert day1["truth_hit"] == pytest.approx(w0 + w1)
    assert day1["pred_miss"] == pytest.approx(0.0)
    assert day1["truth_miss"] == pytest.approx(0.0)

    day2 = counts[counts["date"] == pd.Timestamp("2018-01-02")].iloc[0]
    w2 = np.cos(np.deg2rad(lat[2]))
    assert day2["pred_hit"] == pytest.approx(w2)
    assert day2["truth_hit"] == pytest.approx(w2)


# --------------------------------------------------------------------------- #
# threshold_sweep / best_csi
# --------------------------------------------------------------------------- #

def _prob_truth_three_rows():
    """3 lat rows x 1 lon col, 1 timestep, front=["cold"].

    prob = [0.2, 0.35, 0.6] by row; truth is True only at row 1. At
    threshold 0.3, pred = [False, True, True]: this is a hand-computable
    contingency at dilation 0 (see inline comment for the exact numbers).
    lat is held at 0 for every row so cos(lat) area weighting is exactly 1
    and doesn't complicate the by-hand arithmetic.
    """
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0])
    time = pd.date_range("2018-01-01", periods=1, freq="3h")
    coords = {"time": time, "front": ["cold"], "lat": lat, "lon": lon}

    prob_vals = np.array([0.2, 0.35, 0.6]).reshape(1, 1, 3, 1)
    truth_vals = np.zeros((1, 1, 3, 1), dtype=bool)
    truth_vals[0, 0, 1, 0] = True

    prob = xr.DataArray(prob_vals, dims=("time", "front", "lat", "lon"), coords=coords)
    truth = xr.DataArray(truth_vals, dims=("time", "front", "lat", "lon"), coords=coords)
    return prob, truth


def test_threshold_sweep_matches_hand_case_at_030():
    """At thr=0.3: pred = [False, True, True], truth = [False, True, False].

    dilation 0: pred_hit=1 (row1), pred_miss=1 (row2, false alarm),
    truth_hit=1 (row1 matched), truth_miss=0 -> POD=1, FAR=0.5, CSI=0.5,
    FB=(1+1)/(1+0)=2.
    """
    prob, truth = _prob_truth_three_rows()
    sweep = threshold_sweep(prob, truth, thresholds=[0.3], dilations=(0,))

    assert list(sweep.columns) == ["threshold", "front", "dilation", "csi",
                                   "pod", "far", "fb", "km"]
    row = sweep.iloc[0]
    assert row["threshold"] == pytest.approx(0.3)
    assert row["front"] == "cold"
    assert row["dilation"] == 0
    assert row["pod"] == pytest.approx(1.0)
    assert row["far"] == pytest.approx(0.5)
    assert row["csi"] == pytest.approx(0.5)
    assert row["fb"] == pytest.approx(2.0)


def test_threshold_sweep_pod_is_monotone_nonincreasing_in_threshold():
    """Raising the probability threshold shrinks the predicted-positive set,
    so truth_hit can only fall (or stay flat) and truth_miss can only rise
    -> POD is non-increasing in threshold."""
    prob, truth = _prob_truth_three_rows()
    thresholds = [0.1, 0.25, 0.4, 0.7]
    sweep = threshold_sweep(prob, truth, thresholds=thresholds, dilations=(0,))

    pod_by_thr = (sweep.sort_values("threshold")["pod"]).to_numpy()
    assert np.all(np.diff(pod_by_thr) <= 1e-12)


def test_best_csi_picks_max_csi_row_per_group():
    sweep = pd.DataFrame({
        "threshold": [0.1, 0.3, 0.5, 0.1, 0.3],
        "front": ["cold", "cold", "cold", "warm", "warm"],
        "dilation": [0, 0, 0, 0, 0],
        "csi": [0.2, 0.9, 0.4, 0.7, 0.7],
        "pod": [0.5, 0.8, 0.6, 0.7, 0.6],
        "far": [0.5, 0.1, 0.3, 0.2, 0.3],
        "fb": [1.0, 1.0, 1.0, 1.0, 1.0],
        "km": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    best = best_csi(sweep)

    cold_row = best[best["front"] == "cold"].iloc[0]
    assert cold_row["threshold"] == pytest.approx(0.3)
    assert cold_row["csi"] == pytest.approx(0.9)

    # tie broken by idxmax -> first occurrence (threshold 0.1 for "warm")
    warm_row = best[best["front"] == "warm"].iloc[0]
    assert warm_row["threshold"] == pytest.approx(0.1)
    assert warm_row["csi"] == pytest.approx(0.7)
