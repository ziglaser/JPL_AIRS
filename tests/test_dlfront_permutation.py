"""``dl_front.permutation.single_pass`` on a fake model (C6, 2026-08-18).

Step 2 of the AIRS-channel work: before spending GPU on the stage-A channel
ladder, size how much skill lives in SLP/U10M/V10M (which AIRS never sees --
SLP is copied clean from MERRA-2, the winds are the WRF-27km met driving
HYSPLIT) by shuffling one input channel at a time through an existing
checkpoint.

The module confines every model interaction to ``predict_classes`` exactly
so this file can drive it with a numpy stub and no TensorFlow (the same
trick ``front_finder.permutation`` plays with ``predict.predict_batch``).
The stub below reads ONE channel and ignores the rest, which makes the
expected answer analytic: permuting the channel it reads must cost CSI,
permuting a channel it ignores must cost exactly nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_front import config, permutation

N_CLASSES = 6
NONE = N_CLASSES - 1
CHANNELS = ["T2M", "QV2M"]
#: Rows the synthetic fronts sit on, 6 apart so a shuffle can never be
#: rescued by the widest neighborhood dilation (EVAL_DILATIONS tops out at
#: 2 cells), and all inside the box the synthetic scoring mask covers.
ROWS = (24, 30, 36, 42)
COLS = slice(75, 85)


@pytest.fixture
def scoring_mask():
    """A box standing in for ``dataset.analysis_domain()``.

    Synthetic so the test needs no land-mask file, but the same shape and
    role: only cells inside it are scored, which is what makes a delta here
    the same currency as an evaluate_test CSI delta.
    """
    mask = np.zeros(config.GRID_SHAPE, bool)
    mask[22:44, 64:108] = True
    return mask


def _sample(k: int) -> np.ndarray:
    """Truth grid for sample ``k``: a cold-front line at ``ROWS[k]``."""
    y = np.full(config.GRID_SHAPE, NONE, np.uint8)
    y[ROWS[k], COLS] = 0                              # class 0 = cold
    return y


@pytest.fixture
def synthetic_case():
    """(x, y, times) where channel 0 encodes the truth and channel 1 is noise.

    x[:, :, :, 0] is the constant field ``k`` for sample k, so a model that
    reads it can reproduce the truth exactly; channel 1 is random and
    carries nothing.
    """
    n = len(ROWS)
    rng = np.random.default_rng(7)
    x = np.zeros((n, *config.GRID_SHAPE, 2), np.float32)
    for k in range(n):
        x[k, ..., 0] = float(k)
        x[k, ..., 1] = rng.normal(size=config.GRID_SHAPE)
    y = np.stack([_sample(k) for k in range(n)])
    times = pd.date_range("2016-06-01 21:00", periods=n, freq="1D")
    return x, y, times


class ChannelZeroModel:
    """Duck-typed model: perfect from channel 0 alone, blind to channel 1.

    Reads one pixel of channel 0 (the field is constant per sample), so
    permuting channel 0 hands it the wrong day's answer while permuting
    channel 1 changes nothing at all.
    """

    def __init__(self):
        self.n_predicted = 0

    def predict(self, x, batch_size=128, verbose=0):
        self.n_predicted += len(x)
        probs = np.zeros((len(x), *config.GRID_SHAPE, N_CLASSES), np.float32)
        probs[..., NONE] = 1.0
        for i in range(len(x)):
            k = int(round(float(x[i, 0, 0, 0])))
            probs[i, ROWS[k], COLS, NONE] = 0.0
            probs[i, ROWS[k], COLS, 0] = 1.0
        return probs


def _cold(df, channel, dilation=0, stat="raw"):
    """The cold-front rows of one channel at one dilation."""
    return df[(df.channel == channel) & (df.front == "cold")
              & (df.dilation == dilation) & (df.stat == stat)]


def test_single_pass_frame_shape_and_baseline_row(synthetic_case,
                                                  scoring_mask):
    """Frozen CSV interface + exactly one baseline row per (front, dilation).

    The baseline lives in the SAME table as the permuted rows (channel
    ``"<baseline>"``, zero deltas) so a reader never has to fetch the eval
    leg's CSV to interpret a delta; duplicating it per channel would make
    ``df[df.channel == BASELINE]`` a trap.
    """
    x, y, times = synthetic_case
    df = permutation.single_pass(ChannelZeroModel(), x, y, times, N_CLASSES,
                                 CHANNELS, np.random.default_rng(0),
                                 mask=scoring_mask)
    assert list(df.columns) == permutation.CSV_COLUMNS
    for col in ("channel", "repeat", "front", "dilation", "km", "csi", "pod",
                "far", "fb", "csi_delta", "pod_delta"):
        assert col in df.columns                      # the contract's columns

    base = df[df.channel == permutation.BASELINE]
    assert base.groupby(["front", "dilation"]).size().eq(1).all()
    assert (base.csi_delta == 0.0).all() and (base.pod_delta == 0.0).all()
    assert (base.repeat == permutation.NO_REPEAT).all()
    assert set(df.channel) == {permutation.BASELINE, *CHANNELS}
    # one row per (channel, repeat, front, dilation), repeats=1
    n_cells = len(base)
    assert len(df) == n_cells * (1 + len(CHANNELS))
    np.testing.assert_allclose(_cold(df, permutation.BASELINE).csi.iloc[0],
                               1.0)                   # the stub is perfect


def test_deltas_are_baseline_minus_permuted(synthetic_case, scoring_mask):
    """Sign convention: POSITIVE delta = the model got worse without the
    channel = the channel matters.  Asserted arithmetically on every row,
    because a sign flip here would invert the ablation decision the whole
    step-2 run exists to inform."""
    x, y, times = synthetic_case
    df = permutation.single_pass(ChannelZeroModel(), x, y, times, N_CLASSES,
                                 CHANNELS, np.random.default_rng(0),
                                 mask=scoring_mask)
    base = df[df.channel == permutation.BASELINE].set_index(
        ["front", "dilation"])
    perm = df[df.channel != permutation.BASELINE]
    for _, row in perm.iterrows():
        ref = base.loc[(row.front, row.dilation)]
        np.testing.assert_allclose(row.csi_delta, ref.csi - row.csi,
                                   equal_nan=True)
        np.testing.assert_allclose(row.pod_delta, ref.pod - row.pod,
                                   equal_nan=True)


def test_permuting_an_ignored_channel_costs_nothing(synthetic_case,
                                                    scoring_mask):
    """The calibration of the whole method: a channel the model does not
    read must produce EXACTLY zero delta, while the channel it depends on
    produces a large positive one.

    A nonzero delta on the ignored channel would mean the shuffle is
    perturbing something other than that channel (a broadcasting slip in
    the gather, or the spatial field being shuffled apart from its step).
    """
    x, y, times = synthetic_case
    df = permutation.single_pass(ChannelZeroModel(), x, y, times, N_CLASSES,
                                 CHANNELS, np.random.default_rng(0),
                                 mask=scoring_mask)
    ignored = _cold(df, "QV2M")
    np.testing.assert_array_equal(ignored.csi_delta.values, 0.0)
    np.testing.assert_array_equal(ignored.pod_delta.values, 0.0)
    for dilation in (0, 1, 2):
        used = _cold(df, "T2M", dilation=dilation)
        assert used.csi_delta.iloc[0] > 0.0
        assert used.pod_delta.iloc[0] > 0.0
    # x itself must come back unmodified: single_pass permutes a COPY, and
    # the caller reuses x for the next channel and for the run json's
    # step counts
    assert (x[:, 0, 0, 0] == np.arange(len(ROWS))).all()


def test_repeats_add_mean_std_aggregate(synthetic_case, scoring_mask):
    """With repeats > 1 the per-channel mean/std rows are emitted in the
    same frame, and the raw rows are still recoverable exactly.

    Spread across shuffles is the only handle on how much of a delta is the
    luck of one permutation, so the operator gets it without a second file
    or a groupby of their own.
    """
    x, y, times = synthetic_case
    df = permutation.single_pass(ChannelZeroModel(), x, y, times, N_CLASSES,
                                 CHANNELS, np.random.default_rng(1),
                                 mask=scoring_mask, repeats=3)
    raw = df[df.stat == "raw"]
    assert set(raw[raw.channel != permutation.BASELINE].repeat) == {0, 1, 2}
    assert set(df.stat) == {"raw", "mean", "std"}
    agg = df[df.stat == "mean"]
    assert permutation.BASELINE not in set(agg.channel)   # not a distribution
    assert (agg.repeat == permutation.NO_REPEAT).all()
    cold_mean = _cold(df, "T2M", stat="mean").csi_delta.iloc[0]
    np.testing.assert_allclose(cold_mean,
                               _cold(raw, "T2M").csi_delta.mean())
    assert _cold(df, "QV2M", stat="std").csi_delta.iloc[0] == 0.0


def test_single_pass_defaults_to_the_eval_scoring_mask(synthetic_case,
                                                       monkeypatch,
                                                       scoring_mask):
    """mask=None must fall back to the SAME mask evaluate_test scores under
    (analysis_domain() for 6 classes), or a permutation delta would not be
    comparable to a cross-leg CSI delta -- the stated point of reusing
    evaluate.csi_counts instead of inventing a metric."""
    x, y, times = synthetic_case
    calls = []
    monkeypatch.setattr(permutation.dataset, "analysis_domain",
                        lambda: calls.append(1) or scoring_mask)
    df = permutation.single_pass(ChannelZeroModel(), x, y, times, N_CLASSES,
                                 CHANNELS, np.random.default_rng(0))
    assert calls                                       # the 6-class branch
    np.testing.assert_allclose(_cold(df, permutation.BASELINE).csi.iloc[0],
                               1.0)


def test_channel_name_count_mismatch_raises(synthetic_case, scoring_mask):
    """x's trailing axis IS config.INPUT_CHANNELS in order; a name list of a
    different length means the loader and the caller disagree about the
    channel set, which would silently label every delta with the wrong
    channel.  Errors name the fix (house rule)."""
    x, y, times = synthetic_case
    with pytest.raises(ValueError, match="--channels"):
        permutation.single_pass(ChannelZeroModel(), x, y, times, N_CLASSES,
                                config.SFC_VARS, np.random.default_rng(0),
                                mask=scoring_mask)
