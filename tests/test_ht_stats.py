"""Tests for convection_skill.stats -- analytic/synthetic answers.

The load-bearing claims: (1) weighted Gini == the paper's Gini at unit weights;
(2) the day-block bootstrap matches the naive bootstrap on iid data but is
WIDER on day-correlated data (the whole point of building it); (3) conditional
Gini vanishes for a predictor collinear with its control and preserves the
marginal for an independent one; (4) the event-rate curve recovers a planted
non-monotonic response; (5) BH-FDR behaves at the boundaries; (6) both
inference conventions run through the one entry point and agree on iid data.
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_skill import gini as paper_gini
from convection_skill import stats as S


def test_weighted_gini_matches_paper_full_resolution():
    rng = np.random.default_rng(1)
    n = 50_000
    pred = rng.random(n)
    flags = rng.random(n) < pred**2 * 0.05
    g_ours = S.weighted_gini(pred, flags, rng=np.random.default_rng(9))
    g_paper = paper_gini.gini(pred, flags, n_bins=n, rng=np.random.default_rng(9))
    assert g_ours == pytest.approx(g_paper, abs=2e-3)


def test_weighted_gini_weights_replicate_rows():
    """Weight 2 on a row == that row appearing twice."""
    rng = np.random.default_rng(2)
    n = 20_000
    pred = rng.random(n)
    flags = (rng.random(n) < pred * 0.05).astype(float)
    w = rng.integers(0, 3, size=n).astype(float)
    idx = np.repeat(np.arange(n), w.astype(int))
    g_w = S.weighted_gini(pred, flags, weights=w, rng=np.random.default_rng(3))
    g_rep = S.weighted_gini(pred[idx], flags[idx], rng=np.random.default_rng(3))
    assert g_w == pytest.approx(g_rep, abs=5e-3)


def _day_structured_sample(n_days=400, cells=60, day_effect=0.0, seed=0):
    """Cell-day rows where predictor = day-level signal + cell noise, and events
    depend on the predictor. day_effect>0 makes rows within a day strongly
    dependent (shared day term), which iid resampling wrongly ignores."""
    rng = np.random.default_rng(seed)
    day_signal = rng.normal(0, 1, n_days)
    pred = (day_effect * np.repeat(day_signal, cells)
            + rng.normal(0, 1, n_days * cells))
    prob = 0.03 * (1 / (1 + np.exp(-pred)))
    flags = rng.random(n_days * cells) < prob
    day = np.repeat(np.arange(n_days), cells)
    return pred, flags, day


def test_block_bootstrap_matches_naive_when_iid():
    pred, flags, day = _day_structured_sample(day_effect=0.0, seed=4)
    r = S.block_bootstrap_gini(pred, flags, day, n_reps=300, block_days=5,
                               rng=np.random.default_rng(5))
    assert r.se_block == pytest.approx(r.se_naive, rel=0.35)  # same order


def _day_level_sample(seed, n_days=300, cells=100):
    """Purely day-level predictor (constant within day; e.g. a domain-day
    anomaly): effective sample = days, the worst case for iid-row errors."""
    rng = np.random.default_rng(seed)
    day_signal = rng.normal(0, 1, n_days)
    pred = np.repeat(day_signal, cells)
    prob = 0.02 * (1 / (1 + np.exp(-2 * day_signal)))
    flags = rng.random(n_days * cells) < np.repeat(prob, cells)
    return pred, flags, np.repeat(np.arange(n_days), cells)


def test_block_bootstrap_is_calibrated_and_naive_underestimates():
    """The real correctness claim (verified by simulation during development):
    against the TRUE sampling SD of the Gini (SD of point estimates across
    independent worlds), the day-block bootstrap is ~unbiased while the iid-row
    bootstrap underestimates. (For rare events the iid bias is moderate --
    event-count noise dominates -- but it is systematic.)"""
    points = []
    for s in range(40):
        p, f, d = _day_level_sample(s)
        order = np.argsort(p, kind="mergesort")
        points.append(S._gini_from_sorted(f[order].astype(float), np.ones(f.size)))
    true_sd = np.std(points, ddof=1)

    sb, sn = [], []
    for s in range(6):
        p, f, d = _day_level_sample(s)
        r = S.block_bootstrap_gini(p, f, d, n_reps=300, block_days=5,
                                   rng=np.random.default_rng(s + 99))
        sb.append(r.se_block)
        sn.append(r.se_naive)
    assert np.mean(sb) == pytest.approx(true_sd, rel=0.25)   # calibrated
    assert np.mean(sn) < np.mean(sb)                          # naive is smaller


def test_block_bootstrap_p_significant_for_real_signal():
    pred, flags, day = _day_structured_sample(day_effect=0.5, seed=8)
    r = S.block_bootstrap_gini(pred, flags, day, n_reps=300,
                               rng=np.random.default_rng(9))
    assert r.gini > 0.1
    assert r.p_value < 0.05
    assert r.ci_lo < r.gini < r.ci_hi


def test_conditional_gini_collinear_is_zero():
    """Predictor == control -> no residual skill within control bins."""
    rng = np.random.default_rng(10)
    n = 100_000
    control = rng.random(n)
    flags = rng.random(n) < control**2 * 0.05
    res = S.conditional_gini(control, flags, control, n_control_bins=10,
                             rng=np.random.default_rng(11))
    marginal = S.weighted_gini(control, flags, rng=np.random.default_rng(11))
    assert marginal > 0.3                       # strongly skilled marginally
    assert abs(res["conditional_gini"]) < 0.05  # ~no skill beyond itself


def test_conditional_gini_preserves_independent_signal():
    """A predictor independent of the control keeps its marginal skill."""
    rng = np.random.default_rng(12)
    n = 100_000
    control = rng.random(n)
    pred = rng.random(n)
    flags = rng.random(n) < (0.5 * pred + 0.5 * control) ** 2 * 0.05
    res = S.conditional_gini(pred, flags, control, n_control_bins=10,
                             rng=np.random.default_rng(13))
    marginal = S.weighted_gini(pred, flags, rng=np.random.default_rng(13))
    assert res["conditional_gini"] > 0.5 * marginal  # most skill survives


def test_event_rate_curve_recovers_nonmonotonic_shape():
    """Plant an inverted-U response (the A2 CIN hypothesis shape)."""
    rng = np.random.default_rng(14)
    n = 200_000
    pred = rng.uniform(-1, 1, n)
    prob = 0.05 * np.exp(-((pred - 0.0) ** 2) / 0.1)  # peak at pred=0
    flags = rng.random(n) < prob
    day = rng.integers(0, 300, n)
    c = S.event_rate_curve(pred, flags, day, n_bins=10, n_reps=50,
                           rng=np.random.default_rng(15))
    peak_bin = int(np.argmax(c["rate"]))
    assert 3 <= peak_bin <= 6                # interior peak
    assert c["rate"][peak_bin] > 2 * c["rate"][0]
    assert c["rate"][peak_bin] > 2 * c["rate"][-1]
    # a corresponding Gini washes out relative to the peak contrast
    g = S.weighted_gini(pred, flags)
    assert abs(g) < 0.15


def test_iid_method_matches_naive_se_and_finds_signal():
    """method='iid' drives CIs/p from the paper's construction; on genuinely
    iid data it agrees with the block SEs and still detects a real signal."""
    pred, flags, day = _day_structured_sample(day_effect=0.0, seed=21)
    r = S.bootstrap_gini(pred, flags, day, method="iid", n_reps=300,
                         rng=np.random.default_rng(22))
    assert r.method == "iid"
    assert r.p_value < 0.05 and r.gini > 0
    assert r.ci_lo < r.gini < r.ci_hi
    # both SE columns populated (day_index given), same order on iid data
    assert r.se_naive == pytest.approx(r.se_block, rel=0.5)


def test_iid_method_without_day_index():
    """The paper convention needs no day structure at all."""
    pred, flags, _ = _day_structured_sample(day_effect=0.0, seed=23)
    r = S.bootstrap_gini(pred, flags, method="iid", n_reps=200,
                         rng=np.random.default_rng(24))
    assert np.isfinite(r.p_value) and np.isfinite(r.se_naive)
    assert np.isnan(r.se_block)  # no day structure -> no block SE


def test_block_method_requires_day_index():
    pred, flags, _ = _day_structured_sample(seed=25)
    with pytest.raises(ValueError, match="day_index"):
        S.bootstrap_gini(pred, flags, method="block")


def test_conditional_gini_iid_and_block_agree_on_iid_data():
    rng = np.random.default_rng(26)
    n = 60_000
    control = rng.random(n)
    pred = rng.random(n)
    flags = rng.random(n) < (0.5 * pred + 0.5 * control) ** 2 * 0.05
    day = rng.integers(0, 300, n)
    kw = dict(n_control_bins=10, n_reps=200)
    r_iid = S.conditional_gini(pred, flags, control, day_index=day,
                               method="iid", rng=np.random.default_rng(27), **kw)
    r_blk = S.conditional_gini(pred, flags, control, day_index=day,
                               method="block", rng=np.random.default_rng(27), **kw)
    assert r_iid["conditional_gini"] == pytest.approx(r_blk["conditional_gini"])
    width_iid = r_iid["ci_hi"] - r_iid["ci_lo"]
    width_blk = r_blk["ci_hi"] - r_blk["ci_lo"]
    assert width_iid == pytest.approx(width_blk, rel=0.6)  # same order on iid


def test_fdr_bh_boundaries():
    p = np.array([0.001, 0.002, 0.5, 0.9, np.nan])
    sig = S.fdr_bh(p, alpha=0.10)
    assert sig[0] and sig[1] and not sig[2] and not sig[3] and not sig[4]
    assert not S.fdr_bh(np.array([0.5, 0.8]), alpha=0.10).any()
