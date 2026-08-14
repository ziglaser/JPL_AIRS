"""Tests for the event-curve inversion (inversion.py) -- analytic answers.

Load-bearing claims: (1) a planted non-monotone response with raw Gini ~ 0
scores high once inverted, under BOTH estimators; (2) inversion is a no-op
(rank-preserving) for a monotone predictor; (3) cross-fitting kills in-sample
leakage: a pure-noise predictor's inverted Gini is ~0; (4) the quadratic
estimator recovers the exact Bayes ordering in the Gaussian
unequal-variance case it is derived from; (5) the suite wiring produces the
inverted rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_skill import stats as S
from convection_skill.inversion import (
    binned_rate_index, crossfit_index, quadratic_logistic_index,
)


def _inverted_u_sample(n=120_000, seed=0):
    """Events peak at x = 0 (the A2 'optimal cap' shape): raw rank skill ~ 0."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    prob = 0.06 * np.exp(-(x ** 2) / 0.08)
    flags = (rng.random(n) < prob).astype(float)
    day = rng.integers(0, 400, n)
    return x, flags, day


def _gini(pred, flags):
    return S.weighted_gini(pred, flags, rng=np.random.default_rng(1))


@pytest.mark.parametrize("method", ["quadratic", "binned"])
def test_inversion_recovers_nonmonotone_skill(method):
    x, flags, day = _inverted_u_sample()
    raw = _gini(x, flags)
    idx = crossfit_index(x, flags, day, method=method)
    inv = _gini(idx, flags)
    assert abs(raw) < 0.1          # the rank statistic washes out, as designed
    assert inv > 0.4               # the index recovers the ordering skill


@pytest.mark.parametrize("method", ["quadratic", "binned"])
def test_inversion_is_noop_for_monotone_predictor(method):
    """Rank invariance: for a monotone response, Gini(index) ~ Gini(raw)."""
    rng = np.random.default_rng(2)
    n = 100_000
    x = rng.random(n)
    flags = (rng.random(n) < 0.05 * x ** 2).astype(float)
    day = rng.integers(0, 400, n)
    idx = crossfit_index(x, flags, day, method=method)
    assert _gini(idx, flags) == pytest.approx(_gini(x, flags), abs=0.03)


@pytest.mark.parametrize("method", ["quadratic", "binned"])
def test_crossfitting_prevents_leakage_on_pure_noise(method):
    """THE honesty test: an uninformative predictor must stay at Gini ~ 0.

    (In-sample fitting of the binned curve would score well above zero here.)
    """
    rng = np.random.default_rng(3)
    n = 80_000
    x = rng.random(n)
    flags = (rng.random(n) < 0.03).astype(float)   # independent of x
    day = rng.integers(0, 400, n)
    idx = crossfit_index(x, flags, day, method=method)
    assert abs(_gini(idx, flags)) < 0.05


def test_in_sample_binned_fit_would_leak():
    """Documents WHY cross-fitting exists: same noise data, in-sample binned
    transform, visibly inflated Gini."""
    rng = np.random.default_rng(4)
    n = 80_000
    x = rng.random(n)
    flags = (rng.random(n) < 0.03).astype(float)
    leaky = binned_rate_index(x, flags, x)          # fit == eval rows
    honest = crossfit_index(x, flags, rng.integers(0, 400, n), method="binned")
    assert _gini(leaky, flags) > abs(_gini(honest, flags)) + 0.03


def test_quadratic_recovers_gaussian_bayes_ordering():
    """Unequal-variance Gaussians: the Bayes index IS quadratic; the estimator
    should reach the Bayes Gini (computed from the true eta)."""
    rng = np.random.default_rng(5)
    n = 150_000
    flags = (rng.random(n) < 0.05).astype(float)
    x = np.where(flags > 0, rng.normal(0.0, 0.4, n), rng.normal(0.0, 1.0, n))
    # true eta up to monotone transform: log LR = quadratic in x
    pi = 0.05
    def gauss(v, s): return np.exp(-(v ** 2) / (2 * s * s)) / s
    eta_true = pi * gauss(x, 0.4) / (pi * gauss(x, 0.4) + (1 - pi) * gauss(x, 1.0))
    bayes = _gini(eta_true, flags)
    est = quadratic_logistic_index(x, flags, x)
    assert _gini(est, flags) == pytest.approx(bayes, abs=0.02)
    assert bayes > 0.3


def test_quadratic_atom_term_captures_zero_inflation():
    """CIN-like predictor: point mass at 0 with its OWN event rate, plus an
    inverted-U on the continuous part. The atom offset makes the quadratic
    competitive; without it the smooth curve must smear the discontinuity."""
    rng = np.random.default_rng(6)
    n = 120_000
    at_zero = rng.random(n) < 0.4                  # 40% uncapped columns
    x = np.where(at_zero, 0.0, -np.abs(rng.normal(0, 60.0, n)))
    prob = np.where(at_zero, 0.002,                # capped-off: low rate at 0
                    0.05 * np.exp(-((x + 30.0) ** 2) / 400.0))  # peak at -30
    flags = (rng.random(n) < prob).astype(float)
    day = rng.integers(0, 400, n)

    quad = _gini(crossfit_index(x, flags, day, method="quadratic"), flags)
    binned = _gini(crossfit_index(x, flags, day, method="binned"), flags)
    assert quad > 0.4
    assert quad == pytest.approx(binned, abs=0.1)  # both estimators agree


def test_crossfit_handles_nan_and_validates_method():
    x, flags, day = _inverted_u_sample(n=5_000)
    x[:100] = np.nan
    idx = crossfit_index(x, flags, day, method="quadratic")
    assert np.isnan(idx[:100]).all() and np.isfinite(idx[100:]).all()
    with pytest.raises(ValueError, match="method"):
        crossfit_index(x, flags, day, method="spline")


def test_diagnostics_report_fold_stability():
    x, flags, day = _inverted_u_sample(n=60_000)
    _, diag = crossfit_index(x, flags, day, method="quadratic",
                             return_diagnostics=True)
    assert diag["fold_curves"].shape[1] == diag["grid"].size
    assert diag["fold_stability_spearman"] > 0.9   # a real signal is stable


def test_suite_wiring_runs_inverted_specs():
    """test_hypothesis applies the inversion when spec.invert is set."""
    import pandas as pd
    from convection_skill.config import AnalysisConfig
    from convection_skill.dataset import Prepared
    from convection_skill.hypotheses import HypothesisSpec
    from convection_skill.suite import test_hypothesis

    x, flags, day_int = _inverted_u_sample(n=40_000, seed=7)
    days = np.datetime64("2019-03-01") + day_int.astype("timedelta64[D]")
    table = pd.DataFrame({
        "mu_cin": x, "day": days,
        "season": "JJA", "is_east": True, "is_late_slot": False,
        "fcst_q": np.random.default_rng(8).random(x.size),
        "sm_cell_clim": 0.3, "wind": 5.0,
    })
    prepared = Prepared(cfg=AnalysisConfig(years=(2019,), n_boot_reps=100),
                        table=table, thresholds={},
                        flags={"heavy": flags.astype(bool)}, onset=None)
    raw_spec = HypothesisSpec("raw", "raw CIN", "mu_cin", target="heavy")
    inv_spec = HypothesisSpec("inv", "inverted CIN", "mu_cin", target="heavy",
                              invert="quadratic")
    raw_rows, _ = test_hypothesis(raw_spec, prepared)
    inv_rows, _ = test_hypothesis(inv_spec, prepared)
    raw_g = raw_rows[0]["gini"]
    inv_g = inv_rows[0]["gini"]
    assert abs(raw_g) < 0.1 and inv_g > 0.4
