"""Tests for the predictor-agnostic Gini core.

These use analytically known answers so a reader can confirm the implementation
matches the paper's definition without trusting any real data:

- uniform/shuffled predictor   -> Gini ~ 0
- perfectly separating predictor (events are the top fraction f) -> Gini = 1 - f
- anti-correlated predictor    -> Gini < 0
- cross-check against the classification identity  Gini == 2 * AUC - 1
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from convection_skill import gini
from convection_skill.gini import (
    detection_cdf,
    exceedance_flags,
    gini_from_cdf,
)


def test_uniform_predictor_is_uninformative():
    """A predictor unrelated to events gives Gini ~ 0 (the diagonal)."""
    rng = np.random.default_rng(1)
    n = 200_000
    predictor = rng.random(n)
    flags = rng.random(n) < 0.05  # events independent of predictor
    assert abs(gini.gini(predictor, flags, rng=rng)) < 0.01


def test_shuffled_predictor_matches_uniform():
    """Shuffling an informative predictor destroys its skill (Gini -> 0)."""
    rng = np.random.default_rng(2)
    n = 200_000
    predictor = rng.random(n)
    flags = rng.random(n) < predictor ** 3 * 0.05
    g_shuffled = gini.gini(rng.permutation(predictor), flags, rng=rng)
    assert abs(g_shuffled) < 0.01


@pytest.mark.parametrize("event_fraction_f", [0.01, 0.05, 0.10, 0.25])
def test_perfect_predictor_gives_one_minus_f(event_fraction_f):
    """If the events are exactly the highest-predictor fraction f, Gini = 1 - f.

    Derivation (see gini.gini_from_cdf): the capture curve is flat at 0 until the
    sample fraction reaches 1 - f, then rises linearly to 1, giving area
    0.5 - 0.5 f between it and the diagonal, so Gini = 1 - f.
    """
    n = 100_000
    predictor = np.arange(n, dtype=float)  # strictly increasing, no ties
    k = int(round(event_fraction_f * n))
    flags = np.zeros(n, dtype=bool)
    flags[-k:] = True  # events are the top-k predictor values
    g = gini.gini(predictor, flags)
    assert g == pytest.approx(1.0 - event_fraction_f, abs=1e-3)


def test_anti_correlated_predictor_is_negative():
    """Events at LOW predictor values give a negative Gini."""
    n = 100_000
    predictor = np.arange(n, dtype=float)
    flags = np.zeros(n, dtype=bool)
    flags[: n // 20] = True  # events are the lowest 5%
    assert gini.gini(predictor, flags) < -0.5


def test_gini_matches_cap_auc_identity():
    """Cross-check against the exact CAP/ROC identity  Gini = (2*AUC - 1)*(1 - f).

    The paper defines the Gini from the (un-normalised) CDF/Lorenz area, whose
    perfect-predictor limit is 1 - f (not 1). That equals the classification
    accuracy ratio 2*AUC-1 scaled by the non-event fraction (1 - f). Full
    resolution (one bin per sample) removes binning discretisation so the match is
    tight. This is the precise sense in which the paper's Gini and the ROC "give
    the same conclusions".
    """
    rng = np.random.default_rng(3)
    n = 50_000
    predictor = rng.random(n)
    flags = rng.random(n) < predictor ** 2 * 0.2
    f = flags.mean()
    g_full = gini.gini(predictor, flags, n_bins=n, rng=rng)
    expected = (2.0 * roc_auc_score(flags, predictor) - 1.0) * (1.0 - f)
    assert g_full == pytest.approx(expected, abs=2e-3)


def test_gini_approaches_two_auc_minus_one_for_rare_events():
    """For rare events (small f) the paper's Gini ~ 2*AUC - 1 to within ~f.

    This is the regime of the actual analysis (QPE99.95 has f = 5e-4), so the
    distinction between the Lorenz-area Gini and the classification accuracy ratio
    is negligible there.
    """
    rng = np.random.default_rng(7)
    n = 500_000
    predictor = rng.random(n)
    flags = rng.random(n) < predictor ** 2 * 0.002  # f ~ 6.7e-4
    f = flags.mean()
    g_full = gini.gini(predictor, flags, n_bins=n, rng=rng)
    accuracy_ratio = 2.0 * roc_auc_score(flags, predictor) - 1.0
    assert abs(g_full - accuracy_ratio) < 2.0 * f


def test_hundred_bins_close_to_full_resolution():
    """The paper's 100-bin construction is close to the full-resolution Gini."""
    rng = np.random.default_rng(4)
    n = 200_000
    predictor = rng.random(n)
    flags = rng.random(n) < predictor ** 2 * 0.2
    g_100 = gini.gini(predictor, flags, n_bins=100, rng=rng)
    g_full = gini.gini(predictor, flags, n_bins=n, rng=rng)
    assert g_100 == pytest.approx(g_full, abs=5e-3)


def test_zero_tiebreak_makes_cdf_linear_over_zero_range():
    """Many zero-predictor samples with random events should give a ~linear CDF
    segment (the behaviour the paper's +/-1e-10 perturbation is designed to produce).
    """
    rng = np.random.default_rng(5)
    n = 100_000
    predictor = np.zeros(n)  # everything is zero-CAPE
    predictor[-n // 2 :] = rng.random(n // 2)  # upper half has positive CAPE
    flags = rng.random(n) < 0.05  # events independent of predictor
    x, y = detection_cdf(predictor, flags, rng=rng)
    # Over the zero range (first half of samples) the capture curve should track
    # the diagonal because events are randomly ordered among the zeros.
    half = len(x) // 2
    assert np.allclose(y[:half], x[:half], atol=0.03)


def test_exceedance_flags():
    """Flags are strict exceedances of the pooled percentile threshold."""
    target = np.arange(101, dtype=float)  # 0..100
    flags = exceedance_flags(target, 99.0)
    assert flags.sum() == 1  # only the value 100 exceeds the 99th percentile


def test_detection_cdf_endpoints():
    """The capture curve always runs from (0,0) to (1,1)."""
    rng = np.random.default_rng(6)
    predictor = rng.random(10_000)
    flags = rng.random(10_000) < 0.1
    x, y = detection_cdf(predictor, flags, rng=rng)
    assert x[0] == 0.0 and y[0] == 0.0
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)


def test_no_events_raises():
    """A degenerate all-zero flag array is a caller error, not a silent 0."""
    predictor = np.arange(100, dtype=float)
    flags = np.zeros(100, dtype=bool)
    with pytest.raises(ValueError, match="no events"):
        gini.gini(predictor, flags)
