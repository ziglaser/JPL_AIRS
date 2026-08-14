"""Tests for ``front_finder.degrade`` -- analytic answers, no TensorFlow.

Mirrors the house style of ``tests/test_front_dataset.py`` / ``test_gini.py``:
small synthetic arrays / MERRA-2-shaped datasets, exact or tightly-bounded
checks, no real data, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from front_finder import config, degrade

LEVS = list(config.TARGET_LEVELS_HPA)


# --------------------------------------------------------------------------- #
# 1. vertical_mixing_matrix
# --------------------------------------------------------------------------- #

def test_mixing_matrix_rows_sum_to_one():
    m = degrade.vertical_mixing_matrix(fwhm_km=degrade.T_FWHM_KM)
    assert m.shape == (5, 5)
    assert np.allclose(m.sum(axis=1), 1.0)


def test_mixing_matrix_fwhm_km_is_required():
    """fwhm_km has no default (2026-08-10): T and q resolve differently, so
    the caller must pass T_FWHM_KM or Q_FWHM_KM explicitly."""
    with pytest.raises(TypeError):
        degrade.vertical_mixing_matrix()
    with pytest.raises(TypeError):
        degrade.vertical_mixing_matrix(fwhm_km=None)


def test_per_variable_fwhm_constants():
    assert degrade.T_FWHM_KM == 1.5
    assert degrade.Q_FWHM_KM == 2.5
    assert not hasattr(degrade, "DEFAULT_FWHM_KM")


def test_mixing_matrix_fwhm_zero_is_identity():
    m = degrade.vertical_mixing_matrix(fwhm_km=0.0)
    assert np.allclose(m, np.eye(5))


def test_mixing_matrix_tiny_fwhm_approaches_identity():
    m = degrade.vertical_mixing_matrix(fwhm_km=1e-6)
    assert np.allclose(m, np.eye(5), atol=1e-6)


def test_mixing_matrix_symmetric_for_equidistant_levels():
    """Levels equispaced in log-p (hence in z) -> the pre-normalization
    kernel depends only on |z_i - z_j|, so the matrix is invariant under
    simultaneously reversing both the row and column order (reflecting the
    equispaced level ladder about its center); and each row itself is
    palindromic (weights fall off symmetrically from the diagonal, up to
    the row's own edge-truncation renormalization)."""
    levels = (1000.0, 900.0, 810.0, 729.0, 656.1)  # ratio 0.9 each step
    m = degrade.vertical_mixing_matrix(levels_hpa=levels, fwhm_km=2.0)
    assert np.allclose(m, m[::-1, ::-1])
    # the middle row has full symmetric support on both sides, so it alone
    # is guaranteed to be an exact palindrome.
    mid = m[2]
    assert np.allclose(mid, mid[::-1])


def test_mixing_matrix_double_smoothing_increases_spread():
    """A delta profile (all mass on one level) smoothed twice has larger
    variance (across levels, weighted by the resulting profile) than
    smoothed once -- i.e. repeated application spreads the profile further."""
    m = degrade.vertical_mixing_matrix(fwhm_km=degrade.Q_FWHM_KM)
    delta = np.zeros(5)
    delta[2] = 1.0
    once = m[2]                       # row 2 IS the once-smoothed delta
    twice = once @ m
    idx = np.arange(5)
    mean_once = (once * idx).sum()
    var_once = (once * (idx - mean_once) ** 2).sum()
    mean_twice = (twice * idx).sum()
    var_twice = (twice * (idx - mean_twice) ** 2).sum()
    assert var_twice > var_once


# --------------------------------------------------------------------------- #
# 2. smooth_profiles
# --------------------------------------------------------------------------- #

def test_smooth_profiles_conserves_constant_profile():
    t = np.full((3, 4, 5), 280.0)   # (lat, lon, lev)
    q = np.full((3, 4, 5), 0.01)
    t_s, q_s = degrade.smooth_profiles(t, q, axis=-1)
    assert np.allclose(t_s, 280.0)
    assert np.allclose(q_s, 0.01)


def test_smooth_profiles_conserves_constant_profile_with_one_nan_level():
    t = np.full((2, 5), 280.0)
    t[:, 2] = np.nan            # one below-ground level everywhere
    q = np.full((2, 5), 0.01)
    q[:, 2] = np.nan
    t_s, q_s = degrade.smooth_profiles(t, q, axis=-1)
    # the NaN level stays NaN...
    assert np.all(np.isnan(t_s[:, 2]))
    assert np.all(np.isnan(q_s[:, 2]))
    # ...and every finite level is unchanged (renormalized weights over
    # finite entries of a constant profile still yield the same constant).
    finite_levels = [0, 1, 3, 4]
    assert np.allclose(t_s[:, finite_levels], 280.0)
    assert np.allclose(q_s[:, finite_levels], 0.01)


def test_smooth_profiles_nan_level_does_not_propagate_to_neighbors():
    """A single non-constant profile with one NaN level: neighboring finite
    levels are still finite (not NaN-poisoned) after smoothing."""
    t = np.array([300.0, 295.0, np.nan, 285.0, 280.0])
    q = np.full(5, 0.01)
    t_s, _ = degrade.smooth_profiles(t, q, axis=-1)
    assert np.isnan(t_s[2])
    assert np.all(np.isfinite(t_s[[0, 1, 3, 4]]))


# --------------------------------------------------------------------------- #
# 3. add_noise
# --------------------------------------------------------------------------- #

def test_add_noise_t_sigma_within_tolerance():
    rng = np.random.default_rng(0)
    t = np.full((20000, 5), 280.0)
    q = np.full((20000, 5), 0.01)
    t_n, _ = degrade.add_noise(t, q, rng, axis=-1)
    sigma = np.std(t_n - t)
    assert sigma == pytest.approx(degrade.T_NOISE_SIGMA_K, rel=0.2)


def test_add_noise_q_mean_preserving_and_log_std():
    rng = np.random.default_rng(1)
    t = np.full((20000, 5), 280.0)
    q = np.full((20000, 5), 0.01)
    _, q_n = degrade.add_noise(t, q, rng, axis=-1)
    ratio = q_n / q
    assert np.mean(ratio) == pytest.approx(1.0, abs=0.02)
    sigma_ln = np.sqrt(np.log(1.0 + degrade.Q_NOISE_FRAC_SIGMA ** 2))
    assert np.std(np.log(ratio)) == pytest.approx(sigma_ln, rel=0.2)


def test_add_noise_level_lag1_correlation_matches_rho():
    rng = np.random.default_rng(2)
    n = 200000
    t = np.full((n, 5), 280.0)
    q = np.full((n, 5), 0.01)
    t_n, _ = degrade.add_noise(t, q, rng, axis=-1)
    e = t_n - t
    lag1 = np.corrcoef(e[:, 0], e[:, 1])[0, 1]
    assert lag1 == pytest.approx(degrade.LEVEL_NOISE_RHO, abs=0.1)
    lag1_b = np.corrcoef(e[:, 2], e[:, 3])[0, 1]
    assert lag1_b == pytest.approx(degrade.LEVEL_NOISE_RHO, abs=0.1)


def test_add_noise_zero_sigma_is_identity():
    rng = np.random.default_rng(3)
    t = np.full((4, 5), 280.0)
    q = np.full((4, 5), 0.01)
    t_n, q_n = degrade.add_noise(t, q, rng, t_sigma_k=0.0, q_frac_sigma=0.0,
                                 axis=-1)
    assert np.allclose(t_n, t)
    assert np.allclose(q_n, q)


# --------------------------------------------------------------------------- #
# 4. severity_ramp
# --------------------------------------------------------------------------- #

def test_severity_ramp_endpoints_and_linearity():
    assert degrade.DEFAULT_RAMP_EPOCHS == 41
    assert degrade.severity_ramp(0) == 0.0
    assert degrade.severity_ramp(41) == 1.0
    assert degrade.severity_ramp(80) == 1.0  # clipped past ramp_epochs
    assert degrade.severity_ramp(10) == pytest.approx(10 / 41)


# --------------------------------------------------------------------------- #
# 5. degrade_day
# --------------------------------------------------------------------------- #

LAT = np.arange(10, 15, 1, dtype=np.float64)
LON = np.arange(-20, -15, 1, dtype=np.float64)


def _make_day():
    times = pd.date_range("2003-01-01", periods=2, freq="3h")
    shape = (len(times), len(LEVS), len(LAT), len(LON))
    rng = np.random.default_rng(42)
    T = rng.uniform(250.0, 300.0, size=shape).astype(np.float32)
    QV = rng.uniform(0.001, 0.02, size=shape).astype(np.float32)
    U = rng.uniform(-20.0, 20.0, size=shape).astype(np.float32)
    V = rng.uniform(-20.0, 20.0, size=shape).astype(np.float32)
    PS = np.full((len(times), len(LAT), len(LON)), 101000.0, dtype=np.float32)
    return xr.Dataset(
        {
            "T": (("time", "lev", "lat", "lon"), T),
            "QV": (("time", "lev", "lat", "lon"), QV),
            "U": (("time", "lev", "lat", "lon"), U),
            "V": (("time", "lev", "lat", "lon"), V),
            "PS": (("time", "lat", "lon"), PS),
        },
        coords={"time": times, "lev": LEVS, "lat": LAT, "lon": LON},
    )


def test_degrade_day_severity_zero_is_identity():
    day = _make_day()
    rng = np.random.default_rng(0)
    out = degrade.degrade_day(day, rng, severity=0.0)
    assert np.allclose(out["T"].values, day["T"].values)
    assert np.allclose(out["QV"].values, day["QV"].values)
    assert np.allclose(out["U"].values, day["U"].values)
    assert np.allclose(out["V"].values, day["V"].values)
    assert np.allclose(out["PS"].values, day["PS"].values)


def test_degrade_day_severity_one_changes_thermo_not_winds():
    day = _make_day()
    rng = np.random.default_rng(1)
    out = degrade.degrade_day(day, rng, severity=1.0)
    assert not np.allclose(out["T"].values, day["T"].values)
    assert not np.allclose(out["QV"].values, day["QV"].values)
    assert np.array_equal(out["U"].values, day["U"].values)
    assert np.array_equal(out["V"].values, day["V"].values)
    assert np.array_equal(out["PS"].values, day["PS"].values)


def test_degrade_day_preserves_shape_and_dims():
    day = _make_day()
    rng = np.random.default_rng(2)
    out = degrade.degrade_day(day, rng, severity=0.5)
    assert out["T"].dims == day["T"].dims
    assert out["T"].shape == day["T"].shape
    assert out["QV"].shape == day["QV"].shape
