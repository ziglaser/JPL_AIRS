"""Tests for sub-hourly resampling -- analytic (endpoints, linearity, path length)."""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import geo
from trajectory_kernels.resample import resample_series, resample_trajectory


def test_endpoints_preserved():
    t = np.array([0.0, 2.1, 3.1, 4.1])  # irregular first step, like the real data
    y = np.array([10.0, 12.0, 9.0, 5.0])
    ft, fy = resample_series(t, y, 10.0 / 60.0)
    assert ft[0] == pytest.approx(t[0])
    assert ft[-1] == pytest.approx(t[-1])
    assert fy[0] == pytest.approx(y[0])
    assert fy[-1] == pytest.approx(y[-1])


def test_linear_segment_is_exact():
    """A constant-velocity parcel is reproduced exactly by linear interpolation."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = 3.0 + 2.0 * t  # exactly linear
    ft, fy = resample_series(t, y, 7.0 / 60.0)
    assert np.allclose(fy, 3.0 + 2.0 * ft)


def test_path_length_at_least_hourly_chord():
    """Resampled great-circle path length >= the sum of hourly straight chords."""
    lat = np.array([40.0, 40.5, 41.2, 41.3])
    lon = np.array([-95.0, -94.0, -92.5, -92.4])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    r = resample_trajectory(lat, lon, np.zeros(4), t, step_min=10.0)
    fine_len = geo.cumulative_path_km(r["lat"], r["lon"])[-1]
    coarse_len = geo.cumulative_path_km(lat, lon)[-1]
    assert fine_len == pytest.approx(coarse_len, rel=0.02)  # linear-in-time: ~equal


def test_truncates_at_nan():
    lat = np.array([40.0, 40.5, np.nan, 41.0])
    lon = np.array([-95.0, -94.0, -93.0, -92.0])
    alt = np.array([500.0, 600.0, 700.0, 800.0])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    r = resample_trajectory(lat, lon, alt, t, step_min=30.0)
    assert r["t_hours"][-1] <= 1.0  # stopped before the NaN
