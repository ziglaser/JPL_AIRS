"""Tests for the PBL models -- analytic checks on the diurnal curve and gradient."""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import config
from trajectory_kernels.pbl import ClimatologicalPBL, ConstantPBL, local_solar_hour


def _dt(hour_utc: float):
    return np.datetime64("2019-06-05T00:00:00") + np.timedelta64(int(hour_utc * 3600), "s")


def test_constant_pbl_is_uniform_and_broadcasts():
    m = ConstantPBL(1500.0)
    out = m(np.array([30.0, 40.0]), np.array([-90.0, -100.0]), _dt(21))
    assert out.shape == (2,)
    assert np.allclose(out, 1500.0)


def test_local_solar_hour():
    # UTC noon at lon -90 -> 12 - 6 = 06:00 local
    assert local_solar_hour(-90.0, _dt(12)) == pytest.approx(6.0, abs=1e-6)
    # wraps: UTC 02 at lon -90 -> 2 - 6 = -4 -> 20:00 local
    assert local_solar_hour(-90.0, _dt(2)) == pytest.approx(20.0, abs=1e-6)


def test_climatological_diurnal_endpoints():
    m = ClimatologicalPBL()
    lon = -90.0
    # nocturnal at 03 local (UTC 09 at lon -90)
    night = float(m(0.0, lon, _dt(9)))
    assert night == pytest.approx(config.PBL_NOCTURNAL_M, abs=1.0)
    # near the afternoon peak (15 local -> UTC 21 at lon -90), deepest of the day
    noon = float(m(0.0, lon, _dt(21)))
    assert noon > night
    # the peak equals the (gradient-adjusted) daytime maximum
    daytime_max = config.PBL_DAYTIME_M + config.PBL_WEST_DEEPENING_M_PER_DEG * (
        config.PBL_REFERENCE_LON - (-90.0)
    )
    assert float(noon) == pytest.approx(daytime_max, abs=1.0)


def test_west_is_deeper_in_daytime():
    """At the same local afternoon hour, a western point has a deeper PBL."""
    m = ClimatologicalPBL()
    # 15:00 local at each longitude -> different UTC; use local peak directly
    # lon -105 (west) vs -80 (east), both at their 15:00 local
    west = m(0.0, -105.0, _dt(21 + 105 / 15 - 90 / 15))  # keep 15 local at -105
    east = m(0.0, -80.0, _dt(21 + 80 / 15 - 90 / 15))
    assert float(west) > float(east)


def test_diurnal_is_smooth_and_peaks_once():
    """Sweep a full day in LOCAL hour at one point: single afternoon maximum,
    nocturnal floor, monotone up to the peak and down after."""
    m = ClimatologicalPBL()
    lon = -90.0
    local_hours = np.arange(0.0, 24.0, 0.5)
    # utc = local - lon/15 = local + 6 at lon -90 (datetime handles day rollover)
    depths = np.array([float(m(0.0, lon, _dt(lh - lon / 15.0))) for lh in local_hours])
    assert depths.min() == pytest.approx(config.PBL_NOCTURNAL_M, abs=1.0)
    peak_idx = int(np.argmax(depths))
    assert local_hours[peak_idx] == pytest.approx(config.PBL_PEAK_HOUR_LOCAL, abs=0.5)
    assert np.all(np.diff(depths[: peak_idx + 1]) >= -1e-9)
    assert np.all(np.diff(depths[peak_idx:]) <= 1e-9)
