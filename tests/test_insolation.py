"""Tests for the clear-sky available-energy weight (insolation.py).

Anchors are analytic: at the Cooper-declination zero (doy 81, i.e.
2019-03-22T00:00 UTC) the declination term vanishes EXACTLY, and putting the
sun at local noon (lon = 180 at 00 UTC -> solar_hour = 12) reduces the zenith
formula to cosZ = cos(lat), so any target cosZ can be dialed in via
lat = arccos(cosZ). The DSWF reference values then follow from the module
docstring: cosZ = 0.9 -> ~955 W/m2, 0.5 -> ~476, 0 -> 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import config
from trajectory_kernels.insolation import (
    ClearSkyAvailableEnergy,
    UniformEnergy,
    available_energy_flux,
    clear_sky_dswf,
    cos_solar_zenith,
)

# Cooper declination is exactly zero at doy = 81 (360*(284+81)/365 = 360 deg),
# and doy is computed with fractional days, so only the 00 UTC instant hits it.
_EQUINOX_00Z = np.datetime64("2019-03-22T00:00:00")
_NOON_LON = 180.0  # solar_hour = 0 + 180/15 = 12 -> hour angle exactly 0


def _lat_for_cosz(cosz: float) -> float:
    """At decl = 0 and hour angle = 0, cosZ = cos(lat)."""
    return float(np.degrees(np.arccos(cosz)))


# --------------------------------------------------------------------------- #
# cos_solar_zenith
# --------------------------------------------------------------------------- #
def test_cosz_equator_equinox_noon_is_one():
    """Exact construction: decl = 0, hour angle = 0, lat = 0 -> cosZ = 1."""
    assert cos_solar_zenith(0.0, _NOON_LON, _EQUINOX_00Z) == pytest.approx(1.0, abs=1e-12)


def test_cosz_equator_march_noon_near_one_within_declination():
    """Conventional anchor: equator, 12 UTC at lon 0 near the equinox. The
    fractional-doy declination is ~0.2 deg, so cosZ is 1 within that tolerance."""
    cosz = cos_solar_zenith(0.0, 0.0, np.datetime64("2019-03-22T12:00:00"))
    assert cosz == pytest.approx(1.0, abs=1e-3)


def test_cosz_local_midnight_is_zero():
    """Local mean midnight (lon/15 offsets 06 UTC to 00 solar) -> sun below the
    horizon at midlatitudes -> clipped to exactly 0."""
    cosz = cos_solar_zenith(40.0, -90.0, np.datetime64("2019-06-05T06:00:00"))
    assert cosz == 0.0


def test_cosz_clipped_at_night_never_negative():
    """A full diurnal cycle: night hours clip to exactly 0, day hours are > 0."""
    hours = np.arange(0, 24, dtype="timedelta64[h]")
    times = np.datetime64("2019-06-05T00:00:00") + hours
    cosz = cos_solar_zenith(40.0, -90.0, times)
    assert np.all(cosz >= 0.0)
    assert np.any(cosz == 0.0)   # night exists (clipped, not small-negative)
    assert np.any(cosz > 0.9)    # early-June midday sun at 40N is high


# --------------------------------------------------------------------------- #
# clear_sky_dswf: docstring reference values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target_cosz, ref_wm2", [(0.9, 955.0), (0.5, 476.0)])
def test_dswf_docstring_reference_values(target_cosz, ref_wm2):
    lat = _lat_for_cosz(target_cosz)
    cosz = cos_solar_zenith(lat, _NOON_LON, _EQUINOX_00Z)
    assert cosz == pytest.approx(target_cosz, abs=1e-12)  # construction worked
    dswf = clear_sky_dswf(lat, _NOON_LON, _EQUINOX_00Z)
    # exact against the bulk-transmissivity formula ...
    expected = config.SOLAR_CONSTANT_WM2 * cosz * (
        config.CLEARSKY_B0 + config.CLEARSKY_B1 * cosz)
    assert dswf == pytest.approx(expected, rel=1e-12)
    # ... and matches the module-docstring anchor to the quoted precision
    assert dswf == pytest.approx(ref_wm2, abs=1.0)


def test_dswf_zero_at_night():
    assert clear_sky_dswf(40.0, -90.0, np.datetime64("2019-06-05T06:00:00")) == 0.0


# --------------------------------------------------------------------------- #
# available_energy_flux and the callable wrappers
# --------------------------------------------------------------------------- #
def test_available_energy_is_a_times_dswf_exactly():
    lats = np.array([0.0, 25.0, 40.0, 60.0])
    lons = np.array([-100.0, -95.0, -90.0, -85.0])
    times = np.datetime64("2019-06-05T18:00:00") + np.arange(4).astype("timedelta64[h]")
    ae = available_energy_flux(lats, lons, times)
    dswf = clear_sky_dswf(lats, lons, times)
    assert np.array_equal(ae, config.AVAILABLE_ENERGY_COEF * dswf)


def test_clear_sky_available_energy_wrapper_scales_with_a():
    lat, lon, t = 40.5, -90.5, np.datetime64("2019-06-05T18:00:00")
    dswf = clear_sky_dswf(lat, lon, t)
    assert dswf > 0  # a daytime anchor, so the scaling is actually exercised
    model = ClearSkyAvailableEnergy(a=0.45)
    assert model(lat, lon, t) == pytest.approx(0.45 * dswf, rel=1e-12)
    # the default-coefficient wrapper matches the bare function
    default = ClearSkyAvailableEnergy()
    assert default(lat, lon, t) == pytest.approx(
        available_energy_flux(lat, lon, t), rel=1e-12)
    assert repr(model) == "ClearSkyAvailableEnergy(a=0.45)"


def test_uniform_energy_broadcasts_scalar_lat_array_time():
    times = np.datetime64("2019-06-05T00:00:00") + np.arange(5).astype("timedelta64[h]")
    out = UniformEnergy()(40.5, -90.5, times)
    assert out.shape == (5,)
    assert np.array_equal(out, np.ones(5))


def test_uniform_energy_broadcasts_array_lat_scalar_time():
    lats = np.array([30.5, 35.5, 40.5])
    lons = np.array([-100.5, -95.5, -90.5])
    out = UniformEnergy()(lats, lons, np.datetime64("2019-06-05T18:00:00"))
    assert out.shape == (3,)
    assert np.array_equal(out, np.ones(3))
