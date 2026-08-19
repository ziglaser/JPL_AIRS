"""Available surface energy at a parcel's ground point: the missing hour-to-hour
weight.

``footprint.py``'s kernel is normalized per lag hour (a plotting/mapping
convention); the physical hour-to-hour mass is carried separately as the
``lag_weight`` variable and multiplied back in by the Psi contraction
(``apply.apply_kernel(lag_weights=...)`` via ``predictors.psi``), so with this
module's weight applied, contact hours count in proportion to their available
energy. Physically they must: a parcel in surface contact at 01 UTC over the Plains
(dark, stable) receives no energy from the soil, while the same contact time at
20 UTC receives ~700 W/m2 of available energy. Weighting contact time by the
available energy flux is what turns a *geometric* residence-time footprint into a
*physical* soil-influence measure.

The nocturnal PBL collapse in ``pbl.ClimatologicalPBL``
(``PBL_NOCTURNAL_M = 200`` m) already suppresses much of the night through the
contact gate, so this module is complementary rather than redundant: it also
graduates the daytime hours (a 17 LT contact hour is worth roughly a third of a
13 LT one) and handles the evening window where the PBL is still deep but the sun
is nearly down.

Available energy, not net radiation, is the right quantity: it is the reservoir
that the Bowen-ratio partitioning divides between sensible and latent heating,

    H + LE = Rn - G ~= a * DSWF,      a = (Rn - G)/DSWF ~= 0.55

Literature / provenance:
- Cooper 1969            : solar declination approximation.
- Liu & Jordan 1960 form : bulk clear-sky transmissivity as a function of the
  cosine of the solar zenith angle.
- a = 0.55: Rn/DSWF ~= 0.60-0.70 for vegetated land (albedo 0.15-0.25, net
  longwave -60 to -100 W/m2) times (1 - G/Rn) with G/Rn ~= 0.05-0.15 vegetated,
  0.2-0.4 bare soil. Cross-checked in the method note against H + LE = 400 W/m2
  for a high-plains June afternoon: agreement to 3%.
"""

from __future__ import annotations

import numpy as np

from . import config


def cos_solar_zenith(lat, lon, time_utc) -> np.ndarray:
    """Cosine of the solar zenith angle, clipped at 0 (night).

    ``lat``, ``lon`` in degrees (lon negative west); ``time_utc`` any
    numpy-datetime64-compatible array. Broadcasts. Uses mean solar time
    (longitude/15) rather than true solar time -- the equation-of-time
    correction is at most ~16 min, i.e. ~4 degrees of hour angle, which is
    well inside the uncertainty of ``a``.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    t = np.asarray(time_utc, dtype="datetime64[s]")

    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    doy = ((t - year_start) / np.timedelta64(1, "D")).astype(float) + 1.0
    day_start = t.astype("datetime64[D]").astype("datetime64[s]")
    utc_hour = ((t - day_start) / np.timedelta64(1, "h")).astype(float)

    # Cooper (1969) declination, degrees
    decl = np.deg2rad(23.44 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0)))
    solar_hour = utc_hour + lon / 15.0
    hour_angle = np.deg2rad(15.0 * (solar_hour - 12.0))

    lat_r = np.deg2rad(lat)
    cosz = (np.sin(lat_r) * np.sin(decl)
            + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle))
    return np.clip(cosz, 0.0, None)


def clear_sky_dswf(lat, lon, time_utc) -> np.ndarray:
    """Clear-sky downward shortwave flux at the surface, W/m2.

    ``DSWF = S0 * cosZ * (b0 + b1 * cosZ)``, the standard bulk-transmissivity
    form: transmissivity rises with sun elevation because the air mass falls.
    Reference values: cosZ = 0.9 -> 955 W/m2; cosZ = 0.5 -> 476 W/m2; cosZ = 0
    -> 0.
    """
    cosz = cos_solar_zenith(lat, lon, time_utc)
    return config.SOLAR_CONSTANT_WM2 * cosz * (
        config.CLEARSKY_B0 + config.CLEARSKY_B1 * cosz)


def available_energy_flux(lat, lon, time_utc,
                          a: float = config.AVAILABLE_ENERGY_COEF) -> np.ndarray:
    """Surface available energy ``Rn - G ~= a * DSWF``, W/m2.

    This is the quantity to multiply contact time by. Same call signature as
    ``pbl.PBLModel.__call__``, so it drops into
    ``footprint.footprint_from_trajectories`` beside the PBL model.

    Reference value: 0.55 * 750 = 413 W/m2 for a mid-afternoon high-plains
    clear-sky hour, against H + LE = 400 W/m2 typical for that setting.

    NOTE this is a CLEAR-SKY upper bound: there is no cloud field in the
    trajectory dataset. Cloud cover reduces the true flux, most on exactly the
    pre-convective days of interest, so treat ``Omega``/``Phi`` as upper bounds
    and rank-based comparisons as the trustworthy use. If the ``fullgrid`` twin
    carries a radiation or cloud field, prefer it (documented extension).
    """
    return a * clear_sky_dswf(lat, lon, time_utc)


class ClearSkyAvailableEnergy:
    """Callable wrapper, so the weight is pluggable by interface like the PBL
    and fuzz models: ``energy_fn=ClearSkyAvailableEnergy(a=0.45)``."""

    def __init__(self, a: float = config.AVAILABLE_ENERGY_COEF):
        self.a = float(a)

    def __call__(self, lat, lon, time_utc) -> np.ndarray:
        return available_energy_flux(lat, lon, time_utc, a=self.a)

    def __repr__(self) -> str:
        return f"ClearSkyAvailableEnergy(a={self.a})"


class UniformEnergy:
    """No-op weight (returns 1.0), reproducing the current equal-hour behaviour.

    Keep it for the A/B test in step 3 of the implementation note: if Psi is
    materially different under ``UniformEnergy`` vs
    ``ClearSkyAvailableEnergy``, the hour-to-hour weighting mattered and should
    be reported.
    """

    def __call__(self, lat, lon, time_utc) -> np.ndarray:
        return np.ones(np.broadcast(np.asarray(lat, float),
                                    np.asarray(lon, float),
                                    np.asarray(time_utc)).shape)

    def __repr__(self) -> str:
        return "UniformEnergy()"
