"""Synthetic AIRS gap/swath fields for stage-B degraded pretraining.

The real gap bank (:mod:`.mask_bank`) is the ground truth for coverage
statistics, but it currently holds a single day -- training against one
swath geometry teaches the model that geometry, not gap-robustness.  Until
the multi-year fullgrid harvest lands, stage B draws synthetic
valid-fraction fields built from two simple, documented heuristics
(2026-08-10; deliberately crude, revisit against the grown bank):

1. **Swath geometry.** Aqua is sun-synchronous (inclination 98.2 deg); an
   afternoon ascending pass crosses the domain as a roughly north-south
   band ~1650 km wide (AIRS scan swath, Aumann et al. 2003), tilted
   westward with latitude by the inclination plus Earth rotation
   (~0.33 deg lon per deg lat at 40 N).  Pixels outside the band were
   never observed: valid fraction 0.
2. **Cloud-correlated holes.** Retrieval failures cluster under cloud
   systems, which are contiguous mesoscale objects, not salt-and-pepper.
   A horizontally correlated Gaussian field (e-folding ``CLOUD_CORR_KM``)
   is mapped to a uniform "cloudiness" via the normal CDF; the retrieval
   then fails smoothly where cloudiness exceeds a per-level threshold.
   Lower levels fail first -- cloud-clearing degrades toward the surface
   -- so 1000 hPa has the lowest yield and 500 hPa the highest, matching
   the qualitative AIRS yield profile (~50-70 % overall).

Output matches ``mask_bank`` semantics exactly: per-level valid fraction
in [0, 1] on the unpadded (lat 68, lon 141, lev 5) label grid, consumed by
``mask_bank.apply_mask``.
"""
from __future__ import annotations

import numpy as np

from . import config
from .degrade import horizontally_correlated_noise

#: AIRS/AMSU scan swath width (Aumann et al. 2003).
SWATH_WIDTH_KM = 1650.0
#: Ground-track slope d(lon)/d(lat) of an afternoon ascending pass:
#: heading ~14 deg west of north at 40 N (98.2 deg inclination + Earth
#: rotation), divided by cos(40) for the degree conversion.
TRACK_DLON_DLAT = -0.33
#: 1-sigma jitter on the slope between draws (orbit-to-orbit variety).
TRACK_DLON_DLAT_JITTER = 0.08
#: Horizontal e-folding length of the synthetic cloud field: mesoscale-to-
#: synoptic cloud system scale.  Heuristic, like NOISE_CORR_KM.
CLOUD_CORR_KM = 400.0
#: Per-level cloudiness threshold above which the retrieval fails
#: (cloudiness is uniform on [0, 1], so each threshold IS that level's
#: expected clear-sky yield inside the swath).  Ordered like
#: config.TARGET_LEVELS_HPA = (1000, 925, 850, 700, 500).
LEVEL_YIELD = (0.55, 0.60, 0.65, 0.75, 0.80)
#: Width of the smooth pass/fail transition in cloudiness units (gives the
#: graded valid fractions real retrievals show at cloud edges).
YIELD_SOFTNESS = 0.08

_LAT = np.arange(10.0, 77.1, 1.0)                     # label-grid latitudes
_LON = np.arange(-171.0, -30.9, 1.0)                  # label-grid longitudes


def synthetic_valid_fraction(rng: np.random.Generator,
                             month: int | None = None) -> np.ndarray:
    """One synthetic (lat 68, lon 141, lev 5) valid-fraction field.

    ``month`` is accepted for signature parity with
    ``mask_bank.sample_mask`` (seasonal conditioning is a real-bank
    feature; the synthetic heuristics are season-free for now).
    """
    n_lev = len(config.TARGET_LEVELS_HPA)
    lat = _LAT[:, None]                                # (68, 1)
    lon = _LON[None, :]                                # (1, 141)

    # ---- 1. swath band -----------------------------------------------------
    slope = TRACK_DLON_DLAT + TRACK_DLON_DLAT_JITTER * rng.standard_normal()
    lon0 = rng.uniform(_LON[0], _LON[-1])              # track lon at 40 N
    track_lon = lon0 + slope * (lat - 40.0)
    half_width_deg = (SWATH_WIDTH_KM / 2.0) / (
        config.KM_PER_ITERATION * np.cos(np.deg2rad(lat)))
    in_swath = np.abs(lon - track_lon) <= half_width_deg   # (68, 141)

    # ---- 2. cloud-correlated holes ------------------------------------------
    corr_px = CLOUD_CORR_KM / config.KM_PER_ITERATION
    c = horizontally_correlated_noise((len(_LAT), len(_LON)), rng, corr_px,
                                      horizontal_axes=(0, 1))
    from scipy.special import ndtr                     # standard normal CDF
    cloudiness = ndtr(c)                               # uniform marginal [0, 1]

    vf = np.empty((len(_LAT), len(_LON), n_lev), dtype=np.float32)
    for k, level_yield in enumerate(LEVEL_YIELD):
        # smooth step: 1 well below the threshold, 0 well above
        vf[..., k] = 1.0 / (1.0 + np.exp((cloudiness - level_yield)
                                         / YIELD_SOFTNESS))
    vf *= in_swath[..., None]
    return vf
