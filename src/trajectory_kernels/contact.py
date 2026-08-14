"""Surface-coupling weight: how strongly the land surface can talk to a parcel
at a given altitude, given the local PBL depth.

Following STILT (Lin+2003; Fasoli+2018), a parcel feels the surface only within
the lower ``CONTACT_FRACTION`` of the boundary layer: STILT uses ~0.5*PBLH,
Sodemann+2008 uses 1.5*PBLH (entrainment zone included); the default 1.0 is a
compromise. The weight tapers smoothly across the layer top rather than cutting.
"""

from __future__ import annotations

import numpy as np

from . import config


def contact_weight(
    alt_m,
    pbl_depth_m,
    fraction: float = config.CONTACT_FRACTION,
    taper_fraction: float = config.CONTACT_TAPER_FRACTION,
) -> np.ndarray:
    """Surface-coupling weight in [0, 1] for altitude ``alt_m`` under PBL depth.

    - ``w = 1`` for a parcel below ``(1 - taper_fraction) * fraction * PBL`` (well
      inside the contact layer),
    - linearly ramps ``1 -> 0`` across the top ``taper_fraction`` of the layer,
    - ``w = 0`` above ``fraction * PBL``.

    Altitudes below ground (``alt < 0``) are clamped to full contact. Inputs
    broadcast; non-finite altitude or non-positive PBL depth yields 0.
    """
    alt = np.asarray(alt_m, dtype=float)
    pbl = np.asarray(pbl_depth_m, dtype=float)

    layer_top = fraction * pbl
    taper_start = (1.0 - taper_fraction) * layer_top

    with np.errstate(invalid="ignore", divide="ignore"):
        ramp = (layer_top - alt) / (layer_top - taper_start)
        w = np.clip(ramp, 0.0, 1.0)
        w = np.where(alt <= taper_start, 1.0, w)  # full contact below the taper
        w = np.where(alt > layer_top, 0.0, w)

    w = np.where(np.isfinite(alt) & (pbl > 0.0), w, 0.0)
    return w
