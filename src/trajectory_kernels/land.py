"""Land-fraction lookup from the global 1-deg land-sea mask (``data/lsm.nc``).

Bilinear interpolation of ``lsm`` gives any position a smooth 0-1 land weight
(soil moisture only exists over land; ocean weights to 0).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from . import config


def make_land_lookup(path=config.LSM_PATH) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return ``land_frac(lat, lon) -> [0, 1]`` bilinearly interpolated from lsm.

    Points outside the mask return 0 (treated as non-land). Inputs broadcast.
    """
    with xr.open_dataset(path) as ds:
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
        frac = np.clip(ds["lsm"].values.astype(float), 0.0, 1.0)

    interp = RegularGridInterpolator(
        (lat, lon), frac, method="linear", bounds_error=False, fill_value=0.0
    )

    def land_frac(qlat, qlon) -> np.ndarray:
        qlat = np.asarray(qlat, dtype=float)
        qlon = np.asarray(qlon, dtype=float)
        pts = np.stack([qlat.ravel(), qlon.ravel()], axis=-1)
        out = interp(pts).reshape(qlat.shape)
        return np.clip(out, 0.0, 1.0)

    return land_frac
