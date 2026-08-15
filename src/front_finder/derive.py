"""Derived thermodynamic predictors from (T, q, P).

All physics comes from the vendored FrontFinder implementation
(``fronts/utils/variables.py``: P in Pa, q in kg/kg, temperatures in K) so
pretraining and AIRS fine-tuning share one set of formulas.  Derivations are
applied to PROFILE-RESOLUTION data before any level extraction/regridding
decisions downstream (workplan section 2: nonlinear derived variables from
cell-mean T/q would be biased).
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from . import config


def _variables():
    """Lazy import of fronts/utils/variables.py (pulls in tensorflow).

    Deferred so that consumers of this module that never call
    ``thermo_channels`` (e.g. mask_bank -> ingest_hysplit in the numpy-only
    .venv) do not need a TensorFlow install.
    """
    sys.path.insert(0, str(config.FRONTS_REPO))
    from utils import variables as V  # noqa: E402 (fronts/utils/variables.py)
    return V


def thermo_channels(t: xr.DataArray, q: xr.DataArray,
                    p_pa: xr.DataArray) -> xr.Dataset:
    """The 7 thermodynamic channels of workplan section 3.3.

    t [K], q [kg/kg], p_pa [Pa]; any matching dims.  Returns T, q, r, Td,
    theta_e, Tv, RH (names per ``config.THERMO_VARS``).
    """
    # Argument orders/names follow the upstream ai2es/fronts master submodule
    # (2026-08-14 signature migration: dewpoint gained T, mixing ratio and
    # theta_e reordered, relative_humidity_from_dewpoint renamed).
    V = _variables()
    td = xr.apply_ufunc(V.dewpoint_from_specific_humidity, p_pa, t, q)
    r = xr.apply_ufunc(V.mixing_ratio_from_dewpoint, td, p_pa)
    return xr.Dataset({
        "T": t,
        "q": q,
        "r": r,
        "Td": td,
        "theta_e": xr.apply_ufunc(V.equivalent_potential_temperature, t, td, p_pa),
        "Tv": xr.apply_ufunc(V.virtual_temperature_from_mixing_ratio, t, r),
        "RH": xr.apply_ufunc(V.relative_humidity, t, td),
    })


def merra2_channels(day: xr.Dataset, winds: bool = True) -> xr.Dataset:
    """Channel dataset for one MERRA-2 daily label-grid file.

    ``day`` holds T/QV/U/V (time, lev[hPa], lat, lon).  Below-ground points
    (MERRA-2 fills where lev > surface pressure) stay NaN; the dataset layer
    imputes them and zeroes the mask channel.
    """
    p_pa = (day["lev"] * 100.0).broadcast_like(day["T"])
    ds = thermo_channels(day["T"], day["QV"], p_pa)
    if winds:
        ds["u"] = day["U"]
        ds["v"] = day["V"]
    return ds
