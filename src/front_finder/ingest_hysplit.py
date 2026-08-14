"""Ingest HYSPLIT ``fullgrid`` AIRS-FCST files onto the label grid.

Source schema (audit 2026-08-04, one sample day 2019-06-05):
``fullgrid_*_1p00deg_*.nc`` with dims (time=7, level=33, lat=28, lon=43) --
slot 0 = AIRS overpass, slots 1-6 = 21,22,23,00,01,02 UTC; level = 30-hPa bin
centers 115..1075 hPa; lat 25.5..52.5, lon -106.5..-64.5 (HALF-degree cell
centers, offset 0.5 deg from the integer-degree label grid); fill -9999;
``q`` is MIXING RATIO in g/kg (attr "mixing_ratio"), unlike MERRA-2's
specific humidity in kg/kg; ``N`` = parcels per box (0/fill = unobserved).

Pipeline: fill->NaN and unit fixes -> vertical interpolation of t/q/u/v to
``config.TARGET_LEVELS_HPA`` -> bilinear horizontal interpolation to the
integer-degree label points -> embed into the full 68x141 label grid (NaN
off-swath).  Derived thermo channels use the SAME formulas and nominal
level pressures as the MERRA-2 pretraining corpus (consistency beats the
marginal accuracy of per-bin mean parcel pressure).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, derive

FILL = -9999.0
#: Interpolation to a label point uses the 4 surrounding half-degree cells;
#: the point is "observed" if the interpolated observed-indicator exceeds
#: this (i.e. at least ~half the surrounding cells saw parcels).
OBSERVED_MIN_FRACTION = 0.5


def load_fullgrid(path) -> xr.Dataset:
    """Open one fullgrid file: fills -> NaN, q -> specific humidity kg/kg."""
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    ds = ds.drop_dims("levels", errors="ignore")   # duplicate coord dim
    ds = ds.where(ds != FILL)
    r = ds["q"] / 1000.0                           # mixing ratio g/kg -> kg/kg
    ds["q"] = r / (1.0 + r)                        # -> specific humidity
    ds["q"].attrs.update(units="kg/kg", long_name="specific_humidity")
    ds.load()
    return ds


def overpass_time(path) -> pd.Timestamp:
    """Mid-time of the overpass window encoded in the filename.

    ``fullgrid_..._YYYYMMDD_HHMM-HHMM.nc`` -> midpoint of the window (the
    per-pixel ``parceltime`` is available for finer pairing later).
    """
    stem = Path(path).stem
    datepart, window = stem.split("_")[-2], stem.split("_")[-1]
    t0 = pd.Timestamp(f"{datepart} {window.split('-')[0]}")
    t1 = pd.Timestamp(f"{datepart} {window.split('-')[1]}")
    return t0 + (t1 - t0) / 2


def nearest_bulletin(t: pd.Timestamp) -> pd.Timestamp:
    """Nearest 3-hourly CSB bulletin time (workplan 3.7: within +-1.5 h)."""
    return t.round("3h")


def to_label_grid(ds: xr.Dataset, slot: int = 0,
                  winds: bool = True) -> xr.Dataset:
    """One time slot -> channel dataset on the (lat 68, lon 141) label grid.

    Returns config.THERMO_VARS [+ u, v] with dims (lat, lon, lev) plus
    ``observed`` (bool, any-level) and ``valid_fraction`` (per level, the
    interpolated observed-indicator in [0, 1]).
    """
    one = ds.isel(time=slot)
    fields = one[["t", "q", "u", "v"] if winds else ["t", "q"]]
    # observed indicator BEFORE interpolation so swath geometry survives
    obs = (one["N"].fillna(0) > 0).astype("float64")

    # vertical: linear in level (nominal bin centers, hPa) to the target set
    fields = fields.interp(level=list(config.TARGET_LEVELS_HPA),
                           method="linear")
    obs = obs.interp(level=list(config.TARGET_LEVELS_HPA), method="linear")

    # horizontal: bilinear from half-degree centers to integer label points
    lat_t = np.arange(np.ceil(float(one.lat[0])), np.floor(float(one.lat[-1])) + 0.1)
    lon_t = np.arange(np.ceil(float(one.lon[0])), np.floor(float(one.lon[-1])) + 0.1)
    fields = fields.interp(lat=lat_t, lon=lon_t, method="linear")
    obs = obs.interp(lat=lat_t, lon=lon_t, method="linear")

    # embed into the full label grid
    lat_full = np.arange(10.0, 77.1, 1.0)
    lon_full = np.arange(-171.0, -30.9, 1.0)
    fields = fields.reindex(lat=lat_full, lon=lon_full)
    obs = obs.reindex(lat=lat_full, lon=lon_full).fillna(0.0)

    p_pa = (fields["level"] * 100.0).broadcast_like(fields["t"])
    ch = derive.thermo_channels(fields["t"], fields["q"], p_pa)
    if winds:
        ch["u"], ch["v"] = fields["u"], fields["v"]
    ch = ch.rename({"level": "lev"})
    obs = obs.rename({"level": "lev"})
    ch["valid_fraction"] = obs
    ch["observed"] = (obs >= OBSERVED_MIN_FRACTION).any("lev")
    # cells "observed" but NaN after interpolation (edge effects) and cells
    # unobserved: both are imputation territory; keep NaN for make_x's mask
    for var in config.THERMO_VARS + (config.WIND_VARS if winds else ()):
        ch[var] = ch[var].where(obs >= OBSERVED_MIN_FRACTION)
    return ch.transpose("lat", "lon", "lev", missing_dims="ignore")
