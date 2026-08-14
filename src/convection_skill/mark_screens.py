"""Mark's reference screening code (email 2026-07-21), VERBATIM.

The functions below are copied unchanged from ``src/email_zach_data_loader.txt``
(@author: markr) -- they are the gold-standard sample definition this pipeline
now follows. Do not edit their bodies; :mod:`convection_skill.dataset` derives
its screen columns from the same expressions, and
``tests/test_mark_screens.py`` asserts exact parity against these originals.

The ONE adaptation our files need lives in :func:`add_gridav`: Mark's file cut
carries ``MRMS_GaugeCorrQPE01H_gridav`` (the 1x1 grid-cell mean QPE); ours does
not, so we reconstruct it from the audited identity

    gridav = _av * _cnt / 81    (wet-area mean x wet fraction; see config QPE note)

and attach it under Mark's variable name so his code runs verbatim.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path

from . import config

#: Mark's screen parameters (make_master_mask defaults in his walkthrough).
Z_THRESH_M: float = 1000.0        # FCST_alt ceiling, ALL 7 hours
QPE_DRY_THRESH: float = 0.001     # dry-start QPE bar, hours 0 and 1
GRIDAV_VAR: str = "MRMS_GaugeCorrQPE01H_gridav"
MUST_BE_VALID: tuple[str, ...] = ("FCST_MU_CAPE", "FCST_MU_CIN", GRIDAV_VAR)


def add_gridav(ds: xr.Dataset) -> xr.Dataset:
    """Attach Mark's ``gridav`` variable, reconstructed from ``_av * _cnt / 81``.

    No-op if the file already carries it (Mark's file versions do; ours don't).
    Our MRMS variables live on the ``nhours`` axis while Mark's code indexes
    ``gridav`` on ``time``; the two 7-slot axes are hour-aligned (slot k = the
    same forecast hour), so the reconstruction is emitted on ``time`` dims.
    """
    if GRIDAV_VAR not in ds:
        gridav = ds[config.QPE_VAR] * (
            ds[config.QPE_CNT_VAR] / config.WET_CELL_MAX_CNT)
        axis = next(d for d in gridav.dims if d in ("time", "nhours"))
        vals = gridav.transpose("date", axis, "lat", "lon").values
        ds[GRIDAV_VAR] = (("date", "time", "lat", "lon"), vals)
    return ds


# =========================================================================== #
# VERBATIM from email_zach_data_loader.txt -- do not edit
# =========================================================================== #
def lsm_ds(lsm_f='../../data/lsm.nc',lats=slice(32,53),lons=slice(-107,-64),
           land_thresh=0.5):
    """
    """

    with xr.open_dataset(lsm_f) as lds:
        land = lds['lsm'].sel(lat=lats,lon=lons) > land_thresh

    return land




def load_dataset(file_paths, lats=slice(32,53), lons=slice(-107,-64)):
    """
    Loads netCDF forecast files without needing dask/open_mfdataset.
    """
    datasets = []
    for p in file_paths:
        # Open single file and drop problem variable
        ds = xr.open_dataset(p, drop_variables=['FCST_parceltime'])

        # Spatial slice if requested
        if lats is not None:
            ds = ds.sel(lat=lats)
        if lons is not None:
            ds = ds.sel(lon=lons)

        datasets.append(ds)

    # Combine manually along date or time dimension
    combined_ds = xr.concat(datasets, dim='date')
    return combined_ds




def make_master_mask(ds, land_mask, z_thresh=1000, qpe_thresh=0.001,
                     must_be_valid=[
                     'FCST_MU_CAPE', 'FCST_MU_CIN','MRMS_GaugeCorrQPE01H_gridav',
                     ],
                     qpe_k='MRMS_GaugeCorrQPE01H_gridav',):
    """
    Processes multi-dimensional weather data to identify valid (date, lat, lon) event sites.

    Constraints applied:
    1. Land-only constraint.
    2. Altitude <= z_thresh for ALL 7 hours.
    3. Dry start: QPE <= qpe_thresh for initial 2 hours (time index 0 and 1).
    4. Data integrity: No missing values across critical predictor variables.

    Returns:
    - master_mask: 3D Boolean DataArray with dimensions (date, lat, lon)
    """

    # 1. Align Land-Sea Mask to Dataset coordinates
    # Ensures grid matching even if floating-point lats/lons differ slightly
    land_aligned = land_mask.reindex_like(ds, method='nearest') > 0

    # 2. Altitude Constraint: lowest parcel must be below z_thresh for ALL 7 hours
    alt_ok = (ds['FCST_alt'] < z_thresh).all(dim='time')

    # 3. Dry Start Constraint: QPE ~ 0 for first 2 hours
    is_dry = (ds[qpe_k].isel(time=[0, 1]) <= qpe_thresh).all(dim='time')

    # 4. Validity Constraint: No NaNs for specific vars across all 7 hours
    valid_data = ds[must_be_valid].to_array().notnull().all(dim=('variable', 'time'))

    # 5. Combine into Master Mask (date, lat, lon)
    master_mask = land_aligned & alt_ok & is_dry & valid_data

    return master_mask




def extract_samples(ds, master_mask, target_hours=slice(1, None)):
    """
    Applies the master mask to extract flattened 2D arrays ready for ML models.

    Parameters:
    - ds: xr.Dataset
    - master_mask: 3D boolean array (date, lat, lon)
    - target_hours: Slice for hours to retain (default: slice(1, None) keeps hours 1 through end)

    Returns:
    - ds_flat: xr.Dataset stacked into (sample, time) where sample = (date, lat, lon)
    """
    # Subset time to hours 1 onwards (keeping initial hours isolated for conditions)
    ds_sub = ds.isel(time=target_hours)

    # Filter dataset using 3D spatial/temporal mask
    ds_masked = ds_sub.where(master_mask)

    # Stack space-time coordinates into a single 'sample' dimension
    # Resulting shape for variables: (sample, time)
    ds_flat = ds_masked.stack(sample=('date', 'lat', 'lon')).dropna(dim='sample', how='all')

    return ds_flat
# =========================================================================== #
# end verbatim block
# =========================================================================== #


def master_mask_components(raw: xr.Dataset) -> xr.Dataset:
    """Mark's mask, one (date, lat, lon) field per constraint so config toggles
    can apply them independently. Each expression is Mark's own line from
    :func:`make_master_mask`; combined with ``&`` they reproduce it exactly
    (parity-tested). ``alt_max`` / ``dry_start_qpe`` are the sufficient
    statistics (``skipna=False``: any NaN hour fails the comparison, exactly
    like Mark's ``.all(dim='time')`` over a NaN comparison).
    """
    ds = add_gridav(raw)
    out = xr.Dataset()
    # sufficient statistics -- threshold at analysis time, cache once
    out["alt_max"] = ds["FCST_alt"].max(dim="time", skipna=False)
    out["dry_start_qpe"] = ds[GRIDAV_VAR].isel(time=[0, 1]).max(
        dim="time", skipna=False)
    # Mark's completeness constraint, verbatim expression
    out["valid7"] = ds[list(MUST_BE_VALID)].to_array().notnull().all(
        dim=("variable", "time"))
    return out
