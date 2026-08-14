"""Load and prepare front labels (CODSUS) and the published benchmark fronts.

Both archives share one schema (audit 2026-08-04): ``fronts(time, front, lat,
lon)`` ubyte {0, 1, 2=fill} with ``front_type = [cold, warm, stationary,
occluded, none]`` on the 1 deg MERRA-2 grid.  CODSUS omits missing bulletins
from its time axis; the benchmark axis is complete -> pair with an inner join.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

from . import config


def _open_fronts(path) -> xr.Dataset:
    """Open one label/benchmark file, dropping the giant ``history`` attr."""
    ds = xr.open_dataset(path)
    ds.attrs.pop("history", None)
    return ds


def load_codsus(year: int, width: int = 1, masked: bool = False) -> xr.Dataset:
    """One year of analyst-drawn CSB fronts (3-hourly, gaps omitted).

    Only the ``codsus_masked_*`` files (LABEL_FILL outside the >40-crossings
    region mask) survived the 2026-08-11 data reorg, so both ``masked``
    settings open the same file.  The flag is kept for call-site
    compatibility: every consumer either wants masked labels anyway or
    applies the region mask downstream as a loss/metric weight, where
    off-mask fill and off-mask zeros are indistinguishable.
    """
    # manifest reorg 2026-08-13: WPC subdirs are now {w}wide, not 1deg_{w}wide
    path = (config.CODSUS_DIR / f"{width}wide"
            / f"codsus_masked_merra2-1deg_{width}wide_{year}.nc")
    return _open_fronts(path)


def load_noaa(year: int, width: int = 1, masked: bool = False) -> xr.Dataset:
    """One year of NOAA-XML analyst fronts in the CODSUS schema (has dryline).

    The rasterizer writes NaN (not the CODSUS byte 2) outside its domain;
    converted to ``LABEL_FILL`` here so :func:`valid_mask` and
    :func:`front_stack` see identical semantics for both sources.  There is
    no pre-masked file variant: ``masked=True`` applies the CODSUS
    >40-crossings region mask as fill, which is exactly what the
    ``codsus_masked_*`` files encode.
    """
    path = (config.NOAA_LABELS_DIR / f"{width}wide"
            / f"noaa_fronts_merra2-1deg_{width}wide_{year}.nc")
    ds = _open_fronts(path)
    fr = ds["fronts"].values
    fr = np.where(np.isnan(fr), config.LABEL_FILL, fr).astype(np.uint8)
    if masked:
        with xr.open_dataset(config.REGION_MASK_PATH) as mask_ds:
            inside = mask_ds["codsus_mask"].values == 1
        fr[:, :, ~inside] = config.LABEL_FILL
    ds["fronts"] = (ds["fronts"].dims, fr)
    return ds


def load_fronts(year: int, width: int = 1, masked: bool = False) -> xr.Dataset:
    """The year's labels from the configured source (``config.LABEL_SOURCE``).

    Every training/evaluation label load goes through here so that
    ``JPL_FRONT_LABELS=noaa`` switches the whole pipeline -- including the
    dryline class -- in one place.
    """
    loader = load_noaa if config.LABEL_SOURCE == "noaa" else load_codsus
    return loader(year, width=width, masked=masked)


def load_benchmark(year: int, width: int = 1, freq: str = "3hr") -> xr.Dataset:
    """One year of the authors' published model fronts (hourly or 3-hourly).

    Layout (unchanged inner tree, manifest reorg 2026-08-13):
    ``predicted_fronts/bk19/1deg_{width}wide/{1hr,3hr}/``; the hourly files
    carry no frequency tag in their name, the 3-hourly do.
    """
    hourly = freq == "hourly"
    path = (config.BENCHMARK_DIR / f"1deg_{width}wide"
            / ("1hr" if hourly else freq)
            / f"merra2_merra2-1deg_{width}wide{'' if hourly else f'_{freq}'}"
              f"_{year}.nc")
    return _open_fronts(path)


def front_stack(ds: xr.Dataset) -> xr.DataArray:
    """``fronts`` restricted to the four physical classes, ordered per config.

    Returns a boolean DataArray (time, front, lat, lon); fill pixels (2) are
    False here -- validity is carried separately by :func:`valid_mask`.
    """
    front_type = [str(s) for s in ds["front_type"].values]
    idx = [front_type.index(name) for name in config.FRONT_TYPES]
    fronts = ds["fronts"].isel(front=idx)
    fronts = fronts.assign_coords(front=list(config.FRONT_TYPES))
    return fronts == 1


def valid_mask(ds: xr.Dataset) -> xr.DataArray:
    """True where the label is defined (not the fill value 2), any class."""
    return (ds["fronts"] != config.LABEL_FILL).all("front")


def align_times(truth: xr.Dataset, pred: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Inner-join the two archives on time (CODSUS has omitted bulletins)."""
    common = np.intersect1d(truth["time"].values, pred["time"].values)
    return truth.sel(time=common), pred.sel(time=common)


def dilate(binary: np.ndarray, iterations: int) -> np.ndarray:
    """8-connected dilation of the trailing (lat, lon) axes.

    Mirrors ``fronts/utils/data_utils.py::expand_fronts`` (one pixel per
    iteration, queen's-move neighborhood), vectorized over leading axes.
    """
    if iterations == 0:
        return binary
    struct = np.ones((3, 3), dtype=bool)
    full = struct[(np.newaxis,) * (binary.ndim - 2)]   # no dilation across time
    return ndimage.binary_dilation(binary, structure=full, iterations=iterations)
