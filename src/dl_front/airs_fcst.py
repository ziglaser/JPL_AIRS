"""Read HYSPLIT AIRS-FCST ``fullgrid`` files into DL-FRONT surface channels.

Source schema (audit 2026-08-04, sample day 2019-06-05):
``fullgrid_wrf27km_GOOD_1p00deg_YYYYMMDD_HHMM-HHMM[.nc]`` with dims
(time=7, level=33, lat=28, lon=43): slot 0 = the AIRS overpass (per-pixel
``parceltime``, ~18:53-20:39 UTC spread), slots 1-6 = uniform hourly
forecasts at 21,22,23,00,01,02 UTC; level = 30-hPa bin centers 115..1075
hPa; lat 25.5..52.5, lon -106.5..-64.5 (HALF-degree cell centers, offset
0.5 deg from the integer-degree label grid); fill -9999; ``q`` is MIXING
RATIO in g/kg (MERRA-2's QV2M is specific humidity in kg/kg); ``N`` =
parcels per box (100 % non-fill; N == 0 means unobserved).  There is no
surface-pressure/SLP variable.

The conversion pipeline mirrors the proven ``front_finder.ingest_hysplit``
(fill -> NaN, unit fixes, observed indicator BEFORE any interpolation,
vertical then bilinear horizontal interpolation, embed into the 68 x 141
label grid) but targets the SINGLE near-surface level
``config.AIRS_SURFACE_LEVEL_HPA`` and emits the DL-FRONT channel names
T2M/QV2M/U10M/V10M.

Vertical subtlety: the level axis holds the 30-hPa BIN CENTERS 115..1075
hPa, the bottom three bins (1015/1045/1075 hPa) are always empty and
985 hPa (~10 % observed over the swath) is the deepest real retrieval, so
``config.AIRS_SURFACE_LEVEL_HPA`` defaults to 985 -- an actual bin, where
the linear interp of both fields and indicator is exact.  (A between-bin
target such as 1000 hPa would average the indicator with the always-zero
1015-hPa bin and mathematically cap ``valid_frac`` at 0.5, i.e. at the
``>= OBSERVED_MIN_FRACTION`` threshold itself; audit 2026-08-12.)  Fields
are additionally filled DOWNWARD (each column's deepest finite bin extends
to the bins below) before the linear interp so an off-bin target still
yields the deepest real retrieval instead of NaN; the observed indicator
keeps the pure linear interp, so the ``valid_frac >= OBSERVED_MIN_FRACTION``
cut still requires the retrieval bin itself to be observed around the pixel.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config

FILL = -9999.0
#: A label point is "observed" if the bilinearly interpolated observed
#: indicator reaches this, i.e. at least ~half the surrounding half-degree
#: cells saw parcels.  Aliased from the ``degradation.observed_min_fraction``
#: tunable (configs/dl_front.yaml) so a JPL_DLFRONT_CONFIG override changes
#: the on-the-fly stage-B degradation AND every kriged-cache build together;
#: the tracked default (0.5) equals front_finder.ingest_hysplit's constant
#: (asserted in tests/test_dlfront_model.py).
OBSERVED_MIN_FRACTION = config.OBSERVED_MIN_FRACTION
#: fullgrid variable -> DL-FRONT surface channel it stands in for.
CHANNEL_MAP = {"t": "T2M", "q": "QV2M", "u": "U10M", "v": "V10M"}


# --------------------------------------------------------------------------- #
# File discovery & loading
# --------------------------------------------------------------------------- #

#: One-time-per-process recursive archive scans: {root: {YYYYMMDD: path}}.
#: An AIRS archive is sparse by design, so per-date recursive globbing would
#: walk the whole (NFS) tree once per uncovered date -- thousands of times in
#: a multi-year cache build; the index costs exactly one walk per root.
_FULLGRID_INDEX: dict[Path, dict[str, Path]] = {}


def _archive_index(root: Path) -> dict[str, Path]:
    """{YYYYMMDD: fullgrid path} from ONE recursive scan of ``root``."""
    idx = _FULLGRID_INDEX.get(root)
    if idx is None:
        idx = {}
        for p in sorted(root.rglob("fullgrid_*")):
            m = re.search(r"(\d{8})_\d{4}-\d{4}", p.name)
            if m:
                idx.setdefault(m.group(1), p)      # first hit in sorted order
        _FULLGRID_INDEX[root] = idx
    return idx


def find_fullgrid(date, root=None) -> Path | None:
    """Locate the fullgrid file for one overpass day, or None.

    Tries the JPL archive layout ``root/YYYY/wrf27km_YYYYMMDD/fullgrid_*``
    first, then falls back to a recursive index of the whole tree (built
    once per process; the local demo tree nests
    ``wrf27km_20190605/wrf27km_20190605/``).  Files with or without the
    ``.nc`` suffix are accepted.
    """
    root = Path(root) if root is not None else config.AIRS_FCST_ROOT
    day = f"{pd.Timestamp(date):%Y%m%d}"
    subdir = root / day[:4] / f"wrf27km_{day}"
    hits = sorted(subdir.glob(f"fullgrid_*{day}*")) if subdir.is_dir() else []
    if hits:
        return hits[0]
    return _archive_index(root).get(day)


def load_fullgrid(path) -> xr.Dataset:
    """Open one fullgrid file: drop dup dim, fills -> NaN, q -> kg/kg.

    Same fixes as front_finder.ingest_hysplit.load_fullgrid: the duplicate
    'levels' dim is dropped, -9999 becomes NaN, and q (mixing ratio, g/kg)
    becomes specific humidity in kg/kg to match MERRA-2 QV2M.
    """
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    ds = ds.drop_dims("levels", errors="ignore")   # duplicate coord dim
    ds = ds.where(ds != FILL)
    r = ds["q"] / 1000.0                           # mixing ratio g/kg -> kg/kg
    ds["q"] = r / (1.0 + r)                        # -> specific humidity
    ds["q"].attrs.update(units="kg/kg", long_name="specific_humidity")
    ds.load()
    return ds


# --------------------------------------------------------------------------- #
# Period (time-slot) selection
# --------------------------------------------------------------------------- #

def overpass_midpoint(path) -> pd.Timestamp:
    """Midpoint of the ``..._YYYYMMDD_HHMM-HHMM`` filename window."""
    stem = Path(path).name.replace(".nc", "")
    datepart, window = stem.split("_")[-2], stem.split("_")[-1]
    t0 = pd.Timestamp(f"{datepart} {window.split('-')[0]}")
    t1 = pd.Timestamp(f"{datepart} {window.split('-')[1]}")
    return t0 + (t1 - t0) / 2


def slot_timestamp(ds: xr.Dataset, slot: int) -> pd.Timestamp:
    """Actual time of one slot: nan-median of per-pixel ``parceltime``.

    Slots 1-6 are uniform (every pixel shares the timestamp), so the median
    rounded to the hour IS the forecast time; for slot 0 it lands mid-swath.
    """
    med = float(np.nanmedian(ds["parceltime"].isel(time=slot).values))
    return pd.Timestamp(med, unit="s").round("h")


def _select_slot(ds: xr.Dataset, path, hour: int) -> tuple[int, pd.Timestamp]:
    """Hour in config.AIRS_HOURS -> (time slot, period timestamp).

    FORECAST SLOTS ONLY (user decision 2026-08-15): the returned slot is
    the uniform forecast slot whose parceltime matches ``hour`` (0 means
    next-day 00 UTC).  Slot 0 -- the overpass itself -- is never selected:
    its per-pixel obs times span ~2.6 h across the domain (Aqua 13:30 LT
    ascending), so no single 3-hourly label hour is correct for it, and
    the earlier conventions (always-18Z clamp, review 2026-08-13; then a
    nearest-label 19:30 rule) both left either misregistration or a mixed
    retrieval/forecast input distribution.  Forecast slots are per-pixel
    time-uniform, so every step is exactly label-aligned.  ValueError when
    the file has no forecast slot at ``hour`` (callers skip-with-note).
    """
    for slot in range(1, ds.sizes["time"]):
        t = slot_timestamp(ds, slot)
        if t.hour == hour:
            return slot, t
    raise ValueError(f"{Path(path).name}: no forecast slot at hour {hour}")


# --------------------------------------------------------------------------- #
# Regridding to the label grid
# --------------------------------------------------------------------------- #

def _fill_down(da: xr.DataArray) -> xr.DataArray:
    """Extend each column's deepest finite value downward along ``level``.

    Level increases with pressure, so "down" (toward the surface) is the
    increasing-index direction; columns with no finite value stay NaN.
    Equivalent to a bottleneck-free ``ffill('level')``.
    """
    da = da.transpose("level", ...)
    vals = da.values
    finite = np.isfinite(vals)
    idx = np.where(finite, np.arange(vals.shape[0]).reshape(
        -1, *([1] * (vals.ndim - 1))), 0)
    idx = np.maximum.accumulate(idx, axis=0)
    return da.copy(data=np.take_along_axis(vals, idx, axis=0))


def period_fields(path, hour: int, ds: xr.Dataset | None = None) -> xr.Dataset:
    """One AIRS period -> surface channels on the full 68 x 141 label grid.

    Returns T2M, QV2M, U10M, V10M (float32, physical units, NaN where
    unobserved or off-swath), ``valid_frac`` (float32 in [0, 1], the
    bilinearly interpolated binarized observed indicator = the fraction of
    neighbor weight that saw a retrieval, 0 off-swath) and a scalar ``time``
    coordinate holding the period timestamp.

    ``ds``: an already-:func:`load_fullgrid`-ed dataset for ``path``, so
    multi-hour callers (swath.build_swath_bank reads all three periods of
    every archive day) load each file once instead of once per hour
    (review 2026-08-13); default: load ``path`` here.
    """
    if ds is None:
        ds = load_fullgrid(path)
    slot, when = _select_slot(ds, path, hour)
    one = ds.isel(time=slot)
    # observed indicator BEFORE interpolation so swath geometry survives
    obs = (one["N"].fillna(0) > 0).astype("float64")
    fields = one[list(CHANNEL_MAP)]

    # vertical: fields fill-down + linear, indicator pure linear (docstring
    # at module top explains why they differ), both to the single target.
    # A target ON a bin center is selected exactly: interp evaluates the
    # LEFT interval even at a node, so a NaN in the bin above (e.g. 955 hPa
    # unobserved over an observed 985) would poison the exact-bin value.
    target = float(config.AIRS_SURFACE_LEVEL_HPA)
    fields = fields.map(_fill_down)
    if np.isin(target, fields["level"].values):
        fields = fields.sel(level=target)
        obs = obs.sel(level=target)
    else:
        fields = fields.interp(level=target, method="linear")
        obs = obs.interp(level=target, method="linear")
    # binarized native indicator: the retrieval bin itself must be observed
    # AND carry every channel (a cell with parcels but a missing retrieval
    # for any variable would otherwise enter the average as garbage)
    ind = obs >= OBSERVED_MIN_FRACTION
    for var in fields.data_vars:
        ind = ind & np.isfinite(fields[var])
    ind = ind.astype("float64")

    # horizontal: MASKED bilinear from half-degree centers to integer label
    # points.  A plain bilinear NaN-poisons every label point with even one
    # unobserved neighbor, which would leave NaN fields at pixels whose
    # valid_frac clears the threshold (0.5 = half the neighbor weight);
    # instead the observed neighbors are averaged with their bilinear
    # weights renormalized: value = interp(ind * f) / interp(ind).  Where
    # all neighbors are observed this is the exact bilinear value, and a
    # field value exists wherever valid_frac > 0 by construction.
    lat_t = np.arange(np.ceil(float(one.lat[0])), np.floor(float(one.lat[-1])) + 0.1)
    lon_t = np.arange(np.ceil(float(one.lon[0])), np.floor(float(one.lon[-1])) + 0.1)
    den = ind.interp(lat=lat_t, lon=lon_t, method="linear")
    num = fields.where(ind > 0).fillna(0.0).interp(lat=lat_t, lon=lon_t,
                                                   method="linear")
    fields = num / den.where(den > 0)
    obs = den

    # embed into the full label grid; off-swath = unobserved
    lat_full = np.asarray(config.LABEL_LATS)
    lon_full = np.asarray(config.LABEL_LONS)
    fields = fields.reindex(lat=lat_full, lon=lon_full)
    obs = obs.reindex(lat=lat_full, lon=lon_full).fillna(0.0)

    out = fields.rename(CHANNEL_MAP)
    for var in out.data_vars:
        out[var] = out[var].where(obs >= OBSERVED_MIN_FRACTION).astype("float32")
    out["valid_frac"] = obs.astype("float32")
    out = out.drop_vars("level", errors="ignore")
    return out.assign_coords(time=when)
