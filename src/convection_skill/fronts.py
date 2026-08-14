"""CODSUS surface-front flags on the analysis grid.

Source: NCICS "Coded Surface Bulletins" front masks on the MERRA2 1-degree
grid (Biard & Kunkel 2019, https://zenodo.org/records/2651361), 3-hourly
human-analyst front positions in two line widths:

- ``1wide``: the front line itself (one grid cell wide);
- ``3wide``: the line dilated to three cells (a "near a front" neighborhood).

Files cover 2003-2018 only; for years without a file every front column is
emitted as all-NaN so the base-table schema is identical across years.

Two alignment steps, both deliberately simple and interpretable:

1. **Grid**: the front grid is centered on integer degrees, ours on half
   degrees, so each of our cells overlaps exactly four front cells. A flag is
   ON if ANY of the four overlapping front cells is flagged (2x2 max-pool) --
   "a front touches this cell".
2. **Time**: bulletins are 3-hourly (00, 03, ... 21 UTC); each forecast slot
   takes the most recent analysis AT OR BEFORE its hour, so slots 1-3
   (21-23 UTC) use the same-day 21 UTC analysis and slots 4-6 (00-02 UTC,
   next calendar day) use the next day's 00 UTC analysis. These are
   CONCURRENT with the 21-02 UTC target window (Zach 2026-08-05: concurrent
   flags only) -- fronts here are a synoptic-environment covariate like CAPE,
   not a timing-guarded antecedent predictor.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import config

FRONTS_DIR = (config.DATA_DIR / "fronts" / "CODSUS_netCDF_MERRA2_2003-2018"
              / "CODSUS" / "MERRA2")
#: Fallback for years after 2018: files REGENERATED from the raw WPC bulletins
#: in the IEM archive (src/codsus_regen.py; same schema and naming). The
#: published product always wins when both exist. Regenerated lines agree with
#: the published ones at IoU ~0.8 cell-for-cell (the original rasterizer's
#: internals are unpublished); after the 2x2 pooling below the flags agree
#: substantially better -- codsus_regen.validate_against_published quantifies.
REGEN_FRONTS_DIR = config.DATA_DIR / "fronts" / "CODSUS_regen" / "MERRA2"
FRONT_FILE_TEMPLATE: str = "codsus_masked_merra2-1deg_{width}wide_{year}.nc"
FRONT_WIDTHS: tuple[int, ...] = (1, 3)
#: The file's ``none`` channel (no front) is redundant with the complement of
#: ``any`` and is not carried.
FRONT_TYPES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded")
BULLETIN_INTERVAL_H: int = 3


def front_columns() -> tuple[str, ...]:
    """All front column names, e.g. ``front_cold_1w`` ... ``front_any_3w``."""
    return tuple(f"front_{t}_{w}w"
                 for w in FRONT_WIDTHS for t in FRONT_TYPES + ("any",))


def _analysis_offsets(slots: tuple[int, ...]) -> dict[int, int]:
    """Slot -> hours from the row's date to its governing bulletin time.

    Forecast hours below the window's first hour (21 UTC) belong to the NEXT
    calendar day (the 21-02 UTC window straddles midnight), then the hour is
    floored to the preceding 3-hourly bulletin. Slots without a forecast-hour
    mapping (e.g. the overpass slot 0) are omitted and stay NaN.
    """
    slot_to_hour = dict(zip(config.FORECAST_SLOTS, config.FORECAST_HOURS_UTC))
    pivot = config.FORECAST_HOURS_UTC[0]
    out = {}
    for s in slots:
        hour = slot_to_hour.get(int(s))
        if hour is None:
            continue
        hour_from_date = hour if hour >= pivot else hour + 24
        out[int(s)] = (hour_from_date // BULLETIN_INTERVAL_H) * BULLETIN_INTERVAL_H
    return out


def _pool_to_half_degree(fronts: xr.DataArray,
                         lats: np.ndarray, lons: np.ndarray) -> xr.DataArray:
    """2x2 max-pool the integer-centered front grid onto half-degree centers."""
    corners = []
    for dlat in (-0.5, 0.5):
        for dlon in (-0.5, 0.5):
            corner = fronts.sel(lat=lats + dlat, lon=lons + dlon)
            corners.append(corner.assign_coords(lat=lats, lon=lons))
    pooled = corners[0]
    for corner in corners[1:]:
        pooled = np.fmax(pooled, corner)
    return pooled


def _nan_flags(coords: dict) -> xr.Dataset:
    """The all-NaN fallback for years without front files (schema-stable)."""
    shape = tuple(len(v) for v in coords.values())
    nan = np.full(shape, np.nan, dtype=np.float32)
    return xr.Dataset(
        {name: (tuple(coords), nan.copy()) for name in front_columns()},
        coords=coords)


def year_front_flags(
    year: int,
    dates: np.ndarray,
    slots: tuple[int, ...],
    lats: np.ndarray,
    lons: np.ndarray,
) -> xr.Dataset:
    """Front flags for one year on the (date, slot, lat, lon) analysis grid.

    Binary 0/1 as float32; NaN where the governing bulletin is missing (a
    whole year without files, or Dec 31 slots whose 00 UTC bulletin falls in
    the next year's file).
    """
    coords = {"date": dates, "slot": np.asarray(slots), "lat": lats, "lon": lons}
    offsets = _analysis_offsets(tuple(slots))

    out = _nan_flags(coords)
    for width in FRONT_WIDTHS:
        name = FRONT_FILE_TEMPLATE.format(width=width, year=year)
        path = next((d / name for d in (FRONTS_DIR, REGEN_FRONTS_DIR)
                     if (d / name).exists()), None)
        if path is None:
            continue
        wanted = {s: dates + np.timedelta64(off, "h") for s, off in offsets.items()}
        all_times = np.unique(np.concatenate(list(wanted.values())))
        with xr.open_dataset(path) as f:
            types = [str(t) for t in f["front_type"].values]
            fronts = (f["fronts"]
                      .isel(front=[types.index(t) for t in FRONT_TYPES])
                      .reindex(time=all_times)  # missing bulletins -> NaN
                      .sel(lat=slice(lats.min() - 0.5, lats.max() + 0.5),
                           lon=slice(lons.min() - 0.5, lons.max() + 0.5))
                      .load())
        pooled = _pool_to_half_degree(fronts, lats, lons)

        for k, ftype in enumerate(FRONT_TYPES + ("any",)):
            flag = (pooled.max("front") if ftype == "any"
                    else pooled.isel(front=k))
            per_slot = [flag.sel(time=wanted[s]).values if s in wanted
                        else np.full((len(dates), len(lats), len(lons)), np.nan)
                        for s in coords["slot"]]
            stacked = np.stack(per_slot, axis=1).astype(np.float32)
            out[f"front_{ftype}_{width}w"] = (tuple(coords), stacked)
    return out
