"""Analyst-drawn (and model-predicted) surface-front flags on the analysis grid.

Source: NCICS "Coded Surface Bulletins" front masks on the MERRA2 1-degree
grid (Biard & Kunkel 2019, https://zenodo.org/records/2651361), 3-hourly
human-analyst front positions in two line widths:

- ``1wide``: the front line itself (one grid cell wide);
- ``3wide``: the line dilated to three cells (a "near a front" neighborhood).

The WPC/CODSUS tree covers 2003-2021 (2019-2021 regenerated from the raw WPC
bulletins in the IEM archive by ``src/codsus_regen.py``); the NOAA-XML
re-rasterization (``src/front_formats/xml_to_codsus.py``) covers 2007-2022 and
is the only source with DRYLINES.  For years without a file every front column
is emitted as all-NaN so the base-table schema is identical across years.

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

:func:`year_front_flags` is the in-memory base-table entry point (unchanged
behaviour and signature).  :func:`file_front_flags` is the same two alignment
steps factored out so that ``scripts/add_front_flags.py`` can write flags for
ANY file in this schema -- both label sources, both widths, and our own
bk19-schema model predictions -- into the FCST_SMAP_MRMS year files and be
bit-for-bit identical to what the base table computes (deliverable 2,
2026-08-18).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# front_finder.config owns the canonical label-tree constants and honours
# $JPL_AIRS_DATA, so both packages resolve ONE tree on the cluster.
from front_finder import config as ff_config

from . import config

#: PATH FIX 2026-08-18 (not cosmetic): these pointed at
#: ``DATA_DIR/fronts/CODSUS_netCDF_MERRA2_2003-2018/CODSUS/MERRA2`` and
#: ``DATA_DIR/fronts/CODSUS_regen/MERRA2``, both DELETED by the 2026-08-13
#: manifest reorg.  :func:`year_front_flags` skips a missing file with a bare
#: ``continue``, so every front column silently came out ALL-NaN -- i.e. every
#: cached table at DATASET_VERSION=9, the ``front`` stratifier and every
#: F1-F5 result produced since the reorg are void, not merely stale.
#: The year files now live in ``{width}wide/`` subdirectories.
FRONTS_DIR = ff_config.CODSUS_DIR
#: The regenerated 2019-2021 years were merged INTO the published WPC tree by
#: the reorg (same schema and naming), so there is no longer a second
#: directory to search; the name is kept because the search-list below and
#: outside callers still refer to it.  Regenerated lines agree with the
#: published ones at IoU ~0.8 cell-for-cell (the original rasterizer's
#: internals are unpublished); after the 2x2 pooling below the flags agree
#: substantially better -- codsus_regen.validate_against_published quantifies.
REGEN_FRONTS_DIR = ff_config.CODSUS_DIR
#: NOAA-XML analyses re-rasterized into the same schema (dateline bug fixed
#: 2026-08-17); the only source carrying the dryline class.
NOAA_FRONTS_DIR = ff_config.NOAA_LABELS_DIR
FRONT_FILE_TEMPLATE: str = "codsus_masked_merra2-1deg_{width}wide_{year}.nc"
NOAA_FILE_TEMPLATE: str = "noaa_fronts_merra2-1deg_{width}wide_{year}.nc"
#: Model predictions are published in the bk19 schema and basename
#: (``predicted_fronts/<tag>/1deg_{w}wide/3hr/``, deliverable 1, 2026-08-18);
#: mirrors ``front_finder.labels.load_benchmark`` so any bk19-schema tree --
#: including bk19 itself -- can be read with the same reader path.
PRED_FILE_TEMPLATE: str = "merra2_merra2-1deg_{width}wide_3hr_{year}.nc"
FRONT_WIDTHS: tuple[int, ...] = (1, 3)
#: The file's ``none`` channel (no front) is redundant with the complement of
#: ``any`` and is not carried.
FRONT_TYPES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded")
#: Present in the NOAA-XML source only, so it is NOT part of the base table's
#: column set (which must be schema-stable across sources and years); callers
#: that want it ask for it explicitly (see :func:`file_front_types`).
DRYLINE_TYPE: str = "dryline"
BULLETIN_INTERVAL_H: int = 3
#: Met-drawn label source -> basename template (see :func:`label_path`).
LABEL_FILE_TEMPLATES: dict[str, str] = {
    "wpc": FRONT_FILE_TEMPLATE,
    "noaa": NOAA_FILE_TEMPLATE,
}


def front_columns() -> tuple[str, ...]:
    """All front column names, e.g. ``front_cold_1w`` ... ``front_any_3w``."""
    return tuple(f"front_{t}_{w}w"
                 for w in FRONT_WIDTHS for t in FRONT_TYPES + ("any",))


def label_search_dirs(source: str, width: int) -> tuple[Path, ...]:
    """Directories searched for one label year file, in priority order.

    Post-reorg the year files live in ``{width}wide/`` subdirectories; the bare
    root is kept in the list so a flat tree (the pre-2026-08-13 backups, the
    synthetic trees in ``tests/``) still resolves.  The module-level constants
    are read at CALL time so a caller can repoint them.
    """
    if source not in LABEL_FILE_TEMPLATES:
        raise ValueError(f"label source must be one of "
                         f"{sorted(LABEL_FILE_TEMPLATES)}, got {source!r}")
    root = FRONTS_DIR if source == "wpc" else NOAA_FRONTS_DIR
    return (root / f"{width}wide", root)


def label_path(source: str, width: int, year: int) -> Path | None:
    """Path to one met-drawn label year file, or None if it does not exist."""
    name = LABEL_FILE_TEMPLATES[source].format(width=width, year=year)
    return next((d / name for d in label_search_dirs(source, width)
                 if (d / name).exists()), None)


def prediction_path(pred_dir: Path, width: int, year: int) -> Path | None:
    """Path to one year inside a bk19-schema prediction tree, or None.

    The contract is the DIRECTORY (``.../<tag>/``), not any producer's
    internals: the ``1deg_{w}wide/3hr/`` layout and the basename are fixed by
    the published bk19 product and hard-coded identically in
    ``front_finder.labels.load_benchmark`` and ``dl_front.evaluate_test``.
    """
    path = (Path(pred_dir) / f"1deg_{width}wide" / "3hr"
            / PRED_FILE_TEMPLATE.format(width=width, year=year))
    return path if path.exists() else None


def file_front_types(path: Path) -> tuple[str, ...]:
    """The physical front types this file carries, in canonical order.

    ``none`` is dropped (redundant with the complement of ``any``) and
    ``dryline`` is included only when the file has it, so one call site serves
    the 4-class WPC files, the 5+none NOAA files and our 6-channel model
    predictions.
    """
    with xr.open_dataset(path) as f:
        present = {str(t) for t in f["front_type"].values}
    return tuple(t for t in FRONT_TYPES + (DRYLINE_TYPE,) if t in present)


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


def file_front_flags(
    path: Path,
    types: tuple[str, ...],
    dates: np.ndarray,
    slots: tuple[int, ...],
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict[str, np.ndarray]:
    """Both alignment steps for ONE front file -> ``{type: (date, slot, lat, lon)}``.

    Keys are ``types`` plus ``"any"`` (the max over the requested types).
    Values are binary 0/1 float32 with NaN wherever the governing bulletin is
    unavailable: the slots with no forecast hour (slot 0), dates whose
    bulletin is absent from the file (missing analyses, and Dec 31 slots 4-6
    whose 00 UTC bulletin lives in the next year's file), and -- for model
    predictions -- cells outside the trained domain, which the bk19 schema
    stores as the fill byte 2 and ``xr.open_dataset`` decodes to NaN.  The
    ``np.fmax`` pool ignores NaN, so a cell is NaN only where all four
    overlapping 1-degree cells are.
    """
    offsets = _analysis_offsets(tuple(slots))
    wanted = {s: dates + np.timedelta64(off, "h") for s, off in offsets.items()}
    all_times = np.unique(np.concatenate(list(wanted.values())))
    with xr.open_dataset(path) as f:
        names = [str(t) for t in f["front_type"].values]
        missing = [t for t in types if t not in names]
        if missing:
            raise ValueError(f"{path.name} has no {missing} channel "
                             f"(front_type = {names})")
        # Crop and load BEFORE reindexing: these files are chunked with ~1000
        # time steps per chunk (NOAA: [976, 1, 23, 47]), so pulling ~730
        # scattered bulletin times straight out of the file re-decompresses the
        # same chunks hundreds of times (90 s -> ~2 s for one year, 2026-08-18).
        # Reindexing the in-memory slab is numerically identical.
        fronts = (f["fronts"]
                  .isel(front=[names.index(t) for t in types])
                  .sel(lat=slice(lats.min() - 0.5, lats.max() + 0.5),
                       lon=slice(lons.min() - 0.5, lons.max() + 0.5))
                  .load()
                  .reindex(time=all_times))  # missing bulletins -> NaN
    pooled = _pool_to_half_degree(fronts, lats, lons)

    empty = np.full((len(dates), len(lats), len(lons)), np.nan)
    out = {}
    for k, ftype in enumerate(types + ("any",)):
        flag = pooled.max("front") if ftype == "any" else pooled.isel(front=k)
        per_slot = [flag.sel(time=wanted[s]).values if s in wanted else empty
                    for s in slots]
        out[ftype] = np.stack(per_slot, axis=1).astype(np.float32)
    return out


def year_front_flags(
    year: int,
    dates: np.ndarray,
    slots: tuple[int, ...],
    lats: np.ndarray,
    lons: np.ndarray,
    source: str = "wpc",
) -> xr.Dataset:
    """Front flags for one year on the (date, slot, lat, lon) analysis grid.

    Binary 0/1 as float32; NaN where the governing bulletin is missing (a
    whole year without files, or Dec 31 slots whose 00 UTC bulletin falls in
    the next year's file).  ``source`` defaults to the WPC/CODSUS labels the
    pre-registered F1-F5 hypotheses were written against.
    """
    coords = {"date": dates, "slot": np.asarray(slots), "lat": lats, "lon": lons}

    out = _nan_flags(coords)
    for width in FRONT_WIDTHS:
        path = label_path(source, width, year)
        if path is None:
            continue
        flags = file_front_flags(path, FRONT_TYPES, dates, tuple(slots),
                                 lats, lons)
        for ftype, values in flags.items():
            out[f"front_{ftype}_{width}w"] = (tuple(coords), values)
    return out
