"""NOAA XML front analyses -> Biard & Kunkel (CODSUS) format, plus drylines.

The reverse cross-format: the FrontFinder label source (NOAA 3-hourly XML
analyses, which unlike CODSUS carry DRY_LINE objects) rasterized with the
reverse-engineered CODSUS recipe (:mod:`codsus_regen`):

- stroke rule -- a MERRA2 1-degree cell is marked when its center lies within
  ``width/2`` degrees of the polyline (round caps, both cells at exact ties);
- one type per cell, priority ``warm > occluded > stationary > cold`` with
  **dryline lowest** (a dryline never displaces a front);
- forming/dissipating XML variants fold into their base types; troughs,
  tropical troughs and instability lines are dropped;
- hard-zeroed outside the WPC mask (``codsus_merra2-1deg_mask.nc``);
- yearly files in the published schema -- ``fronts(time, front, lat, lon)``,
  3-hourly time axis, missing analyses -> NaN timesteps -- with front_type
  ``[cold, warm, stationary, occluded, dryline, none]`` (one extra channel vs
  the published files; the suite loader indexes channels by name, so the
  files remain loader-compatible).

CLI::

    python -m front_formats.xml_to_codsus 2016 2017
    python -m front_formats.xml_to_codsus all       # 2006-2022
"""

from __future__ import annotations

import glob
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codsus_regen as cr  # noqa: E402
from convection_skill import config  # noqa: E402

# LOCAL TOOLING (manifest reorg 2026-08-13): the raw NOAA XML archive lives
# only on this machine (not in the cluster manifest); OUT_DIR is a staging
# area -- finished years are merged into the canonical
# front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/{w}wide tree that
# front_finder.labels.load_noaa reads.


def _resolve_xml_dir():
    """Root of the raw NOAA XML analyses, whichever layout is on this disk.

    The archive moved under ``front_id/raw_met_drawn_fronts/`` in a later
    reorg (found 2026-08-18 while regenerating the labels after the dev data
    mount was lost), but older checkouts still have it at the flat
    ``front_id/NOAA_USA_fronts``.  Probe the new location first and fall back,
    so a rebuild works on either -- and return the new path when NEITHER
    exists so the error names the canonical location.  ``JPL_AIRS_XML_DIR``
    overrides both for an archive parked somewhere else entirely.
    """
    env = os.environ.get("JPL_AIRS_XML_DIR", "")
    if env:
        return Path(env)
    root = config.DATA_DIR / "front_id"
    new = root / "raw_met_drawn_fronts" / "NOAA_USA_fronts"
    old = root / "NOAA_USA_fronts"
    return old if (old.is_dir() and not new.is_dir()) else new


XML_DIR = _resolve_xml_dir()
OUT_DIR = config.DATA_DIR / "front_id" / "NOAA_to_CODSUS_staging"
FILE_TEMPLATE = "noaa_fronts_merra2-1deg_{width}wide_{year}.nc"

TYPES = ("cold", "warm", "stationary", "occluded", "dryline")
#: overlap priority: the CODSUS ordering, dryline lowest.
PRIORITY = ("warm", "occluded", "stationary", "cold", "dryline")
PGEN_TO_TYPE = {
    "COLD_FRONT": "cold", "COLD_FRONT_FORM": "cold", "COLD_FRONT_DISS": "cold",
    "WARM_FRONT": "warm", "WARM_FRONT_FORM": "warm", "WARM_FRONT_DISS": "warm",
    "STATIONARY_FRONT": "stationary", "STATIONARY_FRONT_FORM": "stationary",
    "STATIONARY_FRONT_DISS": "stationary",
    "OCCLUDED_FRONT": "occluded", "OCCLUDED_FRONT_FORM": "occluded",
    "OCCLUDED_FRONT_DISS": "occluded",
    "DRY_LINE": "dryline",
}  # TROF / TROPICAL_TROF / INSTABILITY intentionally absent


#: Longitude of the grid centre, used to pick a polyline's 360-degree branch.
_GRID_LON_MID = 0.5 * (cr.GRID_LONS[0] + cr.GRID_LONS[-1])


def unwrap_lon(lon: np.ndarray) -> np.ndarray:
    """Put a polyline's longitudes on ONE continuous 360-degree branch.

    The XML stores every point in [-180, 180], so a front that crosses the
    antimeridian jumps (e.g. -178.1 -> +174.5).  :func:`cr.rasterize_polyline`
    strokes in plain lat/lon space with no wrap awareness, so it reads that
    jump as a 352-degree-long segment and paints a dead-straight horizontal
    line right across the grid -- the bar that showed up at 33N spanning
    113W-66W in the 2017-07-25 00Z labels, well inside the analysis domain.

    Unwrapping (each step taken as the shorter of the two ways round) makes
    the crossing segment run OFF the grid instead, where the rasterizer's
    bounding-box clip drops it.  The whole polyline is then shifted by the
    multiple of 360 that lands it nearest the grid, so a front approaching
    from the far side of the antimeridian (first point at +178, rest in the
    grid) still gets drawn rather than being pushed off it.
    """
    lon = np.asarray(lon, dtype=float)
    out = np.empty_like(lon)
    out[0] = lon[0]
    out[1:] = lon[0] + np.cumsum((np.diff(lon) + 180.0) % 360.0 - 180.0)
    return out - 360.0 * np.round((out.mean() - _GRID_LON_MID) / 360.0)


def split_valid(pts: np.ndarray) -> list[np.ndarray]:
    """Split a polyline at out-of-range points, dropping them.

    A handful of analyses carry a missing-value sentinel as a Point (e.g.
    ``Lat="-9999" Lon="10359"`` at 2014-12-14 18Z).  Kept, it drags a segment
    right off the map -- a wild diagonal before the longitude unwrap and a
    grid-height vertical stripe after it.  The points on either side belong to
    the same front but not to the same stroke, so the line is CUT there rather
    than joined across the gap.
    """
    ok = (np.abs(pts[:, 0]) <= 90.0) & (np.abs(pts[:, 1]) <= 180.0)
    return [run for run in np.split(pts, np.nonzero(np.diff(ok))[0] + 1)
            if np.abs(run[0, 0]) <= 90.0 and np.abs(run[0, 1]) <= 180.0]


def parse_xml(xml_path: str) -> list[tuple[str, np.ndarray]]:
    """One XML analysis -> [(type, polyline (n,2) lat/lon), ...]."""
    root = ET.parse(xml_path, parser=ET.XMLParser(encoding="utf-8")).getroot()
    fronts = []
    for line in root.iter("Line"):
        ftype = PGEN_TO_TYPE.get(line.get("pgenType"))
        if ftype is None:
            continue
        pts = np.array([(float(p.get("Lat")), float(p.get("Lon")))
                        for p in line.iter("Point")])
        if not len(pts):
            continue
        for run in split_valid(pts):
            run = run.copy()
            run[:, 1] = unwrap_lon(run[:, 1])
            fronts.append((ftype, run))
    return fronts


def rasterize_analysis(fronts: list, width: int,
                       mask: np.ndarray) -> np.ndarray:
    """(type, lat, lon) binary channels, CODSUS stroke rule + priority + mask."""
    grids = np.zeros((len(TYPES), len(cr.GRID_LATS), len(cr.GRID_LONS)),
                     dtype=np.float32)
    for ftype, pts in fronts:
        cr.rasterize_polyline(pts, grids[TYPES.index(ftype)], width)
    grids *= mask
    claimed = np.zeros(grids.shape[1:], dtype=bool)
    for ftype in PRIORITY:
        ch = TYPES.index(ftype)
        grids[ch][claimed] = 0.0
        claimed |= grids[ch] > 0
    return grids


def build_year(year: int, widths: tuple[int, ...] = (1, 3)) -> None:
    files = {}
    # the archive is nested year/month/day (it was flat when this first
    # ran); match both layouts so a rebuild finds every analysis
    pattern = str(XML_DIR / "**" / f"pres*_{year}*f000.xml")
    for path in glob.glob(pattern, recursive=True):
        stamp = os.path.basename(path).split("_")[-1].split(".")[0].split("f")[0]
        files[pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} "
                           f"{stamp[8:10]}:00")] = path

    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 21:00", freq="3h")
    with xr.open_dataset(cr.MASK_PATH) as m:
        mask = (m["codsus_mask"].values > 0).astype(np.float32)
    print(f"{year}: {sum(t in files for t in times)}/{len(times)} "
          f"XML analyses found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for width in widths:
        data = np.full((len(times), len(TYPES) + 1,
                        len(cr.GRID_LATS), len(cr.GRID_LONS)), np.nan,
                       dtype=np.float32)
        for k, t in enumerate(times):
            path = files.get(t)
            if path is None:
                continue
            try:
                fronts = parse_xml(path)
            except ET.ParseError:
                continue  # unreadable analysis -> NaN timestep
            grids = rasterize_analysis(fronts, width, mask)
            data[k, :len(TYPES)] = grids
            data[k, len(TYPES)] = (1.0 - grids.max(axis=0)) * mask
        ds = xr.Dataset(
            {"fronts": (("time", "front", "lat", "lon"), data)},
            coords={"time": times, "lat": cr.GRID_LATS, "lon": cr.GRID_LONS,
                    "front_type": ("front",
                                   np.array(list(TYPES) + ["none"],
                                            dtype="<U10"))},
            attrs={"title": ("NOAA XML surface-front analyses (FrontFinder "
                             "label source, incl. drylines) rasterized in the "
                             "Biard & Kunkel CODSUS format "
                             "(front_formats/xml_to_codsus.py)."),
                   "type_priority": " > ".join(PRIORITY)})
        ds["fronts"].attrs = {"long_name": "front line images",
                              "valid_min": 0, "valid_max": 1}
        path = OUT_DIR / FILE_TEMPLATE.format(width=width, year=year)
        ds.to_netcdf(path, encoding={"fronts": {"zlib": True, "complevel": 4}})
        print(f"  wrote {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    years = (list(range(2006, 2023)) if args == ["all"]
             else [int(y) for y in args])
    for y in years:
        build_year(y)
