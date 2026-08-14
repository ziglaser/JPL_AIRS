"""Regenerate CODSUS-style gridded front masks for years the archive lacks.

The published gridded product
(data/front_id/met_drawn_fronts/WPC_CODSUS/WPC_1deg_gridded,
Biard & Kunkel, https://zenodo.org/records/2651361) ends in 2018, but the raw
NWS/WPC "coded surface bulletins" it was built from are issued to this day and
are archived by the Iowa Environmental Mesonet (IEM). This module rebuilds the
same product for later years:

1. **Fetch** monthly batches of CODSUS bulletins from the IEM AFOS archive
   (cached as text under ``data/front_id/csb_raw/``).
2. **Parse** the 1-degree ("LR") bulletins -- the file history of the
   published product shows it was built from the LR JSON polylines
   (``Codsus_*_LR.json`` inputs to ``jsonPolysToMaskedNetCDF.py -w 1``).
   Points are 4-5 digit groups: 2-digit lat, 2-3 digit lon (degrees west).
3. **Rasterize** each front polyline onto the MERRA2 1-degree grid
   (integer-centered, lat 10..77, lon -171..-31): mark every cell a segment
   passes through (dense sampling + rounding); ``3wide`` = the 1-wide line
   dilated by one cell in all eight directions.
4. **Write** per-year netCDF files with the published product's schema
   (``fronts(time, front, lat, lon)``, types cold/warm/stationary/occluded/
   none, 3-hourly time axis; missing bulletins -> NaN timestep) into
   ``data/front_id/CODSUS_regen/``.

:func:`validate_against_published` measures cell-level agreement against the
published files on an overlap year (2016-2018), so the regenerated years'
fidelity is quantified rather than assumed.

CLI::

    python src/codsus_regen.py fetch 2019 2020 2021   # download bulletins
    python src/codsus_regen.py build 2019 2020 2021   # parse+rasterize+write
    python src/codsus_regen.py validate 2016          # parity vs published
"""

from __future__ import annotations

import re
import sys
import time as _time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from convection_skill import config  # noqa: E402

# LOCAL TOOLING (manifest reorg 2026-08-13): csb_raw/ and the regen staging
# area exist only on this machine, not in the cluster manifest.  The
# regenerated years were merged into the published archive's {w}wide/
# subdirs (now under met_drawn_fronts/WPC_CODSUS/WPC_1deg_gridded); OUT_DIR
# is a staging area so re-runs never overwrite the merged archive.
RAW_DIR = config.DATA_DIR / "front_id" / "csb_raw"
OUT_DIR = config.DATA_DIR / "front_id" / "CODSUS_regen"
PUBLISHED_DIR = (config.DATA_DIR / "front_id" / "met_drawn_fronts"
                 / "WPC_CODSUS" / "WPC_1deg_gridded")
IEM_URL = ("https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
           "?pil=CODSUS&sdate={sdate}&edate={edate}&fmt=text&limit=9999")
#: The published product's analysis-domain mask (ships with the download;
#: 1 = inside the WPC reliable-analysis region). The published files ZERO
#: fronts outside it -- e.g. Pacific segments of bulletins are dropped -- so
#: the regeneration must apply it too or regenerated-only fronts appear in
#: masked regions (caught visually in the 2016 comparison animation).
MASK_PATH = config.DATA_DIR / "masks" / "codsus_merra2-1deg_mask.nc"

#: The published product's grid and front channels.
GRID_LATS = np.arange(10.0, 78.0)     # 68 integer-centered cells
GRID_LONS = np.arange(-171.0, -30.0)  # 141
FRONT_TYPES = ("cold", "warm", "stationary", "occluded")
TYPE_OF_KEYWORD = {"COLD": "cold", "WARM": "warm",
                   "STNRY": "stationary", "OCFNT": "occluded"}
#: Bulletin section keywords that END a front's point list.
SECTION_KEYWORDS = {"HIGHS", "LOWS", "TROF", *TYPE_OF_KEYWORD}
STRENGTHS = {"WK", "MDT", "STG"}


# --------------------------------------------------------------------------- #
# Fetch (IEM AFOS archive, cached monthly)
# --------------------------------------------------------------------------- #
def month_cache_path(year: int, month: int) -> Path:
    return RAW_DIR / f"CODSUS_{year}-{month:02d}.txt"


def fetch_month(year: int, month: int, pause_s: float = 1.0) -> Path:
    """Download one month of CODSUS bulletins from IEM (skip if cached)."""
    path = month_cache_path(year, month)
    if path.exists() and path.stat().st_size > 0:
        return path
    nxt = (year + 1, 1) if month == 12 else (year, month + 1)
    url = IEM_URL.format(sdate=f"{year}-{month:02d}-01",
                         edate=f"{nxt[0]}-{nxt[1]:02d}-01")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        path.write_bytes(resp.read())
    _time.sleep(pause_s)  # be polite to the IEM archive
    return path


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def _parse_point_lr(token: str) -> tuple[float, float]:
    """A 1-degree bulletin point: 2-digit lat + 2-3 digit lon (degrees west)."""
    return float(token[:2]), -float(token[2:])


def parse_product(text: str, year_hint: int) -> dict | None:
    """One bulletin -> {valid, fronts:[(type, [(lat, lon), ...]), ...]} or None.

    Only the 1-degree products are parsed (WMO id ``ASUS01``, or point tokens
    of 4-5 digits); the 0.1-degree ``ASUS02`` products are skipped so we mirror
    the published product's LR source. ``VALID MMDDHHZ`` carries no year --
    ``year_hint`` supplies it (with December/January wraparound).
    """
    m = re.search(r"^VALID\s+(\d{2})(\d{2})(\d{2})Z", text, re.MULTILINE)
    if m is None:
        return None
    if re.search(r"^ASUS(\d{2})\s", text, re.MULTILINE):
        if re.search(r"^ASUS02\s", text, re.MULTILINE):
            return None  # high-res product
    month, day, hour = (int(g) for g in m.groups())
    year = year_hint
    # pd.Timestamp keys: datetime64 hashes depend on the time UNIT, so a
    # [m]-unit key silently never matches an [ns] lookup. Timestamps don't.
    valid = pd.Timestamp(year=year, month=month, day=day, hour=hour)

    tokens = text[m.end():].split()
    fronts, current = [], None
    for tok in tokens:
        if tok in TYPE_OF_KEYWORD:
            current = (TYPE_OF_KEYWORD[tok], [])
            fronts.append(current)
        elif tok in SECTION_KEYWORDS or tok in STRENGTHS:
            if tok not in STRENGTHS:
                current = None  # HIGHS/LOWS/TROF section: ignore its points
        elif current is not None and tok.isdigit() and 4 <= len(tok) <= 5:
            current[1].append(_parse_point_lr(tok))
        elif current is not None and tok.isdigit() and len(tok) >= 6:
            return None  # high-res coordinates: not the LR product
        elif current is not None:
            current = None  # prose or anything unexpected ends the section
    fronts = [(t, pts) for t, pts in fronts if pts]
    return {"valid": valid, "fronts": fronts}


def parse_month(path: Path, year: int, month: int) -> dict[pd.Timestamp, dict]:
    """All LR bulletins of one cached month, deduped by valid time.

    Dedup keeps the bulletin with the most coordinate points per valid time
    (see inline note). The year for the ``MMDDHH`` valid stamp wraps at the
    month boundaries of the fetch window.
    """
    out: dict[pd.Timestamp, tuple[int, dict]] = {}
    for chunk in path.read_text(errors="replace").split("\x01"):
        if "CODSUS" not in chunk:
            continue
        year_hint = year
        m = re.search(r"^VALID\s+(\d{2})", chunk, re.MULTILINE)
        if m and month == 1 and int(m.group(1)) == 12:
            year_hint = year - 1
        elif m and month == 12 and int(m.group(1)) == 1:
            year_hint = year + 1
        product = parse_product(chunk, year_hint)
        if product is None:
            continue
        # keep the MOST COMPLETE bulletin per valid time (most points): the
        # archive holds partial retransmissions alongside full analyses (e.g.
        # 2016-01-02 03Z has four products, one with a single OCFNT), and
        # "last wins" was silently keeping partials -- worth +0.02 IoU
        # against the published product on its own.
        n_points = sum(len(pts) for _, pts in product["fronts"])
        prev = out.get(product["valid"])
        if prev is None or n_points > prev[0]:
            out[product["valid"]] = (n_points, product)
    return {v: p for v, (n, p) in out.items()}


# --------------------------------------------------------------------------- #
# Rasterize
# --------------------------------------------------------------------------- #
def rasterize_polyline(points: list[tuple[float, float]],
                       grid: np.ndarray, width: int = 1) -> None:
    """Mark the 1-degree cells along a polyline (in place).

    THE RULE (reverse-engineered against the published 2016 product): a cell
    is marked when its center lies within ``width/2`` degrees of the
    polyline, in plain lat/lon space, with round caps -- i.e. a stroked line
    of width ``width``. The decisive detail is the tie behavior: where the
    line passes exactly mid-way between two cell centers, BOTH cells are
    marked (their tool anti-aliases the stroke and thresholds coverage at
    50%, so exact-boundary cells land inside). Switching to this rule from
    point-sampled rounding took 1-wide IoU 0.855 -> 0.950 on 2016.

    Known residual vs the published files (~0.95/0.87 IoU at 1-/3-wide, with
    the most-complete-bulletin dedup in :func:`parse_month`): front ENDPOINT
    cells sit exactly at the 50% coverage threshold of a flat-capped stroke
    and fall in or out with float jitter in their renderer (66% in, with a
    direction bias; uniform in segment length/orientation) -- no
    deterministic rule can reproduce a threshold tie, and keeping all
    endpoints is optimal in expectation. Rejected variants: point sampling
    (any rounding), exact/integer Bresenham, supercover, spline paths,
    Lambert/NARR-projected geometry (peaks 0.86), high-res bulletin source,
    3x3/plus dilations, flat or reduced caps for 3-wide.
    """
    lat0, lon0 = GRID_LATS[0], GRID_LONS[0]
    r = width / 2.0
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    if len(pts) == 1:
        pts = np.vstack([pts, pts])  # degenerate front: a single r-disc
    for (la1, lo1), (la2, lo2) in zip(pts[:-1], pts[1:]):
        a = np.array([la1 - lat0, lo1 - lon0])
        b = np.array([la2 - lat0, lo2 - lon0])
        i0 = max(int(np.floor(min(a[0], b[0]) - r)), 0)
        i1 = min(int(np.ceil(max(a[0], b[0]) + r)), grid.shape[0] - 1)
        j0 = max(int(np.floor(min(a[1], b[1]) - r)), 0)
        j1 = min(int(np.ceil(max(a[1], b[1]) + r)), grid.shape[1] - 1)
        if i1 < i0 or j1 < j0:
            continue
        I, J = np.mgrid[i0:i1 + 1, j0:j1 + 1]
        P = np.stack([I, J], -1).astype(float)
        d = b - a
        L2 = float((d ** 2).sum())
        t = np.clip(((P - a) @ d) / (L2 if L2 else 1.0), 0.0, 1.0)
        dist = np.sqrt(((P - (a + t[..., None] * d)) ** 2).sum(-1))
        grid[i0:i1 + 1, j0:j1 + 1][dist <= r + 1e-9] = 1


#: The published product is a CATEGORICAL image -- exactly one type per cell
#: (zero multi-type cells in all of 2016, both widths), resolved by this
#: priority where strokes overlap (inferred from the pairwise win matrix on
#: 2016 overlaps; each pair is >99% one-sided, e.g. warm beats cold 1697:5).
TYPE_PRIORITY: tuple[str, ...] = ("warm", "occluded", "stationary", "cold")


def rasterize_product(product: dict, width: int = 1) -> np.ndarray:
    """One bulletin -> (front_type, lat, lon) binary grid, one type per cell."""
    grids = np.zeros((len(FRONT_TYPES), len(GRID_LATS), len(GRID_LONS)),
                     dtype=np.float32)
    for ftype, points in product["fronts"]:
        rasterize_polyline(points, grids[FRONT_TYPES.index(ftype)], width)
    claimed = np.zeros(grids.shape[1:], dtype=bool)
    for ftype in TYPE_PRIORITY:
        ch = FRONT_TYPES.index(ftype)
        grids[ch][claimed] = 0.0
        claimed |= grids[ch] > 0
    return grids


# --------------------------------------------------------------------------- #
# Build one year
# --------------------------------------------------------------------------- #
def build_year(year: int, widths: tuple[int, ...] = (1, 3)) -> list[Path]:
    """Fetch + parse + rasterize one year; write one file per line width.

    The time axis is the complete 3-hourly year; a valid time with no parsed
    bulletin becomes an all-NaN timestep (same missing semantics the analysis
    loader already handles).
    """
    products: dict[pd.Timestamp, dict] = {}
    for month in range(1, 13):
        path = fetch_month(year, month)
        products.update({v: p for v, p in parse_month(path, year, month).items()
                         if v.year == year})

    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 21:00", freq="3h")
    print(f"{year}: {sum(t in products for t in times)}/{len(times)} "
          f"bulletins parsed")

    with xr.open_dataset(MASK_PATH) as m:
        mask = (m["codsus_mask"].values > 0).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for width in widths:
        data = np.full((len(times), len(FRONT_TYPES) + 1,
                        len(GRID_LATS), len(GRID_LONS)), np.nan,
                       dtype=np.float32)
        for k, t in enumerate(times):
            product = products.get(t)
            if product is None:
                continue  # missing bulletin -> all-NaN timestep
            grids = rasterize_product(product, width) * mask
            data[k, :4] = grids
            data[k, 4] = (1.0 - grids.max(axis=0)) * mask  # 'none' = no front
        ds = xr.Dataset(
            {"fronts": (("time", "front", "lat", "lon"), data)},
            coords={"time": times, "lat": GRID_LATS, "lon": GRID_LONS,
                    "front_type": ("front",
                                   np.array(list(FRONT_TYPES) + ["none"],
                                            dtype="<U10"))},
            attrs={"title": ("Coded surface bulletins on a merra2-1deg grid, "
                             "REGENERATED from the IEM CODSUS archive "
                             "(codsus_regen.py) to extend the published "
                             "2003-2018 product."),
                   "source": IEM_URL.split("?")[0]})
        ds["fronts"].attrs = {"long_name": "front line images",
                              "valid_min": 0, "valid_max": 1}
        path = OUT_DIR / f"codsus_masked_merra2-1deg_{width}wide_{year}.nc"
        ds.to_netcdf(path, encoding={"fronts": {"zlib": True, "complevel": 4}})
        written.append(path)
        print(f"  wrote {path}")
    return written


# --------------------------------------------------------------------------- #
# Validation against the published product (overlap years)
# --------------------------------------------------------------------------- #
def validate_against_published(year: int = 2016,
                               widths: tuple[int, ...] = (1, 3)) -> pd.DataFrame:
    """Cell-level agreement of a regenerated year vs the published files."""
    rows = []
    for width in widths:
        ours = xr.open_dataset(
            OUT_DIR / f"codsus_masked_merra2-1deg_{width}wide_{year}.nc")
        pub = xr.open_dataset(
            # manifest reorg 2026-08-13: subdirs are {w}wide, not 1deg_{w}wide
            PUBLISHED_DIR / f"{width}wide"
            / f"codsus_masked_merra2-1deg_{width}wide_{year}.nc")
        common = np.intersect1d(ours.time.values, pub.time.values)
        a = ours.fronts.sel(time=common).values
        types = [str(t) for t in pub.front_type.values]
        b = pub.fronts.sel(time=common).isel(
            front=[types.index(t) for t in FRONT_TYPES + ("none",)]).values
        both = np.isfinite(a) & np.isfinite(b)
        for c, name in enumerate(FRONT_TYPES):
            m = both[:, c]
            av, bv = a[:, c][m] > 0, b[:, c][m] > 0
            inter, union = (av & bv).sum(), (av | bv).sum()
            rows.append({"width": width, "front": name,
                         "agreement": float((av == bv).mean()),
                         "iou": float(inter / union) if union else np.nan,
                         "ours_rate": float(av.mean()),
                         "published_rate": float(bv.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cmd, years = sys.argv[1], [int(y) for y in sys.argv[2:]]
    if cmd == "fetch":
        for y in years:
            for m in range(1, 13):
                print(fetch_month(y, m))
    elif cmd == "build":
        for y in years:
            build_year(y)
    elif cmd == "validate":
        for y in years or [2016]:
            print(validate_against_published(y).to_string(index=False))
    else:
        raise SystemExit(f"unknown command {cmd!r}; use fetch|build|validate")
