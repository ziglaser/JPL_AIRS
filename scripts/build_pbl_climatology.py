#!/usr/bin/env python3
"""Grid the Guo et al. (2024) merged-PBLH model output into 1-degree monthly
diurnal climatologies.

Input  : $JPL_AIRS_DATA/PBL_depth/Guo2024_model/<year>/[<year>/]YYYYMMDDHH.nc
         one file per 3-hourly synoptic time, 0.25 deg global (land-only),
         variable "Merged Planetary Boundary Layer Height" [m], fill -999.
Output : one NetCDF with dims (month=12, hour=8, lat, lon) holding the mean,
         the standard deviation and the sample counts of PBL depth in every
         1 deg cell x calendar month x 3-hourly slot of the day.

Two things this product is for:
  1. a PBLH field to merge into FCST_SMAP_MRMS (1 deg CONUS, cell centres at
     X.5 -- ``--domain conus`` reproduces that grid exactly);
  2. context for the HYSPLIT soil-moisture index -- a parcel's surface contact
     is only meaningful relative to the boundary-layer depth it sits in, so
     ``trajectory_kernels.pbl`` can read this file instead of the analytic
     ``ClimatologicalPBL`` curve.

Aggregation rule (one rule, applied once): every *native* 0.25 deg sample that
passes the validity filter is pooled with equal weight into the 1 deg cell that
contains it -- ``cell = floor(coordinate)``, so cell centres land on X.5 and
each cell collects the (at most) 4x4 native points inside it.  The reported
mean is therefore sum(valid values) / count(valid values); it is a *land* mean,
because the source data is NaN over water (~72% of the globe).

Diurnal reference frame (``--time-ref``):
  utc  -- slot = UTC hour / 3.  Use this when joining to a UTC-indexed product.
  lst  -- slot = round((UTC hour + lon/15) / 3) mod 8, i.e. local *solar* time,
          the frame in which the PBL diurnal cycle is actually in phase.
          The per-column shift is an integer number of 3 h slots (lon/45,
          rounded), so no interpolation is invented.  Default, because a
          UTC-hour diurnal composite is meaningless across a global domain.

Usage
-----
    # global local-solar-time climatology, all five years (~30-60 min)
    python scripts/build_pbl_climatology.py

    # the FCST_SMAP_MRMS CONUS grid, UTC frame
    python scripts/build_pbl_climatology.py --domain conus --time-ref utc

    # smoke test: 200 files, no cache
    python scripts/build_pbl_climatology.py --limit 200 --no-cache \
        --out /tmp/pbl_smoke.nc

Per-year partial sums are cached under ``--cache-dir`` (default
``<out dir>/_cache``) so an interrupted run resumes for free: the accumulators
are additive, so a cached year is simply added back in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# --------------------------------------------------------------------------- #
# Facts about the source corpus (audited 2026-08-18 on the full 14 587 files)
# --------------------------------------------------------------------------- #
#: Variable name in the raw files -- note the spaces.
VAR_NAME = "Merged Planetary Boundary Layer Height"

#: Native grid: 600 lat rows from +90.0 southward, 1440 lon columns from 0.0
#: eastward, both on a 0.25 deg lattice.  The files' *stored* lon axis is
#: ``linspace(0, 360, 1440)`` (step 0.250174), which drifts up to +0.25 deg by
#: the antimeridian -- almost certainly a writing artefact of an intended
#: 0.25 deg grid.  ``--lon-grid uniform`` (default) rebuilds lon = 0.25*j;
#: ``--lon-grid file`` uses the stored values verbatim.
NATIVE_NLAT, NATIVE_NLON = 600, 1440
NATIVE_STEP_DEG = 0.25
NATIVE_LAT_FIRST = 90.0
NATIVE_LON_FIRST = 0.0

#: Missing data is stored as -999.0 (also flagged as _FillValue).
FILL_VALUE = -999.0

#: Physical validity window for a retained sample [m].  The corpus min/max over
#: sampled files is ~21 / ~5550 m, so this filter only removes fill values and
#: any unphysical outlier, never real depths.
PBLH_MIN_M, PBLH_MAX_M = 1.0, 8000.0

#: Time axis: 8 synoptic times per day, 00/03/.../21 UTC.
SLOT_HOURS = 3
N_SLOTS = 24 // SLOT_HOURS
N_MONTHS = 12

#: Files are named YYYYMMDDHH.nc; the stamp is authoritative (verified equal to
#: the in-file ``time`` coordinate on a sample of files).
FILENAME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})\.nc$")

#: Target-grid presets.  Bounds are cell *edges*; centres fall on X.5.
DOMAINS = {
    # everything the source covers: lat +90 .. -59.75, all longitudes
    "global": dict(lat_min=-60.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0),
    # exactly the FCST_SMAP_MRMS grid: lat 25.5..52.5 (28), lon -106.5..-64.5 (43)
    "conus": dict(lat_min=25.0, lat_max=53.0, lon_min=-107.0, lon_max=-64.0),
}

DEFAULT_ROOT = Path(
    os.environ.get("JPL_AIRS_DATA", "/mnt/d/JPL_AIRS/data")
) / "PBL_depth" / "Guo2024_model"


# --------------------------------------------------------------------------- #
# 1. File inventory
# --------------------------------------------------------------------------- #
def find_files(root: Path, years: list[int] | None) -> list[tuple[Path, datetime]]:
    """All ``YYYYMMDDHH.nc`` under *root*, sorted by valid time.

    The corpus nests inconsistently (``2017/*.nc`` but ``2019/2019/*.nc``), so
    the search is recursive and the year comes from the filename, not the path.
    """
    out: list[tuple[Path, datetime]] = []
    for path in root.rglob("*.nc"):
        m = FILENAME_RE.match(path.name)
        if m is None:
            continue
        y, mo, d, h = (int(g) for g in m.groups())
        if years is not None and y not in years:
            continue
        out.append((path, datetime(y, mo, d, h)))
    out.sort(key=lambda p: p[1])
    return out


# --------------------------------------------------------------------------- #
# 2. Grids and the native -> 1 deg index map
# --------------------------------------------------------------------------- #
def native_coords(lon_grid: str, sample_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Native (lat, lon) axes in degrees; lon wrapped to [-180, 180)."""
    lat = NATIVE_LAT_FIRST - NATIVE_STEP_DEG * np.arange(NATIVE_NLAT)
    if lon_grid == "uniform":
        lon = NATIVE_LON_FIRST + NATIVE_STEP_DEG * np.arange(NATIVE_NLON)
    else:
        with Dataset(sample_file) as nc:
            lon = np.asarray(nc.variables["lon"][:], dtype=float)
    return lat, ((lon + 180.0) % 360.0) - 180.0


def target_grid(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    """1 deg cell-centre axes for the requested edge bounds (lat descending)."""
    lat = np.arange(bounds["lat_max"] - 0.5, bounds["lat_min"] - 0.5 + 1e-9, -1.0)
    lon = np.arange(bounds["lon_min"] + 0.5, bounds["lon_max"] - 0.5 + 1e-9, 1.0)
    return np.round(lat, 1), np.round(lon, 1)


def build_index(lat_n, lon_n, lat_t, lon_t):
    """Map each native point to a flat target-cell index.

    Returns ``(keep, flat_idx, n_native)`` where *keep* selects the native
    points that fall inside the target domain, *flat_idx* is their cell index
    (row-major, same length as the number of kept points), and *n_native* is
    the per-cell count of native points -- the denominator for "how much of
    this 1 deg cell does the source grid even cover".
    """
    # cell = floor(coordinate): a point at 25.0..25.75 belongs to centre 25.5
    row = np.searchsorted(-lat_t, -(np.floor(lat_n) + 0.5))
    col = np.searchsorted(lon_t, np.floor(lon_n) + 0.5)
    row_ok = (row < len(lat_t)) & np.isclose(lat_t[np.clip(row, 0, len(lat_t) - 1)],
                                             np.floor(lat_n) + 0.5)
    col_ok = (col < len(lon_t)) & np.isclose(lon_t[np.clip(col, 0, len(lon_t) - 1)],
                                             np.floor(lon_n) + 0.5)
    keep2d = row_ok[:, None] & col_ok[None, :]
    flat2d = row[:, None] * len(lon_t) + col[None, :]
    keep = keep2d.ravel()
    flat_idx = flat2d.ravel()[keep].astype(np.int32)
    n_native = np.bincount(flat_idx, minlength=len(lat_t) * len(lon_t)).astype(np.int32)
    return keep, flat_idx, n_native


def lst_slot_offset(lon_t: np.ndarray) -> np.ndarray:
    """Whole-slot local-solar-time shift per target longitude column.

    local solar hour = UTC hour + lon/15, and one slot is 3 h, so the shift is
    ``lon / 45`` slots, rounded.  Integer by construction: no interpolation.
    Ties (lon = +-22.5, +-67.5, ... -- all of which are 1 deg cell centres) are
    broken half-up rather than with numpy's round-half-to-even, so the mapping
    is the same monotone step everywhere instead of alternating direction.
    """
    return np.floor(lon_t / (15.0 * SLOT_HOURS) + 0.5).astype(int)


# --------------------------------------------------------------------------- #
# 3. Per-file reduction (runs in worker processes)
# --------------------------------------------------------------------------- #
_W: dict = {}


def _init_worker(keep, flat_idx, ncell):
    _W["keep"] = keep
    _W["flat_idx"] = flat_idx
    _W["ncell"] = ncell


def reduce_file(path: Path):
    """(sum, sumsq, count) per target cell for one raw file, or None if unreadable."""
    keep, flat_idx, ncell = _W["keep"], _W["flat_idx"], _W["ncell"]
    try:
        with Dataset(path) as nc:
            var = nc.variables[VAR_NAME]
            var.set_auto_mask(False)
            values = np.asarray(var[0], dtype=np.float32).ravel()[keep]
    except Exception as exc:  # unreadable/truncated file: report, do not abort
        print(f"  !! skipping {path.name}: {exc}", file=sys.stderr)
        return None
    ok = np.isfinite(values) & (values >= PBLH_MIN_M) & (values <= PBLH_MAX_M)
    idx = flat_idx[ok]
    val = values[ok].astype(np.float64)
    return (
        np.bincount(idx, val, minlength=ncell),
        np.bincount(idx, val * val, minlength=ncell),
        np.bincount(idx, minlength=ncell).astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# 4. Accumulation into (month, slot, cell)
# --------------------------------------------------------------------------- #
class Accumulator:
    """Additive (month, slot, cell) sums -- the whole climatology is a sum, so
    years can be computed independently, cached, and added back together."""

    def __init__(self, ncell: int):
        shape = (N_MONTHS, N_SLOTS, ncell)
        self.total = np.zeros(shape, dtype=np.float64)   # sum of values
        self.total_sq = np.zeros(shape, dtype=np.float64)  # sum of squares
        self.n_obs = np.zeros(shape, dtype=np.int64)     # valid native samples
        self.n_times = np.zeros(shape, dtype=np.int32)   # timesteps contributing

    def add_file(self, stamp: datetime, red, col_groups):
        """Fold one file's per-cell (sum, sumsq, count) into the accumulator.

        *col_groups* is ``[(slot_shift, cell_mask), ...]``: for UTC it is a
        single group with shift 0; for local solar time, one group per distinct
        longitudinal slot shift.
        """
        s, sq, c = red
        month = stamp.month - 1
        utc_slot = stamp.hour // SLOT_HOURS
        for shift, mask in col_groups:
            slot = (utc_slot + shift) % N_SLOTS
            self.total[month, slot][mask] += s[mask]
            self.total_sq[month, slot][mask] += sq[mask]
            self.n_obs[month, slot][mask] += c[mask]
            self.n_times[month, slot][mask] += (c[mask] > 0)

    def __iadd__(self, other: "Accumulator"):
        self.total += other.total
        self.total_sq += other.total_sq
        self.n_obs += other.n_obs
        self.n_times += other.n_times
        return self

    def save(self, path: Path):
        np.savez_compressed(path, total=self.total, total_sq=self.total_sq,
                            n_obs=self.n_obs, n_times=self.n_times)

    @classmethod
    def load(cls, path: Path, ncell: int) -> "Accumulator":
        z = np.load(path)
        acc = cls(ncell)
        acc.total, acc.total_sq = z["total"], z["total_sq"]
        acc.n_obs, acc.n_times = z["n_obs"], z["n_times"]
        return acc


def column_groups(ncell: int, nlon: int, offsets: np.ndarray | None):
    """Cell masks sharing one slot shift (UTC: everything shifts by 0)."""
    if offsets is None:
        return [(0, np.ones(ncell, dtype=bool))]
    col_of_cell = np.arange(ncell) % nlon
    groups = []
    for shift in np.unique(offsets):
        groups.append((int(shift), np.isin(col_of_cell, np.nonzero(offsets == shift)[0])))
    return groups


# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def run_year(files, acc: Accumulator, col_groups, jobs: int, index):
    keep, flat_idx, ncell = index
    if jobs <= 1:
        _init_worker(keep, flat_idx, ncell)
        for path, stamp in files:
            red = reduce_file(path)
            if red is not None:
                acc.add_file(stamp, red, col_groups)
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(keep, flat_idx, ncell)) as pool:
        for (path, stamp), red in zip(files, pool.map(reduce_file,
                                                      [f for f, _ in files],
                                                      chunksize=4)):
            if red is not None:
                acc.add_file(stamp, red, col_groups)


def write_output(out_path: Path, acc: Accumulator, lat_t, lon_t, n_native,
                 args, files, missing_note: str):
    n = acc.n_obs.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, acc.total / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, (acc.total_sq - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    shape = (N_MONTHS, N_SLOTS, len(lat_t), len(lon_t))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("month", N_MONTHS)
        nc.createDimension("hour", N_SLOTS)
        nc.createDimension("lat", len(lat_t))
        nc.createDimension("lon", len(lon_t))

        def coord(name, values, **attrs):
            v = nc.createVariable(name, "f4" if values.dtype.kind == "f" else "i4", (name,))
            v[:] = values
            v.setncatts(attrs)

        hour_label = ("local solar" if args.time_ref == "lst" else "UTC")
        coord("month", np.arange(1, 13, dtype=np.int32), long_name="calendar month")
        # deliberately no ``units`` attribute: "hours" would make CF readers
        # decode this label into a timedelta instead of an integer hour
        coord("hour", np.arange(0, 24, SLOT_HOURS, dtype=np.int32),
              long_name=f"{hour_label} hour (0,3,...,21) labelling the 3 h slot",
              time_reference=args.time_ref)
        coord("lat", lat_t.astype("f4"), units="degrees_north",
              long_name="1 deg cell centre latitude")
        coord("lon", lon_t.astype("f4"), units="degrees_east",
              long_name="1 deg cell centre longitude (-180..180)")

        def field(name, data, dtype, fill=None, **attrs):
            # counts carry no fill value (0 == "no data"); a fill value would
            # force CF readers to promote the integer counts to float
            v = nc.createVariable(name, dtype, ("month", "hour", "lat", "lon"),
                                  zlib=True, complevel=4, fill_value=fill)
            v[:] = data.reshape(shape)
            v.setncatts(attrs)

        field("pblh_mean", mean, "f4", np.float32(np.nan), units="m",
              long_name="mean planetary boundary layer depth (land only)",
              cell_methods="lat,lon: mean (0.25 deg native samples) "
                           "time: mean over years within month and hour slot")
        field("pblh_std", std, "f4", np.float32(np.nan), units="m",
              long_name="standard deviation of PBL depth across all pooled samples",
              comment="spread over BOTH the within-cell space samples and the "
                      "day-to-day/interannual samples; not a standard error")
        field("n_obs", acc.n_obs.astype(np.int32), "i4", units="1",
              long_name="number of valid 0.25 deg samples pooled into the mean")
        field("n_times", acc.n_times, "i4", units="1",
              long_name="number of 3-hourly files contributing at least one valid sample")

        v = nc.createVariable("n_native", "i4", ("lat", "lon"), zlib=True)
        v[:] = n_native.reshape(len(lat_t), len(lon_t))
        v.long_name = ("number of 0.25 deg native grid points inside each 1 deg cell "
                       "(16 in the interior; fewer at the domain/data edge)")
        v.comment = ("land fraction of a cell ~ n_obs / (n_times * n_native); the source "
                     "field is undefined over water")

        stamps = [s for _, s in files]
        nc.setncatts(dict(
            title="1 deg monthly diurnal PBL-depth climatology from the Guo et al. "
                  "(2024) merged PBLH model output",
            source=str(args.root),
            source_variable=VAR_NAME,
            source_resolution="0.25 deg, 3-hourly, land only",
            time_reference=args.time_ref,
            domain=args.domain,
            lon_grid=args.lon_grid,
            years=",".join(str(y) for y in sorted({s.year for s in stamps})),
            first_time=min(stamps).isoformat(), last_time=max(stamps).isoformat(),
            n_files=len(files),
            missing_source_times=missing_note,
            validity_filter=f"{PBLH_MIN_M} m <= PBLH <= {PBLH_MAX_M} m "
                            f"(fill value {FILL_VALUE} excluded)",
            aggregation="equal-weight pool of every valid native sample in the cell "
                        "(cell = floor(coordinate), centres on X.5)",
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            command=" ".join(sys.argv),
            history=f"{datetime.now(timezone.utc):%Y-%m-%d} "
                    f"scripts/build_pbl_climatology.py",
        ))
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"raw Guo2024_model tree (default: {DEFAULT_ROOT})")
    p.add_argument("--years", type=str, default=None,
                   help="comma list and/or ranges, e.g. 2017,2019-2021 (default: all)")
    p.add_argument("--domain", choices=sorted(DOMAINS), default="global")
    for edge in ("lat-min", "lat-max", "lon-min", "lon-max"):
        p.add_argument(f"--{edge}", type=float, default=None,
                       help=f"override the preset {edge.replace('-', ' ')} (cell edge)")
    p.add_argument("--time-ref", choices=("lst", "utc"), default="lst",
                   help="diurnal reference frame (default: lst = local solar time)")
    p.add_argument("--lon-grid", choices=("uniform", "file"), default="uniform",
                   help="native lon axis: rebuilt 0.25 deg lattice, or the stored "
                        "(slightly drifting) values")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true", help="do not read or write per-year caches")
    p.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)))
    p.add_argument("--limit", type=int, default=None, help="process only the first N files (smoke test)")
    return p.parse_args(argv)


def parse_years(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    years: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main(argv=None) -> Path:
    args = parse_args(argv)
    bounds = dict(DOMAINS[args.domain])
    for key in ("lat_min", "lat_max", "lon_min", "lon_max"):
        override = getattr(args, key)
        if override is not None:
            bounds[key] = override

    files = find_files(args.root, parse_years(args.years))
    if not files:
        raise SystemExit(f"no YYYYMMDDHH.nc files found under {args.root}")
    if args.limit:
        files = files[: args.limit]
    expected = {y: (366 if y % 4 == 0 else 365) * N_SLOTS
                for y in sorted({s.year for _, s in files})}
    got = {y: sum(1 for _, s in files if s.year == y) for y in expected}
    missing_note = "; ".join(f"{y}: {expected[y] - got[y]} of {expected[y]} times absent"
                             for y in expected if got[y] != expected[y]) or "none"

    lat_n, lon_n = native_coords(args.lon_grid, files[0][0])
    lat_t, lon_t = target_grid(bounds)
    keep, flat_idx, n_native = build_index(lat_n, lon_n, lat_t, lon_t)
    ncell = len(lat_t) * len(lon_t)
    offsets = lst_slot_offset(lon_t) if args.time_ref == "lst" else None
    groups = column_groups(ncell, len(lon_t), offsets)

    tag = f"{args.domain}_{args.time_ref}"
    out_path = args.out or (args.root.parent / "derived" /
                            f"PBL_climatology_1deg_{tag}_"
                            f"{min(expected)}-{max(expected)}.nc")
    cache_dir = args.cache_dir or out_path.parent / "_cache"
    use_cache = not (args.no_cache or args.limit)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{tag}.json").write_text(json.dumps(
            dict(bounds=bounds, time_ref=args.time_ref, lon_grid=args.lon_grid,
                 nlat=len(lat_t), nlon=len(lon_t)), indent=2))

    print(f"{len(files)} files | domain={args.domain} "
          f"({len(lat_t)}x{len(lon_t)} cells) | time_ref={args.time_ref} | "
          f"jobs={args.jobs}\nmissing source times -> {missing_note}\nout: {out_path}")

    acc = Accumulator(ncell)
    for year in sorted(expected):
        year_files = [(f, s) for f, s in files if s.year == year]
        cache_path = cache_dir / f"{tag}_{year}.npz"
        if use_cache and cache_path.exists():
            print(f"  {year}: cached ({len(year_files)} files)")
            acc += Accumulator.load(cache_path, ncell)
            continue
        year_acc = Accumulator(ncell)
        t0 = datetime.now()
        run_year(year_files, year_acc, groups, args.jobs, (keep, flat_idx, ncell))
        print(f"  {year}: {len(year_files)} files in "
              f"{(datetime.now() - t0).total_seconds() / 60:.1f} min")
        if use_cache:
            year_acc.save(cache_path)
        acc += year_acc

    write_output(out_path, acc, lat_t, lon_t, n_native, args, files, missing_note)
    covered = float((acc.n_obs.sum(axis=(0, 1)) > 0).mean())
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB); "
          f"{covered:.1%} of cells have data")
    return out_path


if __name__ == "__main__":
    main()
