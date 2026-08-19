#!/usr/bin/env python3
"""Grid the Guo et al. (2024) merged-PBLH model output to 1 degree, keeping
every 3-hourly timestamp -- the per-day assessed PBLH input for the upwind
soil-moisture pipeline.

Input  : $JPL_AIRS_DATA/PBL_depth/Guo2024_model/<year>/[<year>/]YYYYMMDDHH.nc
         one file per 3-hourly synoptic time, 0.25 deg global (land-only),
         variable "Merged Planetary Boundary Layer Height" [m], fill -999.
Output : one NetCDF with dims (time, lat, lon) holding the 1 deg mean PBL
         depth and sample count at every source timestamp -- the same
         aggregation as ``build_pbl_climatology.py`` with the pooling over
         days removed.

What this product is for (UPWIND_INDEX_REVIEW.md sections 1.5, 1.8, 4.1d):
  1. ``trajectory_kernels.pbl.GriddedPBL`` -- the day's actual boundary-layer
     depth for the contact gate and m*, replacing the information-free
     climatological curve (review F2);
  2. ``gamma_gap = z_LFC - z_i`` per (day, slot, cell) -- one subtraction
     against the FCST_MU/MML_LFC already in the match-up files.

Aggregation rule (identical to the climatology, one rule applied once): every
*native* 0.25 deg sample that passes the validity filter is pooled with equal
weight into the 1 deg cell that contains it -- ``cell = floor(coordinate)``,
centres on X.5 -- and the reported value is sum(valid) / count(valid).  It is
a *land* mean, because the source is NaN over water.

Domain: the padded CONUS box, cell centres lat 19.5..58.5 and lon
-112.5..-58.5 -- the FCST_SMAP_MRMS 1 deg grid (25.5..52.5, -106.5..-64.5)
padded by 6 deg, the halfwidth of the trajectory source window, so an upwind
path launched from any CONUS receptor never falls off the PBLH grid.  Source
0..360 longitudes are wrapped to -180..180 before pooling.

Time axis: the true datetime of every source file present.  Missing source
times (21 files in October 2021: 2021-10-13 00Z .. 2021-10-15 06Z plus
2021-10-19 15Z and 18Z) are simply absent rows, not NaN rows -- consumers
must look up by timestamp, never by position, and fall back (to the
climatology) when a lookup misses.  The gap is recorded in the
``missing_source_times`` attribute.

Usage
-----
    # full build, 2017-2021 (~14 600 timestamps; I/O-bound, use --jobs)
    python scripts/build_pblh_3hrly_1deg.py --jobs 10

    # one year
    python scripts/build_pblh_3hrly_1deg.py --years 2019

    # smoke test: 16 files, no cache
    python scripts/build_pblh_3hrly_1deg.py --limit 16 --no-cache \
        --out-dir /tmp/pblh_smoke

Per-year (time, sum, count) blocks are cached under ``--cache-dir`` (default
``<out dir>/_cache``) so an interrupted run resumes for free: timestamps never
overlap between years, so a cached year is simply concatenated back in.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pbl_climatology import (  # noqa: E402  (shared source-corpus facts)
    DEFAULT_ROOT,
    FILL_VALUE,
    PBLH_MAX_M,
    PBLH_MIN_M,
    VAR_NAME,
    _init_worker,
    build_index,
    find_files,
    native_coords,
    parse_years,
    reduce_file,
    target_grid,
)

# --------------------------------------------------------------------------- #
# The padded-CONUS target grid
# --------------------------------------------------------------------------- #
#: Cell *edges*; centres fall on X.5.  The FCST_SMAP_MRMS CONUS box
#: (lat 25..53, lon -107..-64) padded by 6 deg -- the trajectory source-window
#: halfwidth -- on every side: 40 lat x 55 lon cells.
PADDED_CONUS = dict(lat_min=19.0, lat_max=59.0, lon_min=-113.0, lon_max=-58.0)

#: Encoding of the output time axis.  Whole hours since a round epoch keeps
#: the values human-readable in ncdump and exact in int64.
TIME_UNITS = "hours since 2017-01-01 00:00:00"
TIME_CALENDAR = "standard"


# --------------------------------------------------------------------------- #
# Per-year block: (times, sum, count) -- the additive unit of this build
# --------------------------------------------------------------------------- #
def run_year(files, jobs: int, index):
    """Reduce one year's files to per-timestamp (sum, count) over target cells.

    Returns ``(stamps, total, count)`` where *total* is float64 (n_times,
    ncell) and *count* int32.  Unreadable files (``reduce_file`` -> None) are
    dropped from the time axis entirely, consistent with the absent-row
    convention for source gaps.
    """
    keep, flat_idx, ncell = index

    if jobs <= 1:
        _init_worker(keep, flat_idx, ncell)
        reduced = (reduce_file(path) for path, _ in files)
        pairs = [(stamp, red) for (_, stamp), red in zip(files, reduced)
                 if red is not None]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(keep, flat_idx, ncell)) as pool:
            reduced = pool.map(reduce_file, [f for f, _ in files], chunksize=4)
            pairs = [(stamp, red) for (_, stamp), red in zip(files, reduced)
                     if red is not None]

    stamps = [stamp for stamp, _ in pairs]
    total = np.zeros((len(pairs), ncell), dtype=np.float64)
    count = np.zeros((len(pairs), ncell), dtype=np.int32)
    for row, (_, (s, _sq, c)) in enumerate(pairs):
        total[row] = s
        count[row] = c
    return stamps, total, count


def save_block(path: Path, stamps, total, count):
    np.savez_compressed(
        path,
        time_hours=date2num(stamps, TIME_UNITS, TIME_CALENDAR).astype(np.int64),
        total=total, count=count)


def load_block(path: Path):
    from netCDF4 import num2date

    z = np.load(path)
    stamps = [datetime(d.year, d.month, d.day, d.hour)
              for d in num2date(z["time_hours"], TIME_UNITS, TIME_CALENDAR)]
    return stamps, z["total"], z["count"]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_output(out_path: Path, stamps, total, count, lat_t, lon_t, n_native,
                 args, missing_note: str):
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    shape = (len(stamps), len(lat_t), len(lon_t))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(stamps))
        nc.createDimension("lat", len(lat_t))
        nc.createDimension("lon", len(lon_t))

        v = nc.createVariable("time", "i8", ("time",))
        v[:] = date2num(stamps, TIME_UNITS, TIME_CALENDAR).astype(np.int64)
        v.units = TIME_UNITS
        v.calendar = TIME_CALENDAR
        v.long_name = ("source synoptic time (UTC); missing source files are "
                       "absent rows, not NaN rows")

        v = nc.createVariable("lat", "f4", ("lat",))
        v[:] = lat_t.astype("f4")
        v.units = "degrees_north"
        v.long_name = "1 deg cell centre latitude"

        v = nc.createVariable("lon", "f4", ("lon",))
        v[:] = lon_t.astype("f4")
        v.units = "degrees_east"
        v.long_name = "1 deg cell centre longitude (-180..180)"

        v = nc.createVariable("pblh", "f4", ("time", "lat", "lon"),
                              zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))
        v[:] = mean.reshape(shape).astype(np.float32)
        v.units = "m"
        v.long_name = ("mean planetary boundary layer depth over the valid "
                       "0.25 deg native samples in the cell (land only)")
        v.cell_methods = "lat,lon: mean (0.25 deg native samples)"

        v = nc.createVariable("n_obs", "i2", ("time", "lat", "lon"),
                              zlib=True, complevel=4)
        v[:] = count.reshape(shape).astype(np.int16)
        v.units = "1"
        v.long_name = ("number of valid 0.25 deg samples pooled into the mean "
                       "(16 max per interior cell; 0 == all water or all fill)")

        v = nc.createVariable("n_native", "i4", ("lat", "lon"), zlib=True)
        v[:] = n_native.reshape(len(lat_t), len(lon_t))
        v.long_name = ("number of 0.25 deg native grid points inside each "
                       "1 deg cell (16 in the interior)")
        v.comment = "land fraction of a cell ~ n_obs / n_native"

        nc.setncatts(dict(
            title="1 deg 3-hourly PBL depth from the Guo et al. (2024) merged "
                  "PBLH model output, one field per source timestamp",
            source=str(args.guo_root),
            source_variable=VAR_NAME,
            source_resolution="0.25 deg, 3-hourly, land only",
            domain="padded CONUS: FCST_SMAP_MRMS 1 deg grid padded by the "
                   "6 deg trajectory source-window halfwidth; cell centres "
                   "lat 19.5..58.5, lon -112.5..-58.5",
            years=",".join(str(y) for y in sorted({s.year for s in stamps})),
            first_time=min(stamps).isoformat(), last_time=max(stamps).isoformat(),
            n_times=len(stamps),
            missing_source_times=missing_note,
            validity_filter=f"{PBLH_MIN_M} m <= PBLH <= {PBLH_MAX_M} m "
                            f"(fill value {FILL_VALUE} excluded)",
            aggregation="equal-weight pool of every valid native sample in the "
                        "cell (cell = floor(coordinate), centres on X.5)",
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            command=" ".join(sys.argv),
            history=f"{datetime.now(timezone.utc):%Y-%m-%d} "
                    f"scripts/build_pblh_3hrly_1deg.py",
        ))
    return out_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--guo-root", type=Path, default=DEFAULT_ROOT,
                   help=f"raw Guo2024_model tree (default: {DEFAULT_ROOT})")
    p.add_argument("--years", type=str, default=None,
                   help="comma list and/or ranges, e.g. 2017,2019-2021 (default: all)")
    p.add_argument("--out-dir", type=Path,
                   default=DEFAULT_ROOT.parent / "derived",
                   help="directory for PBLH_1deg_3hrly_{y0}-{y1}.nc")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="do not read or write per-year caches")
    p.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)))
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N files (smoke test)")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)

    files = find_files(args.guo_root, parse_years(args.years))
    if not files:
        raise SystemExit(f"no YYYYMMDDHH.nc files found under {args.guo_root}")
    if args.limit:
        files = files[: args.limit]
    n_slots = 8
    expected = {y: (366 if y % 4 == 0 else 365) * n_slots
                for y in sorted({s.year for _, s in files})}
    got = {y: sum(1 for _, s in files if s.year == y) for y in expected}
    missing_note = "; ".join(f"{y}: {expected[y] - got[y]} of {expected[y]} times absent"
                             for y in expected if got[y] != expected[y]) or "none"

    lat_n, lon_n = native_coords("uniform", files[0][0])
    lat_t, lon_t = target_grid(PADDED_CONUS)
    keep, flat_idx, n_native = build_index(lat_n, lon_n, lat_t, lon_t)
    ncell = len(lat_t) * len(lon_t)

    out_path = args.out_dir / (f"PBLH_1deg_3hrly_"
                               f"{min(expected)}-{max(expected)}.nc")
    cache_dir = args.cache_dir or args.out_dir / "_cache"
    use_cache = not (args.no_cache or args.limit)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} files | padded CONUS "
          f"({len(lat_t)}x{len(lon_t)} cells) | jobs={args.jobs}\n"
          f"missing source times -> {missing_note}\nout: {out_path}")

    stamps, totals, counts = [], [], []
    for year in sorted(expected):
        year_files = [(f, s) for f, s in files if s.year == year]
        cache_path = cache_dir / f"pblh_3hrly_{year}.npz"
        if use_cache and cache_path.exists():
            block = load_block(cache_path)
            print(f"  {year}: cached ({len(block[0])} times)")
        else:
            t0 = datetime.now()
            block = run_year(year_files, args.jobs, (keep, flat_idx, ncell))
            print(f"  {year}: {len(year_files)} files in "
                  f"{(datetime.now() - t0).total_seconds() / 60:.1f} min")
            if use_cache:
                save_block(cache_path, *block)
        stamps.extend(block[0])
        totals.append(block[1])
        counts.append(block[2])

    write_output(out_path, stamps, np.concatenate(totals),
                 np.concatenate(counts), lat_t, lon_t, n_native,
                 args, missing_note)
    covered = float((np.concatenate(counts).sum(axis=0) > 0).mean())
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB); "
          f"{covered:.1%} of cells have data")
    return out_path


if __name__ == "__main__":
    main()
