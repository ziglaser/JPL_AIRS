"""Estimate the size of the full AIRS-FCST HYSPLIT trajectory dataset.

The cluster will receive one directory per forecast day, matching the layout of
the local demo day (data/HYSPLIT_demo/wrf27km_20190605/wrf27km_20190605/):

  - N   nogrid_*.nc   files (one per AIRS granule overpassing that day).
        Dims are fixed (time:7, level:57, fieldx:45, fieldy:30), so every
        nogrid file is the same size regardless of how many columns are real.
  - 1   fullgrid_*.nc file (1-deg box-average of the parcels).

This script therefore estimates:

  total files = n_days * (granules_per_day + 1)
  total bytes = n_days * (granules_per_day * nogrid_bytes + fullgrid_bytes)

where
  * nogrid_bytes / fullgrid_bytes are MEASURED from the demo day on disk,
  * n_days is COUNTED from the FCST_SMAP_MRMS yearly files (a day counts if
    FCST_MU_CAPE has any finite value on that date — the HYSPLIT runs are
    driven by the same forecast days),
  * granules_per_day defaults to the demo day's count (5) and is the one
    genuinely uncertain knob, so a low/high range is printed too.

Usage:
    python scripts/estimate_hysplit_dataset.py
    python scripts/estimate_hysplit_dataset.py --granules-per-day 4 --years 2019 2020
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
DEMO_DAY_DIR = REPO / "data/HYSPLIT_demo/wrf27km_20190605/wrf27km_20190605"
FCST_DIR = REPO / "data/FCST_SMAP_MRMS"
ALL_YEARS = (2016, 2017, 2018, 2019, 2020, 2021)


def human(nbytes: float) -> str:
    """Format a byte count as a human-readable string (binary units)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PiB"


def measure_demo_day(demo_dir: Path) -> tuple[int, int, int]:
    """Return (nogrid_bytes, fullgrid_bytes, demo_granule_count) from disk.

    nogrid files are constant-size (fixed dims), so one measurement suffices;
    we still average over all present as a sanity check.
    """
    nogrid = sorted(demo_dir.glob("nogrid_*.nc"))
    fullgrid = sorted(demo_dir.glob("fullgrid_*.nc"))
    if not nogrid or not fullgrid:
        raise FileNotFoundError(f"Demo day incomplete in {demo_dir}")
    nogrid_sizes = [f.stat().st_size for f in nogrid]
    if len(set(nogrid_sizes)) > 1:
        print(f"WARNING: nogrid sizes vary: {nogrid_sizes} — using the mean.")
    return int(np.mean(nogrid_sizes)), fullgrid[0].stat().st_size, len(nogrid)


def count_forecast_days(year: int) -> int:
    """Count dates in FCST_SMAP_MRMS_<year>.nc with any finite FCST_MU_CAPE."""
    path = FCST_DIR / f"FCST_SMAP_MRMS_{year}.nc"
    with xr.open_dataset(path) as ds:
        cape = ds["FCST_MU_CAPE"]
        # A date is a real forecast day if CAPE is finite anywhere on it.
        non_date_dims = [d for d in cape.dims if d != "date"]
        valid = np.isfinite(cape).any(dim=non_date_dims).values
    return int(valid.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--granules-per-day", type=int, default=None,
        help="AIRS granules (nogrid files) per day. Default: the demo day's count.",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=list(ALL_YEARS),
        help=f"Years to include (default: {ALL_YEARS}).",
    )
    args = parser.parse_args()

    nogrid_bytes, fullgrid_bytes, demo_granules = measure_demo_day(DEMO_DAY_DIR)
    granules = args.granules_per_day or demo_granules

    print("Per-file sizes (measured from the demo day, 2019-06-05):")
    print(f"  nogrid  (per AIRS granule) : {human(nogrid_bytes)}")
    print(f"  fullgrid (one per day)     : {human(fullgrid_bytes)}")
    print(f"  granules on the demo day   : {demo_granules}")
    print()

    day_bytes = granules * nogrid_bytes + fullgrid_bytes
    files_per_day = granules + 1
    print(f"Assuming {granules} granules/day -> "
          f"{files_per_day} files, {human(day_bytes)} per day.")
    print()

    total_days = 0
    print(f"{'year':>6} {'forecast days':>14} {'files':>8} {'size':>12}")
    for year in args.years:
        n_days = count_forecast_days(year)
        total_days += n_days
        print(f"{year:>6} {n_days:>14} {n_days * files_per_day:>8} "
              f"{human(n_days * day_bytes):>12}")

    print("-" * 44)
    print(f"{'TOTAL':>6} {total_days:>14} {total_days * files_per_day:>8} "
          f"{human(total_days * day_bytes):>12}")
    print()

    # Granule count is the uncertain input: bracket it. The demo day's two
    # overpass swaths gave 5 granules; days with wider coverage could see more.
    print("Sensitivity to granules/day (total dataset size):")
    for g in (3, 4, 5, 6, 8):
        size = total_days * (g * nogrid_bytes + fullgrid_bytes)
        marker = "  <- default" if g == granules else ""
        print(f"  {g} granules/day: {total_days * (g + 1):>6} files, "
              f"{human(size)}{marker}")


if __name__ == "__main__":
    main()
