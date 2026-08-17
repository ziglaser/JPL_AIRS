#!/usr/bin/env python3
"""One-off diagnostic: why is the western analysis domain never covered?

Two checks, run on the cluster (fronts-tf env, full data access):

1. Raw swath_bank.npz frequency, west vs east, per cycle day/hour -- is
   west coverage a hard zero (never observed) or just below the 0.075
   footprint threshold (rare but present)?
2. Per-file overpass geometry across a broad sample of the archive -- does
   the RAW overpass (slot 0, before any hour selection) ever cross the
   western half, and does its longitude position actually shift with
   cycle_day as Aqua's 16-day ground-track repeat predicts?  If the
   observed-centroid longitude clusters near a fixed value regardless of
   cycle_day, the archive's day-selection is not sampling the true orbital
   cycle (or cycle_day itself is wrong).

Usage::

    PYTHONPATH=src python scripts/diagnose_swath_coverage.py --n-sample 400
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from dl_front import airs_fcst, config, dataset, swath


def _check_env():
    """Refuse to run against the repo-local data/ fallback: config.py
    silently resolves DATA_ROOT/AIRS_FCST_ROOT to <repo>/data when
    JPL_AIRS_DATA/JPL_AIRS_FCST aren't exported in THIS shell (a recurring
    footgun this session -- env vars set in one shell don't carry to a new
    one), which would scan the tiny local demo tree instead of the real
    15-year GPFS archive and produce misleadingly clean/empty results."""
    missing = [v for v in ("JPL_AIRS_DATA", "JPL_AIRS_FCST")
              if not os.environ.get(v)]
    if missing:
        sys.exit(f"FATAL: {', '.join(missing)} not set in this shell -- "
                 f"would silently scan {config.REPO_ROOT / 'data'} instead "
                 f"of the real GPFS archive. Export them first, e.g.:\n"
                 f"  export JPL_AIRS_DATA=/gpfs/scratch/smap-convection/"
                 f"AIRS_SMAP_Front_data\n"
                 f"  export JPL_AIRS_FCST=/gpfs/scratch/smap-convection/"
                 f"AIRS_FCST_1deg")
    n_files = len(airs_fcst._archive_index(config.AIRS_FCST_ROOT))
    print(f"JPL_AIRS_DATA={os.environ['JPL_AIRS_DATA']}")
    print(f"JPL_AIRS_FCST={os.environ['JPL_AIRS_FCST']} "
         f"({n_files} fullgrid files indexed)")
    if n_files < 1000:
        sys.exit(f"FATAL: only {n_files} fullgrid files found under "
                 f"AIRS_FCST_ROOT={config.AIRS_FCST_ROOT} -- expected "
                 f"thousands (the full archive is ~6700 files); this looks "
                 f"like the wrong root, not real archive sparsity.")


def check_bank(west, east):
    if not config.SWATH_BANK_PATH.exists():
        print(f"no swath bank at {config.SWATH_BANK_PATH}; skipping check 1")
        return
    with np.load(config.SWATH_BANK_PATH) as z:
        freq, hours = z["freq"], z["hours"]
    print(f"\n=== check 1: swath_bank.npz raw frequency, west vs east ===")
    print(f"hours archived: {list(hours)}")
    for h_idx, h in enumerate(hours):
        print(f"\n-- hour {h}Z --")
        for cyc in range(swath.CYCLE_DAYS):
            f = freq[cyc, h_idx]
            print(f"  cycle {cyc:2d}: west max={f[west].max():.4f} "
                 f"mean={f[west].mean():.5f}  |  "
                 f"east max={f[east].max():.4f} mean={f[east].mean():.4f}")


def check_raw_overpass(n_sample: int):
    print(f"\n=== check 2: raw overpass (slot 0) geometry, {n_sample} files ===")
    all_files = sorted(airs_fcst._archive_index(config.AIRS_FCST_ROOT).values())
    if len(all_files) > n_sample:
        rng = np.random.default_rng(0)
        all_files = [all_files[i] for i in
                    sorted(rng.choice(len(all_files), n_sample, replace=False))]
    lons = np.asarray(config.LABEL_LONS)
    rows = []
    for p in all_files:
        try:
            ds = airs_fcst.load_fullgrid(p)
        except Exception as err:
            continue
        one = ds.isel(time=0)
        # any retrieval bin at/below the surface scan floor (the
        # terrain-following extraction's candidate set, 2026-08-16)
        obs3d = one["N"].fillna(0) > 0
        obs = obs3d.sel(level=obs3d["level"]
                        >= float(config.AIRS_SURFACE_SCAN_FLOOR_HPA)
                        ).any("level")
        if not bool(obs.any()):
            continue
        native_lon = one["lon"].values          # native half-degree grid
        cols_with_obs = np.nonzero(obs.any(axis=0).values)[0]
        lon_centroid = float(native_lon[cols_with_obs].mean())
        date = airs_fcst.overpass_midpoint(p).normalize()
        any_west = bool((native_lon[cols_with_obs] < -95).any())
        rows.append((date, swath.cycle_day(date), lon_centroid, any_west))
    df = pd.DataFrame(rows, columns=["date", "cycle_day", "lon_centroid",
                                    "any_west_native_grid"])
    print(df.groupby("cycle_day")["lon_centroid"].agg(["mean", "std", "count"]))
    print(f"\noverall lon_centroid range: {df['lon_centroid'].min():.1f} to "
         f"{df['lon_centroid'].max():.1f}  (spread near-zero => archive is "
         f"NOT sampling the full orbital cycle across cycle_day bins)")
    print(f"files with ANY native-grid observed pixel west of -95: "
         f"{df['any_west_native_grid'].sum()} / {len(df)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sample", type=int, default=400)
    a = ap.parse_args(argv)

    _check_env()
    domain = dataset.analysis_domain()
    lons = np.asarray(config.LABEL_LONS)
    west = domain & (lons[None, :] < -95)
    east = domain & (lons[None, :] >= -95)

    check_bank(west, east)
    check_raw_overpass(a.n_sample)


if __name__ == "__main__":
    main()
