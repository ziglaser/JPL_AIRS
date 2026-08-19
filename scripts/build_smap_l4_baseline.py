"""Per-cell monthly SMAP L4 surface soil-moisture baseline from FCST_SMAP_MRMS.

Purpose: the reference field for the *cardinal* soil-moisture anomaly used by
the upwind influence index (src/trajectory_kernels/UPWIND_INDEX_REVIEW.md,
section 1.7):

    S'(date, hour, cell) = SMAP_L4_smsfc_av(date, hour, cell)
                           - baseline(month(date), cell)

Design decisions (Zach, 19 Aug 2026):
- Anomaly is a DIFFERENCE in m3/m3 (cardinal), not a percentile (ordinal):
  it stays linear in water, the same currency as the latent flux it
  modulates, and preserves the magnitude of departures.
- Baseline = mean over ALL days of that calendar month and ALL 5 L4 analysis
  hours, pooled over 2016-2021 (the full current record; intended to be
  extended as more years arrive -- rebuild by rerunning this script).
- Hours are pooled because the L4 surface-SM diurnal cycle is small next to
  day-to-day variability; months are calendar months of the `date` coord.
- No smoothing, no gap filling: a cell/month with no valid samples is NaN,
  never fabricated. `n_obs` records the pool size per cell/month.

Because the kernel contraction is linear, convolving S' directly through the
trajectory kernels is exactly equivalent to convolving raw SM and subtracting
the convolved baseline -- the spatially varying reference is handled by
convolving the anomaly field itself.

Output: SMAP_L4_smsfc_monthly_baseline_{y0}-{y1}.nc next to the inputs, with
  sm_baseline (month, lat, lon)  mean surface soil moisture, m3/m3
  sm_std      (month, lat, lon)  sample std of the same pool (variability,
                                 NOT a standard error; samples are dependent)
  n_obs       (month, lat, lon)  valid samples in the pool

Usage:
  python scripts/build_smap_l4_baseline.py            # defaults
  python scripts/build_smap_l4_baseline.py --data-dir /path --out /path/x.nc
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

VARIABLE = "SMAP_L4_smsfc_av"
HOUR_DIM = "L4_nhours"


def build_baseline(paths: list[str]) -> xr.Dataset:
    """Accumulate sum / sum-of-squares / count per (month, lat, lon)."""
    total = sq_total = count = None
    coords = None
    for path in sorted(paths):
        with xr.open_dataset(path) as ds:
            sm = ds[VARIABLE].load()  # (date, L4_nhours, lat, lon)
            months = sm["date"].dt.month
            if coords is None:
                coords = {"lat": sm["lat"].values, "lon": sm["lon"].values}
                shape = (12, coords["lat"].size, coords["lon"].size)
                total = np.zeros(shape)
                sq_total = np.zeros(shape)
                count = np.zeros(shape, dtype=np.int64)
            for m in range(1, 13):
                vals = sm.sel(date=(months == m)).values  # (days, hours, lat, lon)
                finite = np.isfinite(vals)
                total[m - 1] += np.where(finite, vals, 0.0).sum(axis=(0, 1))
                sq_total[m - 1] += np.where(finite, vals**2, 0.0).sum(axis=(0, 1))
                count[m - 1] += finite.sum(axis=(0, 1))
        print(f"  accumulated {os.path.basename(path)}")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        var = np.where(count > 1,
                       (sq_total - count * mean**2) / (count - 1), np.nan)
        std = np.sqrt(np.clip(var, 0.0, None))

    dims = ("month", "lat", "lon")
    coords = {"month": np.arange(1, 13, dtype=np.int32), **coords}
    return xr.Dataset(
        {
            "sm_baseline": (dims, mean.astype(np.float32),
                            {"units": "m3 m-3",
                             "long_name": "monthly-mean SMAP L4 surface soil moisture"}),
            "sm_std": (dims, std.astype(np.float32),
                       {"units": "m3 m-3",
                        "long_name": "sample std of the monthly pool (variability, not SE)"}),
            "n_obs": (dims, count.astype(np.int32),
                      {"long_name": "valid samples pooled (days x analysis hours x years)"}),
        },
        coords=coords,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="/mnt/d/JPL_AIRS/data/FCST_SMAP_MRMS")
    p.add_argument("--out-dir", default="/mnt/d/JPL_AIRS/data/soil_moisture",
                   help="directory for derived soil-moisture products")
    p.add_argument("--out", default=None,
                   help="full output path (overrides --out-dir; years in name)")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "FCST_SMAP_MRMS_*.nc")))
    if not paths:
        raise SystemExit(f"no FCST_SMAP_MRMS_*.nc under {args.data_dir}")
    years = [os.path.basename(f).split("_")[-1].removesuffix(".nc") for f in paths]

    out = build_baseline(paths)
    out.attrs.update({
        "source_variable": VARIABLE,
        "source_files": ", ".join(os.path.basename(f) for f in paths),
        "years": f"{years[0]}-{years[-1]}",
        "aggregation": ("mean over all days of the calendar month and all "
                        f"{HOUR_DIM} analysis hours, pooled across years; "
                        "NaN where no valid samples"),
        "purpose": ("reference for the cardinal soil-moisture anomaly "
                    "S' = SM - baseline(month, cell); see "
                    "src/trajectory_kernels/UPWIND_INDEX_REVIEW.md section 1.7"),
    })
    os.makedirs(args.out_dir, exist_ok=True)
    dest = args.out or os.path.join(
        args.out_dir, f"SMAP_L4_smsfc_monthly_baseline_{years[0]}-{years[-1]}.nc")
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(dest, encoding=encoding)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
