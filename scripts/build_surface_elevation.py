#!/usr/bin/env python3
"""Build the static surface-elevation map for terrain-following AIRS surface
extraction -> $JPL_AIRS_DATA/masks/surface_elevation.nc.

Elevation per label-grid cell from MERRA-2 climatology via the hypsometric
relation  z_s = (Rd * <T2M> / g) * ln(<SLP> / <PS>):  <PS> from the daily
multi-level files (front_id/reanalysis/MERRA2/daily), <SLP>/<T2M> from
sfc_daily.  A multi-year mean makes the transient pressure-weather term
vanish; what survives is the (static) terrain elevation, on exactly the
68 x 141 grid every other product uses -- no external DEM, no regridding
assumptions beyond the ones MERRA-2 itself already makes.

The output is small (~75 kB) but data/ is gitignored, so run this ONCE per
machine (locally and on the cluster; both have the needed years on disk):

    PYTHONPATH=src python scripts/build_surface_elevation.py --years 2008-2012
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dl_front import config
from front_finder import config as fd_config
from dl_front.krige_fill import parse_years

RD = 287.04          # J/kg/K, dry air
G = 9.80665          # m/s^2


def _year_means(year: int) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """(<PS>, <SLP>, <T2M>) time-mean maps for one year."""
    daily_dir = fd_config.MERRA2_DIR / "daily" / str(year)
    sfc_dir = config.SFC_DIR / str(year)
    for d in (daily_dir, sfc_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"{d} does not exist -- surface elevation needs the daily "
                f"(PS) and sfc_daily (SLP/T2M) MERRA-2 corpora for the "
                f"requested years")
    ps, slp, t2m, n = 0.0, 0.0, 0.0, 0
    for f in sorted(daily_dir.iterdir()):
        with xr.open_dataset(f) as ds:
            ps = ps + ds["PS"].mean("time").load()
        n += 1
    for f in sorted(sfc_dir.iterdir()):
        with xr.open_dataset(f) as ds:
            slp = slp + ds["SLP"].mean("time").load()
            t2m = t2m + ds["T2M"].mean("time").load()
    return ps / n, slp / n, t2m / n


def build(years, out_path: Path | None = None) -> Path:
    ps, slp, t2m = 0.0, 0.0, 0.0
    for y in years:
        p, s, t = _year_means(y)
        ps, slp, t2m = ps + p, slp + s, t2m + t
        print(f"{y}: accumulated", flush=True)
    ps, slp, t2m = ps / len(years), slp / len(years), t2m / len(years)

    elev = (RD * t2m / G) * np.log(slp / ps)
    elev = elev.clip(min=0.0)          # ocean/coast numerical noise -> 0
    out = xr.Dataset(
        {"elev_m": elev.astype(np.float32),
         "ps_mean_pa": ps.astype(np.float32)},
        attrs=dict(
            description="MERRA-2 hypsometric surface elevation, "
                        "z = (Rd*<T2M>/g)*ln(<SLP>/<PS>)",
            years=[int(y) for y in years],
            built_by="scripts/build_surface_elevation.py"))
    out["elev_m"].attrs.update(units="m", long_name="surface elevation ASL")
    out["ps_mean_pa"].attrs.update(units="Pa",
                                   long_name="mean surface pressure")
    out_path = (Path(out_path) if out_path is not None
                else fd_config.DATA_ROOT / "masks/surface_elevation.nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    out.to_netcdf(tmp)
    tmp.replace(out_path)
    print(f"wrote {out_path}  (elev range {float(elev.min()):.0f}.."
          f"{float(elev.max()):.0f} m)", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2008-2012",
                    help="climatology years (default 2008-2012; any span "
                         "with daily+sfc_daily on disk works -- the answer "
                         "is static terrain)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    build(parse_years(a.years), a.out)


if __name__ == "__main__":
    main()
