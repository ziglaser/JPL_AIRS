"""Download the DL-FRONT input corpus: MERRA-2 single-level surface fields.

Source: M2I1NXASM (inst1_2d_asm_Nx, 0.5 x 0.625 deg, hourly) via GES DISC
OPeNDAP, subset server-side to 3-hourly steps over the label domain plus a
2-cell margin, then remapped locally to the 1 deg label grid by bicubic
interpolation (paper "Data availability": "remapped by bicubic interpolation
to a 1x1 deg latitude-longitude grid").

Transport layer (cookie jars, stream numbers, retry/backoff, netCDF-3
fallback, atomic write) is shared with ``front_finder.acquire_merra2``.

CLI:  PYTHONPATH=src python -m dl_front.acquire_merra2_sfc 2003 [2004 ...]
"""
from __future__ import annotations

import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from front_finder.acquire_merra2 import (MAX_WORKERS, _fetch, _jars,
                                            _nc_lock, _stream)

from . import config

BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I1NXASM.5.12.4"

#: Generous physical bounds -- OPeNDAP transfers occasionally deliver
#: silently corrupt bytes that still parse as HDF (~1.6% of first pulls in
#: the M2I3NPASM corpus); every download is gated on these.
PHYSICAL_BOUNDS = {"T2M": (180.0, 340.0), "QV2M": (-1e-6, 0.05),
                   "SLP": (85_000.0, 110_000.0),
                   "U10M": (-100.0, 100.0), "V10M": (-100.0, 100.0)}


def is_physical(ds: xr.Dataset) -> bool:
    """True iff every SFC variable is finite and inside PHYSICAL_BOUNDS."""
    for var, (lo, hi) in PHYSICAL_BOUNDS.items():
        if var not in ds:
            return False
        v = ds[var].values
        if not np.isfinite(v).all():   # single-level fields have no fill
            return False
        if v.min() < lo or v.max() > hi:
            return False
    return True


def bicubic_to_label_grid(ds: xr.Dataset) -> xr.Dataset:
    """Native 0.5 x 0.625 deg subset -> 1 deg label grid, bicubic per field.

    RectBivariateSpline(kx=3, ky=3) on the native rectilinear grid evaluated
    at the integer-degree label coordinates; exact for the native points that
    coincide with label latitudes (all of them) and cubic in longitude.
    """
    lats = np.asarray(config.LABEL_LATS)
    lons = np.asarray(config.LABEL_LONS)
    out = {}
    for var in config.SFC_VARS:
        da = ds[var]
        vals = np.empty((da.sizes["time"], len(lats), len(lons)), np.float32)
        for t in range(da.sizes["time"]):
            spl = RectBivariateSpline(da["lat"].values, da["lon"].values,
                                      da.isel(time=t).values, kx=3, ky=3)
            vals[t] = spl(lats, lons)
        out[var] = xr.DataArray(vals, dims=("time", "lat", "lon"),
                                coords={"time": ds["time"].values,
                                        "lat": lats, "lon": lons})
    return xr.Dataset(out)


def _url(date: pd.Timestamp, stream: int) -> str:
    la0, la1 = config.MERRA2_SFC_LAT_SLICE
    lo0, lo1 = config.MERRA2_SFC_LON_SLICE
    fname = f"MERRA2_{stream}.inst1_2d_asm_Nx.{date:%Y%m%d}.nc4"
    dims = f"%5B0:3:23%5D%5B{la0}:{la1}%5D%5B{lo0}:{lo1}%5D"
    fields = ",".join(f"{v}{dims}" for v in config.SFC_VARS)
    coords = f"lat%5B{la0}:{la1}%5D,lon%5B{lo0}:{lo1}%5D,time%5B0:3:23%5D"
    return f"{BASE}/{date:%Y}/{date:%m}/{fname}.nc4?{fields},{coords}"


def day_path(date: pd.Timestamp) -> Path:
    return config.SFC_DIR / f"{date:%Y}" / f"m2sfc_{date:%Y%m%d}.nc"


def download_day(date: pd.Timestamp, retries: int = 2) -> bool:
    """One day -> compact label-grid file (5 surface vars x 8 3-hourly steps)."""
    out = day_path(date)
    if out.exists():
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    url = _url(date, _stream(date.year))
    jar = _jars.get()
    try:
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
            dest = Path(tmp) / "native.nc4"
            ok = _fetch(url, dest, jar)
            if not ok and "MERRA2_400" in url:   # reprocessed months use 401
                ok = _fetch(url.replace("MERRA2_400", "MERRA2_401"), dest, jar)
            if not ok:                           # Hyrax nc4 writer fallback
                ok = _fetch(url.replace(".nc4.nc4?", ".nc4.nc?"), dest, jar)
            backoff = 30.0
            while not ok and retries > 0:
                time.sleep(backoff)
                backoff = min(backoff * 2, 300.0)
                retries -= 1
                ok = _fetch(url, dest, jar)
            if not ok:
                return False
            with _nc_lock:
                ds = xr.open_dataset(dest).load()
                if not is_physical(ds):
                    return False
                ds = bicubic_to_label_grid(ds)
                enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                       for v in ds.data_vars}
                ds.to_netcdf(Path(tmp) / "day.nc", encoding=enc)
                (Path(tmp) / "day.nc").rename(out)
        return True
    finally:
        _jars.put(jar)


def download_year(year: int) -> None:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    missing = [d for d in dates if not day_path(d).exists()]
    print(f"{year}: {len(dates) - len(missing)} present, {len(missing)} to fetch",
          flush=True)
    failed, streak = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_day, d, 4) for d in missing]
        for i, (d, fut) in enumerate(zip(missing, futures)):
            if fut.result():
                streak = 0
            else:
                failed.append(d)
                streak += 1
                if streak >= 5:      # persistent outage: stop, rerun later
                    for f in futures:
                        f.cancel()
                    print(f"{year}: aborting after {streak} consecutive "
                          f"failures; {len(failed)} failed so far", flush=True)
                    return
            if (i + 1) % 50 == 0:
                print(f"  {year}: {i + 1}/{len(missing)} "
                      f"({len(failed)} failed)", flush=True)
    if failed:
        print(f"{year}: FAILED days: {[f'{d:%m-%d}' for d in failed]}", flush=True)
    else:
        print(f"{year}: complete", flush=True)


if __name__ == "__main__":
    for y in sys.argv[1:]:
        download_year(int(y))
