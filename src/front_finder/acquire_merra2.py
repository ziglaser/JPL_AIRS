"""Download the MERRA-2 pretraining corpus onto the 1 deg label grid.

Source: M2I3NPASM (inst3_3d_asm_Np) via GES DISC OPeNDAP, subset server-side
to the label domain -- lat stride 2 lands exactly on the integer-degree label
latitudes (10..77 N); longitudes arrive at native 0.625 deg and are
interpolated to the integer-degree label longitudes locally.

Auth: ~/.netrc entry for urs.earthdata.nasa.gov (curl -n).  Session cookies
persist in per-worker jars under MERRA2_DIR -- the URS OAuth redirect chain
costs ~20 s versus ~2 s for a cookie'd request, so each worker pays it once,
not once per request.  Resumable: days whose compact file exists are skipped.

CLI:  PYTHONPATH=src python -m front_finder.acquire_merra2 2015 [2014 ...]
"""
from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config

BASE = "https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I3NPASM.5.12.4"

# GES DISC asks users to keep concurrent connections modest; 4 workers is
# well under their limit and saturates the ~2 s/request subset latency.
MAX_WORKERS = 4

# Pool of persistent cookie jars, one per in-flight day, so concurrent curl
# processes never write the same jar.
_jars: queue.Queue[Path] = queue.Queue()
for _i in range(MAX_WORKERS):
    _jars.put(config.MERRA2_DIR / f".urs_cookies_{_i}")

# HDF5/netCDF is not thread-safe: only curl runs concurrently; all
# xarray open/merge/write happens under this lock.
_nc_lock = threading.Lock()

#: generous physical bounds per variable -- OPeNDAP transfers occasionally
#: deliver silently corrupt bytes that still parse as HDF (29+/6940 files in
#: the first 2003-2021 pull, garbage in any subset of variables).
PHYSICAL_BOUNDS = {"T": (150.0, 350.0), "QV": (-1e-6, 0.05),
                   "U": (-200.0, 200.0), "V": (-200.0, 200.0),
                   "PS": (30_000.0, 115_000.0)}


def is_physical(ds: xr.Dataset) -> bool:
    """True iff every variable's finite values sit inside PHYSICAL_BOUNDS."""
    for var, (lo, hi) in PHYSICAL_BOUNDS.items():
        if var not in ds:
            return False
        v = ds[var].values
        if np.isinf(v).any():          # NaN = below-ground, fine; inf = never
            return False
        v = v[np.isfinite(v)]
        if v.size == 0 or v.min() < lo or v.max() > hi:
            return False
    return True


def _stream(year: int) -> int:
    """MERRA-2 production stream number (GMAO file-spec section 5)."""
    if year <= 1991:
        return 100
    if year <= 2000:
        return 200
    if year <= 2010:
        return 300
    return 400


def _projection(var: str, lev_sel: str) -> str:
    la0, la1, las = config.MERRA2_LAT_SLICE
    lo0, lo1 = config.MERRA2_LON_SLICE
    return f"{var}%5B0:7%5D%5B{lev_sel}%5D%5B{la0}:{las}:{la1}%5D%5B{lo0}:{lo1}%5D"


def _urls(date: pd.Timestamp, stream: int) -> list[str]:
    """Two subset URLs/day: irregular level picks need two strided ranges
    (0:3:6 -> 1000/925/850 hPa; 12:4:16 -> 700/500 hPa)."""
    la0, la1, las = config.MERRA2_LAT_SLICE
    lo0, lo1 = config.MERRA2_LON_SLICE
    fname = f"MERRA2_{stream}.inst3_3d_asm_Np.{date:%Y%m%d}.nc4"
    stem = f"{BASE}/{date:%Y}/{date:%m}/{fname}.nc4"
    coords = (f"lat%5B{la0}:{las}:{la1}%5D,lon%5B{lo0}:{lo1}%5D,"
              f"time%5B0:7%5D")
    low = ",".join(_projection(v, "0:3:6") for v in config.MERRA2_VARS_3D)
    high = ",".join(_projection(v, "12:4:16") for v in config.MERRA2_VARS_3D)
    ps = (f"PS%5B0:7%5D%5B{la0}:{las}:{la1}%5D%5B{lo0}:{lo1}%5D")
    return [f"{stem}?{low},lev%5B0:3:6%5D,{coords}",
            f"{stem}?{high},{ps},lev%5B12:4:16%5D,{coords}"]


def _fetch(url: str, dest: Path, jar: Path) -> bool:
    r = subprocess.run(
        ["curl", "-sS", "-n", "-L", "--fail", "--max-time", "300",
         "-c", str(jar), "-b", str(jar), "-o", str(dest), url],
        capture_output=True, text=True)
    return r.returncode == 0


def day_path(date: pd.Timestamp) -> Path:
    return config.MERRA2_DIR / "daily" / f"{date:%Y}" / f"m2_{date:%Y%m%d}.nc"


def download_day(date: pd.Timestamp, retries: int = 2) -> bool:
    """One day -> compact label-grid file (T/QV/U/V x 5 levels + PS)."""
    out = day_path(date)
    if out.exists():
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = _stream(date.year)
    jar = _jars.get()
    try:
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
            paths = []
            for i, url in enumerate(_urls(date, stream)):
                dest = Path(tmp) / f"part{i}.nc4"
                ok = _fetch(url, dest, jar)
                if not ok and stream == 400:      # reprocessed months use 401
                    ok = _fetch(url.replace("MERRA2_400", "MERRA2_401"),
                                dest, jar)
                if not ok:
                    # Hyrax's netCDF-4 file-out writer dies with "NetCDF:
                    # HDF error" on a few granules (e.g. 2003-01-14); the
                    # netCDF-3 writer handles them fine.
                    ok = _fetch(url.replace(".nc4.nc4?", ".nc4.nc?"),
                                dest, jar)
                # GES DISC OPeNDAP has intermittent BES outages (observed
                # 2026-08-04: nc_create permission errors / socket resets) --
                # back off rather than hammer.
                backoff = 30.0
                while not ok and retries > 0:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300.0)
                    retries -= 1
                    ok = _fetch(url, dest, jar)
                if not ok:
                    return False
                paths.append(dest)
            with _nc_lock:
                ds = xr.merge([xr.open_dataset(p).load() for p in paths])
                if not is_physical(ds):   # reject silently corrupt transfers
                    return False
                ds = ds.interp(lon=np.arange(-171.0, -30.9, 1.0),
                               method="linear")
                ds = ds.reindex(lev=list(config.TARGET_LEVELS_HPA))
                enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                       for v in ds.data_vars}
                # Write then rename so a crash never leaves a truncated
                # file at the resumable-skip path.
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
    failed = []
    streak = 0
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
                          f"failures (server outage?); {len(failed)} failed "
                          f"so far", flush=True)
                    return
            if (i + 1) % 25 == 0:
                print(f"  {year}: {i + 1}/{len(missing)} "
                      f"({len(failed)} failed)", flush=True)
    if failed:
        print(f"{year}: FAILED days: {[f'{d:%m-%d}' for d in failed]}", flush=True)
    else:
        print(f"{year}: complete", flush=True)


if __name__ == "__main__":
    for y in sys.argv[1:]:
        download_year(int(y))
