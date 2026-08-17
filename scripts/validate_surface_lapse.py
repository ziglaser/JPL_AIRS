#!/usr/bin/env python3
"""Validate the terrain-following surface-T extrapolation against reanalysis.

For a sample of archive days (spread across months), compares
airs_fcst.period_fields T2M under several lapse schemes against the MERRA-2
sfc_daily T2M at the same timestamps, over observed analysis-domain pixels.
Run on the CLUSTER (needs the fullgrid archive + sfc_daily + the elevation
map); the local checkout has only one demo day, which a 2026-08-16 spot
check already showed: extrapolation >> none (RMSE 8.3 -> ~6 K), derived
LSQ ~ fixed 6.5 in June, fixed 9.8 best on THAT afternoon -- but 9.8 would
over-warm winter inversions, which is what this multi-season sweep decides.

    PYTHONPATH=src python scripts/validate_surface_lapse.py --n-days 120

Output: per-scheme, per-season bias/RMSE (all + west-of--95), printed and
written to results/dl_front/lapse_validation.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from dl_front import airs_fcst, config, dataset

SCHEMES = [
    ("lsq_derived", True, None),
    ("fixed_6.5", False, 6.5),
    ("fixed_9.8", False, 9.8),
    ("no_extrap", False, 0.0),
]
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260816)
    a = ap.parse_args(argv)

    files = sorted(airs_fcst._archive_index(config.AIRS_FCST_ROOT).items())
    if len(files) < 50:
        raise SystemExit(f"only {len(files)} fullgrid files under "
                         f"{config.AIRS_FCST_ROOT} -- run on the cluster "
                         f"with JPL_AIRS_FCST exported")
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(files), min(a.n_days, len(files)),
                             replace=False))
    domain = dataset.analysis_domain()
    lons = np.asarray(config.LABEL_LONS)
    west = domain & (lons[None, :] < -95)

    rows = []
    for i in pick:
        day, path = files[i]
        try:
            ds = airs_fcst.load_fullgrid(path)
        except (ValueError, KeyError, IndexError, OSError):
            continue
        for hour in config.AIRS_HOURS:
            for name, derived, rate in SCHEMES:
                config.AIRS_SURFACE_LAPSE_DERIVED = derived
                if rate is not None:
                    config.AIRS_SURFACE_LAPSE_K_PER_KM = rate
                try:
                    per = airs_fcst.period_fields(path, hour, ds=ds)
                except (ValueError, KeyError, IndexError, OSError):
                    continue
                when = pd.Timestamp(per["time"].values)
                rea_path = (config.SFC_DIR / f"{when:%Y}"
                            / f"m2sfc_{when:%Y%m%d}.nc")
                if not rea_path.exists():
                    continue
                with xr.open_dataset(rea_path) as rea:
                    if when not in pd.DatetimeIndex(rea["time"].values):
                        continue
                    t_rea = rea["T2M"].sel(time=when).values
                t_air = per["T2M"].values
                for region, mask in (("all", domain), ("west", west)):
                    obs = np.isfinite(t_air) & mask
                    if obs.sum() < 10:
                        continue
                    d = t_air[obs] - t_rea[obs]
                    rows.append({"date": day, "hour": hour, "scheme": name,
                                 "region": region, "season": SEASON[when.month],
                                 "n": int(obs.sum()),
                                 "bias": float(d.mean()),
                                 "rmse": float(np.sqrt((d ** 2).mean()))})
        print(f"{day}: done", flush=True)

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "lapse_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = (df.groupby(["scheme", "region", "season"])
                 .apply(lambda g: pd.Series(
                     {"bias": np.average(g["bias"], weights=g["n"]),
                      "rmse": np.sqrt(np.average(g["rmse"] ** 2,
                                                 weights=g["n"])),
                      "steps": len(g)}))
                 .round(2))
    print(summary.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
