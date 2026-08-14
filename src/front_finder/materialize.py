"""Materialize training samples to per-year memmap shards (post-mortem fix).

Why this exists (2026-08-09): TF 2.15's file-backed ``dataset.cache(filename)``
writer accumulates ~0.4 MB of native memory per element until the cache
finalizes -- a full pass over the 12-year corpus never completed on a 15 GB
WSL VM, which OOM-thrashed the machine at epoch 43.  Instead, samples are
derived ONCE here, in a plain bounded-memory Python loop, and training reads
the finished shards; the leaky cache writer is never used.

Shards store the WIND-inclusive channel set (superset); thermo-only training
selects its channels by index at read time (dataset.make_shard_tf_dataset),
so one materialization serves both E1a and E1b.

Layout under ``config.SHARD_DIR``::

    x_{year}.npy      float32 (n, 72, 144, 5, 10)   normalized, imputed
    y_{year}.npy      float32 (n, 72, 144, 6)       one-hot + weight channel
    times_{year}.npy  datetime64[ns] (n,)           CODSUS bulletin times
    meta_{year}.json  n, channels, dilation, masked_labels, norm-stats echo

Restartable at year granularity: finished years are skipped, partial years
are written to ``*.tmp`` and atomically renamed only on success.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr

from . import config, dataset, labels
from .acquire_merra2 import day_path

#: Samples are stored with the full wind-inclusive channel set.
SHARD_CHANNELS = dataset.channel_names(winds=True)


def _year_paths(year: int) -> dict:
    d = config.SHARD_DIR
    return {"x": d / f"x_{year}.npy", "y": d / f"y_{year}.npy",
            "times": d / f"times_{year}.npy", "meta": d / f"meta_{year}.json"}


def year_done(year: int) -> bool:
    """All four shard files exist AND the meta parses.

    A zero-byte/corrupt meta means the year's final writes never reached
    disk (2026-08-10: a hard WSL reboot left meta_2012-2014.json empty
    while x/y looked plausible) -- the whole year is untrustworthy, so it
    reports not-done and gets fully rebuilt.
    """
    paths = _year_paths(year)
    if not all(p.exists() for p in paths.values()):
        return False
    try:
        json.loads(paths["meta"].read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return True


def count_samples(year: int, masked_labels: bool = True) -> int:
    """Exact sample count for ``year`` without deriving anything.

    Mirrors dataset.year_samples' skip rules: a CODSUS timestep yields a
    sample iff its day's MERRA-2 file exists and contains that hour.
    """
    truth_ds = labels.load_fronts(year, width=1, masked=masked_labels)
    times = pd.DatetimeIndex(truth_ds["time"].values)
    truth_ds.close()
    n = 0
    for date, idx in times.groupby(times.normalize()).items():
        p = day_path(pd.Timestamp(date))
        if not p.exists():
            continue
        with xr.open_dataset(p) as day:
            day_hours = pd.DatetimeIndex(day["time"].values)
        n += sum(1 for t in idx if (day_hours == t).any())
    return n


def materialize_year(year: int, stats: dict | None = None,
                     dilation: int = config.LABEL_DILATION,
                     masked_labels: bool = True,
                     overwrite: bool = False) -> int:
    """Write one year's shard; returns the sample count (0-cost if done)."""
    paths = _year_paths(year)
    if year_done(year) and not overwrite:
        return json.loads(paths["meta"].read_text())["n"]

    stats = stats or dataset.load_norm_stats()
    n = count_samples(year, masked_labels)
    config.SHARD_DIR.mkdir(parents=True, exist_ok=True)

    tmp = {k: p.with_suffix(p.suffix + ".tmp") for k, p in paths.items()}
    x_mm = np.lib.format.open_memmap(
        tmp["x"], mode="w+", dtype=np.float32,
        shape=(n, *config.PADDED_SHAPE, len(config.TARGET_LEVELS_HPA),
               len(SHARD_CHANNELS)))
    y_mm = np.lib.format.open_memmap(
        tmp["y"], mode="w+", dtype=np.float32,
        shape=(n, *config.PADDED_SHAPE, len(dataset.CLASS_NAMES) + 1))
    times = np.empty(n, dtype="datetime64[ns]")

    i = 0
    for x, y, t in dataset.year_samples(year, stats, winds=True,
                                        dilation=dilation,
                                        masked_labels=masked_labels,
                                        return_times=True):
        x_mm[i], y_mm[i], times[i] = x, y, np.datetime64(t)
        i += 1
    if i != n:
        raise RuntimeError(f"{year}: generated {i} samples, counted {n} -- "
                           "corpus changed mid-run?")
    x_mm.flush(); y_mm.flush()
    del x_mm, y_mm
    with open(tmp["times"], "wb") as f:   # np.save(path) would append ".npy"
        np.save(f, times)
    tmp["meta"].write_text(json.dumps({
        "n": n, "channels": list(SHARD_CHANNELS), "dilation": dilation,
        "masked_labels": masked_labels,
        "label_source": config.LABEL_SOURCE,
        "classes": list(dataset.CLASS_NAMES),
        "norm_stats": str(dataset.NORM_STATS_PATH)}, indent=1))
    for k in ("x", "y", "times", "meta"):        # atomic publish, meta last
        tmp[k].rename(paths[k])
    return n


def labels_stale(year: int) -> bool:
    """True when the year's y shard was built at a different dilation (or
    the meta predates the dilation key)."""
    paths = _year_paths(year)
    if not year_done(year):
        return False                       # nothing to refresh; needs full run
    meta = json.loads(paths["meta"].read_text())
    return meta.get("dilation") != config.LABEL_DILATION


def rematerialize_labels(year: int, dilation: int = config.LABEL_DILATION,
                         masked_labels: bool = True) -> int:
    """Rebuild y_{year}.npy (labels only) at the current dilation.

    The x shard and times shard are reused untouched -- rebuilding y needs
    no netCDF/derivation work, only the label files, so this runs in
    minutes where a full materialize takes hours.  y is regenerable at any
    dilation from the labels, so the old y is simply replaced.
    """
    paths = _year_paths(year)
    if not year_done(year):
        raise FileNotFoundError(f"{year}: no complete shard to relabel -- "
                                "run a full materialize first")
    meta = json.loads(paths["meta"].read_text())
    if meta.get("label_source", "codsus") != config.LABEL_SOURCE:
        raise ValueError(f"{year}: shard has label_source "
                         f"{meta.get('label_source', 'codsus')!r}, config says "
                         f"{config.LABEL_SOURCE!r} -- full rematerialize needed")
    times = np.load(paths["times"])

    truth_ds = labels.load_fronts(year, width=1, masked=masked_labels)
    fronts = labels.front_stack(truth_ds).values
    fronts = labels.dilate(fronts.reshape(-1, *config.GRID_SHAPE),
                           dilation).reshape(fronts.shape)
    valid = labels.valid_mask(truth_ds).values
    label_times = truth_ds["time"].values.astype("datetime64[ns]")
    truth_ds.close()

    tmp = paths["y"].with_suffix(paths["y"].suffix + ".tmp")
    y_mm = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32,
        shape=(len(times), *config.PADDED_SHAPE, len(dataset.CLASS_NAMES) + 1))
    for i, t in enumerate(times):
        j = np.flatnonzero(label_times == t)
        if len(j) == 0:
            raise RuntimeError(f"{year}: shard time {t} missing from labels "
                               "-- label files changed since materialize?")
        y_mm[i] = dataset.make_y(fronts[j[0]], valid[j[0]])
    y_mm.flush()
    del y_mm
    meta["dilation"] = dilation
    meta["masked_labels"] = masked_labels
    tmp.rename(paths["y"])                        # atomic swap, then meta
    paths["meta"].write_text(json.dumps(meta, indent=1))
    return len(times)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("years", nargs="+", type=int)
    ap.add_argument("--workers", type=int, default=1,
                    help="years materialized in parallel (separate processes;"
                         " netCDF/HDF5 is not thread-safe)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="rebuild only y_{year}.npy at config.LABEL_DILATION "
                         "(fast; use after changing the dilation)")
    a = ap.parse_args(argv)

    if a.labels_only:
        todo = [y for y in a.years if a.overwrite or labels_stale(y)]
        print(f"relabeling {len(todo)}/{len(a.years)} years at dilation "
              f"{config.LABEL_DILATION} -> {config.SHARD_DIR}", flush=True)
        for y in todo:
            print(f"  {y}: {rematerialize_labels(y)} samples", flush=True)
        return

    todo = [y for y in a.years if a.overwrite or not year_done(y)]
    print(f"materializing {len(todo)}/{len(a.years)} years -> "
          f"{config.SHARD_DIR}", flush=True)
    if a.workers > 1 and len(todo) > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            counts = pool.map(materialize_year, todo)
    else:
        counts = [materialize_year(y) for y in todo]
    for y, n in zip(todo, counts):
        print(f"  {y}: {n} samples", flush=True)


if __name__ == "__main__":
    main()
