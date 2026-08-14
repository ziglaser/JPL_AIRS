"""tf.data pipelines pairing MERRA-2 predictors with analyst front labels.

Labels come from ``config.LABEL_SOURCE`` (labels.load_fronts): CODSUS gives
the 4 classical types; ``JPL_FRONT_LABELS=noaa`` adds the dryline class and
every tensor below grows one channel.

Element spec (workplan sections 3.3-3.5):
  x      float32 (72, 144, 5, C)        channels = THERMO_VARS [+ u,v] + mask
  y_true float32 (72, 144, n_cls + 1)   one-hot (none first, then
                                        config.FRONT_TYPES) + trailing
                                        pixel-weight channel for the masked FSS

Normalization is min-max per (variable, level) -- paper section 2b -- with
constants computed ONCE from the pretraining train years and frozen for every
later stage (recomputing on AIRS would silently change what the imputation
value means).  Invalid pixels are imputed to 0.5 (the normalized midpoint)
AFTER the mask channel is attached; grid padding pixels carry weight 0.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

from . import config, derive, labels
from .acquire_merra2 import day_path

NORM_STATS_PATH = config.MERRA2_DIR / "norm_stats.json"
#: none-class first: index 0 = "no front", 1..4 = config.FRONT_TYPES.
CLASS_NAMES = ("none",) + config.FRONT_TYPES
PAD_LAT = config.PADDED_SHAPE[0] - config.GRID_SHAPE[0]   # 4
PAD_LON = config.PADDED_SHAPE[1] - config.GRID_SHAPE[1]   # 3
IMPUTE_VALUE = 0.5   # normalized midpoint; distinguishable via mask channel


def channel_names(winds: bool) -> tuple:
    return config.THERMO_VARS + (config.WIND_VARS if winds else ()) + ("mask",)


#: Paper section 2c augmentation: each horizontal dimension is flipped
#: independently with 25% probability, so the model cannot learn geographic
#: orientation priors (e.g. "dry air is always west of a dryline").
FLIP_CHANCE = 0.25


def random_flip(x: np.ndarray, y: np.ndarray, rng) -> tuple:
    """Independently flip the lat and lon axes of one (x, y) pair.

    Both arrays lead with (lat, lon); everything trailing (levels, channels,
    classes, the loss-weight channel) flips with its pixel.  Applied to
    stage A/B TRAINING streams only -- never validation, never stage C
    (workplan 3.6: lon-crop only there).
    """
    if rng.random() < FLIP_CHANCE:
        x, y = x[::-1], y[::-1]
    if rng.random() < FLIP_CHANCE:
        x, y = x[:, ::-1], y[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


# --------------------------------------------------------------------------- #
# Normalization constants (frozen)
# --------------------------------------------------------------------------- #

def compute_norm_stats(years=config.PRETRAIN_TRAIN_YEARS, step_days: int = 3,
                       path=NORM_STATS_PATH) -> dict:
    """Per (variable, level) min/max over the pretraining train years.

    Every ``step_days``-th day is scanned (min-max is insensitive to modest
    subsampling; values falling slightly outside [0, 1] later are harmless).
    """
    lo, hi = {}, {}
    for year in years:
        for date in pd.date_range(f"{year}-01-01", f"{year}-12-31",
                                  freq=f"{step_days}D"):
            p = day_path(date)
            if not p.exists():
                continue
            with xr.open_dataset(p) as day:
                ch = derive.merra2_channels(day.load(), winds=True)
                for var in ch.data_vars:
                    for lev in config.TARGET_LEVELS_HPA:
                        key = f"{var}_{lev}"
                        v = ch[var].sel(lev=lev).values
                        if np.isnan(v).all():
                            continue
                        lo[key] = min(lo.get(key, np.inf), float(np.nanmin(v)))
                        hi[key] = max(hi.get(key, -np.inf), float(np.nanmax(v)))
    stats = {k: [lo[k], hi[k]] for k in sorted(lo)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=1))
    return stats


def load_norm_stats(path=NORM_STATS_PATH) -> dict:
    return json.loads(path.read_text())


def class_weights(years=config.PRETRAIN_TRAIN_YEARS, cap: float = 20.0,
                  dilation: int = config.LABEL_DILATION,
                  masked_labels: bool = True) -> list:
    """Inverse-sqrt-frequency class weights -- DIAGNOSTICS ONLY (2026-08-10).

    Training deliberately passes NO class weights (train.py/model.build):
    the paper trains its FSS loss unweighted at worse imbalance than ours
    (their occluded 0.22%/dryline 0.06% of pixels vs our ~0.3-1.1% per
    class at 1 deg -- line-pixel fraction scales with grid spacing), the
    epoch-41 quicklook showed no class collapse (max probability 0.92-0.99
    for every class), and up-weighting rare classes shifts the operating
    point toward over-forecasting -- the exact FB pathology the dilation-0
    retrain is meant to remove.  Revisit ONLY if a class collapses at every
    threshold of the eval sweep, and then gently (3-5x on that class, not
    the capped-20 schedule).  This function stays for imbalance monitoring;
    "none" anchored at 1.0, front classes capped at ``cap``.
    Order matches CLASS_NAMES.
    """
    counts = np.zeros(len(CLASS_NAMES))
    for year in years:
        try:
            ds = labels.load_fronts(year, width=1, masked=masked_labels)
        except FileNotFoundError:
            continue
        fr = labels.front_stack(ds).values
        fr = labels.dilate(fr.reshape(-1, *config.GRID_SHAPE),
                           dilation).reshape(fr.shape)
        valid = labels.valid_mask(ds).values
        counts[0] += (valid & ~fr.any(1)).sum()
        counts[1:] += (fr & valid[:, None]).sum(axis=(0, 2, 3))
        ds.close()
    freq = counts / counts.sum()
    w = np.sqrt(freq[0] / freq)                 # none-class anchored at 1
    return list(np.minimum(w, cap).round(3))


# --------------------------------------------------------------------------- #
# Sample assembly
# --------------------------------------------------------------------------- #

def _pad(a: np.ndarray) -> np.ndarray:
    """Zero-pad the leading (lat, lon) axes 68x141 -> 72x144 (workplan 3.1)."""
    return np.pad(a, ((2, PAD_LAT - 2), (1, PAD_LON - 1))
                  + ((0, 0),) * (a.ndim - 2))


def make_x(ch: xr.Dataset, stats: dict, winds: bool) -> np.ndarray:
    """(72, 144, 5, C) normalized inputs from a single-timestep channel set."""
    names = channel_names(winds)
    x = np.empty((*config.GRID_SHAPE, len(config.TARGET_LEVELS_HPA), len(names)),
                 dtype=np.float32)
    for c, var in enumerate(names[:-1]):
        for k, lev in enumerate(config.TARGET_LEVELS_HPA):
            mn, mx = stats[f"{var}_{lev}"]
            x[..., k, c] = (ch[var].sel(lev=lev).values - mn) / (mx - mn)
    valid = np.isfinite(x[..., 0])                    # (lat, lon, lev) via T
    x[..., -1] = valid.astype(np.float32)             # mask channel
    x = np.nan_to_num(x, nan=IMPUTE_VALUE)
    x[..., :-1] = np.where(~valid[..., None], IMPUTE_VALUE, x[..., :-1])
    return _pad(x)


def make_y(fronts_t: np.ndarray, valid_t: np.ndarray) -> np.ndarray:
    """(72, 144, 6) one-hot labels + trailing weight channel.

    fronts_t: bool (front, lat, lon), ALREADY dilated; valid_t: bool
    (lat, lon) label validity (fill pixels / outside analysis mask).
    Overlapping classes resolve to the first True in FRONT_TYPES order.
    """
    n_cls = len(CLASS_NAMES)
    cls = np.zeros(config.GRID_SHAPE, dtype=np.int64)
    for k in range(len(config.FRONT_TYPES) - 1, -1, -1):
        cls[fronts_t[k]] = k + 1
    y = np.eye(n_cls, dtype=np.float32)[cls]          # (lat, lon, 6? no: n_cls)
    w = valid_t.astype(np.float32)[..., None]
    return _pad(np.concatenate([y, w], axis=-1))


# --------------------------------------------------------------------------- #
# Stage B: degraded-reanalysis samples ("AIRS simulator", workplan 3.6-B)
# --------------------------------------------------------------------------- #

#: Bulletin hours nearest the AIRS afternoon overpasses (~18:50/20:35 UTC).
OVERPASS_HOURS = (18, 21)


def degraded_year_samples(year: int, stats: dict, winds: bool,
                          severity, rng, vf_sampler,
                          dilation: int = config.LABEL_DILATION,
                          masked_labels: bool = True,
                          hours: tuple = OVERPASS_HOURS):
    """Stage-B generator: degrade -> derive -> gap-mask, overpass hours only.

    ``severity`` is a float or a zero-arg callable (read per day so a
    ramp callback can move it mid-stream).  Degradation (vertical mixing +
    horizontally correlated noise) hits T/QV BEFORE derived variables;
    ``vf_sampler(rng, month)`` supplies the gap field -- a REAL AIRS
    valid-fraction from the bank once it is big enough, else a synthetic
    swath+cloud field (see make_degraded_tf_dataset / synth_gaps).
    The loss weight is unchanged -- gaps stay scored (workplan 3.4).
    """
    from . import degrade, labels as L, mask_bank

    truth_ds = L.load_fronts(year, width=1, masked=masked_labels)
    fronts = L.front_stack(truth_ds).values
    fronts = L.dilate(fronts.reshape(-1, *config.GRID_SHAPE),
                      dilation).reshape(fronts.shape)
    valid = L.valid_mask(truth_ds).values
    times = pd.DatetimeIndex(truth_ds["time"].values)
    truth_ds.close()

    for date, idx in times.groupby(times.normalize()).items():
        p = day_path(pd.Timestamp(date))
        if not p.exists():
            continue
        s = severity() if callable(severity) else severity
        with xr.open_dataset(p) as day:
            day = degrade.degrade_day(day.load(), rng, severity=s)
        ch = derive.merra2_channels(day, winds=winds)
        day_hours = pd.DatetimeIndex(day["time"].values)
        vf = vf_sampler(rng, pd.Timestamp(date).month)
        # blend toward all-valid at low severity (curriculum)
        vf_s = 1.0 - s * (1.0 - vf)
        for t in idx:
            if t.hour not in hours:
                continue
            i = np.flatnonzero(times == t)[0]
            j = np.flatnonzero(day_hours == t)
            if len(j) == 0:
                continue
            x = make_x(ch.isel(time=int(j[0])), stats, winds)
            if s > 0:
                x = mask_bank.apply_mask(x, vf_s, IMPUTE_VALUE)
            yield x, make_y(fronts[i], valid[i])


def make_vf_sampler():
    """Gap-field sampler for stage B: real bank if it is big enough, else
    synthetic swath+cloud fields (synth_gaps heuristics, 2026-08-10).

    Returns ``(sampler, source_name)``; ``sampler(rng, month) -> vf
    (lat, lon, lev)``.  The real bank wins once it holds
    ``mask_bank.MIN_REAL_BANK`` fields (seasonal/geometry diversity) --
    it will after the multi-year fullgrid harvest.
    """
    from . import mask_bank, synth_gaps

    try:
        bank_vf, bank_dates = mask_bank.load_bank()
    except FileNotFoundError:
        bank_vf = None
    if bank_vf is not None and len(bank_vf) >= mask_bank.MIN_REAL_BANK:
        return (lambda rng, month: mask_bank.sample_mask(
                    bank_vf, rng, month=month, dates=bank_dates),
                f"real bank (n={len(bank_vf)})")
    n = 0 if bank_vf is None else len(bank_vf)
    return (synth_gaps.synthetic_valid_fraction,
            f"synthetic (real bank has {n} < {mask_bank.MIN_REAL_BANK})")


def make_degraded_tf_dataset(years, winds: bool, batch_size: int,
                             severity, seed: int | None = None,
                             stats: dict | None = None, shuffle: bool = True,
                             freeze_realization: bool = False,
                             augment: bool = False):
    """Batched stage-B dataset.  No cache: noise/masks resample every pass.

    ``freeze_realization=True`` (validation): the RNG is re-seeded at every
    generator restart, so every epoch sees the IDENTICAL noise and gap
    realization -- early stopping then compares model changes, not noise
    draws.  Training leaves it False so realizations stay fresh.
    ``augment=True`` adds the paper's 25%-per-axis flips (training only).
    """
    import tensorflow as tf

    stats = stats or load_norm_stats()
    vf_sampler, vf_source = make_vf_sampler()
    print(f"stage-B gap fields: {vf_source}", flush=True)
    base_seed = config.BOOT_SEED if seed is None else seed
    rng = np.random.default_rng(base_seed)
    n_ch = len(channel_names(winds))
    sig = (tf.TensorSpec((*config.PADDED_SHAPE, 5, n_ch), tf.float32),
           tf.TensorSpec((*config.PADDED_SHAPE, len(CLASS_NAMES) + 1),
                         tf.float32))

    def gen():
        r = np.random.default_rng(base_seed) if freeze_realization else rng
        for year in years:
            for x, y in degraded_year_samples(year, stats, winds, severity,
                                              r, vf_sampler):
                yield random_flip(x, y, r) if augment else (x, y)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if shuffle:
        ds = ds.shuffle(256, seed=config.BOOT_SEED,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Stage C: real-AIRS fine-tune samples (fullgrid AIRS-FCST files)
# --------------------------------------------------------------------------- #

def airs_x(path, stats: dict, winds: bool, slot: int = 0):
    """One fullgrid file -> (x, observed, bulletin_time).

    x is (72, 144, 5, C) exactly as the pretraining inputs (same frozen
    normalization, same imputation, mask channel = AIRS validity);
    ``observed`` is the unpadded (68, 141) bool swath mask for the loss
    weight; ``bulletin_time`` the paired 3-hourly CSB time (workplan 3.7).
    """
    from . import ingest_hysplit as ih

    ch = ih.to_label_grid(ih.load_fullgrid(path), slot=slot, winds=winds)
    x = make_x(ch, stats, winds)
    # make_x's mask channel is binary validity; refine it to the graded
    # valid_fraction (workplan 3.3) where observed
    vf = ch["valid_fraction"].transpose("lat", "lon", "lev").values
    x[..., -1] = _pad(np.nan_to_num(vf.astype(np.float32)))
    t = ih.nearest_bulletin(ih.overpass_time(path))
    return x, ch["observed"].values, t


def airs_samples(paths, stats: dict, winds: bool,
                 dilation: int = config.LABEL_DILATION,
                 masked_labels: bool = True, slot: int = 0):
    """Yield (x, y) fine-tune pairs from fullgrid files with CSB labels.

    Loss weight = label validity AND the AIRS swath mask -- off-swath pixels
    are never penalized (workplan 3.4).  Files whose paired bulletin has no
    CODSUS label (year > 2018 until the CSB extension lands) are skipped
    with a warning rather than failing the whole run.
    """
    import warnings

    label_cache: dict = {}
    for path in paths:
        x, observed, t = airs_x(path, stats, winds, slot)
        year = t.year
        if year not in label_cache:
            try:
                truth_ds = labels.load_fronts(year, width=1,
                                              masked=masked_labels)
            except FileNotFoundError:
                label_cache[year] = None
            else:
                fr = labels.front_stack(truth_ds).values
                fr = labels.dilate(fr.reshape(-1, *config.GRID_SHAPE),
                                   dilation).reshape(fr.shape)
                label_cache[year] = (pd.DatetimeIndex(truth_ds["time"].values),
                                     fr, labels.valid_mask(truth_ds).values)
                truth_ds.close()
        if label_cache[year] is None:
            warnings.warn(f"{path}: no CODSUS labels for {year} "
                          "(CSB extension pending); skipped")
            continue
        times, fronts, valid = label_cache[year]
        i = np.flatnonzero(times == t)
        if len(i) == 0:
            warnings.warn(f"{path}: bulletin {t} missing from CODSUS; skipped")
            continue
        yield x, make_y(fronts[i[0]], valid[i[0]] & observed)


def make_airs_tf_dataset(paths, winds: bool, batch_size: int,
                         shuffle: bool = True, stats: dict | None = None):
    """Batched tf.data.Dataset of fine-tune pairs (mirrors make_tf_dataset)."""
    import tensorflow as tf

    stats = stats or load_norm_stats()
    n_ch = len(channel_names(winds))
    sig = (tf.TensorSpec((*config.PADDED_SHAPE, 5, n_ch), tf.float32),
           tf.TensorSpec((*config.PADDED_SHAPE, len(CLASS_NAMES) + 1),
                         tf.float32))
    paths = [str(p) for p in paths]

    def gen():
        yield from airs_samples(paths, stats, winds)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if shuffle:
        ds = ds.shuffle(256, seed=config.BOOT_SEED,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Generators / tf.data
# --------------------------------------------------------------------------- #

def year_samples(year: int, stats: dict, winds: bool,
                 dilation: int = config.LABEL_DILATION,
                 masked_labels: bool = True, return_times: bool = False):
    """Yield (x, y) for every labeled timestep of ``year`` with MERRA-2 data.

    ``return_times=True`` yields (x, y, timestep) instead -- used by
    predict.predict_year to keep inference inputs identical to training ones.
    """
    truth_ds = labels.load_fronts(year, width=1, masked=masked_labels)
    fronts = labels.front_stack(truth_ds).values      # (time, front, lat, lon)
    fronts = labels.dilate(
        fronts.reshape(-1, *config.GRID_SHAPE), dilation).reshape(fronts.shape)
    valid = labels.valid_mask(truth_ds).values
    times = pd.DatetimeIndex(truth_ds["time"].values)
    truth_ds.close()

    for date, idx in times.groupby(times.normalize()).items():
        p = day_path(pd.Timestamp(date))
        if not p.exists():
            continue
        with xr.open_dataset(p) as day:
            day = day.load()
        ch = derive.merra2_channels(day, winds=winds)
        day_hours = pd.DatetimeIndex(day["time"].values)
        for t in idx:
            i = np.flatnonzero(times == t)[0]
            j = np.flatnonzero(day_hours == t)
            if len(j) == 0:
                continue
            x = make_x(ch.isel(time=int(j[0])), stats, winds)
            y = make_y(fronts[i], valid[i])
            yield (x, y, t) if return_times else (x, y)


def make_tf_dataset(years, winds: bool, batch_size: int, shuffle: bool = True,
                    stats: dict | None = None, cache: bool = False,
                    augment: bool = False):
    """Batched, prefetched tf.data.Dataset over the given years.

    ``cache=True`` writes decoded samples to a tf.data cache file on the
    first pass -- DO NOT enable for multi-epoch training: TF 2.15's file
    cache writer accumulates ~0.4 MB/element of native memory until the
    cache finalizes, which OOM-killed the first E1b run at epoch 43
    (post-mortem 2026-08-09).  Prefer materialized shards
    (make_shard_tf_dataset) for repeated passes.  ``augment=True`` adds the
    paper's 25%-per-axis flips (training only; incompatible with cache=True
    since cached samples would freeze one flip draw).
    """
    import tensorflow as tf

    if augment and cache:
        raise ValueError("augment=True with cache=True would cache a single "
                         "flip realization -- disable one of them")
    stats = stats or load_norm_stats()
    rng = np.random.default_rng(config.BOOT_SEED)
    n_ch = len(channel_names(winds))
    sig = (tf.TensorSpec((*config.PADDED_SHAPE, 5, n_ch), tf.float32),
           tf.TensorSpec((*config.PADDED_SHAPE, len(CLASS_NAMES) + 1),
                         tf.float32))

    def gen():
        for year in years:
            for x, y in year_samples(year, stats, winds):
                yield random_flip(x, y, rng) if augment else (x, y)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if cache:
        tag = f"{min(years)}-{max(years)}_{'wind' if winds else 'thermo'}"
        cache_dir = config.MERRA2_DIR / "tfcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_dir / tag))
    if shuffle:
        ds = ds.shuffle(256, seed=config.BOOT_SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Materialized shards (materialize.py) -- the post-2026-08-09 training path
# --------------------------------------------------------------------------- #

def _thermo_channel_idx() -> list:
    """Indices of the thermo-only channel set within the wind-inclusive one.

    Shards store channel_names(True); thermo training selects
    channel_names(False) out of them (mask is last in both orderings).
    """
    full = channel_names(True)
    return [full.index(name) for name in channel_names(False)]


def shards_exist(years) -> bool:
    from . import materialize
    return all(materialize.year_done(y) for y in years)


def make_shard_tf_dataset(years, winds: bool, batch_size: int,
                          shuffle: bool = True, seed: int | None = None,
                          augment: bool = False):
    """Batched tf.data.Dataset over materialized shards.

    Samples are read from per-year memmaps: bounded memory (OS page cache
    only), true whole-corpus random order per epoch (vs the streaming
    pipeline's 256-sample shuffle window), no derivation cost.  Each fresh
    iteration (tf re-invokes the generator, e.g. per .repeat() cycle) draws
    a new permutation from ``rng``.  ``augment=True`` adds the paper's
    25%-per-axis flips (training only) -- applied at read time, so the
    stored shards stay flip-free.
    """
    import tensorflow as tf

    from . import materialize

    xs, ys, index = [], [], []
    for yi, year in enumerate(sorted(years)):
        p = materialize._year_paths(year)
        meta = json.loads(p["meta"].read_text())
        # shards bake the labels in; pre-switch CODSUS shards have no key
        if meta.get("label_source", "codsus") != config.LABEL_SOURCE:
            raise ValueError(
                f"shard {p['x'].name} was materialized from "
                f"{meta.get('label_source', 'codsus')!r} labels but "
                f"config.LABEL_SOURCE is {config.LABEL_SOURCE!r} -- "
                f"re-run front_finder.materialize (each source has its own "
                f"shard dir)")
        # shards also bake the label dilation into y (pre-2026-08-10 shards
        # were built at dilation 1)
        if meta.get("dilation") != config.LABEL_DILATION:
            raise ValueError(
                f"shard y_{year}.npy was materialized at dilation "
                f"{meta.get('dilation')!r} but config.LABEL_DILATION is "
                f"{config.LABEL_DILATION} -- rebuild the label shards with "
                f"'python -m front_finder.materialize <years> --labels-only' "
                f"(x shards are unaffected)")
        xs.append(np.load(p["x"], mmap_mode="r"))
        ys.append(np.load(p["y"], mmap_mode="r"))
        index.extend((yi, i) for i in range(xs[-1].shape[0]))
    index = np.asarray(index, dtype=np.int64)
    ch = None if winds else np.asarray(_thermo_channel_idx())
    n_ch = len(channel_names(winds))
    sig = (tf.TensorSpec((*config.PADDED_SHAPE, 5, n_ch), tf.float32),
           tf.TensorSpec((*config.PADDED_SHAPE, len(CLASS_NAMES) + 1),
                         tf.float32))
    rng = np.random.default_rng(config.BOOT_SEED if seed is None else seed)

    def gen():
        order = rng.permutation(len(index)) if shuffle else range(len(index))
        for k in order:
            yi, i = index[k]
            x = xs[yi][i]
            x, y = (x if ch is None else x[..., ch]), ys[yi][i]
            yield random_flip(x, y, rng) if augment else (x, y)

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
