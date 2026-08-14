"""Pair MERRA-2 surface fields with rasterized front labels (DL-FRONT).

Element spec (paper section 3.1):
  x      float32 (68, 141, 5)          standardized T2M/QV2M/SLP/U10M/V10M
  y_true float32 (68, 141, n_cls + 1)  one-hot classes + trailing pixel
                                       weight = the Fig. 2 region mask

No padding: the DL-FRONT CNN is fully convolutional with 'same' convolutions
and works on the native 68 x 141 grid.  Normalization constants are computed
once from the training years and frozen for every later stage (house rule,
front_finder.dataset).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

from front_finder import config as fd_config
from front_finder import labels as fd_labels

from . import config
from .acquire_merra2_sfc import day_path


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def load_label_ds(year: int, n_classes: int) -> xr.Dataset:
    """One year of 3wide labels: CODSUS (5-class) or NOAA-schema (6-class)."""
    if n_classes == 5:
        return fd_labels.load_codsus(year, width=config.LABEL_WIDTH,
                                     masked=False)
    if n_classes == 6:
        # unmasked like the CODSUS branch: the loss-weight mask (crop /
        # analysis domain per stage, train.loss_mask_for) enters as the
        # per-pixel weight, not as label fill
        return fd_labels.load_noaa(year, width=config.LABEL_WIDTH,
                                   masked=False)
    raise ValueError(f"n_classes must be 5 or 6, got {n_classes}")


def class_names(n_classes: int) -> tuple:
    return {5: config.CLASS_NAMES_5, 6: config.CLASS_NAMES_6}[n_classes]


def class_grid(ds: xr.Dataset, n_classes: int) -> np.ndarray:
    """(time, lat, lon) uint8 class index from the per-type channels.

    The paper's labels assign "one and only one category" per cell
    (section 3.1); where the published per-type rasters overlap (crossing
    fronts), ties resolve by the renderer's painter priority
    ``config.TYPE_PRIORITY`` (warm highest).  'none' = the last class index.
    """
    names = class_names(n_classes)
    front_type = [str(s) for s in ds["front_type"].values]
    cls = np.full(ds["fronts"].shape[::2][:1] + tuple(ds["fronts"].shape[2:]),
                  len(names) - 1, dtype=np.uint8)      # (time, lat, lon) none
    # paint lowest priority first so higher priority overwrites
    ordered = [n for n in reversed(config.TYPE_PRIORITY) if n in names]
    for name in ordered:
        hit = ds["fronts"].isel(front=front_type.index(name)).values == 1
        cls[hit] = names.index(name)
    return cls


def valid_label_steps(lab: xr.Dataset, n_classes: int = 5) -> np.ndarray:
    """Boolean (time,): True where the step holds a real analysis.

    The NOAA-XML year files pad missing analyses as all-LABEL_FILL time
    steps instead of omitting them (the archive only begins 2006-12-20, so
    2006 is ~97 % padding; audit 2026-08-10 found fill inside the region
    mask is always all-or-nothing).  A fill cell means "no analysis", not
    "no front", so any step with fill inside the mask must not be trained
    or scored.  CODSUS files omit gaps and never contain fill.

    The guard mask follows the track (review 2026-08-13): the 5-class
    paper replication trains/scores inside :func:`region_mask`, but the
    6-class track's widest loss/scoring mask is :func:`crop_domain` (stage
    A), 274 of whose pixels lie OUTSIDE the region mask -- a hypothetical
    partial-fill step confined to that southern/oceanic band would
    otherwise train on LABEL_FILL as if it were front labels.
    """
    fill = (lab["fronts"].values == fd_config.LABEL_FILL).any(axis=1)
    mask = crop_domain() if n_classes == 6 else region_mask().astype(bool)
    return ~(fill & mask).any(axis=(1, 2))


def region_mask() -> np.ndarray:
    """The Fig. 2 training/evaluation mask (>40 CSB crossings/yr envelope),
    float32 (68, 141) in {0, 1}.  Still the mask of the 5-class paper
    replication; the 6-class dryline/AIRS track uses :func:`analysis_domain`
    / :func:`crop_domain` instead (user decision 2026-08-13: the full codsus
    mask was 'far too large' for the AIRS product)."""
    with xr.open_dataset(config.REGION_MASK_PATH) as ds:
        return np.nan_to_num(ds["codsus_mask"].values).astype(np.float32)


# --------------------------------------------------------------------------- #
# Analysis domain of the 6-class dryline/AIRS track (user decision 2026-08-13)
# --------------------------------------------------------------------------- #

#: Module-level cache of the domain masks, keyed by every config value that
#: enters them so a monkeypatched constant (tests, experiment YAMLs) cannot
#: serve a stale mask.
_DOMAIN_CACHE: dict = {}


def _domain_box(lat_range, lon_range) -> np.ndarray:
    """(68, 141) bool: cells inside the INCLUSIVE lat/lon box."""
    lat = np.asarray(config.LABEL_LATS)
    lon = np.asarray(config.LABEL_LONS)
    return ((lat >= lat_range[0]) & (lat <= lat_range[1]))[:, None] \
        & ((lon >= lon_range[0]) & (lon <= lon_range[1]))[None, :]


def _land_fraction() -> np.ndarray:
    """(68, 141) float: global 1-deg land fraction ('lsm', half-degree cell
    centers) bilinearly interpolated onto the integer label grid."""
    with xr.open_dataset(config.LAND_MASK_PATH) as ds:
        return ds["lsm"].interp(lat=list(config.LABEL_LATS),
                                lon=list(config.LABEL_LONS),
                                method="linear").values


def analysis_domain() -> np.ndarray:
    """The 6-class ANALYSIS domain, (68, 141) bool: every scored pixel.

    Box lat 32..53 N, lon -107..-64 E (inclusive) intersected with
    interpolated land fraction >= LAND_FRACTION_MIN (user decision
    2026-08-13).  ALL 6-class scoring is restricted to it: training loss in
    stages B/C, every evaluate_test leg (bk19 included), and the
    krige_validate metrics.
    """
    key = ("analysis", config.ANALYSIS_LAT_RANGE, config.ANALYSIS_LON_RANGE,
           config.LAND_FRACTION_MIN, str(config.LAND_MASK_PATH))
    if key not in _DOMAIN_CACHE:
        box = _domain_box(config.ANALYSIS_LAT_RANGE, config.ANALYSIS_LON_RANGE)
        land = _land_fraction() >= config.LAND_FRACTION_MIN
        _DOMAIN_CACHE[key] = box & land
    return _DOMAIN_CACHE[key]


def halo_px() -> int:
    """The network receptive-field radius in pixels (= degrees at 1 deg).

    DERIVED, not tuned (user decision 2026-08-13): each of the
    N_CONV_LAYERS 'same' convolutions plus the softmax head conv (the +1)
    widens the receptive field by KERNEL_SIZE // 2 pixels, so beyond
    box + halo nothing can influence an in-box prediction.  Default config:
    (3 + 1) * (5 // 2) = 8 px.

    Per-layer reach is k // 2, NOT (k - 1) // 2 (review 2026-08-13): TF
    'same' padding is asymmetric for even kernels (pad_end = k // 2), so
    an even tunable kernel_size reaches k // 2 pixels on the bottom/right
    side; k // 2 covers both sides and equals (k - 1) // 2 at every odd k
    (exact at the default k = 5).
    """
    return (config.N_CONV_LAYERS + 1) * (config.KERNEL_SIZE // 2)


def crop_domain() -> np.ndarray:
    """The CROP domain, (68, 141) bool: analysis box + minimal kriging halo.

    The box expanded by :func:`halo_px` degrees on all sides (clipped to
    the label grid), with NO land or codsus intersection -- this is the
    kriging extent of the stage-B/C cache builds (halo filled from real
    observations, kriged AIRS-informed data, accepted drift and all), the
    stage-A loss mask, and the input-context extent; outside it nothing is
    filled (imputed standardized 0 at load, zero loss, gap_type -1).
    """
    key = ("crop", config.ANALYSIS_LAT_RANGE, config.ANALYSIS_LON_RANGE,
           halo_px())
    if key not in _DOMAIN_CACHE:
        h = halo_px()
        lat0, lat1 = config.ANALYSIS_LAT_RANGE
        lon0, lon1 = config.ANALYSIS_LON_RANGE
        _DOMAIN_CACHE[key] = _domain_box((lat0 - h, lat1 + h),
                                         (lon0 - h, lon1 + h))
    return _DOMAIN_CACHE[key]


# --------------------------------------------------------------------------- #
# Normalization constants (frozen)
# --------------------------------------------------------------------------- #

def compute_norm_stats(years=config.TRAIN_YEARS_5, step_days: int = 3,
                       path=config.NORM_STATS_PATH) -> dict:
    """Per-variable mean/std over the training years -> frozen JSON."""
    acc = {v: [0.0, 0.0, 0] for v in config.SFC_VARS}   # sum, sumsq, n
    for year in years:
        for date in pd.date_range(f"{year}-01-01", f"{year}-12-31",
                                  freq=f"{step_days}D"):
            p = day_path(date)
            if not p.exists():
                continue
            with xr.open_dataset(p) as day:
                for v in config.SFC_VARS:
                    a = day[v].values.astype(np.float64)
                    acc[v][0] += a.sum()
                    acc[v][1] += (a * a).sum()
                    acc[v][2] += a.size
    stats = {}
    for v, (s, ss, n) in acc.items():
        mean = s / n
        stats[v] = [mean, float(np.sqrt(ss / n - mean * mean))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=1))
    return stats


def load_norm_stats(path=config.NORM_STATS_PATH) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Sample assembly
# --------------------------------------------------------------------------- #

def sfc_x(day: xr.Dataset, stats: dict) -> np.ndarray:
    """(time, 68, 141, 5) standardized inputs from one daily surface file."""
    x = np.stack([(day[v].values - stats[v][0]) / stats[v][1]
                  for v in config.SFC_VARS], axis=-1)
    return x.astype(np.float32)


def year_arrays(year: int, n_classes: int, stats: dict
                ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """All (x, class-index) pairs of one year, inner-joined on time.

    Returns x (n, 68, 141, 5) float16, y (n, 68, 141) uint8, times.
    x is stored half-precision so the 5-7 training years fit alongside
    TensorFlow's tensor copy in 15 GB of RAM (standardized values are
    ~N(0,1); float16's ~1e-3 relative resolution is far below retrieval
    error) and cast back to float32 per batch.
    """
    with load_label_ds(year, n_classes) as lab:
        keep = valid_label_steps(lab, n_classes)
        cls = class_grid(lab, n_classes)[keep]
        label_times = pd.DatetimeIndex(lab["time"].values)[keep]

    xs, ys, ts = [], [], []
    for date, idx in label_times.groupby(label_times.normalize()).items():
        p = day_path(pd.Timestamp(date))
        if not p.exists():
            continue
        with xr.open_dataset(p) as day:
            day = day.load()
        day_times = pd.DatetimeIndex(day["time"].values)
        x_day = sfc_x(day, stats)
        for t in idx:
            j = np.flatnonzero(day_times == t)
            if len(j) == 0:
                continue
            xs.append(x_day[int(j[0])])
            ys.append(cls[label_times.get_loc(t)])
            ts.append(t)
    return (np.stack(xs).astype(np.float16), np.stack(ys),
            pd.DatetimeIndex(ts))


def stack_years(years, n_classes: int, stats: dict):
    """Concatenate :func:`year_arrays` over ``years``."""
    parts = [year_arrays(y, n_classes, stats) for y in years]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            parts[0][2].append([p[2] for p in parts[1:]]))


# --------------------------------------------------------------------------- #
# Kriged gap-filled caches (dl_front.krige_fill output)
# --------------------------------------------------------------------------- #

# ``--source`` value -> cache directory: config.KRIGED_SOURCE_DIRS holds the
# full Paths since the manifest reorg 2026-08-13 (the old KRIGED_DIR root and
# this module's source->subdir map are both retired).


def kriged_year_arrays(year: int, n_classes: int, stats: dict, source: str
                       ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """:func:`year_arrays`, but with inputs from a kriged cache file.

    ``source`` is one of :data:`config.KRIGED_SOURCE_DIRS`.  The cache files
    (``kriged_sfc_{year}.nc``) hold SFC_VARS in PHYSICAL units on the label
    grid, gap-filled inside the analysis domain, so they z-score with the
    SAME frozen stats as the reanalysis and pair with the labels by the
    same exact-timestamp inner join.  Returns x (n, 68, 141, 5) float16,
    y (n, 68, 141) uint8, times -- identical spec to :func:`year_arrays`.

    Schema v3 caches (user decision 2026-08-13) are NaN everywhere outside
    the CROP domain (:func:`crop_domain` = analysis box + receptive-field
    halo); after z-scoring those pixels are imputed to 0.0 -- the
    standardized mean, the same convention degrade_sfc uses for gap
    imputation -- so x stays finite for the CNN.  Because the halo is
    exactly the receptive-field radius, the imputed pixels can never
    influence an in-box prediction.  NaN INSIDE the crop means a corrupt
    cache and raises loudly.  The ``gap_type`` variable is carried in the
    file but not consumed here yet (future extra-channel experiment).

    Schema guard (review 2026-08-13): a v1 cache (kriged over the FULL
    grid, no ``gap_type``, no ``schema_version`` attr) contains no NaN at
    all, so it passes the corrupt-cache check while silently feeding real
    out-of-crop values where v3 feeds the 0.0 impute -- and the builder
    and chain both skip existing files, so it would never be migrated.
    A v2 cache is kriged over the old (region-mask) domain, which does not
    cover the new crop.  The loader therefore refuses any cache whose
    ``schema_version`` attr is not 3 and names the rebuild command.
    """
    path = config.KRIGED_SOURCE_DIRS[source] / f"kriged_sfc_{year}.nc"
    with load_label_ds(year, n_classes) as lab:
        keep = valid_label_steps(lab, n_classes)
        cls = class_grid(lab, n_classes)[keep]
        label_times = pd.DatetimeIndex(lab["time"].values)[keep]

    with xr.open_dataset(path) as kriged:
        kriged = kriged.load()
    build_cmd = {"kriged-degraded": "build-degraded",
                 "kriged-airs": "build-airs"}[source]
    rebuild = (f"rebuild it with 'python -m dl_front.krige_fill "
               f"{build_cmd} --years {year} --force'")
    version = kriged.attrs.get("schema_version")   # missing attr = v1 cache
    if version != 3:
        raise ValueError(
            f"{path}: schema_version={version!r} -- this loader requires "
            f"schema v3 (crop-domain fills: analysis box + halo, user "
            f"decision 2026-08-13; a missing attr means a v1 cache kriged "
            f"over the full grid, and v2 caches cover the old region-mask "
            f"domain, which the new crop is not a subset of).  " +
            rebuild.capitalize() + ".")
    # A v3 cache is self-describing: the builders record the resolved
    # domain decision in its attrs precisely so a cache built under a
    # DIFFERENT box/halo (both are tunables) is refused with the right
    # diagnosis instead of silently mismatched provenance -- or, when the
    # crop grew, a misleading 'cache is corrupt' NaN error (review
    # 2026-08-13).  land_fraction_min is deliberately NOT compared: it
    # gates only the analysis (scoring) mask, never the crop the cache
    # fills, so a cache stays valid across land-threshold changes.
    stored = {"domain_lat_range": tuple(
                  np.asarray(kriged.attrs.get("domain_lat_range", ()),
                             float).tolist()),
              "domain_lon_range": tuple(
                  np.asarray(kriged.attrs.get("domain_lon_range", ()),
                             float).tolist()),
              "halo_px": int(kriged.attrs.get("halo_px", -1))}
    expected = {"domain_lat_range": tuple(float(v) for v in
                                          config.ANALYSIS_LAT_RANGE),
                "domain_lon_range": tuple(float(v) for v in
                                          config.ANALYSIS_LON_RANGE),
                "halo_px": halo_px()}
    if stored != expected:
        diffs = ", ".join(f"{k}: cache={stored[k]!r} != config={expected[k]!r}"
                          for k in expected if stored[k] != expected[k])
        raise ValueError(
            f"{path}: schema v3 cache built under a DIFFERENT domain "
            f"configuration ({diffs}) -- its crop extent does not match "
            f"the current configs/dl_front.yaml domain/halo, so its fills "
            f"would carry mismatched provenance; {rebuild}.")
    kriged_times = pd.DatetimeIndex(kriged["time"].values)
    common = kriged_times.intersection(label_times)
    x = sfc_x(kriged.sel(time=common), stats)
    crop = crop_domain()
    bad = np.isnan(x) & crop[None, :, :, None]
    if bad.any():
        steps = common[bad.any(axis=(1, 2, 3))]
        raise ValueError(
            f"{path}: {int(bad.sum())} NaN pixel(s) INSIDE the crop "
            f"domain at {list(steps[:5])} -- schema v3 guarantees the "
            f"crop gap-free, so this cache is corrupt; rebuild it with "
            f"'python -m dl_front.krige_fill ... --force'")
    x = np.nan_to_num(x, nan=0.0)       # out-of-crop -> standardized mean
    y = cls[label_times.get_indexer(common)]
    return x.astype(np.float16), y, common


def stack_kriged_years(years, n_classes: int, stats: dict, source: str):
    """Concatenate :func:`kriged_year_arrays` over ``years``."""
    parts = [kriged_year_arrays(y, n_classes, stats, source) for y in years]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            parts[0][2].append([p[2] for p in parts[1:]]))


def filter_hours(x: np.ndarray, y: np.ndarray, times: pd.DatetimeIndex,
                 hours) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Subset an (x, y, times) triple to steps whose UTC hour is in ``hours``.

    Used to restrict training/evaluation to the AIRS-covered label hours
    (``config.AIRS_HOURS``), so reanalysis and kriged-AIRS runs see
    identical time steps.
    """
    keep = np.isin(times.hour, list(hours))
    return x[keep], y[keep], times[keep]


def fold_split(n: int, fold: int, n_folds: int = config.N_FOLDS,
               seed: int = config.FOLD_SEED):
    """Paper section 4.1: each fold trains on a random 2/3 of the samples.

    A seeded permutation is cut into ``n_folds`` contiguous validation
    chunks; fold k validates on chunk k and trains on the rest.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    perm = np.random.default_rng(seed).permutation(n)
    chunks = np.array_split(perm, n_folds)
    val = chunks[fold]
    train = np.concatenate([c for i, c in enumerate(chunks) if i != fold])
    return train, val


def make_tf_dataset(x: np.ndarray, y: np.ndarray, n_classes: int,
                    batch_size: int = config.BATCH_SIZE,
                    shuffle: bool = True, weights: np.ndarray | None = None):
    """In-memory tf.data pipeline: one-hot + trailing region-mask weight.

    ``weights``: per-pixel loss weight, (68, 141) applied to every sample or
    (n, 68, 141) per sample (stage C swath masks); defaults to the region
    mask.  y stays uint8 in memory; one-hot happens on the fly.
    """
    import tensorflow as tf

    w = region_mask() if weights is None else weights.astype(np.float32)
    per_sample = w.ndim == 3
    w_t = tf.constant(w) if not per_sample else None

    if per_sample:
        ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=config.FOLD_SEED,
                        reshuffle_each_iteration=True)

    def to_pair(*args):
        xi, yi = args[0], args[1]
        wi = args[2] if per_sample else w_t
        y_true = tf.concat([tf.one_hot(tf.cast(yi, tf.int32), n_classes),
                            wi[..., None]], axis=-1)
        return tf.cast(xi, tf.float32), y_true

    return (ds.map(to_pair, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))
