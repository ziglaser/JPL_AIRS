"""Pair MERRA-2 surface fields with rasterized front labels (DL-FRONT).

Element spec (paper section 3.1):
  x      float32 (68, 141, n_ch)       standardized config.INPUT_CHANNELS
                                       (default all five SFC_VARS:
                                       T2M/QV2M/SLP/U10M/V10M)
  y_true float32 (68, 141, n_cls + 1)  one-hot classes + trailing pixel
                                       weight = the Fig. 2 region mask

No padding: the DL-FRONT CNN is fully convolutional with 'same' convolutions
and works on the native 68 x 141 grid.  Normalization constants are computed
once from the training years and frozen for every later stage (house rule,
front_finder.dataset).
"""
from __future__ import annotations

import hashlib
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
    """Standardized model inputs from one surface file (daily or cache).

    Returns (time, 68, 141, len(config.INPUT_CHANNELS)) float32.  The
    trailing axis is the MODEL's channel set, not the file's: the file
    always carries all five ``config.SFC_VARS`` (that is the on-disk
    schema), and this stacks only the subset ``config.INPUT_CHANNELS``
    selects, in SFC_VARS order (channel-ladder work, user decision
    2026-08-18).  Subsetting by NAME is what lets the frozen norm-stats
    JSON keep all five keys and stay untouched -- ``stats`` is looked up
    per channel name, never positionally.

    Called from BOTH :func:`year_arrays` (reanalysis sfc_daily day files)
    and :func:`kriged_year_arrays` (kriged cache files), so a channel
    subset applies identically to every stage.
    """
    x = np.stack([(day[v].values - stats[v][0]) / stats[v][1]
                  for v in config.INPUT_CHANNELS], axis=-1)
    return x.astype(np.float32)


def year_arrays(year: int, n_classes: int, stats: dict
                ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """All (x, class-index) pairs of one year, inner-joined on time.

    Returns x (n, 68, 141, len(config.INPUT_CHANNELS)) float16,
    y (n, 68, 141) uint8, times.
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


def _cache_kriged_split(kriged, version, path, rebuild) -> tuple:
    """The channels a cache file gap-filled by kriging.

    Caches written before the ``kriged_channels`` attr existed (v3) carry
    their split implicitly, so it is recovered from the documented
    historical constant rather than the file being refused wholesale.
    """
    if "kriged_channels" in kriged.attrs:
        return tuple(str(v) for v in
                     np.atleast_1d(kriged.attrs["kriged_channels"]).tolist())
    legacy = config.KRIGE_LEGACY_KRIGED_CHANNELS.get(version, ())
    if not legacy:
        raise ValueError(
            f"{path}: schema v{version} carries no 'kriged_channels' attr "
            f"and no legacy split is recorded for it in "
            f"config.KRIGE_LEGACY_KRIGED_CHANNELS, so which channels are "
            f"satellite-derived cannot be established; {rebuild}.")
    return legacy


def _sfc_x_split(kriged, times, from_cache, from_rea, stats, crop, path
                 ) -> np.ndarray:
    """Standardized inputs assembled from the cache AND the reanalysis.

    ``from_cache`` channels (the kriged ones) come out of the cache file;
    ``from_rea`` channels (the clean ones) are read from ``sfc_daily`` at the
    same timestamps -- see the sourcing note in :func:`kriged_year_arrays`.
    Reanalysis fields are masked to the crop domain first so both sources
    share the schema's "nothing defined outside the crop" convention and the
    caller's single ``nan_to_num`` impute covers them identically.
    """
    n = len(times)
    x = np.full((n, *config.GRID_SHAPE, len(config.INPUT_CHANNELS)),
                np.nan, dtype=np.float32)
    idx = {c: i for i, c in enumerate(config.INPUT_CHANNELS)}
    for c in from_cache:
        mean, std = stats[c]
        x[..., idx[c]] = (kriged[c].values - mean) / std
    if not from_rea:
        return x
    # One day file per calendar day, same lookup year_arrays uses -- but read
    # as LITTLE of each as possible (review 2026-08-18).  The first version
    # ``.load()``ed the WHOLE file (all five SFC_VARS at all eight 3-hourly
    # steps) for the <= 3 label steps and 1-3 clean channels it actually
    # uses, ~365 times per year, in EVERY training job and EVERY eval leg.
    # ``drop_variables`` stops the unread channels being decompressed at all;
    # the single ``isel(time=steps).load()`` asks for only the needed steps.
    # Pure I/O: the arithmetic below is untouched, so x is BIT-IDENTICAL to
    # the old loop's (verified by A/B over a full 2016 year, 1098 steps,
    # np.array_equal on the returned float16 array).
    #
    # Honest accounting of what that buys, measured on the real 2016
    # sfc_daily corpus (1098 steps, warm cache, WSL /mnt/d): the ladder's
    # 3-channel case (from_rea = SLP only) 23.9 s -> 19.1 s per cache-year,
    # the 5-channel case (from_rea = SLP/U10M/V10M) 23.8 s -> 22.7 s.  The
    # residual is NOT decode but the 365 file OPENS themselves (11.7 s of it
    # here), which nothing inside this loader can remove; and the step
    # selection saves no decode today because the day files are written as
    # ONE zlib chunk per variable ([8, 68, 141]) -- a whole variable is
    # inflated no matter how few steps are asked for.  Both remain correct
    # and both pay off if the files are ever rechunked, so the cheap read
    # stays; a real fix for the open cost is a per-YEAR reanalysis file.
    #
    # Independent re-measurement (verification 2026-08-18), same machine but
    # a 732-step 2016 cache, 3 interleaved repeats of each loop: 3-channel
    # 24.3 s -> 22.6 s (a win, as above), but the DEFAULT 5-channel case
    # 24.2 s -> 26.2 s -- i.e. when from_rea is SLP/U10M/V10M the fancy
    # ``isel(time=steps)`` on single-chunk variables costs MORE than the
    # whole-file read it replaces, so the 5-channel number above does not
    # reproduce and the change is a ~8 % regression on the main-chain path.
    # Kept anyway (bit-identical output, and it is the 2/3-channel ladder
    # rungs that read this loader most), but do NOT quote a 5-channel
    # speed-up: the honest summary is "helps when channels are dropped,
    # slightly hurts when none are".
    drop = [v for v in config.SFC_VARS if v not in from_rea]
    for date, when in pd.Series(times, index=times).groupby(
            times.normalize()):
        day_path_ = day_path(pd.Timestamp(date))
        if not day_path_.exists():
            raise FileNotFoundError(
                f"{path} covers {when.iloc[0]:%Y-%m-%d} but the reanalysis "
                f"day file {day_path_} is missing, and "
                f"{list(from_rea)} are CLEAN channels read from the "
                f"reanalysis rather than the cache (config "
                f"airs.kriged_channels={list(config.KRIGED_CHANNELS)}).  "
                f"Fetch it with 'python -m dl_front.acquire_merra2_sfc "
                f"{pd.Timestamp(date).year}'.")
        with xr.open_dataset(day_path_, drop_variables=drop) as day:
            day_times = pd.DatetimeIndex(day["time"].values)
            steps, rows = [], []
            for t in when:
                j = np.flatnonzero(day_times == t)
                if len(j) == 0:
                    raise ValueError(
                        f"{day_path_} has no step at {t}, needed for the "
                        f"clean channels {list(from_rea)} of {path}.")
                steps.append(int(j[0]))
                rows.append(int(np.flatnonzero(times == t)[0]))
            sub = day[list(from_rea)].isel(time=steps).load()
        for k, row in enumerate(rows):
            for c in from_rea:
                mean, std = stats[c]
                grid = np.where(crop, sub[c].values[k], np.nan)
                x[row, ..., idx[c]] = (grid - mean) / std
    return x


def _raise_in_crop_nan(bad, times, from_cache, from_rea, cache_path,
                       version, rebuild) -> None:
    """Report in-crop NaN with the message the OFFENDING SOURCE deserves.

    Why this is split by provenance (review 2026-08-18): since the channel
    sourcing decision the crop is filled from TWO files -- the kriged cache
    (``from_cache``) and the ``sfc_daily`` reanalysis day file
    (``from_rea``) -- but the check inherited from the cache-only loader
    blamed the cache for every NaN and told the operator to spend hours
    rebuilding it.  Reproduced with a pristine cache and one NaN in the
    reanalysis SLP: the file named was not at fault, the rebuild could not
    have fixed it, and the message named neither the channel nor its source.
    So: cache channels keep the corrupt-cache message, reanalysis channels
    get their own naming the CHANNEL, the step, the day file and the
    acquire command.  ``cache_path`` still opens the header line because it
    is the file the caller asked for, but it is no longer accused by it.
    """
    idx = {c: i for i, c in enumerate(config.INPUT_CHANNELS)}

    def hits(channels):
        out = []
        for c in channels:
            m = bad[..., idx[c]]
            if m.any():
                out.append((c, int(m.sum()), times[m.any(axis=(1, 2))]))
        return out

    parts = []
    for c, n, steps in hits(from_cache):
        parts.append(
            f"channel {c} (read FROM THE CACHE, config "
            f"airs.kriged_channels={list(config.KRIGED_CHANNELS)}): {n} NaN "
            f"pixel(s) at {list(steps[:5])} -- schema v{version} guarantees "
            f"the crop gap-free, so this cache is corrupt; {rebuild}.")
    for c, n, steps in hits(from_rea):
        days = sorted({str(day_path(pd.Timestamp(t).normalize()))
                       for t in steps[:5]})
        years = sorted({pd.Timestamp(t).year for t in steps})
        parts.append(
            f"channel {c} (read FROM THE REANALYSIS, not from the cache: it "
            f"is not in airs.kriged_channels="
            f"{list(config.KRIGED_CHANNELS)}): {n} NaN pixel(s) at "
            f"{list(steps[:5])}, i.e. in the sfc_daily day file(s) {days}.  "
            f"The cache named above is NOT at fault and rebuilding it cannot "
            f"fix this; delete the offending day file(s) and re-fetch with "
            f"'python -m dl_front.acquire_merra2_sfc "
            f"{' '.join(str(y) for y in years)}'.")
    raise ValueError(
        f"{cache_path}: NaN pixel(s) INSIDE the crop domain, by source -- "
        + "  ".join(parts))


def kriged_year_arrays(year: int, n_classes: int, stats: dict, source: str
                       ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """:func:`year_arrays`, but with inputs from a kriged cache file.

    ``source`` is one of :data:`config.KRIGED_SOURCE_DIRS`.  The cache files
    (``kriged_sfc_{year}.nc``) hold SFC_VARS in PHYSICAL units on the label
    grid, gap-filled inside the analysis domain, so they z-score with the
    SAME frozen stats as the reanalysis and pair with the labels by the
    same exact-timestamp inner join.  Returns x (n, 68, 141,
    len(config.INPUT_CHANNELS)) float16, y (n, 68, 141) uint8, times --
    identical spec to :func:`year_arrays`.

    Schema v3 caches (user decision 2026-08-13) are NaN everywhere outside
    the CROP domain (:func:`crop_domain` = analysis box + receptive-field
    halo); after z-scoring those pixels are imputed to 0.0 -- the
    standardized mean, the same convention degrade_sfc uses for gap
    imputation -- so x stays finite for the CNN.  Because the halo is
    exactly the receptive-field radius, the imputed pixels can never
    influence an in-box prediction.  NaN INSIDE the crop always raises, and
    since the split sourcing it raises against the file that actually holds
    it -- the cache OR the reanalysis day file (:func:`_raise_in_crop_nan`,
    review 2026-08-18).  The ``gap_type`` variable is carried in the file
    but not consumed here yet (future extra-channel experiment).

    Refuses outright (review 2026-08-18) a channel set with no kriged
    channel in it: that run reads nothing from the cache, so calling it
    ``kriged-*`` in the CSVs would be a provenance lie -- see the guard at
    the top of the body.

    Schema guard (review 2026-08-13): a v1 cache (kriged over the FULL
    grid, no ``gap_type``, no ``schema_version`` attr) contains no NaN at
    all, so it passes the corrupt-cache check while silently feeding real
    out-of-crop values where v3 feeds the 0.0 impute -- and the builder
    and chain both skip existing files, so it would never be migrated.
    A v2 cache is kriged over the old (region-mask) domain, which does not
    cover the new crop.  Both are genuine FORMAT breaks and stay unreadable.

    v3 is not: it differs from v4 only in that its U10M/V10M are kriged
    rather than clean reanalysis, and under the sourcing rule below those
    copies are never read, so v3 loads at any ``--channels`` width
    (``config.KRIGE_SCHEMA_READABLE``; user decision 2026-08-18 -- "only
    reducing the number of kriged variables is backwards compatible").  The
    one provenance check that survives is the reverse case: a channel the
    config calls kriged that the cache holds CLEAN, which no inspection of
    the values could reveal and which would quietly train on reanalysis
    where the configuration promises AIRS information.
    """
    path = config.KRIGED_SOURCE_DIRS[source] / f"kriged_sfc_{year}.nc"
    # Which channels come from which file (see the CHANNEL SOURCING note
    # below).  Resolved BEFORE the cache is touched, because of the footgun
    # guard that follows.
    from_cache = tuple(c for c in config.INPUT_CHANNELS
                       if c in config.KRIGED_CHANNELS)
    from_rea = tuple(c for c in config.INPUT_CHANNELS
                     if c not in config.KRIGED_CHANNELS)
    if not from_cache:
        # Footgun guard (review 2026-08-18): a channel set with NO kriged
        # channel (e.g. --channels SLP,U10M,V10M) would load 100 % from the
        # reanalysis, open and schema-check the cache, never read a byte of
        # it, and still be recorded under this --source in every metrics CSV
        # and _run.json -- an "AIRS" number containing zero satellite
        # information, indistinguishable after the fact from a real one.
        # Not on the current channel ladder; this is a footgun guard.
        raise ValueError(
            f"--source {source} with --channels "
            f"{','.join(config.INPUT_CHANNELS)}: none of these channels is "
            f"kriged (config airs.kriged_channels="
            f"{list(config.KRIGED_CHANNELS)}), so every channel would be "
            f"read from the MERRA-2 reanalysis and the cache {path} would "
            f"be opened, schema-checked and never read -- the run would "
            f"contain no satellite information at all while being recorded "
            f"as '{source}'.  Either run it as '--source reanalysis' "
            f"(identical inputs, honest provenance), or put a kriged "
            f"channel ({', '.join(config.KRIGED_CHANNELS)}) back in "
            f"--channels.")
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
    if version not in config.KRIGE_SCHEMA_READABLE:
        raise ValueError(
            f"{path}: schema_version={version!r} -- this loader reads "
            f"{list(config.KRIGE_SCHEMA_READABLE)}.  A missing attr means a "
            f"v1 cache kriged over the FULL grid (it has no NaN at all, so "
            f"it passes the corrupt-cache check while silently feeding real "
            f"out-of-crop values where the current schema feeds the 0.0 "
            f"impute), and a v2 cache covers the old region-mask domain, "
            f"which the current crop is not a subset of.  Both are genuine "
            f"format breaks; " + rebuild + ".")
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
            f"{path}: schema v{version} cache built under a DIFFERENT domain "
            f"configuration ({diffs}) -- its crop extent does not match "
            f"the current configs/dl_front.yaml domain/halo, so its fills "
            f"would carry mismatched provenance; {rebuild}.")
    # CHANNEL SOURCING (user decision 2026-08-18).  A cache's "clean"
    # channels are BY DEFINITION the MERRA-2 reanalysis, so there is no
    # reason to trust a copy of them baked into the cache file -- we read
    # them straight from sfc_daily at the same timestamp instead.  Only the
    # KRIGED channels (the ones carrying satellite-shaped gap fills, which
    # exist nowhere else) come out of the cache.
    #
    # This is what makes the v3 -> v4 wind change a non-event: a v3 cache
    # kriged U10M/V10M, but under the current config those are clean
    # channels, so we never read v3's copies of them and the file is
    # perfectly usable at any --channels width.  No rebuild, no schema
    # negotiation.  (It is also strictly more honest: the model now sees the
    # SAME reanalysis SLP/winds at train and eval time regardless of which
    # cache generation produced the T2M/QV2M fills.)
    cache_channels = _cache_kriged_split(kriged, version, path, rebuild)
    # (from_cache / from_rea were resolved at the top of the function)
    # The one hazard left: a channel the config calls kriged that this cache
    # holds CLEAN.  Then the cache has no satellite information for it and
    # we would silently train on reanalysis while believing otherwise.
    missing = [c for c in from_cache if c not in cache_channels]
    if missing:
        raise ValueError(
            f"{path}: {missing} are kriged channels per config "
            f"airs.kriged_channels={list(config.KRIGED_CHANNELS)}, but this "
            f"cache records kriged_channels={list(cache_channels)} -- it "
            f"holds a CLEAN reanalysis copy where a satellite-shaped gap "
            f"fill is expected, and the two cannot be told apart by looking "
            f"at the values.  Training on it would quietly use reanalysis "
            f"where the configuration promises AIRS information.  Either "
            f"drop {missing} from airs.kriged_channels (they are then read "
            f"from the reanalysis, which is what the cache actually has), "
            f"or {rebuild}.")

    kriged_times = pd.DatetimeIndex(kriged["time"].values)
    common = kriged_times.intersection(label_times)
    crop = crop_domain()
    x = _sfc_x_split(kriged.sel(time=common), common, from_cache, from_rea,
                     stats, crop, path)
    bad = np.isnan(x) & crop[None, :, :, None]
    if bad.any():
        # blame the file that actually holds the NaN, and quote the version
        # this cache really is (review 2026-08-18) -- see _raise_in_crop_nan
        _raise_in_crop_nan(bad, common, from_cache, from_rea, path,
                           version, rebuild)
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


def fold_split(times, fold: int, n_folds: int = config.N_FOLDS,
               seed: int = config.FOLD_SEED, years=None):
    """Day-keyed CV split: a seeded permutation of CALENDAR DAYS.

    The paper (section 4.1) permuted individual time steps; this splits
    whole days instead, for two reasons (audit + user decision 2026-08-15):

    1. The old positional permutation was keyed to the SAMPLE COUNT, and
       every curriculum stage has a different one (reanalysis carries ~8
       label steps/day, the kriged caches 3, the AIRS archive has missing
       days) -- so "fold k" named different timestamps at each stage and
       ~2/3 of a fine-tune stage's validation steps had been in the
       previous stage's training set.  Keying folds to the calendar day
       makes membership identical across stages regardless of a source's
       hours or coverage.
    2. Same-day 18/21/00Z steps are strongly autocorrelated; keeping a
       day's steps on one side of the split stops the val_loss driving
       early stopping from being scored on near-duplicates of training
       samples.

    ``times``: per-sample timestamps.  ``years``: the day universe to
    permute (default: the years present in ``times``); pass the
    stage-independent training years so sparse sources still agree with
    the full calendar.  Returns (train_idx, val_idx) into the sample axis.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    times = pd.DatetimeIndex(times)
    if years is None:
        years = sorted(set(times.year))
    universe = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(f"{y}-01-01", f"{y}-12-31").values for y in years]))
    days = times.normalize()
    unknown = ~days.isin(universe)
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} samples fall outside the fold universe "
            f"years {list(years)} (first: {times[unknown][0]})")
    perm = np.random.default_rng(seed).permutation(len(universe))
    chunks = np.array_split(perm, n_folds)
    is_val = days.isin(universe[chunks[fold]])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


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


# --------------------------------------------------------------------------- #
# Label provenance (new 2026-08-18, owner: the evaluation tooling)
# --------------------------------------------------------------------------- #

def label_digest(years, n_classes: int = 6) -> str:
    """Short SHA-1 fingerprint of the LABEL CONTENT scored over ``years``.

    Why this exists (user decision 2026-08-18): the antimeridian-crossing
    polyline bug fixed 2026-08-17 (whole horizontal bars painted across the
    grid) was repaired by REGENERATING the label files in place, which
    silently invalidated every metric CSV computed before it -- nothing in
    the pipeline could tell a number scored on the old labels from one
    scored on the new.  Each ``_run.json`` now carries this digest, and the
    chain scripts refuse to reuse a metrics CSV whose digest differs from
    the labels currently on disk.

    This is a CONTENT digest of the SCORED label cells, not a cryptographic
    hash of the files: for each year it counts cells per class over the
    valid analysis steps (:func:`valid_label_steps` -- the SAME filter the
    evaluation applies, so the digest cannot move for reasons unrelated to
    the labels) inside the scoring mask (:func:`analysis_domain` for
    n_classes == 6, :func:`region_mask` for 5), and hashes those counts
    together with the resolved label directory and width.  Consequences of
    that choice, stated plainly:

    * it moves iff the labels move where they are scored -- a relabelled
      cell outside the scoring mask, or a change to bytes the evaluation
      never reads, leaves it alone (by design: such a change cannot alter
      any reported metric);
    * it is not collision-proof; a permutation that preserves every
      per-class cell count would collide.  It is a staleness detector, not
      a security control.

    Cost: one pass over each year's label file (seconds), so a chain script
    can call it once per invocation.
    """
    names = class_names(n_classes)
    mask = (analysis_domain() if n_classes == 6
            else region_mask().astype(bool))
    # The directory is resolved at call time, not import time, so pointing
    # front_finder.config at a backup tree (scripts/eval_decision_rule.py's
    # --labels old) really does change the digest.
    labels_dir = (fd_config.NOAA_LABELS_DIR if n_classes == 6
                  else fd_config.CODSUS_DIR)
    lines = [f"labels_dir:{labels_dir}", f"width:{config.LABEL_WIDTH}",
             f"classes:{n_classes}"]
    for year in years:
        with load_label_ds(int(year), n_classes) as lab:
            keep = valid_label_steps(lab, n_classes)
            cls = class_grid(lab, n_classes)[keep]
        counts = np.bincount(cls[:, mask].ravel(), minlength=len(names))
        lines.append(f"{int(year)}:" +
                     ",".join(str(int(c)) for c in counts))
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()
