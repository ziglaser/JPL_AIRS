"""Checkpoint -> BK19-schema hard-class prediction rasters (netCDF product).

Turns one or more trained 6-class checkpoints into a PUBLISHABLE prediction
archive that is byte-schema-compatible with the published Biard & Kunkel
(2019) DL-FRONT rasters, so every existing BK19 consumer reads our own model
output with ZERO code change::

    data/front_id/predicted_fronts/<tag>/1deg_3wide/3hr/
        merra2_merra2-1deg_3wide_3hr_{YYYY}.nc
        merra2_merra2-1deg_3wide_3hr_{YYYY}_run.json

The basename is kept byte-identical to BK19's because
``evaluate_test.bk19_path`` and ``front_finder.labels.load_benchmark`` both
HARDCODE it under a configurable root; identity therefore lives in the
directory ``<tag>`` plus honest global attrs (reconnaissance decision,
2026-08-17).  That makes the exporter self-verifying::

    JPL_BK19_DIR=$JPL_AIRS_DATA/front_id/predicted_fronts/dlfront_D6C-f0_kriged-airs \
        python -m dl_front.evaluate_test --source bk19 --classes 6

Deliberate, documented DEVIATIONS from the published BK19 files (they are
schema-compatible extensions, not accidents -- see ``front`` / time / fill
notes below):

* ``front = 6``: ``front_type`` is ``config.CLASS_NAMES_6`` (cold, warm,
  stationary, occluded, DRYLINE, none) rather than BK19's five.  Every
  consumer indexes ``front_type`` by NAME -- ``evaluate_test.bk19_class_grid``
  skips names it cannot find, ``convection_skill.fronts`` picks its four by
  name, and ``front_finder.front_stack`` REQUIRES 'dryline' when
  ``JPL_FRONT_LABELS=noaa`` -- so keeping the class that motivated the
  6-class track costs one channel and breaks nothing.
* SPARSE time axis (~730 steps/year, not BK19's dense 2928): the inputs only
  exist at ``config.AIRS_HOURS`` = 21 Z and 00 Z (user decision 2026-08-15),
  which are exactly the bulletins ``convection_skill.fronts`` needs.  The
  ``3hr`` directory level is kept because it names the CADENCE the stamps
  snap to (and is what ``bk19_path`` expects); the axis is NOT padded with
  all-fill steps.
* FILL IS EXERCISED.  The published files declare ``_FillValue = 2UB`` but
  contain zero fill bytes; ours write 2 in ALL SIX channels outside
  ``dataset.analysis_domain()`` (784 of 9588 px; box 32-53 N / 107-64 W
  land>=0.5, user decision 2026-08-13), because the network is untrained
  there and 0 would read as a confident "no front".  NOTE for consumers:
  ``bk19_class_grid`` decodes fill to NaN and therefore scores fill cells as
  'none' -- intersect with ``dataset.analysis_domain()``, or read ``fronts``
  with ``mask_and_scale=False`` and test ``== 2``.

Only ``1deg_3wide`` is written (and the CLI refuses another width): BK19's
1wide is a separately trained product, not a thinning of 3wide (flagged-cell
ratio 3.36, dilation IoU 0.90), and every checkpoint we trained saw
``config.LABEL_WIDTH = 3`` labels only, so a hard per-cell argmax has no
defensible 1wide analogue.

Decision rule: plain ``probs.argmax(-1)``, which is what every reported CSI
uses.  ``--class-scale`` multiplies individual softmax channels before the
argmax; it is OFF by default and, when used, lands in the tag, the filename's
directory, the global attrs AND the run json, because the CSI-optimal factors
sit at frequency bias 1.5-1.7 (memory 2026-08-17: thicker lines gaming
neighborhood matching) -- a shipped product must not do that silently.

Multi-fold (folds 0-2 of ``dataset.fold_split``, whose day universe is the
TRAIN years only, so 2016-2021 is common held-out data for all folds):
passing ``--ckpt`` more than once writes one per-fold archive per checkpoint
AND a softmax-AVERAGED ensemble archive (averaging, not majority vote: no
3-way ties), reusing a single input load per year.

Provenance is written twice, deliberately: a sibling ``<basename>_run.json``
in ``evaluate_test.write_outputs`` style, and global attrs on the file
itself, because a stray .nc must be self-describing.  Both carry each
checkpoint's resolved path AND its SHA-1, the git revision and the
``configs/dl_front.yaml`` SHA-1 -- the 2026-08-17 dateline label bug means a
reader MUST be able to tell which checkpoint generation produced a file.

CLI::

    python -m dl_front.export_predictions --ckpt results/dl_front/models/D6C-f0/D6C-f0.h5 \
        --source kriged-airs [--years 2016-2021] [--force]
    python -m dl_front.export_predictions --ckpt .../D6C-f0.h5 --ckpt .../D6C-f1.h5 \
        --ckpt .../D6C-f2.h5 --source kriged-airs        # per-fold + ens3
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from front_finder import config as fd

from . import config, dataset, predict
from .evaluate_test import (_git_rev, check_kriged_split,
                            check_model_channels, load_year, parse_years,
                            resolve_channels)

#: Root of the prediction archive: the SIBLING of ``config.BK19_DIR``'s
#: default (front_finder.config.BENCHMARK_DIR = <data>/front_id/
#: predicted_fronts/bk19).  Resolved off ``fd.DATA_ROOT`` rather than
#: ``config.BK19_DIR.parent`` on purpose: JPL_BK19_DIR may have been pointed
#: at an out-of-tree BK19 archive (or at one of OUR tags, which is how the
#: exporter is self-verified), and our writes must never follow it.
PREDICTED_FRONTS_DIR = fd.DATA_ROOT / "front_id/predicted_fronts"

#: BK19's filename template, kept byte-identical so ``bk19_path`` and
#: ``front_finder.labels.load_benchmark`` compose our paths unchanged.
FILE_TEMPLATE = "merra2_merra2-1deg_{width}wide_3hr_{year}.nc"

#: The only width we may write: every checkpoint trained on 3wide labels
#: (configs/dl_front.yaml labels.label_width), and BK19's 1wide product is
#: separately trained, not a thinning (see module docstring).
EXPORT_WIDTH = 3

#: Grid cadence directory level (BK19's own; our axis is a sparse subset of
#: it -- ``config.AIRS_HOURS`` only).
FREQ_DIR = "3hr"

#: ``fronts`` fill byte = ``config.LABEL_FILL``'s value in the BK19 schema:
#: written in ALL channels outside ``dataset.analysis_domain()``.
FILL_BYTE = 2

#: BK19 storage parity (free, and keeps per-timestep random access cheap for
#: the six_panel/plot consumers): zlib level 4 + shuffle, one time step per
#: chunk on ``fronts``, contiguous uncompressed coordinates.
ZLIB_COMPLEVEL = 4


# --------------------------------------------------------------------------- #
# Decision rule: likelihoods -> hard classes -> BK19 channels
# --------------------------------------------------------------------------- #

def parse_class_scale(spec: str | None) -> dict[str, float]:
    """``'warm=1.3,occluded=1.35'`` -> {name: factor}; None/'' -> {}.

    Strict: an unknown class name or a non-positive factor raises, because a
    typo here silently changes the shipped product's operating point.
    """
    if not spec:
        return {}
    scale: dict[str, float] = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"--class-scale item {item!r} is not "
                             f"'name=factor'")
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in config.CLASS_NAMES_6:
            raise ValueError(f"--class-scale: unknown class {name!r}; "
                             f"expected one of {list(config.CLASS_NAMES_6)}")
        factor = float(value)
        if not factor > 0:
            raise ValueError(f"--class-scale {name}={factor}: factor must be "
                             f"> 0")
        scale[name] = factor
    return scale


def hard_classes(probs: np.ndarray, n_classes: int,
                 class_scale: dict[str, float] | None = None) -> np.ndarray:
    """(n, lat, lon, n_classes) softmax -> (n, lat, lon) uint8 class index.

    The established decision rule is a plain ``argmax`` over the softmax
    (that is what every reported CSI and every ``evaluate_ckpt`` call uses).
    ``class_scale`` multiplies named channels first -- an operating-point
    knob, default empty; see the module docstring for why it is not the
    default.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if class_scale:
        names = dataset.class_names(n_classes)
        probs = probs.copy()
        for name, factor in class_scale.items():
            if name not in names:
                raise ValueError(f"--class-scale {name!r} is not one of the "
                                 f"{n_classes} classes {list(names)}")
            probs[..., names.index(name)] *= factor
    return probs.argmax(-1).astype(np.uint8)


def class_channels(cls: np.ndarray, n_classes: int,
                   valid: np.ndarray | None = None) -> np.ndarray:
    """(n, lat, lon) class index -> (n, front, lat, lon) ubyte BK19 channels.

    The exact inverse of ``dataset.class_grid``: a plain EXCLUSIVE one-hot in
    ``dataset.class_names`` order, no ``config.TYPE_PRIORITY`` painting on
    the way out (an argmax output is already single-valued).
    ``evaluate.onehot_da`` cannot be reused -- it drops the 'none' channel,
    while the real BK19 files carry it and their channels sum to exactly 1.

    ``valid`` is an optional (lat, lon) bool domain mask; every channel is
    set to :data:`FILL_BYTE` where it is False (a time-invariant mask, so
    ``labels.valid_mask``'s ``(fronts != 2).all('front')`` intent holds and a
    ``mask_and_scale=False`` reader recovers the untrained region exactly).
    """
    cls = np.asarray(cls)
    chan = np.zeros((cls.shape[0], n_classes, *cls.shape[1:]), dtype=np.uint8)
    for k in range(n_classes):
        chan[:, k] = (cls == k)
    if valid is not None:
        chan[:, :, ~np.asarray(valid, dtype=bool)] = FILL_BYTE
    return chan


# --------------------------------------------------------------------------- #
# The BK19-schema writer (raw netCDF4: header order + verbatim attr dtypes)
# --------------------------------------------------------------------------- #

def write_bk19_netcdf(path: Path, channels: np.ndarray,
                      times: pd.DatetimeIndex, front_type,
                      global_attrs: dict) -> Path:
    """Write one year of hard-class channels in the BK19 netCDF schema.

    Raw ``netCDF4`` rather than xarray, verified against
    ``ncdump -h`` on the published file down to a ZERO-line header diff
    (reconnaissance 2026-08-17).  xarray gets three things wrong that matter
    for a drop-in product: the ``coordinates`` attr is illegal in its
    ``encoding=`` dict, it rewrites the time units string (dropping
    ``' 00:00:00'``), and it cannot reproduce the dim/var DECLARATION order.
    A raw writer also puts every attribute and its dtype at the call site,
    which is what this repo's interpretability rule asks for.

    Two dtype traps are handled explicitly:

    * ``valid_min``/``valid_max`` must be ``np.array(..., 'int64')`` written
      through ``setncattr`` -- netCDF4-python's ``Variable.__setattr__``
      special-cases those names (with ``valid_range``/``missing_value``) and
      silently coerces them to the variable's ubyte type (ncdump ``0UB``
      instead of BK19's ``0LL``); ``setncattr`` keeps the int64.
    * ``crs`` is created but LEFT UNWRITTEN, i.e. holds
      ``default_fillvals['f8']``, which is why ncdump prints ``crs = _``.
      Writing NaN there would be a visible difference.
    """
    import netCDF4

    channels = np.asarray(channels, dtype=np.uint8)
    front_type = [str(s) for s in front_type]
    n_time, n_front, n_lat, n_lon = channels.shape
    if (n_front, n_lat, n_lon) != (len(front_type), len(config.LABEL_LATS),
                                   len(config.LABEL_LONS)):
        raise ValueError(
            f"channels shape {channels.shape} does not match "
            f"(time, {len(front_type)} front_type, "
            f"{len(config.LABEL_LATS)} lat, {len(config.LABEL_LONS)} lon)")
    if n_time != len(times):
        raise ValueError(f"{n_time} channel steps vs {len(times)} timestamps")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # BK19 declaration order: dims front, time(UNLIMITED), lat, lon; then
    # vars crs, front_type, fronts, lat, lon, time.
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("front", n_front)
        nc.createDimension("time", None)             # UNLIMITED, as BK19
        nc.createDimension("lat", n_lat)
        nc.createDimension("lon", n_lon)

        crs = nc.createVariable("crs", "f8")
        crs.grid_mapping_name = "latitude_longitude"  # value stays unwritten

        ft = nc.createVariable("front_type", str, ("front",))
        ft.long_name = "kind of front"

        fronts = nc.createVariable(
            "fronts", "u1", ("time", "front", "lat", "lon"),
            fill_value=np.uint8(FILL_BYTE), zlib=True,
            complevel=ZLIB_COMPLEVEL, shuffle=True,
            chunksizes=(1, 1, n_lat, n_lon))
        fronts.long_name = "front line images"
        # setncattr, NOT attribute assignment: netCDF4's Variable.__setattr__
        # special-cases valid_min/valid_max/valid_range/missing_value and
        # coerces them to the VARIABLE's dtype (ncdump 0UB instead of BK19's
        # 0LL); setncattr writes the numpy dtype we hand it.
        fronts.setncattr("valid_min", np.array(0, dtype="int64"))
        fronts.grid_mapping = "crs"
        fronts.setncattr("valid_max", np.array([1], dtype="int64"))
        fronts.coordinates = "front_type lat lon"

        lat = nc.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat.long_name = "latitude"
        lat.standard_name = "latitude"
        lat.axis = "Y"

        lon = nc.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon.long_name = "longitude"
        lon.standard_name = "longitude"
        lon.axis = "X"

        time = nc.createVariable("time", "f8", ("time",), zlib=True,
                                 complevel=ZLIB_COMPLEVEL, shuffle=True,
                                 chunksizes=(1,))
        time.units = "days since 1970-01-01 00:00:00"
        time.long_name = "time"
        time.calendar = "gregorian"
        time.standard_name = "time"
        time.axis = "T"

        for key, value in global_attrs.items():
            nc.setncattr(key, value)

        ft[:] = np.array(front_type, dtype=object)
        lat[:] = np.asarray(config.LABEL_LATS, dtype="f8")
        lon[:] = np.asarray(config.LABEL_LONS, dtype="f8")
        # fractional days since the epoch, hand-encoded (BK19's 0.125 step)
        stamps = pd.DatetimeIndex(times).values.astype("datetime64[ns]")
        time[:] = ((stamps - np.datetime64("1970-01-01T00:00:00"))
                   / np.timedelta64(1, "D")).astype("f8")
        # set_auto_mask off: our 2s ARE data (the untrained-region mask), and
        # netCDF4 would otherwise round-trip them through a masked array.
        fronts.set_auto_mask(False)
        fronts[:] = channels
    return path


# --------------------------------------------------------------------------- #
# Tags, paths and idempotency
# --------------------------------------------------------------------------- #

def _ckpt_stem(ckpt: Path | str) -> str:
    """'.../D6C-f0/D6C-f0.h5' -> 'D6C-f0' (the eval CSVs' leg naming)."""
    return Path(ckpt).stem


def ensemble_stem(ckpts) -> str:
    """Stem of a softmax-averaged multi-checkpoint archive.

    ``[D6C-f0, D6C-f1, D6C-f2]`` -> ``D6C-ens3`` when every stem shares a
    base with a trailing ``-f<k>`` fold suffix; otherwise the stems are
    joined with ``+`` so the tag never claims a fold family it does not have.
    """
    stems = [_ckpt_stem(c) for c in ckpts]
    bases = {s.rsplit("-f", 1)[0] for s in stems
             if "-f" in s and s.rsplit("-f", 1)[1].isdigit()}
    if len(bases) == 1 and len(stems) == len(set(stems)):
        return f"{bases.pop()}-ens{len(stems)}"
    return "+".join(stems)


def export_tag(stem: str, source: str,
               class_scale: dict[str, float] | None = None) -> str:
    """Directory tag: ``dlfront_<stem>_<source>[_scale-<k><f>...]``.

    Both the checkpoint AND the input leg are in the tag: the same
    checkpoint scored on reanalysis vs kriged-airs gives different
    predictions.  A non-default decision rule is in the tag too, so a
    CSI-chasing operating point can never masquerade as the plain-argmax
    product on disk.
    """
    tag = f"dlfront_{stem}_{source}"
    if class_scale:
        parts = "-".join(f"{name}{factor:g}"
                         for name, factor in sorted(class_scale.items()))
        tag += f"_scale-{parts}"
    return tag


def output_path(tag: str, year: int, width: int = EXPORT_WIDTH,
                root: Path | None = None) -> Path:
    """``<root>/<tag>/1deg_{w}wide/3hr/merra2_...
    _{w}wide_3hr_{year}.nc``.

    Mirrors ``evaluate_test.bk19_path`` exactly one level down, which is what
    makes ``JPL_BK19_DIR=<root>/<tag>`` work with no code change.
    """
    root = PREDICTED_FRONTS_DIR if root is None else Path(root)
    return (root / tag / f"1deg_{width}wide" / FREQ_DIR
            / FILE_TEMPLATE.format(width=width, year=year))


def run_json_path(nc_path: Path) -> Path:
    """Sibling provenance file of one exported year (``..._run.json``)."""
    nc_path = Path(nc_path)
    return nc_path.with_name(f"{nc_path.stem}_run.json")


def _sha1_file(path: Path) -> str | None:
    """SHA-1 of a file's bytes, None if unreadable (checkpoint identity)."""
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _config_sha1() -> str | None:
    """SHA-1 of ``configs/dl_front.yaml`` (every tunable that shaped the run)."""
    try:
        return hashlib.sha1(config.CONFIG_YAML.read_bytes()).hexdigest()
    except OSError:
        return None


def is_done(nc_path: Path) -> bool:
    """Done-marker in ``scripts/dlfront_jpl_chain.sh`` style.

    A year counts as exported only when BOTH the netCDF and its provenance
    sibling exist: a half-written pair (an interrupted job) is re-done rather
    than silently accepted.
    """
    return Path(nc_path).exists() and run_json_path(nc_path).exists()


def resolve_output_path(ckpts, year: int, source: str,
                        class_scale: dict[str, float] | None = None,
                        width: int = EXPORT_WIDTH, root: Path | None = None,
                        tag: str | None = None) -> Path:
    """Where one year of one run lands, tag composed if not given.

    Shared by :func:`export_year` (which writes it) and :func:`export_years`
    (which probes it to tell an already-complete year apart from a failed
    one, so the exit code can distinguish them -- audit 2026-08-18).
    """
    if tag is None:
        ckpts = [Path(c) for c in ckpts]
        stem = (_ckpt_stem(ckpts[0]) if len(ckpts) == 1
                else ensemble_stem(ckpts))
        tag = export_tag(stem, source, dict(class_scale or {}))
    return output_path(tag, year, width, root)


# --------------------------------------------------------------------------- #
# Per-year export
# --------------------------------------------------------------------------- #

def _provenance(ckpts, source: str, year: int, hours, n_steps: int,
                times, class_scale: dict[str, float], n_classes: int,
                ensemble: bool) -> dict:
    """The dict written to ``_run.json`` AND (stringified) to global attrs.

    Same fields and spirit as ``evaluate_test.write_outputs``' run json --
    resolved checkpoint paths, source, hours, step counts, a SHA-1 of every
    exported timestamp, git rev, UTC creation stamp -- plus the two things a
    stray .nc needs to identify its checkpoint GENERATION (memory
    2026-08-17: D6A/D6C were trained on dateline-contaminated labels):
    each checkpoint's file SHA-1 and the config YAML's SHA-1.
    """
    stamps = sorted(str(t) for t in pd.DatetimeIndex(times))
    return {
        "checkpoints": [str(Path(c).resolve()) for c in ckpts],
        "checkpoint_sha1": [_sha1_file(Path(c)) for c in ckpts],
        "ensemble": ensemble,
        "combination": ("mean softmax over checkpoints, then argmax"
                        if ensemble else "single checkpoint argmax"),
        "source": source,
        "year": int(year),
        "hours": [int(h) for h in hours],
        "n_steps": int(n_steps),
        # Which of the model's input channels were read from the kriged
        # cache (the rest come clean from MERRA-2) -- the same value
        # evaluate_test.check_kriged_split returns and an eval _run.json
        # records; main has already refused a checkpoint whose training
        # split disagrees, so this states the split actually exported.
        "kriged_channels": ([c for c in config.INPUT_CHANNELS
                             if c in config.KRIGED_CHANNELS]
                            if source in config.KRIGED_SOURCE_DIRS else []),
        "times_sha1": hashlib.sha1("\n".join(stamps).encode()).hexdigest(),
        "decision_rule": ("argmax" if not class_scale else
                          "argmax after per-class softmax scaling"),
        "class_scale": {k: float(v) for k, v in sorted(class_scale.items())},
        "class_names": list(dataset.class_names(n_classes)),
        "label_width": int(config.LABEL_WIDTH),
        "domain": ("dataset.analysis_domain(): box lat "
                   f"{list(config.ANALYSIS_LAT_RANGE)} lon "
                   f"{list(config.ANALYSIS_LON_RANGE)} intersected with land "
                   f"fraction >= {config.LAND_FRACTION_MIN}; every cell "
                   f"outside it is written as the fill byte "
                   f"{FILL_BYTE} in all channels"),
        "git_rev": _git_rev(),
        "dlfront_config_sha1": _config_sha1(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _global_attrs(prov: dict) -> dict:
    """Provenance -> netCDF global attrs (flat, all str/number).

    Deliberately NOT BK19's globals: its ``title`` is stale inherited
    metadata ("Coded surface bulletins ...", already flagged in
    ``dl_front.config``) and its ``history`` is 817 KB of NCO command lines.
    """
    stems = ", ".join(_ckpt_stem(c) for c in prov["checkpoints"])
    return {
        "title": (f"dl_front UNET3+ hard-class front predictions "
                  f"({stems}) on a merra2-1deg grid, BK19 schema"),
        "model": stems,
        "checkpoints": " ".join(prov["checkpoints"]),
        "checkpoint_sha1": " ".join(str(s) for s in prov["checkpoint_sha1"]),
        "combination": prov["combination"],
        "input_source": prov["source"],
        "kriged_channels": " ".join(prov["kriged_channels"]),
        "decision_rule": prov["decision_rule"],
        "class_scale": json.dumps(prov["class_scale"]),
        "class_names": " ".join(prov["class_names"]),
        "label_width": np.int32(prov["label_width"]),
        "hours_utc": " ".join(str(h) for h in prov["hours"]),
        "analysis_domain": prov["domain"],
        "times_sha1": prov["times_sha1"],
        "git_rev": str(prov["git_rev"]),
        "dlfront_config_sha1": str(prov["dlfront_config_sha1"]),
        "created": prov["created"],
        "Conventions": "CF-1.7",
    }


def export_year(models, ckpts, year: int, source: str,
                n_classes: int = 6, stats: dict | None = None,
                hours=None, class_scale: dict[str, float] | None = None,
                width: int = EXPORT_WIDTH, root: Path | None = None,
                force: bool = False, batch_size: int = 64,
                loader=None, tag: str | None = None) -> Path | None:
    """Export ONE year for one model or one softmax-averaged ensemble.

    ``models``: list of objects with ``.predict(x, batch_size=, verbose=)``
    returning (n, lat, lon, n_classes) softmax (a keras model, or a stub in
    the tests); more than one is averaged BEFORE the argmax.
    ``ckpts``: the matching checkpoint paths, for the tag and provenance.
    ``loader``: optional ``loader(year) -> (x, y, times)`` override of
    ``evaluate_test.load_year`` (tests inject synthetic data here), exactly
    as in ``evaluate_test.evaluate_ckpt``.

    Returns the written path, or None when the year was skipped (already
    done, or no steps at ``hours``).  Skipping is per-year and loud: a
    six-year job must not die because one kriged cache is missing (the
    failure mode that killed phase 3a on 2026-08-15) -- that guard lives in
    :func:`export_years`.
    """
    hours = config.AIRS_HOURS if hours is None else tuple(hours)
    class_scale = dict(class_scale or {})
    models = list(models)
    ckpts = [Path(c) for c in ckpts]
    path = resolve_output_path(ckpts, year, source, class_scale, width, root,
                              tag)
    if is_done(path) and not force:
        print(f"{year}: {path} already exported, skipped (--force to "
              f"rewrite)", flush=True)
        return None

    if loader is None:
        if stats is None:
            stats = dataset.load_norm_stats()
        loader = lambda y: load_year(y, n_classes, stats, source)
    x, _, times = dataset.filter_hours(*loader(year), hours)
    if len(x) == 0:
        # same precedent (and message shape) as evaluate_ckpt: krige_fill can
        # legitimately write an empty-but-valid cache for an AIRS-void year
        print(f"{year}: no steps at hours {hours}, skipped", flush=True)
        return None

    probs = None
    for model in models:
        part = np.asarray(model.predict(x.astype(np.float32),
                                        batch_size=batch_size, verbose=0),
                          dtype=np.float64)
        probs = part if probs is None else probs + part
    probs /= len(models)

    cls = hard_classes(probs, n_classes, class_scale)
    channels = class_channels(cls, n_classes, valid=dataset.analysis_domain())

    prov = _provenance(ckpts, source, year, hours, len(x), times,
                       class_scale, n_classes, ensemble=len(models) > 1)
    write_bk19_netcdf(path, channels, times,
                      dataset.class_names(n_classes), _global_attrs(prov))
    run_json_path(path).write_text(json.dumps(prov, indent=1))
    print(f"{year}: {len(x)} steps -> {path}", flush=True)
    return path


def export_years(models, ckpts, years, source: str, **kw) -> dict:
    """Export several years, skipping (loudly) the ones that cannot be built.

    Returns ``{'written': [paths], 'present': [paths], 'skipped':
    {year: reason}}``.  Per-year try/except by design:
    ``dataset.kriged_year_arrays`` raises on a missing or stale-schema cache
    and ``dataset.year_arrays`` raises ValueError when a year's ``sfc_daily``
    day files are all absent, and a six-year job must report those years
    rather than abort at year four.

    ``present`` = years that were already complete on disk (the idempotent
    requeue path).  It exists so ``main`` can tell "nothing to do, the
    archive is already there" from "nothing could be built" and exit non-zero
    only for the latter (audit 2026-08-18: an all-years-failed export used to
    exit 0 and turn any ``afterok`` dependant green over an empty archive).
    """
    written, present, skipped = [], [], {}
    for year in years:
        probe = resolve_output_path(
            ckpts, int(year), source,
            kw.get("class_scale"), kw.get("width", EXPORT_WIDTH),
            kw.get("root"), kw.get("tag"))
        was_done = is_done(probe) and not kw.get("force", False)
        try:
            path = export_year(models, ckpts, int(year), source, **kw)
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            print(f"{year}: SKIPPED -- {type(exc).__name__}: {exc}",
                  flush=True)
            skipped[int(year)] = f"{type(exc).__name__}: {exc}"
            continue
        if path is not None:
            written.append(path)
        elif was_done:
            present.append(probe)
    return {"written": written, "present": present, "skipped": skipped}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(
        description="Export trained dl_front checkpoints as BK19-schema "
                    "hard-class prediction rasters under "
                    "data/front_id/predicted_fronts/<tag>/1deg_3wide/3hr/. "
                    "Pass --ckpt more than once for the per-fold archives "
                    "PLUS a softmax-averaged ensemble archive.")
    ap.add_argument("--ckpt", action="append", required=True, metavar="PATH",
                    help="trained .h5 checkpoint; repeatable (folds)")
    ap.add_argument("--classes", type=int, default=6, choices=(5, 6),
                    help="class count of the checkpoints (default 6: the "
                         "dryline/AIRS track)")
    ap.add_argument("--source", default="kriged-airs",
                    choices=("reanalysis", "kriged-airs", "kriged-degraded"),
                    help="input fields to run inference on; the leg is part "
                         "of the output tag (default kriged-airs)")
    ap.add_argument("--years", default="2016-2021",
                    help="'A-B' range or comma list (default 2016-2021, the "
                         "FCST_SMAP_MRMS span; every fold's train universe "
                         "is 2007-2015, so these years are held out)")
    ap.add_argument("--hours", default=None,
                    help="comma-separated UTC hours to export (default: the "
                         f"AIRS label hours {config.AIRS_HOURS})")
    ap.add_argument("--width", type=int, default=EXPORT_WIDTH,
                    help=f"label width of the archive; only {EXPORT_WIDTH} "
                         f"is supported (BK19's 1wide is a separately "
                         f"trained product, not a thinning)")
    ap.add_argument("--class-scale", default=None, metavar="NAME=F[,...]",
                    help="multiply these classes' softmax channels before "
                         "the argmax (default: none -- the plain argmax "
                         "every reported CSI uses; the CSI-optimal factors "
                         "sit at frequency bias 1.5-1.7, memory 2026-08-17). "
                         "Any value used lands in the output tag.")
    ap.add_argument("--ensemble-only", action="store_true",
                    help="with several --ckpt, write ONLY the averaged "
                         "ensemble archive (default: per-fold archives too)")
    ap.add_argument("--root", default=None,
                    help=f"archive root (default {PREDICTED_FRONTS_DIR})")
    ap.add_argument("--force", action="store_true",
                    help="rewrite years whose .nc + _run.json already exist")
    ap.add_argument("--channels", default=None,
                    help="comma-separated input channels EVERY --ckpt "
                         f"consumes, a subset of {','.join(config.SFC_VARS)} "
                         "(default: adopt run_args.channels from the first "
                         "--ckpt's run_config.yaml, else the yaml "
                         "inputs.channels).  A multi-fold ensemble averages "
                         "softmax elementwise, so every fold must share the "
                         "same channels; passed through to "
                         "evaluate_test.resolve_channels/check_model_channels "
                         "so a reduced-channel ladder "
                         "checkpoint (D6A2/D6A3) either exports or is "
                         "refused with an actionable message instead of a "
                         "raw Keras shape error")
    a = ap.parse_args(argv)

    if a.width != EXPORT_WIDTH:
        ap.error(f"--width {a.width} is not supported: only "
                 f"1deg_{EXPORT_WIDTH}wide predictions exist (every "
                 f"checkpoint was trained on {EXPORT_WIDTH}wide labels; "
                 f"BK19's 1wide is a separately trained product, not a "
                 f"thinning of 3wide)")
    if config.LABEL_WIDTH != EXPORT_WIDTH:
        ap.error(f"config.LABEL_WIDTH is {config.LABEL_WIDTH}, not "
                 f"{EXPORT_WIDTH}: this exporter only knows how to write the "
                 f"3wide archive the checkpoints were trained for "
                 f"(configs/dl_front.yaml labels.label_width)")
    try:
        class_scale = parse_class_scale(a.class_scale)
    except ValueError as exc:
        ap.error(str(exc))
    if a.ensemble_only and len(a.ckpt) == 1:
        ap.error("--ensemble-only needs more than one --ckpt (a single "
                 "checkpoint IS its own archive)")

    years = parse_years(a.years)
    hours = (tuple(int(h) for h in a.hours.split(","))
             if a.hours else config.AIRS_HOURS)
    root = Path(a.root) if a.root else None

    missing = [c for c in a.ckpt if not Path(c).exists()]
    if missing:
        ap.error(f"checkpoint(s) not found: {missing}")

    # Channels first (evaluate_test.resolve_channels):
    # every loader below reads config.INPUT_CHANNELS, so the resolved tuple
    # has to be installed before a single array is loaded.  Adopts from the
    # FIRST --ckpt's run_config.yaml unless --channels is given; the
    # per-checkpoint check right after covers the rest (an ensemble whose
    # folds disagree on channels would average softmax over inputs the
    # model never learned in that slot).
    resolve_channels(a.ckpt[0], a.channels)

    # One load per checkpoint, reused across every year and both the
    # per-fold and the ensemble archives.
    models = [predict.load_model(c) for c in a.ckpt]
    for model, ckpt in zip(models, a.ckpt):
        check_model_channels(model, ckpt)   # refuses a mismatch, never warns
        # INPUT provenance guard (evaluate_test.check_kriged_split, same
        # position as evaluate_test.main: before any data is loaded): a
        # kriged --source export of a checkpoint TRAINED under a different
        # airs.kriged_channels would feed it e.g. clean reanalysis winds
        # where it learned kriged ones, and the archive's tag would still
        # claim the checkpoint's provenance.  Raises with the remedy on a
        # true mismatch; no-op ([]) for --source reanalysis.
        check_kriged_split(ckpt, a.source)

    # runs = (models, ckpts) groups to export: each fold on its own, then
    # the softmax-averaged ensemble (spec 2026-08-17: export both, the
    # ensemble is the headline, per-fold files report fold spread).
    runs = []
    if not a.ensemble_only:
        runs += [([m], [c]) for m, c in zip(models, a.ckpt)]
    if len(models) > 1:
        runs.append((models, list(a.ckpt)))

    stats = dataset.load_norm_stats() if a.source != "bk19" else None
    summary = {}
    for group_models, group_ckpts in runs:
        stem = (_ckpt_stem(group_ckpts[0]) if len(group_ckpts) == 1
                else ensemble_stem(group_ckpts))
        tag = export_tag(stem, a.source, class_scale)
        print(f"\n=== {tag} ===", flush=True)
        summary[tag] = export_years(
            group_models, group_ckpts, years, a.source,
            n_classes=a.classes, stats=stats, hours=hours,
            class_scale=class_scale, width=a.width, root=root,
            force=a.force, tag=tag)

    print("\nsummary:")
    for tag, result in summary.items():
        print(f"  {tag}: {len(result['written'])} year(s) written"
              + (f", {len(result['present'])} already present"
                 if result["present"] else "")
              + (f", skipped {sorted(result['skipped'])}"
                 if result["skipped"] else ""))
    hint_root = root or PREDICTED_FRONTS_DIR
    print(f"\nscore an archive with the existing BK19 leg (no code change):\n"
          f"  JPL_BK19_DIR={hint_root}/<tag> python -m "
          f"dl_front.evaluate_test --source bk19 --classes "
          f"{a.classes}")

    # Exit non-zero when an archive tag ended up with nothing at all --
    # neither a year written now nor a year already complete on disk -- while
    # at least one year failed.  Same precedent as
    # evaluate_test.evaluate_ckpt's "no data found for years ..." (audit
    # 2026-08-18): a green job over an empty archive silently satisfies an
    # `--dependency=afterok:` chain link.  Keyed on the tag, not on the whole
    # run, and NOT on `written` alone: a fully-complete idempotent requeue
    # legitimately writes 0 years.
    barren = [tag for tag, r in summary.items()
              if r["skipped"] and not r["written"] and not r["present"]]
    if barren:
        print(f"\nERROR: no year could be exported for {barren} -- every "
              f"requested year failed (see the SKIPPED lines above); "
              f"exiting 1 so an afterok dependant does not run on an empty "
              f"archive", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
