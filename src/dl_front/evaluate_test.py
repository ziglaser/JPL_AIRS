"""Held-out test evaluation: three-way checkpoint/benchmark comparison.

Scores one evaluation "leg" on the held-out test years (yaml
``splits.eval_years_*``; 2016-2018 for 6-class, user decision 2026-08-13),
from one of three input sources:

* ``reanalysis``  -- clean MERRA-2 surface fields (``dataset.year_arrays``),
* ``kriged-airs`` -- kriged gap-filled AIRS-FCST fields
  (``dataset.kriged_year_arrays``; caches built by ``dl_front.krige_fill``),
* ``bk19``        -- the PUBLISHED Biard & Kunkel (2019) DL-FRONT prediction
  rasters (``config.BK19_DIR``; no checkpoint involved -- the files ARE the
  model output, hard binary classes, no dryline).

All legs are restricted to the SAME time steps, two ways:

* an hours filter (default the AIRS-covered label hours ``config.AIRS_HOURS``)
  equalizes the label hours, and
* a TIMESTAMP intersection with the kriged-AIRS cache (default for
  ``--source reanalysis`` and ``--source bk19``; disable with ``--no-match``)
  equalizes the days: the AIRS archive is sparse, so without it those legs
  would score every test-period step while the kriged run scores only
  AIRS-covered days, confounding input quality with sample composition.

The legs are then directly comparable (the only difference is the input
fields / predictor, never the sample set) -- but note the matching is
ONE-DIRECTIONAL: it only DROPS steps absent from the cache, it cannot
resurrect steps a leg is missing on its own (e.g. a partially downloaded
``sfc_daily`` year makes ``dataset.year_arrays`` silently skip days, so the
reanalysis leg could score fewer steps than the others).  The per-year step
counts and a SHA-1 of the scored timestamps therefore land in the
``_run.json`` provenance file, and :func:`compare` cross-checks them --
together with the LABEL digest, since the 2026-08-17 in-place label
regeneration changed the labels without changing the time axis -- across
legs, warning loudly when the legs are not like-for-like.

Outputs under ``results/dl_front/test_eval/`` (created on demand):

* ``<ckpt-stem>_<source>.csv``        tidy neighborhood-CSI table
                                      (front, dilation, km, csi, pod, far, fb)
                                      (the bk19 leg's stem is just ``bk19``)
* ``<ckpt-stem>_<source>_paper.json`` paper metrics: accuracy dict, ROC AUC,
                                      per-class confusion (% of masked cells).
                                      SKIPPED for bk19: hard binary
                                      predictions make the ROC none-scaling
                                      sweep meaningless.
* ``<ckpt-stem>_<source>_run.json``   provenance: ckpt path, source, years,
                                      hours, model input channels, the
                                      kriged/clean INPUT split
                                      (``kriged_channels``, see below), the
                                      LABEL digest (see below), git
                                      revision, timestamp
* ``comparison.csv``                  (``compare`` subcommand) pooled CSI
                                      pivoted (front, dilation_km) x leg

Label provenance (user decision 2026-08-18): the front labels were
REGENERATED in place on 2026-08-17 (the antimeridian-crossing polyline bug
painted full-width horizontal bars), which silently invalidated every metric
CSV written before it.  Every ``_run.json`` therefore records
``labels_sha1`` -- ``dataset.label_digest`` over the scored years -- and
``labels_dir``; the chain scripts refuse to reuse a CSV whose digest differs
from the labels currently on disk.  The ``label-digest`` subcommand prints
just that digest so a shell script can compare it without parsing anything.

Input channels (user decision 2026-08-18): the model's input channels are a
configurable subset of the on-disk ``config.SFC_VARS`` (the stage-A channel
ladder trains 3- and 2-channel rungs).  ``--channels`` selects them, but the
usual path is automatic: when the checkpoint directory has a
``run_config.yaml`` recording ``run_args.channels``, that list is adopted,
and a checkpoint whose input channels disagree with the resolved channels
-- in COUNT, or (when the checkpoint records them) in NAMES, e.g. a
same-arity relabelling T2M,QV2M -> T2M,SLP of a reused ablation checkpoint
-- is REFUSED rather than scored; a silent mismatch would produce plausible
but meaningless numbers.  The checkpoint's own record beats ``--channels``:
the flag says what the operator means to feed the model, run_config.yaml
says what the weights mean.

Input provenance (provenance guard 2026-08-18): under the 2026-08-18
sourcing rule a kriged leg reads ``config.KRIGED_CHANNELS`` out of the
kriged cache and EVERY other input channel from the clean MERRA-2
reanalysis, so ``airs.kriged_channels`` is as much a property of a metric as
the labels are.  Each ``_run.json`` records the resolved split as
``kriged_channels``, and :func:`check_kriged_split` REFUSES a kriged leg
whose split disagrees with the ``tunables.KRIGED_CHANNELS`` the checkpoint
records -- otherwise a "re-score on the fixed labels" would silently also
swap kriged winds for clean ones and the delta would be misattributed.

CLI::

    python -m dl_front.evaluate_test --ckpt runs/xyz/model.h5 --classes 6 \
        --source kriged-airs [--years 2016-2018] [--hours 18,21,0] \
        [--channels T2M,QV2M]
    python -m dl_front.evaluate_test --source bk19 --classes 6
    python -m dl_front.evaluate_test compare
    python -m dl_front.evaluate_test label-digest [--classes 6] \
        [--years 2016-2018]

Evaluation is streamed one year at a time (``PaperMetrics.update`` +
per-year CSI counts), so the test span never sits in memory at once.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, dataset, evaluate, predict

#: Column order of the tidy CSI CSV (frozen interface, 2026-08-12).
CSI_CSV_COLUMNS = ["front", "dilation", "km", "csi", "csi_lo", "csi_hi",
                   "pod", "far", "fb"]


# --------------------------------------------------------------------------- #
# Per-year data loading (with actionable missing-data errors)
# --------------------------------------------------------------------------- #

def _kriged_cache_path(year: int, source: str) -> Path:
    """Path of one kriged cache file; FileNotFoundError names the fix."""
    # manifest reorg 2026-08-13: KRIGED_SOURCE_DIRS holds the full cache dirs
    path = config.KRIGED_SOURCE_DIRS[source] / f"kriged_sfc_{year}.nc"
    if not path.exists():
        build = {"kriged-airs": "build-airs",
                 "kriged-degraded": "build-degraded"}[source]
        raise FileNotFoundError(
            f"no kriged cache for {year}: {path} does not exist; build "
            f"it with 'python -m dl_front.krige_fill {build} "
            f"--years {year}'")
    return path


def kriged_cache_times(year: int, source: str = "kriged-airs"
                       ) -> pd.DatetimeIndex:
    """Time axis of one kriged cache file (cheap: coords only)."""
    import xarray as xr

    with xr.open_dataset(_kriged_cache_path(year, source)) as ds:
        return pd.DatetimeIndex(ds["time"].values)


def bk19_path(year: int) -> Path:
    """Path of one BK19 published-prediction file; actionable error if absent.

    The published archive covers 1980-2018 (which is why the 6-class eval
    years are 2016-2018, user decision 2026-08-13).
    """
    w = config.LABEL_WIDTH
    path = (config.BK19_DIR / f"1deg_{w}wide" / "3hr"
            / f"merra2_merra2-1deg_{w}wide_3hr_{year}.nc")
    if not path.exists():
        raise FileNotFoundError(
            f"no BK19 published predictions for {year}: {path} does not "
            f"exist (the archive covers 1980-2018 only); the default "
            f"resolves inside the data root "
            f"(front_id/predicted_fronts/bk19, manifest reorg 2026-08-13) "
            f"-- set JPL_BK19_DIR only for an out-of-tree archive")
    return path


def bk19_class_grid(bk, n_classes: int) -> np.ndarray:
    """(time, lat, lon) uint8 class index from a BK19 prediction dataset.

    Same painter as ``dataset.class_grid`` (lowest ``config.TYPE_PRIORITY``
    first, so higher priority overwrites; 'none' = last index), with two
    BK19-specific accommodations: ``front_type`` names are matched
    case-insensitively, and classes absent from the file (dryline -- BK19
    never predicts it) are simply never painted.
    """
    names = dataset.class_names(n_classes)
    front_type = [str(s).lower() for s in bk["front_type"].values]
    n_time, _, n_lat, n_lon = bk["fronts"].shape
    cls = np.full((n_time, n_lat, n_lon), len(names) - 1, dtype=np.uint8)
    for name in reversed(config.TYPE_PRIORITY):
        if name not in names or name.lower() not in front_type:
            continue                       # dryline: not in the BK19 files
        sel = bk["fronts"].isel(front=front_type.index(name.lower()))
        hit = sel.values == 1
        cls[hit] = names.index(name)
    return cls


def bk19_year_arrays(year: int, n_classes: int
                     ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One year of the BK19 leg: (pred-as-x, y, times).

    "x" is the published prediction class grid carried as a trailing
    channel, (n, 68, 141, 1) -- :class:`BK19Predictions` one-hot encodes it
    at "inference" time, so the leg runs through the exact
    :func:`evaluate_ckpt` machinery (hours filter, kriged-cache time
    matching, provenance) as the checkpoint legs.  Labels/times come from
    the SAME ``valid_label_steps`` + exact-timestamp inner join as
    ``dataset.year_arrays``.
    """
    import xarray as xr

    with xr.open_dataset(bk19_path(year)) as bk:
        bk = bk.load()
    pred = bk19_class_grid(bk, n_classes)
    bk_times = pd.DatetimeIndex(bk["time"].values)

    with dataset.load_label_ds(year, n_classes) as lab:
        keep = dataset.valid_label_steps(lab, n_classes)
        cls = dataset.class_grid(lab, n_classes)[keep]
        label_times = pd.DatetimeIndex(lab["time"].values)[keep]

    common = bk_times.intersection(label_times)
    x = pred[bk_times.get_indexer(common)][..., None].astype(np.float16)
    y = cls[label_times.get_indexer(common)]
    return x, y, common


class BK19Predictions:
    """Duck-typed 'model' for the bk19 leg: one-hot encode the class grid.

    ``x`` (from :func:`bk19_year_arrays`) already holds the published class
    indices; "prediction" is a certainty one-hot, so downstream argmax
    reproduces the published classes exactly and PaperMetrics still
    accumulates (its ROC sweep is meaningless on hard classes -- the paper
    json is skipped for this leg).
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def predict(self, x, batch_size=64, verbose=0):
        cls = np.rint(np.asarray(x)[..., 0]).astype(np.int64)
        probs = np.zeros((*cls.shape, self.n_classes), dtype=np.float32)
        np.put_along_axis(probs, cls[..., None], 1.0, axis=-1)
        return probs


def load_year(year: int, n_classes: int, stats: dict, source: str
              ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One year of (x, y, times) from the requested source.

    Raises FileNotFoundError naming the missing data AND the command that
    produces it, so a failed cluster run tells the operator exactly what to
    do next.
    """
    if source == "bk19":
        return bk19_year_arrays(year, n_classes)   # stats unused: no z-score

    if source == "reanalysis":
        year_dir = config.SFC_DIR / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(
                f"no reanalysis surface data for {year}: {year_dir} does not "
                f"exist; fetch it with "
                f"'python -m dl_front.acquire_merra2_sfc {year}'")
        return dataset.year_arrays(year, n_classes, stats)

    if source in config.KRIGED_SOURCE_DIRS:
        _kriged_cache_path(year, source)          # actionable error if absent
        return dataset.kriged_year_arrays(year, n_classes, stats, source)

    raise ValueError(f"unknown source {source!r}; expected 'reanalysis', "
                     f"'bk19' or one of {sorted(config.KRIGED_SOURCE_DIRS)}")


# --------------------------------------------------------------------------- #
# Input-channel alignment (user decision 2026-08-18)
# --------------------------------------------------------------------------- #
# The model no longer necessarily consumes all five SFC_VARS: the stage-A
# channel ladder trains 3-channel (T2M,QV2M,SLP) and 2-channel (T2M,QV2M)
# rungs to size how much front skill survives on the fields AIRS actually
# retrieves.  Scoring a 2-channel checkpoint with 5-channel inputs (or the
# reverse) is the exact failure mode this evaluation cleanup exists to
# prevent -- it either crashes or, worse, silently produces plausible
# numbers -- so the alignment is resolved and CHECKED here, once, and
# reused by dl_front.permutation instead of being reimplemented.

def run_config_path(ckpt) -> Path | None:
    """Where ``train.py`` parked the checkpoint's self-description."""
    return None if ckpt is None \
        else Path(ckpt).resolve().parent / "run_config.yaml"


def read_run_config(ckpt) -> dict | None:
    """The parsed ``run_config.yaml`` next to a checkpoint, or None.

    ``train.py`` writes ``run_config.yaml`` into the checkpoint directory
    precisely so a checkpoint is self-describing.  Anything unreadable or
    absent returns None (a pre-2026-08-18 checkpoint predates some of the
    fields) -- this is a convenience, never a gate; the callers below hold
    the actual guards.

    This is the ONE reader: :func:`channels_from_run_config` (which
    channels the weights consume) and :func:`check_kriged_split` (which of
    those channels the weights saw KRIGED) both go through it, so a
    checkpoint directory is parsed by exactly one piece of code
    (provenance guard 2026-08-18).
    """
    path = run_config_path(ckpt)
    if path is None:
        return None
    try:
        import yaml

        run = yaml.safe_load(path.read_text())
    except (OSError, ValueError, ImportError, AttributeError):
        return None
    return run if isinstance(run, dict) else None


def channels_from_run_config(ckpt) -> list[str] | None:
    """The ``run_args.channels`` recorded next to a checkpoint, or None.

    Missing/unreadable/absent key -> None (a pre-2026-08-18 checkpoint
    predates the field and was necessarily all five channels).
    """
    run = read_run_config(ckpt)
    if run is None:
        return None
    try:
        names = (run.get("run_args") or {}).get("channels")
    except AttributeError:
        return None
    if not names:
        return None
    return [str(n) for n in names]


def resolve_channels(ckpt, spec: str | None) -> tuple:
    """Install the model's input channels -> the resolved tuple.

    ``spec``: the ``--channels`` value ("T2M,QV2M") or None.  Explicit
    always wins; otherwise the checkpoint's own ``run_config.yaml`` is
    adopted (and the adoption is PRINTED -- an evaluation that quietly
    changes its own inputs would be worse than one that fails).  With
    neither, ``config.INPUT_CHANNELS`` keeps its yaml/default value.
    Installation goes through :func:`config.set_input_channels` so the
    validation and the SFC_VARS ordering live in exactly one place.
    """
    if spec:
        return config.set_input_channels(spec.split(","))
    adopted = channels_from_run_config(ckpt)
    if adopted is not None:
        resolved = config.set_input_channels(adopted)
        print(f"adopted input channels {list(resolved)} from "
              f"{Path(ckpt).resolve().parent / 'run_config.yaml'} "
              f"(pass --channels to override)", flush=True)
        return resolved
    return config.set_input_channels(config.INPUT_CHANNELS)


def _normalised(names) -> tuple:
    """``names`` in :data:`config.SFC_VARS` order -- the ordering rule of
    :func:`config.set_input_channels`, applied WITHOUT installing anything.

    ``check_model_channels`` has to compare a checkpoint's recorded channel
    list against the already-installed ``config.INPUT_CHANNELS``, and the
    latter is normalised; comparing raw would flag ``QV2M,T2M`` as a
    mismatch with ``T2M,QV2M`` even though the setter maps both to the same
    channel axis.  A name outside SFC_VARS drops out here and so shortens
    the tuple, which the caller reports as a mismatch rather than ignoring.
    """
    got = {str(n).strip() for n in names}
    return tuple(v for v in config.SFC_VARS if v in got)


def check_model_channels(model, ckpt) -> None:
    """Refuse a checkpoint whose input channels are not the config's.

    The CNN is fully convolutional, so ``model.input_shape`` is
    ``(None, None, None, C)`` -- only the LAST axis is meaningful here and
    the spatial dims are deliberately not compared.  A mismatch raises;
    it is never a warning, because both directions of the mismatch score
    the model on inputs it was not trained on and produce numbers that
    look entirely reasonable.

    Two mismatches are caught: the channel COUNT, and -- when the
    checkpoint directory records ``run_args.channels`` and that list has
    the model's arity -- the channel NAMES.  The name check is new (review
    2026-08-18): the count alone let a same-arity relabelling through.  The
    ablation chain passes --channels explicitly to every step-3 eval while
    skip_train reuses an existing ``<name>_final.h5``, so editing
    CHANNEL_SETS to a different same-arity set (D6A2: T2M,QV2M ->
    T2M,SLP) re-scored the OLD D6A2 checkpoint under the NEW names, feeding
    SLP into the slot the model learned as QV2M and producing a complete,
    plausible, meaningless metric table.
    """
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):        # multi-input models: not ours
        shape = shape[0]
    if not shape:
        return                         # duck-typed model (bk19, test fakes)
    n_model = shape[-1]
    n_config = len(config.INPUT_CHANNELS)
    if n_model is None or n_model == n_config:
        # Counts agree -- but the NAMES still have to, when the checkpoint
        # says what it was trained on.  The checkpoint's own record wins
        # over --channels: the flag describes what the operator intends to
        # feed the model, run_config.yaml describes what the weights
        # actually mean, and only the latter can be right.
        recorded = channels_from_run_config(ckpt)
        if n_model is not None and recorded and len(recorded) == n_model:
            want = _normalised(recorded)
            if want != tuple(config.INPUT_CHANNELS):
                raise ValueError(
                    f"channel mismatch: the checkpoint {ckpt} was TRAINED on "
                    f"{','.join(want)} (run_args.channels in the "
                    f"run_config.yaml next to it) but this evaluation "
                    f"resolved config.INPUT_CHANNELS to "
                    f"{','.join(config.INPUT_CHANNELS)} -- same channel "
                    f"count, different fields, so every metric would be "
                    f"computed with the wrong field in each learned slot: "
                    f"plausible numbers, meaningless ones.  --channels "
                    f"cannot override the checkpoint's own record; either "
                    f"drop --channels (or pass "
                    f"--channels {','.join(want)}) to score this "
                    f"checkpoint, or retrain it on "
                    f"{','.join(config.INPUT_CHANNELS)} (the ablation "
                    f"chain: delete the stale <name>_final.h5, or rerun "
                    f"without skip_train, so the new CHANNEL_SETS entry "
                    f"gets its own checkpoint).")
        return
    # Prefer the checkpoint's OWN recorded channel list over a guess.  The
    # first version of this message always suggested SFC_VARS[:n_model],
    # which is right for the 5-channel and D6A3 rungs but wrong for any
    # other subset (a T2M,SLP model would be told to pass T2M,QV2M).
    # Handing the operator a confidently wrong flag in the error that
    # exists to prevent silently wrong numbers is its own trap
    # (integration 2026-08-18), so guess only when nothing is recorded.
    recorded = channels_from_run_config(ckpt)
    if recorded and len(recorded) == n_model:
        fix = (f"--channels {','.join(recorded)} (from run_config.yaml's "
               f"run_args.channels next to the checkpoint -- authoritative)")
    else:
        fix = (f"--channels {','.join(config.SFC_VARS[:n_model])} -- a GUESS "
               f"(the first {n_model} of config.SFC_VARS); this checkpoint "
               f"has no usable run_args.channels in a run_config.yaml, so "
               f"confirm it against how it was trained before trusting the "
               f"numbers, and re-train with --channels to record it")
    raise ValueError(
        f"channel mismatch: the checkpoint {ckpt} takes {n_model} input "
        f"channel(s) but config.INPUT_CHANNELS has {n_config} "
        f"({','.join(config.INPUT_CHANNELS)}); scoring it anyway would "
        f"produce plausible but meaningless metrics.  Pass the channel "
        f"list the checkpoint was trained on: {fix}.")


def check_kriged_split(ckpt, source: str) -> list[str]:
    """Resolve -- and guard -- the INPUT-provenance split of this leg.

    Returns the channels this evaluation will read out of the kriged cache
    (satellite-shaped gap fills); every other input channel comes from the
    clean MERRA-2 reanalysis at the same timestamp
    (``dataset.kriged_year_arrays``, sourcing rule 2026-08-18).  The list
    is what ``main`` records as ``kriged_channels`` in the ``_run.json``.

    WHY this guard exists (provenance guard 2026-08-18).  ``KRIGED_CHANNELS``
    is a yaml tunable (``airs.kriged_channels``), so it can differ between
    the run that TRAINED a checkpoint and the run that re-scores it.  The
    shipped D6B/D6C weights were trained with
    ``KRIGED_CHANNELS: [T2M, QV2M, U10M, V10M]``; today's config says
    ``[T2M, QV2M]``, so a re-score would feed those same weights CLEAN
    reanalysis winds where they were trained on kriged ones.  That is not a
    cosmetic difference: configs/dl_front.yaml documents the kriged winds
    arriving with ~+1.4/+2.1 m/s mean shifts and roughly halved variance
    (+0.28 / +0.47 sigma against the sfc_norm_stats sigmas 5.085 / 4.469, at
    ~0.7x std) on two of five channels.  Re-scoring "for the label fix"
    would then move D6B/D6C for BOTH reasons at once while the reanalysis
    leg moves for the label reason only, and the delta would be
    misattributed to the labels.  Nothing else catches this: the loader's
    guard fires only the other way (cache clean where the config says
    kriged), and the chain's staleness predicate compares only
    ``labels_sha1``.

    Only channels the model actually CONSUMES (``config.INPUT_CHANNELS``)
    matter, and only a kriged ``--source`` matters -- a ``--source
    reanalysis`` (or ``bk19``) leg reads no cache at all, so its split is
    empty by construction and it is deliberately left alone.

    A checkpoint with no ``run_config.yaml``, or one recording no
    ``tunables.KRIGED_CHANNELS`` (they predate the tunable), cannot be
    checked: that WARNS loudly and continues, naming what went unverified.
    """
    resolved = [c for c in config.INPUT_CHANNELS
                if c in config.KRIGED_CHANNELS]
    if source not in config.KRIGED_SOURCE_DIRS:
        # No cache is opened, so no input channel has kriged provenance:
        # [] is the literal truth for this leg, not "unknown".
        return []
    run = read_run_config(ckpt)
    where = run_config_path(ckpt)
    if run is None:
        print(f"WARNING: cannot verify the kriged/clean input split of "
              f"{ckpt}: no readable {where}.  This evaluation reads "
              f"{resolved or 'no channels'} from the {source} cache and the "
              f"rest from the clean reanalysis; if the checkpoint was "
              f"TRAINED with a different airs.kriged_channels the metrics "
              f"below mix an input-distribution shift into whatever change "
              f"they are meant to measure.  Confirm against how it was "
              f"trained (provenance guard 2026-08-18).", file=sys.stderr,
              flush=True)
        return resolved
    recorded = (run.get("tunables") or {}).get("KRIGED_CHANNELS") \
        if isinstance(run.get("tunables"), dict) else None
    if not recorded:
        print(f"WARNING: {where} records no tunables.KRIGED_CHANNELS "
              f"(it predates the tunable), so the kriged/clean input split "
              f"{ckpt} was TRAINED with cannot be verified.  This "
              f"evaluation reads {resolved or 'no channels'} from the "
              f"{source} cache and the rest from the clean reanalysis "
              f"(provenance guard 2026-08-18).", file=sys.stderr, flush=True)
        return resolved
    # Compare only over the channels the model consumes, in SFC_VARS order
    # (_normalised), so a reordered yaml list is not a "mismatch" and a
    # recorded wind entry is irrelevant to a T2M,QV2M model.
    trained = [c for c in _normalised(recorded)
               if c in config.INPUT_CHANNELS]
    if trained == resolved:
        return resolved
    changed = sorted(set(trained) ^ set(resolved))
    raise ValueError(
        f"kriged-split mismatch: the checkpoint {ckpt} was TRAINED with "
        f"airs.kriged_channels={list(recorded)} -> of its input channels "
        f"{list(config.INPUT_CHANNELS)} it saw {trained or 'none'} kriged "
        f"(tunables.KRIGED_CHANNELS in {where}), but this evaluation has "
        f"config.KRIGED_CHANNELS={list(config.KRIGED_CHANNELS)} -> it would "
        f"read {resolved or 'none'} from the {source} cache and take "
        f"{[c for c in config.INPUT_CHANNELS if c not in resolved] or 'none'}"
        f" from the clean MERRA-2 reanalysis.  {changed} would therefore "
        f"change PROVENANCE between training and scoring: the weights would "
        f"see a different input distribution than they were trained on "
        f"(the kriged winds alone carry ~+1.4/+2.1 m/s mean shifts at "
        f"roughly half the variance of the clean ones), and any delta "
        f"against an earlier number would be misattributed -- e.g. to the "
        f"2026-08-17 label fix -- when part of it is this input change.  "
        f"To re-score with the checkpoint's OWN split (which isolates the "
        f"label effect), set 'airs: kriged_channels: "
        f"{list(recorded)}' in configs/dl_front.yaml (or in the yaml named "
        f"by JPL_DLFRONT_CONFIG) and rerun; to deliberately measure the "
        f"input change instead, retrain under the current split.")


def labels_dir(n_classes: int) -> Path:
    """The label tree :func:`dataset.label_digest` fingerprints, for the
    ``_run.json``: NOAA-XML (6-class, has drylines) or CODSUS (5-class).
    Resolved at call time so a monkeypatched backup tree is reported."""
    from front_finder import config as fd_config

    return (fd_config.NOAA_LABELS_DIR if n_classes == 6
            else fd_config.CODSUS_DIR)


# --------------------------------------------------------------------------- #
# Core evaluation (importable; model is anything with .predict)
# --------------------------------------------------------------------------- #

def evaluate_ckpt(model, years, n_classes: int, source: str,
                  hours=None, stats: dict | None = None,
                  loader=None, batch_size: int = 64,
                  match_source: str | None = None,
                  info: dict | None = None
                  ) -> tuple[evaluate.PaperMetrics, pd.DataFrame]:
    """Streamed test evaluation -> (PaperMetrics, pooled CSI scores).

    ``model``: any object with ``.predict(x, batch_size=, verbose=)``
    returning (n, 68, 141, n_classes) softmax outputs.
    ``hours``: UTC label hours to keep; default ``config.AIRS_HOURS`` for
    EVERY source.
    ``loader``: optional ``loader(year) -> (x, y, times)`` override of
    :func:`load_year` (tests inject synthetic data here).
    ``match_source``: name of a kriged cache whose per-year TIME AXIS the
    scored steps are intersected with -- the identical-sample-set guarantee
    for cross-source comparisons (a sparse AIRS archive covers only some
    days; see module docstring).  The CLI sets ``'kriged-airs'`` for
    reanalysis runs.
    ``info``: optional dict, filled with per-year step counts and a SHA-1 of
    every scored timestamp (comparability provenance).

    Scoring mask (user decision 2026-08-13): the 6-class dryline/AIRS
    track scores EVERY source -- reanalysis, kriged-airs AND bk19 -- over
    ``dataset.analysis_domain()`` (box ∩ land), so the three legs are
    compared on identical pixels; the 5-class paper replication keeps the
    Fig. 2 region mask.
    """
    hours = config.AIRS_HOURS if hours is None else tuple(hours)
    if loader is None:
        if stats is None and source != "bk19":   # bk19 has no z-scoring
            stats = dataset.load_norm_stats()
        loader = lambda year: load_year(year, n_classes, stats, source)

    mask = (dataset.analysis_domain() if n_classes == 6
            else dataset.region_mask().astype(bool))
    pm = evaluate.PaperMetrics(n_classes, mask=mask)
    counts, n_steps, scored_times = [], {}, []
    for year in years:
        x, y, times = dataset.filter_hours(*loader(year), hours)
        if match_source is not None and match_source != source:
            ref = kriged_cache_times(year, match_source)
            keep = times.isin(ref)
            if (~keep).any():
                print(f"{year}: {int((~keep).sum())} steps absent from the "
                      f"{match_source} cache dropped for comparability",
                      flush=True)
            x, y, times = x[keep], y[keep], times[keep]
        if len(x) == 0:
            print(f"{year}: no steps at hours {hours}, skipped", flush=True)
            n_steps[year] = 0
            continue
        probs = np.asarray(model.predict(x.astype(np.float32),
                                         batch_size=batch_size, verbose=0))
        pm.update(probs, y)
        counts.append(evaluate.csi_counts(probs.argmax(-1), y, times,
                                          n_classes, mask=mask))
        n_steps[year] = len(x)
        scored_times.extend(times)
        print(f"{year}: {len(x)} steps scored", flush=True)
    if not counts:
        raise RuntimeError(f"no data found for years {list(years)} at hours "
                           f"{hours} (source={source})")
    if info is not None:
        digest = hashlib.sha1("\n".join(
            str(t) for t in sorted(scored_times)).encode()).hexdigest()
        info.update(n_steps_per_year={int(y): int(n)
                                      for y, n in n_steps.items()},
                    times_sha1=digest, match_source=match_source)
    all_counts = pd.concat(counts, ignore_index=True)
    scores = evaluate.csi_scores(all_counts)
    # Day-block bootstrap CIs (audit + user decision 2026-08-15): cross-leg
    # CSI deltas -- dryline especially, ~150-190 event-bearing steps/year
    # and strongly autocorrelated -- are not interpretable without
    # uncertainty; deltas inside overlapping CIs are sampling noise.
    boot = evaluate.block_bootstrap(all_counts)
    scores["csi_lo"] = boot.lo["csi"]
    scores["csi_hi"] = boot.hi["csi"]
    return pm, scores


# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #

def _git_rev() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, text=True,
            capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ckpt_sha1(ckpt) -> str:
    """Hex SHA-1 of the checkpoint file's bytes.

    Written into every ``_run.json`` that scored a checkpoint (the
    checkpoint-free bk19 leg omits it) so the chain scripts can tell that a
    metric CSV belongs to the WEIGHTS currently on disk -- a retrain under
    the same name leaves every path unchanged, and without the digest a
    stale eval is indistinguishable from a fresh one.
    """
    return hashlib.sha1(Path(ckpt).read_bytes()).hexdigest()


def write_outputs(pm: evaluate.PaperMetrics, scores: pd.DataFrame,
                  ckpt: Path | None, source: str, years, hours,
                  out_dir: Path | None = None,
                  info: dict | None = None,
                  write_paper: bool = True) -> dict:
    """Write the CSV + paper json + provenance json; return their paths.

    ``ckpt=None`` is the checkpoint-free bk19 leg: the stem is just the
    source name.  ``write_paper=False`` (bk19) skips the paper json -- hard
    binary predictions make the ROC none-scaling sweep meaningless -- and
    records that in the run json instead.
    """
    out_dir = Path(out_dir) if out_dir is not None \
        else config.RESULTS_DIR / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source if ckpt is None else f"{Path(ckpt).stem}_{source}"

    tidy = scores.reset_index()[CSI_CSV_COLUMNS]
    csv_path = out_dir / f"{stem}.csv"
    tidy.to_csv(csv_path, index=False)
    paths = {"csv": csv_path}

    if write_paper:
        conf = pm.confusion_table(percent=True)
        paper = {"accuracy": pm.accuracy(),
                 "auc": pm.auc(),
                 "confusion_percent": {actual: dict(row)
                                       for actual, row in conf.iterrows()}}
        paper_path = out_dir / f"{stem}_paper.json"
        paper_path.write_text(json.dumps(paper, indent=1))
        paths["paper"] = paper_path

    run_path = out_dir / f"{stem}_run.json"
    run_path.write_text(json.dumps(
        {"ckpt": None if ckpt is None else str(Path(ckpt).resolve()),
         # weights fingerprint: ties this eval to the checkpoint BYTES, not
         # just its path (see ckpt_sha1).  Omitted when the file is absent
         # (synthetic test paths); the chain scripts treat a run.json
         # lacking the key as stale whenever the checkpoint file exists.
         **({} if ckpt is None or not Path(ckpt).is_file()
            else {"ckpt_sha1": ckpt_sha1(ckpt)}),
         "source": source,
         "years": [int(y) for y in years], "hours": [int(h) for h in hours],
         **({} if write_paper
            else {"paper_metrics": "skipped (binary baseline)"}),
         **(info or {}),
         "git_rev": _git_rev(),
         "created": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        indent=1))
    paths["run"] = run_path
    return paths


# --------------------------------------------------------------------------- #
# Leg comparison (the three-way table, user decision 2026-08-13)
# --------------------------------------------------------------------------- #

def _check_same_sample(provenance: dict) -> bool:
    """Warn loudly when the legs did not score the same sample.

    ``provenance`` maps leg stem -> parsed ``_run.json`` dict (or None when
    the file is missing/unreadable).  TWO digests must agree across legs:

    * ``times_sha1`` -- the scored-timestamp SHA-1.  evaluate_ckpt's time
      matching only DROPS steps, so a leg with missing source days (or a
      stale --no-match / partial --years run) scores a different sample
      without any error.
    * ``labels_sha1`` -- the LABEL content digest.  The labels were
      regenerated IN PLACE on 2026-08-17 (antimeridian-crossing polyline
      bug) without moving the scored time axis, so a stale pre-fix leg and
      a fresh post-fix leg have IDENTICAL times_sha1 and would pool
      silently into the normal
      comparison.csv -- exactly what ``FOLDS=0 FORCE_EVAL=1`` produces
      (fold 0 re-scored on clean labels, folds 1-2 left on the old ones).
      A leg with NO labels_sha1 was written before the field existed, i.e.
      necessarily on pre-fix labels, so it counts as a MISMATCH against any
      leg that has one; a table where NO leg records it is left alone (a
      wholly pre-2026-08-18 set of CSVs is self-consistent, and this check
      is not the place to re-litigate it).

    Returns True when both checks pass.
    """
    def _agree(key: str, require_present: bool) -> bool:
        """Do all legs carry the SAME non-None ``key``?

        ``require_present=False`` tolerates the one case where nobody
        recorded the key (every CSV predates the field); a value present on
        some legs and absent on others is always a disagreement, because
        "absent" means "scored on labels we can no longer identify".
        """
        vals = [(run or {}).get(key) for run in provenance.values()]
        seen = {v for v in vals if v is not None}
        if len(seen) > 1:
            return False                      # two different digests
        if not require_present and not seen:
            return True                       # nobody recorded it at all
        return all(v is not None for v in vals)

    same_times = _agree("times_sha1", require_present=True)
    same_labels = _agree("labels_sha1", require_present=False)
    if same_times and same_labels:
        return True
    what = ("score identical time steps" if not same_times
            else "use the same front LABELS")
    print(f"\nWARNING: the legs did NOT {what} -- the "
          "columns below are NOT a like-for-like comparison.  Per-leg "
          "scored-timestamp SHA-1 and step counts:", flush=True)
    for stem, run in sorted(provenance.items()):
        if run is None:
            print(f"  {stem}: no readable {stem}_run.json (unknown sample)",
                  flush=True)
        else:
            lab = run.get("labels_sha1") or ("ABSENT (CSV written before "
                                             "2026-08-18, so scored on the "
                                             "pre-fix labels)")
            print(f"  {stem}: sha1={run.get('times_sha1')} "
                  f"labels_sha1={lab} "
                  f"years={run.get('years')} match_source="
                  f"{run.get('match_source')} "
                  f"n_steps={run.get('n_steps_per_year')}", flush=True)
    if not same_times:
        print("re-run the odd leg(s) out with the default --years and time "
              "matching (no --no-match), then compare again", flush=True)
    if not same_labels:
        # Name the legs by digest GROUP rather than guessing which group is
        # the current one: neither the majority nor the minority is
        # reliably the fresh set (re-scoring one fold of three leaves 6
        # fresh legs against 12 stale ones, and FOLDS="0" with two legs is
        # an exact tie).  The operator resolves it against label-digest,
        # which reads the labels actually on disk.
        groups: dict = {}
        for stem, run in sorted(provenance.items()):
            groups.setdefault((run or {}).get("labels_sha1"), []).append(stem)
        for sha, stems in sorted(groups.items(), key=lambda kv: str(kv[0])):
            print(f"  labels_sha1={sha or 'ABSENT (pre-2026-08-18)'}: {stems}",
                  flush=True)
        print("the front labels were REGENERATED IN PLACE on 2026-08-17 "
              "(antimeridian polyline bug), which does not move the scored "
              "time axis -- so the legs above share a times_sha1 but were "
              "scored on DIFFERENT labels, and pooling them compares label "
              "versions, not models.  Print the digest of the labels "
              "currently on disk with 'python -m dl_front.evaluate_test "
              "label-digest', then re-score every leg not carrying it with "
              "FORCE_EVAL=1 (the chain scripts' knob that ignores an "
              "existing CSV) and compare again", flush=True)
    return False


def compare(out_dir: Path | None = None) -> pd.DataFrame:
    """Pivot pooled CSI across every leg CSV -> comparison.csv + printed table.

    Dumb and robust by design: every ``*.csv`` in the eval dir carrying the
    frozen :data:`CSI_CSV_COLUMNS` is a leg named by its stem; anything else
    (including ``comparison.csv`` itself) is skipped with a note, and a leg
    missing a (front, dilation_km) row simply shows NaN there.

    One check IS performed: the same-sample guarantee, over BOTH digests in
    each leg's ``_run.json`` (see :func:`_check_same_sample`) --
    ``times_sha1`` (``evaluate_ckpt``'s time matching only drops steps, so a
    leg can silently score fewer steps than the others: missing sfc_daily
    day files, a stale ``--no-match`` or partial ``--years`` run) and
    ``labels_sha1`` (the labels were regenerated in place on 2026-08-17
    WITHOUT moving the time axis, so a stale leg and a fresh one agree on
    times_sha1 and would pool silently).  Either
    disagreement is reported loudly here AND encoded in the output name
    (the table is written as ``comparison_MISMATCHED_SAMPLE.csv``; the
    warning tells you which leg to rerun and how).
    """
    out_dir = Path(out_dir) if out_dir is not None \
        else config.RESULTS_DIR / "test_eval"
    legs, provenance = {}, {}
    for path in sorted(out_dir.glob("*.csv")):
        if path.name == "comparison.csv":
            continue
        df = pd.read_csv(path)
        if not set(CSI_CSV_COLUMNS) <= set(df.columns):
            print(f"skipping {path.name}: not a CSI leg CSV", flush=True)
            continue
        idx = df.set_index(["front", "km"])
        legs[path.stem] = idx["csi"]
        legs[f"{path.stem}_lo"] = idx["csi_lo"]
        legs[f"{path.stem}_hi"] = idx["csi_hi"]
        run_path = path.with_name(f"{path.stem}_run.json")
        try:
            provenance[path.stem] = json.loads(run_path.read_text())
        except (OSError, ValueError):
            provenance[path.stem] = None      # reported below
    if not legs:
        raise FileNotFoundError(f"no leg CSVs found in {out_dir}; run some "
                                f"evaluations first")
    same_sample = _check_same_sample(provenance)
    table = pd.DataFrame(legs)                # aligns on the index union
    table.index.names = ["front", "dilation_km"]
    # a non-like-for-like table must not masquerade as the headline result:
    # the stdout warning never reaches CSV consumers, so the mismatch is
    # encoded in the artifact's NAME (audit 2026-08-15)
    out_path = out_dir / ("comparison.csv" if same_sample
                          else "comparison_MISMATCHED_SAMPLE.csv")
    table.to_csv(out_path)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_path}")
    return table


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_years(spec: str) -> list[int]:
    """'2016-2018' (inclusive range) or '2016,2018,2020' -> list of ints."""
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def resolve_years(spec: str | None, n_classes: int) -> list[int]:
    """--years spec, or (None) the yaml eval split of the class count."""
    if spec:
        return parse_years(spec)
    return list(config.EVAL_YEARS_6 if n_classes == 6
                else config.EVAL_YEARS_5)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare"]:               # leg-comparison subcommand
        if argv[1:]:                          # never silently drop arguments
            sys.exit(f"evaluate_test compare takes no arguments (got "
                     f"{argv[1:]}); it always pivots every leg CSV in "
                     f"{config.RESULTS_DIR / 'test_eval'}")
        compare()
        return

    if argv[:1] == ["label-digest"]:          # label-provenance subcommand
        # Its own parser (no --ckpt: this reads labels, not checkpoints),
        # and like 'compare' it refuses to silently drop arguments --
        # argparse errors on anything unexpected.  Prints the digest and
        # NOTHING else, so a chain script can capture it with $(...).
        dp = argparse.ArgumentParser(
            prog="evaluate_test label-digest",
            description="Print the SHA-1 content digest of the front labels "
                        "over the evaluation years (dataset.label_digest); "
                        "compare it with a _run.json's labels_sha1 to find "
                        "metric CSVs computed on stale labels.")
        dp.add_argument("--classes", type=int, default=6, choices=(5, 6))
        dp.add_argument("--years", default=None,
                        help="'A-B' range or comma list (default: the yaml "
                             "eval split of --classes)")
        d = dp.parse_args(argv[1:])
        print(dataset.label_digest(resolve_years(d.years, d.classes),
                                   d.classes))
        return

    ap = argparse.ArgumentParser(
        description="Score one evaluation leg on the held-out test years "
                    "(paper metrics + neighborhood CSI, AIRS label hours): "
                    "a checkpoint on reanalysis or kriged-AIRS inputs, or "
                    "the published BK19 predictions.  The 'compare' "
                    "subcommand pivots every leg CSV into comparison.csv; "
                    "the 'label-digest' subcommand prints the current "
                    "labels' content digest.")
    ap.add_argument("--ckpt", default=None,
                    help="trained .h5 checkpoint (required unless "
                         "--source bk19, which has no checkpoint)")
    ap.add_argument("--classes", type=int, default=6, choices=(5, 6))
    ap.add_argument("--source", default="reanalysis",
                    choices=("reanalysis", "kriged-airs", "bk19"),
                    help="input fields / predictor to score (every leg is "
                         "scored on the same AIRS-hour time steps)")
    ap.add_argument("--years", default=None,
                    help="'A-B' range or comma list (default: the yaml eval "
                         "split of --classes, e.g. "
                         f"{list(config.EVAL_YEARS_6)} for 6)")
    ap.add_argument("--hours", default=None,
                    help="comma-separated UTC hours to score "
                         f"(default: AIRS hours {config.AIRS_HOURS})")
    ap.add_argument("--channels", default=None,
                    help="comma-separated input channels the checkpoint "
                         f"consumes, a subset of {','.join(config.SFC_VARS)} "
                         "(default: adopt run_args.channels from the "
                         "checkpoint's run_config.yaml, else the yaml "
                         "inputs.channels); a checkpoint whose channel "
                         "count disagrees is refused, never scored")
    ap.add_argument("--no-match", action="store_true",
                    help="do NOT intersect a reanalysis/bk19 run's time "
                         "steps with the kriged-airs cache (scores every "
                         "step, so the result is NOT comparable to a "
                         "kriged-airs run on a sparse AIRS archive)")
    a = ap.parse_args(argv)
    if a.source == "bk19" and a.ckpt:
        ap.error("--ckpt does not apply to --source bk19 (the published "
                 "prediction files ARE the model output)")
    if a.source != "bk19" and not a.ckpt:
        ap.error(f"--ckpt is required for --source {a.source}")
    if a.source == "bk19" and a.channels:
        ap.error("--channels does not apply to --source bk19 (its 'inputs' "
                 "are the published prediction rasters, not surface fields)")

    years = resolve_years(a.years, a.classes)
    hours = (tuple(int(h) for h in a.hours.split(","))
             if a.hours else config.AIRS_HOURS)
    match_source = ("kriged-airs"
                    if a.source in ("reanalysis", "bk19") and not a.no_match
                    else None)

    is_bk19 = a.source == "bk19"
    # Channels first: they decide the shape of every array loaded below.
    channels = None if is_bk19 else resolve_channels(a.ckpt, a.channels)
    model = BK19Predictions(a.classes) if is_bk19 \
        else predict.load_model(a.ckpt)
    if not is_bk19:
        check_model_channels(model, a.ckpt)   # never score a mismatch
    # Label provenance BEFORE the (expensive) scoring pass: it costs
    # seconds and it is what makes this CSV comparable to any other.
    # INPUT provenance, recorded next to the LABEL provenance and checked
    # the same way (provenance guard 2026-08-18): which input channels came
    # from the kriged cache and which from the clean reanalysis is a real
    # confound between legs, and -- like labels_sha1 -- it is invisible
    # after the fact unless the number carries it.
    kriged_channels = check_kriged_split(a.ckpt, a.source)
    info: dict = {"channels": None if is_bk19 else list(channels),
                  "kriged_channels": kriged_channels,
                  "labels_sha1": dataset.label_digest(years, a.classes),
                  "labels_dir": str(labels_dir(a.classes))}
    if is_bk19:
        info["dryline"] = ("not predicted: the BK19 files carry no dryline "
                           "class, so dryline CSV rows are all-miss by "
                           "construction")
    pm, scores = evaluate_ckpt(model, years, a.classes, a.source, hours,
                               match_source=match_source, info=info)
    paths = write_outputs(pm, scores,
                          None if is_bk19 else Path(a.ckpt),
                          a.source, years, hours, info=info,
                          write_paper=not is_bk19)

    acc = pm.accuracy()
    line = (f"\naccuracy: all={acc['all_categories']:.4f} "
            f"front/no-front={acc['front_no_front']:.4f}")
    if not is_bk19:                # ROC AUC is meaningless on hard classes
        line += f"  auc={pm.auc():.4f}"
    print(line)
    print(scores.reset_index()[CSI_CSV_COLUMNS].to_string(index=False))
    print("\nwrote " + "\n      ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    sys.exit(main())
