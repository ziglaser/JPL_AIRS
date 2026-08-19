"""Stage-aware DL-FRONT training driver (mirrors front_finder.train).

Stages: A = clean MERRA-2 replication (paper section 4.1); B = degraded
reanalysis ("AIRS simulator", ``--degraded --retrain <A ckpt>``); C = real
AIRS fine-tune on the JPL laptop (``--finetune-glob`` over files in the
sfc_daily schema produced by an AIRS ingest, ``--retrain <B ckpt>``).

CLI:
  PYTHONPATH=src python -m dl_front.train --name R5-fold0 --classes 5
  PYTHONPATH=src python -m dl_front.train --name D6 --classes 6   # + drylines
  PYTHONPATH=src python -m dl_front.train --name B5 --classes 5 --degraded \
      --retrain results/dl_front/models/R5-fold0/R5-fold0.h5

Kriged-cache variants (dl_front.krige_fill output; decision 2026-08-12):
``--source kriged-degraded`` / ``--source kriged-airs`` swap the input
fields for the gap-filled caches (same years, same fold split), and
``--hours 18,21,0`` restricts training to the AIRS-covered label hours.

Channel ablation (user decision 2026-08-18): ``--channels T2M,QV2M`` trains
on a NAMED SUBSET of the on-disk surface variables.  Only T2M/QV2M carry
AIRS information -- U10M/V10M are the WRF-27km driving winds and SLP is
copied clean from MERRA-2 -- so the stage-A ladder 5 -> T2M,QV2M,SLP ->
T2M,QV2M measures how much front skill survives an AIRS-only input set.
The resolved list is written into the checkpoint's run_config.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, dataset, degrade_sfc, model as model_mod

# Stage-B/C learning rates follow the UNET3+ curriculum ratios; all four
# knobs live in configs/dl_front.yaml (stages: section).
DEGRADED_LR = config.LEARNING_RATE * config.DEGRADED_LR_FACTOR
FINETUNE_LR = config.LEARNING_RATE * config.FINETUNE_LR_FACTOR
FINETUNE_PATIENCE = config.FINETUNE_PATIENCE
SEVERITY_RAMP_EPOCHS = config.SEVERITY_RAMP_EPOCHS
MAX_EPOCHS = config.MAX_EPOCHS


def train_years(n_classes: int) -> tuple:
    return {5: config.TRAIN_YEARS_5, 6: config.TRAIN_YEARS_6}[n_classes]


def loss_mask_for(n_classes: int, source: str, degraded: bool = False
                  ) -> tuple[str, np.ndarray]:
    """The per-pixel loss-weight mask of one training run -> (name, mask).

    User decision 2026-08-13 (6-class dryline/AIRS track only; the 5-class
    paper replication keeps the codsus region mask untouched):

    * stage A (clean reanalysis) trains on ``crop_domain()`` -- box + halo,
      to harvest nearby front examples for the translation-invariant
      filters;
    * every gap-degraded stage (kriged caches, and the legacy on-the-fly
      ``--degraded`` stage B) trains on ``analysis_domain()`` only -- the
      box ∩ land pixels that are the actual product.

    The name is recorded in run_config.yaml so a checkpoint's provenance
    states which mask weighted its loss.
    """
    if n_classes != 6:
        return "region_mask", dataset.region_mask()
    if source == "reanalysis" and not degraded:          # stage A
        return "crop_domain", dataset.crop_domain().astype(np.float32)
    return "analysis_domain", dataset.analysis_domain().astype(np.float32)


def label_provenance(years, n_classes: int) -> dict:
    """Which front labels this run TRAINED on -> run_config.yaml fields.

    Why this exists: the labels were regenerated in
    place on 2026-08-17 (antimeridian polyline bug), and ``_run.json``
    already records ``labels_sha1`` so a NUMBER can be traced to the label
    content it was scored against.  A MODEL had no such field: after a
    FORCE_EVAL=1 re-score the D6A/B/C checkpoints keep their PRE-FIX
    training while their metrics are recomputed on clean labels, so a fold
    retrained on clean labels and a fold left over from the buggy ones look
    identical on disk and their CSIs get pooled into one comparison column.
    Recording the digest next to the weights is what separates them.

    The digest is over the TRAINING years actually used (``years`` as
    passed to :func:`run` after the ``--smoke`` truncation), NOT the
    evaluation years: what a checkpoint's weights depend on is the labels
    it fit, and the eval span is already fingerprinted in ``_run.json``.

    It is a CONTENT digest (:func:`dataset.label_digest` -- per-class cell
    counts over the scored steps and mask), not a hash of the label files:
    it moves iff the labels move where they matter, and is a staleness
    detector, not a security control.

    NEVER fatal: a provenance field must not be the reason an overnight
    training job dies, so a missing/unreadable label tree writes
    ``labels_sha1: null`` plus a ``labels_note`` saying why, and training
    proceeds (the load of those same labels a few lines later will raise
    the real, informative error if they are genuinely absent).
    """
    prov = {"labels_years": [int(y) for y in years],
            "labels_sha1": None, "labels_dir": None}
    try:
        # same resolution the _run.json uses, so the two fields are
        # comparable byte-for-byte without a second convention
        from .evaluate_test import labels_dir

        prov["labels_dir"] = str(labels_dir(n_classes))
        prov["labels_sha1"] = dataset.label_digest(years, n_classes)
    except Exception as exc:                       # noqa: BLE001 - see above
        prov["labels_note"] = (
            f"digest unavailable ({type(exc).__name__}: {exc}); training "
            f"continued.  Recompute later with: PYTHONPATH=src python -m "
            f"dl_front.evaluate_test label-digest --classes {n_classes} "
            f"--years {int(years[0])}-{int(years[-1])}")
        print(f"warning: {prov['labels_note']}", file=sys.stderr)
    return prov


def make_degraded_tf_dataset(x, y, n_classes, stats, severity, seed,
                             batch_size=config.BATCH_SIZE, shuffle=True):
    """Stage-B pipeline: noise + real gap masks resampled every pass.

    ``severity`` is a float or zero-arg callable (ramp callback support).
    Loss weights: analysis_domain() for 6-class, region_mask() for 5-class
    (loss_mask_for; user decision 2026-08-13).
    """
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    _, mask = loss_mask_for(n_classes, "reanalysis", degraded=True)

    def gen():
        # Validation (shuffle=False) restarts the rng every pass so each
        # epoch scores the SAME degradations -- val_loss differences then
        # reflect the weights, not the noise draw, which is what the
        # early-stopping/checkpoint callbacks compare.  Training keeps the
        # persistent rng and sees fresh noise every epoch.
        r = rng if shuffle else np.random.default_rng(seed)
        order = r.permutation(len(x)) if shuffle else np.arange(len(x))
        for i in order:
            s = severity() if callable(severity) else severity
            vf = degrade_sfc.surface_gap_field(r)
            yield degrade_sfc.degrade_x(x[i], r, stats, s, vf), y[i]

    sig = (tf.TensorSpec((*config.GRID_SHAPE, x.shape[-1]), tf.float32),
           tf.TensorSpec(config.GRID_SHAPE, tf.uint8))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    w = tf.constant(mask)

    def to_pair(xi, yi):
        y_true = tf.concat([tf.one_hot(tf.cast(yi, tf.int32), n_classes),
                            w[..., None]], axis=-1)
        return xi, y_true

    return (ds.map(to_pair, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def run(name: str, n_classes: int, fold: int = 0,
        lr: float = config.LEARNING_RATE, epochs: int = MAX_EPOCHS,
        patience: int = config.PAPER_PATIENCE, retrain: str | None = None,
        batch_size: int = config.BATCH_SIZE, degraded: bool = False,
        finetune_glob: str | None = None, smoke: bool = False,
        source: str = "reanalysis", hours: tuple | None = None,
        channels: tuple | None = None) -> Path:
    """Train one stage and return its checkpoint directory.

    ``channels``: the resolved model input channels, for provenance only --
    :func:`main` has already installed them with
    ``config.set_input_channels`` BEFORE any data is loaded (the subset has
    to be in force by the time dataset.sfc_x stacks the channel axis).
    Passing None records the config/YAML value in force.
    """
    import tensorflow as tf

    out = config.RESULTS_DIR / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    # <name>_final.h5 is the done-marker the chain scripts key on
    # (skip_train, and the ablation chain's permutation-readiness gate): its
    # existence must mean "the CURRENT weights finished training".  Delete a
    # leftover from a previous run of this name up front; it is re-created
    # by the m.save() at the end iff fit() completes.
    (out / f"{name}_final.h5").unlink(missing_ok=True)
    # Likewise CSVLogger(append=True) would splice this run's curve onto a
    # previous same-name run's; rotate the old file aside so history.csv
    # always holds exactly one training curve.  (There is no
    # resume-in-place flow to preserve -- --retrain warm-starts from an
    # EXTERNAL checkpoint and fit() runs once per invocation.)
    hist = out / "history.csv"
    if hist.exists():
        hist.replace(out / "history.prev.csv")
    # Provenance: freeze the run's resolved tunables + call arguments next to
    # the weights, so a checkpoint is reproducible even after the tracked
    # YAML (or a JPL_DLFRONT_CONFIG override) changes.
    import yaml as _yaml

    mask_name, loss_mask = loss_mask_for(n_classes, source, degraded)
    # Resolved BEFORE the yaml is written so the
    # provenance names the years the run really fits, --smoke truncation
    # included, rather than the full configured span.
    years = train_years(n_classes)
    if smoke:
        years = years[:1]
    # config.load_tunables() re-READS the YAML, so a --channels override
    # would not appear in the snapshot; overwrite INPUT_CHANNELS with the
    # value actually in force (user decision 2026-08-18).  The list is
    # recorded TWICE on purpose -- as run_args.channels (what the operator
    # asked for, which evaluate_test adopts automatically) and inside
    # tunables (what the model was built from) -- so a checkpoint directory
    # is self-describing and a future reader never has to guess what its
    # inputs were.  Both are plain lists, not tuples, to keep the YAML clean.
    tunables = {k: list(v) if isinstance(v, tuple) else v
                for k, v in config.load_tunables().items()}
    channels = tuple(channels) if channels else tuple(config.INPUT_CHANNELS)
    tunables["INPUT_CHANNELS"] = list(channels)
    (out / "run_config.yaml").write_text(_yaml.safe_dump(
        {"tunables_from": str(config.CONFIG_YAML),
         # which per-pixel loss-weight mask this run trained under
         # (user decision 2026-08-13; see loss_mask_for)
         "loss_mask": mask_name,
         # WHICH LABELS THIS CHECKPOINT WAS TRAINED ON (see
         # label_provenance): top level, next to loss_mask rather than
         # buried in run_args, because it is a property of the fitted
         # weights and not of the operator's command line.
         **label_provenance(years, n_classes),
         "tunables": tunables,
         "run_args": {"name": name, "n_classes": n_classes, "fold": fold,
                      "lr": lr, "epochs": epochs, "patience": patience,
                      "retrain": retrain, "batch_size": batch_size,
                      "degraded": degraded, "finetune_glob": finetune_glob,
                      "smoke": smoke, "source": source,
                      "hours": list(hours) if hours is not None else None,
                      "channels": list(channels)}},
        sort_keys=False))
    stats = dataset.load_norm_stats()
    extra_callbacks = []

    if finetune_glob:              # ---- stage C: AIRS-schema files ---------
        import glob

        paths = sorted(glob.glob(finetune_glob))
        if not paths:
            raise FileNotFoundError(f"no files match {finetune_glob}")
        # Files must follow the sfc_daily schema (SFC_VARS on the label
        # grid); pairing/eval years are whatever the files' times hit.
        raise NotImplementedError(
            "stage C runs on the JPL laptop once the AIRS surface ingest "
            "lands; see docs/DLFRONT_WORKPLAN.md section 4")

    if source == "reanalysis":
        x, y, times = dataset.stack_years(years, n_classes, stats)
    elif source in config.KRIGED_SOURCE_DIRS:
        # Kriged caches (stage B' degraded / stage C real AIRS) share the
        # reanalysis training years; fold membership is day-keyed below,
        # so a calendar day sits in the same fold at every stage no matter
        # how many steps each source carries for it.
        x, y, times = dataset.stack_kriged_years(years, n_classes, stats,
                                                 source)
    else:
        raise ValueError(f"unknown source {source!r}; choose 'reanalysis' or "
                         f"one of {sorted(config.KRIGED_SOURCE_DIRS)}")
    if hours is not None:
        x, y, times = dataset.filter_hours(x, y, times, hours)
    tr, va = dataset.fold_split(times, fold, years=years)
    if smoke:
        tr, va = tr[:64], va[:32]

    if degraded:                   # ---- stage B --------------------------- #
        holder = {"v": 0.0}

        class SeverityRamp(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                holder["v"] = min(1.0, epoch / SEVERITY_RAMP_EPOCHS)

        extra_callbacks.append(SeverityRamp())
        train_ds = make_degraded_tf_dataset(
            x[tr], y[tr], n_classes, stats, severity=lambda: holder["v"],
            seed=config.FOLD_SEED, batch_size=batch_size)
        # validation at FULL severity: early stopping judges the end state
        val_ds = make_degraded_tf_dataset(
            x[va], y[va], n_classes, stats, severity=1.0,
            seed=config.FOLD_SEED + 1, batch_size=batch_size, shuffle=False)
    else:                          # ---- stage A / kriged stages ----------- #
        train_ds = dataset.make_tf_dataset(x[tr], y[tr], n_classes, batch_size,
                                           weights=loss_mask)
        val_ds = dataset.make_tf_dataset(x[va], y[va], n_classes, batch_size,
                                         shuffle=False, weights=loss_mask)

    if retrain:
        m = tf.keras.models.load_model(retrain, compile=False)
        n_out = int(m.outputs[0].shape[-1])
        if n_out != n_classes:
            raise ValueError(f"checkpoint {retrain} predicts {n_out} classes "
                             f"but --classes is {n_classes}")
        m.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=model_mod.make_loss(
                      model_mod.class_weight_vector(n_classes)),
                  metrics=[model_mod.masked_accuracy])
    else:
        m = model_mod.build(n_classes, n_channels=x.shape[-1],
                            learning_rate=lr)

    callbacks = extra_callbacks + [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / f"{name}.h5"),
                                           save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv"), append=True),
    ]
    m.fit(train_ds, epochs=(2 if smoke else epochs), validation_data=val_ds,
          callbacks=callbacks, verbose=2)
    m.save(out / f"{name}_final.h5")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", type=int, default=5, choices=(5, 6))
    ap.add_argument("--fold", type=int, default=0)
    # None (not a sentinel VALUE) marks "untouched": an explicit --lr equal
    # to the config default must win over the stage defaults below.
    ap.add_argument("--lr", type=float, default=None,
                    help=f"learning rate (default: {config.LEARNING_RATE}, "
                         "or the stage default for --degraded/kriged sources)")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=None,
                    help=f"early-stopping patience (default: "
                         f"{config.PAPER_PATIENCE}, or FINETUNE_PATIENCE for "
                         "--source kriged-airs)")
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--retrain", default=None,
                    help="checkpoint to continue from (stages B/C)")
    ap.add_argument("--degraded", action="store_true",
                    help="stage B: AIRS-simulator degradation")
    ap.add_argument("--finetune-glob", default=None,
                    help="stage C: glob of AIRS-schema surface files")
    ap.add_argument("--source", default="reanalysis",
                    choices=("reanalysis", "kriged-degraded", "kriged-airs"),
                    help="input fields: clean reanalysis (default) or a "
                         "kriged gap-filled cache (dl_front.krige_fill)")
    ap.add_argument("--hours", default=None,
                    help="comma-separated UTC label hours to train on, "
                         "e.g. '18,21,0' (default: all label hours)")
    ap.add_argument("--channels", default=None,
                    help="comma-separated MODEL input channels, a subset of "
                         f"{','.join(config.SFC_VARS)} (default: the "
                         "'inputs: channels:' list in configs/dl_front.yaml, "
                         "currently "
                         f"{','.join(config.INPUT_CHANNELS)}).  Order is "
                         "irrelevant -- it is normalised to SFC_VARS order. "
                         "Channel-ladder rungs: --channels T2M,QV2M,SLP and "
                         "--channels T2M,QV2M (AIRS-only thermodynamics)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    # BEFORE any data loading or model construction: dataset.sfc_x stacks
    # config.INPUT_CHANNELS and model.build sizes its input from it, so the
    # subset has to be installed first (user decision 2026-08-18).  Going
    # through set_input_channels (not a bare attribute assignment) is what
    # validates the names and fixes the channel order.
    channels = config.INPUT_CHANNELS
    if a.channels is not None:
        channels = config.set_input_channels(a.channels.split(","))
    if a.degraded and a.source != "reanalysis":
        ap.error(f"--degraded applies the stage-B noise+gap degradation on "
                 f"top of the already-kriged '{a.source}' inputs -- no "
                 f"evaluation source matches that; drop one of the two flags")
    hours = (tuple(int(h) for h in a.hours.split(","))
             if a.hours is not None else None)
    if a.lr is None:                      # stage defaults unless overridden
        if a.finetune_glob or a.source == "kriged-airs":
            a.lr = FINETUNE_LR
        elif a.degraded or a.source == "kriged-degraded":
            a.lr = DEGRADED_LR
        else:
            a.lr = config.LEARNING_RATE
    if a.patience is None:
        a.patience = (FINETUNE_PATIENCE   # small fine-tune corpus (stage C)
                      if a.source == "kriged-airs" else config.PAPER_PATIENCE)
    out = run(a.name, a.classes, a.fold, a.lr, a.epochs, a.patience,
              a.retrain, a.batch, a.degraded, a.finetune_glob, a.smoke,
              a.source, hours, channels)
    print(f"model saved under {out}")


if __name__ == "__main__":
    sys.exit(main())
