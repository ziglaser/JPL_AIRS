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


def make_degraded_tf_dataset(x, y, n_classes, stats, severity, seed,
                             batch_size=config.BATCH_SIZE, shuffle=True):
    """Stage-B pipeline: noise + real gap masks resampled every pass.

    ``severity`` is a float or zero-arg callable (ramp callback support).
    Loss weights: analysis_domain() for 6-class, region_mask() for 5-class
    (loss_mask_for; user decision 2026-08-13).
    """
    import tensorflow as tf

    from front_finder import mask_bank

    bank_vf, bank_dates = mask_bank.load_bank()
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
            vf = degrade_sfc.surface_gap_field(bank_vf, r)
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
        source: str = "reanalysis", hours: tuple | None = None) -> Path:
    import tensorflow as tf

    out = config.RESULTS_DIR / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    # Provenance: freeze the run's resolved tunables + call arguments next to
    # the weights, so a checkpoint is reproducible even after the tracked
    # YAML (or a JPL_DLFRONT_CONFIG override) changes.
    import yaml as _yaml

    mask_name, loss_mask = loss_mask_for(n_classes, source, degraded)
    (out / "run_config.yaml").write_text(_yaml.safe_dump(
        {"tunables_from": str(config.CONFIG_YAML),
         # which per-pixel loss-weight mask this run trained under
         # (user decision 2026-08-13; see loss_mask_for)
         "loss_mask": mask_name,
         "tunables": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in config.load_tunables().items()},
         "run_args": {"name": name, "n_classes": n_classes, "fold": fold,
                      "lr": lr, "epochs": epochs, "patience": patience,
                      "retrain": retrain, "batch_size": batch_size,
                      "degraded": degraded, "finetune_glob": finetune_glob,
                      "smoke": smoke, "source": source,
                      "hours": list(hours) if hours is not None else None}},
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

    years = train_years(n_classes)
    if smoke:
        years = years[:1]
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
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
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
              a.source, hours)
    print(f"model saved under {out}")


if __name__ == "__main__":
    sys.exit(main())
