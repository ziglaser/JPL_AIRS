"""Stage-aware training driver (workplan section 3.6).

Stages: A = clean reanalysis pretrain (this file, available now);
B = degraded reanalysis (needs degrade.py); C = real-AIRS fine-tune (needs
ingest).  ``--retrain FROM`` continues from a stage-A/B checkpoint with a new
learning rate -- the reason this driver exists instead of fronts/train_unet.py.

CLI:
  PYTHONPATH=src python -m front_finder.train --name A-thermo --no-winds
  PYTHONPATH=src python -m front_finder.train --name A-wind --winds
  JPL_FRONT_LABELS=noaa PYTHONPATH=src python -m front_finder.train \
      --name D-thermo --no-winds       # NOAA XML labels incl. the DRYLINE class
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, dataset, model as model_mod

#: Paper section 2c pairing, kept EXACTLY (2026-08-10): batch 64 @ Adam
#: 1e-4, 10 steps/epoch = 640 samples/"epoch".  Sized for one A100-80GB;
#: the earlier local batch-4 runs at the same LR were effectively a 4x
#: hotter per-sample step with noisy batch-4 BatchNorm statistics.  Small
#: GPUs: pass --batch 4 --steps 160 AND drop --lr to ~2.5e-5.
BATCH_SIZE = 64
STEPS_PER_EPOCH = 10
PATIENCE = 55                     # ~= one full stage-A pass without improvement
MAX_EPOCHS = 700                  # paper stopped at 699


#: Stage C (workplan 3.6): all layers unfrozen at a low LR, BN adapting,
#: short patience (small data).
FINETUNE_LR = 2e-5
FINETUNE_PATIENCE = 15
#: Stage B (workplan 3.6-B): continue from A at half LR, severity ramped
#: linearly over the first epochs to avoid a loss cliff.  41 epochs = 3
#: round passes of the overpass-hours stage-B corpus (degrade.py).
DEGRADED_LR = 5e-5
SEVERITY_RAMP_EPOCHS = 41


def _severity_ramp_callback(holder: dict, ramp_epochs: int):
    """Keras callback moving holder['v'] 0 -> 1; the stage-B generator reads
    it per day, so the ramp lands with sub-epoch lag (documented, harmless)."""
    import tensorflow as tf

    class SeverityRamp(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            holder["v"] = min(1.0, epoch / ramp_epochs)

    return SeverityRamp()


def run(name: str, winds: bool, lr: float, epochs: int, retrain: str | None,
        batch_size: int = BATCH_SIZE, steps: int = STEPS_PER_EPOCH,
        smoke: bool = False, airs_glob: str | None = None,
        degraded: bool = False, val_fraction: float = 0.15,
        filter_num=None) -> Path:
    import tensorflow as tf

    out = config.RESULTS_DIR / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"labels: {config.LABEL_SOURCE} -> classes {dataset.CLASS_NAMES}",
          flush=True)
    stats = dataset.load_norm_stats()
    extra_callbacks = []

    if degraded:                   # ---- stage B: AIRS-simulator pretrain ----
        holder = {"v": 0.0}
        extra_callbacks.append(
            _severity_ramp_callback(holder, SEVERITY_RAMP_EPOCHS))
        train_years = config.PRETRAIN_TRAIN_YEARS
        if smoke:
            train_years = (2015,)
        train = dataset.make_degraded_tf_dataset(
            train_years, winds, batch_size, severity=lambda: holder["v"],
            stats=stats, augment=True)
        # validation at FULL severity throughout (early stopping must judge
        # the end state, not the ramp) and with a FROZEN noise/gap
        # realization (identical every epoch, so val_loss moves only when
        # the model does); no augmentation on validation
        val = dataset.make_degraded_tf_dataset(
            (config.PRETRAIN_VAL_YEAR,) if not smoke else (2015,), winds,
            batch_size, severity=1.0, seed=config.BOOT_SEED + 1,
            stats=stats, shuffle=False, freeze_realization=True)
    elif airs_glob:                # ---- stage C: fine-tune on AIRS files ----
        import glob

        paths = sorted(glob.glob(airs_glob))
        if not paths:
            raise FileNotFoundError(f"no fullgrid files match {airs_glob}")
        # year-ordered tail as validation; year-based splits once the
        # archive spans multiple years (workplan 3.6)
        n_val = max(1, int(len(paths) * val_fraction))
        train = dataset.make_airs_tf_dataset(paths[:-n_val], winds,
                                             batch_size, stats=stats)
        val = dataset.make_airs_tf_dataset(paths[-n_val:], winds, batch_size,
                                           shuffle=False, stats=stats)
    else:                          # ---- stages A/B: reanalysis corpus ------
        train_years = config.PRETRAIN_TRAIN_YEARS
        val_years = (config.PRETRAIN_VAL_YEAR,)
        if smoke:                  # tiny end-to-end check: one year each
            train_years, val_years = (2015,), (2015,)
        if dataset.shards_exist((*train_years, *val_years)):
            train = dataset.make_shard_tf_dataset(train_years, winds,
                                                  batch_size, augment=True)
            val = dataset.make_shard_tf_dataset(val_years, winds, batch_size,
                                                shuffle=False)
        else:
            # streaming fallback; never file-cache (post-mortem 2026-08-09:
            # the cache writer's memory leak OOM-thrashed the first E1b run)
            print("shards missing -- streaming netCDF directly "
                  "(run front_finder.materialize for fast epochs)")
            train = dataset.make_tf_dataset(train_years, winds, batch_size,
                                            stats=stats, cache=False,
                                            augment=True)
            val = dataset.make_tf_dataset(val_years, winds, batch_size,
                                          shuffle=False, stats=stats,
                                          cache=False)

    if retrain:
        m = tf.keras.models.load_model(retrain, compile=False)
        n_out = int(m.outputs[0].shape[-1])
        if n_out != len(dataset.CLASS_NAMES):
            raise ValueError(
                f"checkpoint {retrain} predicts {n_out} classes but label "
                f"source {config.LABEL_SOURCE!r} has {len(dataset.CLASS_NAMES)} "
                f"({dataset.CLASS_NAMES}) -- a CODSUS-pretrained model cannot "
                f"be retrained on dryline labels without a new head")
        m.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=model_mod.make_loss())
    else:
        m = model_mod.build(winds, learning_rate=lr, filter_num=filter_num)

    patience = FINETUNE_PATIENCE if airs_glob else PATIENCE
    callbacks = extra_callbacks + [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / f"{name}.h5"),
                                           save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv"), append=True),
    ]
    m.fit(train.repeat(), epochs=(2 if smoke else epochs),
          steps_per_epoch=(4 if smoke else steps),
          validation_data=val, validation_steps=(4 if smoke else None),
          callbacks=callbacks, verbose=2)
    m.save(out / f"{name}_final.h5")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--winds", dest="winds", action="store_true")
    ap.add_argument("--no-winds", dest="winds", action="store_false")
    ap.add_argument("--lr", type=float, default=1e-4)   # paper: Adam 1e-4
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--steps", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--retrain", default=None,
                    help="checkpoint path to continue from (stages B/C)")
    ap.add_argument("--airs-glob", default=None,
                    help="stage C: glob of fullgrid_*.nc fine-tune files "
                         "(use with --retrain <stage A/B ckpt> --lr 2e-5)")
    ap.add_argument("--degraded", action="store_true",
                    help="stage B: degraded-reanalysis pretrain "
                         "(use with --retrain <stage A ckpt>)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs x 4 steps on 2015 only")
    ap.add_argument("--filter-num", default=None,
                    help="comma-separated U-Net widths, one per level "
                         f"(default {model_mod.FILTER_NUM}, sized for one "
                         "A100-80GB at batch 64; length must equal "
                         f"model.LEVELS={model_mod.LEVELS})")
    ap.set_defaults(winds=False)
    a = ap.parse_args(argv)
    if a.lr == 1e-4:               # stage defaults unless overridden
        if a.airs_glob:
            a.lr = FINETUNE_LR
        elif a.degraded:
            a.lr = DEGRADED_LR
    fnum = [int(v) for v in a.filter_num.split(",")] if a.filter_num else None
    out = run(a.name, a.winds, a.lr, a.epochs, a.retrain, a.batch, a.steps,
              smoke=a.smoke, airs_glob=a.airs_glob, degraded=a.degraded,
              filter_num=fnum)
    print(f"model saved under {out}")


if __name__ == "__main__":
    sys.exit(main())
