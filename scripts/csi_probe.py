"""Score neighborhood CSI of the live D6 checkpoint on fold-0 validation data.

Runs on CPU so it can execute alongside GPU training:

  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python3 scripts/csi_probe.py \
      --name D6-fold0 --classes 6 --n-samples 1500

fold_split is a seeded permutation, so the validation indices reconstructed
here are exactly the ones train.py holds out.  A random subsample of them is
enough to read the CSI trend between epochs.  Each probe appends one row per
(class, scale) to <model dir>/csi_probe_log.csv, tagged with the checkpoint
mtime and the epoch count from history.csv at probe time.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd

from dl_front import config, dataset, evaluate
from dl_front.train import train_years


def gather_val_samples(years, n_classes, n_samples, fold, seed=0):
    """Load only the fold-`fold` validation samples, year by year."""
    stats = dataset.load_norm_stats()
    lengths = []
    per_year = []          # (x, y, times) kept only long enough to slice
    for yr in years:
        x, y, times = dataset.year_arrays(yr, n_classes, stats)
        lengths.append(len(x))
        per_year.append((x, y, times))
    n = sum(lengths)
    _, va = dataset.fold_split(n, fold)
    rng = np.random.default_rng(seed)
    if n_samples and n_samples < len(va):
        va = rng.choice(va, size=n_samples, replace=False)
    va = np.sort(va)

    starts = np.cumsum([0] + lengths[:-1])
    xs, ys, ts = [], [], []
    for (x, y, times), start, length in zip(per_year, starts, lengths):
        local = va[(va >= start) & (va < start + length)] - start
        if len(local):
            xs.append(x[local])
            ys.append(y[local])
            ts.append(times[local])
    return (np.concatenate(xs), np.concatenate(ys),
            ts[0].append(ts[1:]), len(va))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="D6-fold0")
    ap.add_argument("--classes", type=int, default=6, choices=(5, 6))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    out = config.RESULTS_DIR / "models" / args.name
    ckpt = out / f"{args.name}.h5"
    ckpt_mtime = pd.Timestamp(ckpt.stat().st_mtime, unit="s").round("s")
    hist = out / "history.csv"
    epoch = (len(pd.read_csv(hist)) if hist.exists() else -1)

    x, y_cls, times, n_val = gather_val_samples(
        train_years(args.classes), args.classes, args.n_samples, args.fold)
    print(f"scoring {len(x)}/{n_val} val samples against {ckpt.name} "
          f"(epoch ~{epoch}, saved {ckpt_mtime})")

    import tensorflow as tf
    m = tf.keras.models.load_model(ckpt, compile=False)
    probs = m.predict(x.astype(np.float32), batch_size=args.batch, verbose=0)
    pred_cls = probs.argmax(-1)

    # scoring mask follows the track (user decision 2026-08-13): 6-class
    # CSI over the analysis domain -- the same mask evaluate_test scores,
    # so the probe trend is comparable to the test metric -- 5-class over
    # the Fig. 2 region mask
    mask = (dataset.analysis_domain() if args.classes == 6
            else dataset.region_mask().astype(bool))
    scores = evaluate.csi_scores(
        evaluate.csi_counts(pred_cls, y_cls, times, args.classes, mask=mask))
    scores["epoch"] = epoch
    scores["ckpt_mtime"] = ckpt_mtime
    scores["n_samples"] = len(x)

    log = out / "csi_probe_log.csv"
    scores.to_csv(log, mode="a", header=not log.exists(),
                  index=isinstance(scores.index, pd.MultiIndex))
    pd.set_option("display.width", 140)
    print(scores.round(3).to_string())
    print(f"\nappended to {log}")


if __name__ == "__main__":
    main()
