"""Re-score a dl_front checkpoint on reanalysis under two knobs.

1. ``--labels old|new`` -- swap ``front_finder.config.NOAA_LABELS_DIR`` between
   the shipped (antimeridian-contaminated) label tree and the regenerated one,
   so the label fix's effect on the reported metrics is measured, not guessed.
2. ``--none-scale`` -- multiply the 'none' softmax likelihood by a factor
   before the argmax.  This is the paper's own section-4.2.4 knob (already
   implemented for the ROC sweep in ``evaluate.PaperMetrics``) but the reported
   CSI uses a plain argmax, i.e. factor 1.0.  The confusion matrices show ~50 %
   of true front cells being predicted 'none', so the argmax operating point is
   badly under-forecast; this sweeps for the CSI-optimal factor.

Usage::

    PYTHONPATH=src python scripts/eval_decision_rule.py \\
        --ckpt results/dl_front/models_v2/D6C-f0/D6C-f0.h5 \\
        --labels new --none-scale 1.0 0.5 0.25 0.125
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Pre-fix label backup, relative to the configured data root (the live
#: NOAA_LABELS_DIR resolves under front_finder.config.DATA_ROOT, which honours
#: $JPL_AIRS_DATA; the backup sits beside it).
OLD_LABELS_REL = ("front_id/met_drawn_fronts/NOAA_CODSUS/"
                  "NOAA_1deg_gridded.pre_2026-08-17_datelinebug")


def set_labels(which: str) -> None:
    """Point the loader at the shipped or the regenerated label tree."""
    from front_finder import config as fd_config
    from dl_front import config as dl_config
    if which == "old":
        path = fd_config.DATA_ROOT / OLD_LABELS_REL
        if not path.exists():
            raise SystemExit(f"no pre-fix label backup at {path}")
        fd_config.NOAA_LABELS_DIR = path
        dl_config.NOAA_LABELS_DIR = path
    print(f"labels: {fd_config.NOAA_LABELS_DIR}", flush=True)


def predict_years(model, years, n_classes, stats):
    """Run inference once; the none-scale sweep then reuses the probabilities."""
    from dl_front import dataset, evaluate_test, config

    out = []
    for year in years:
        x, y, times = dataset.filter_hours(
            *evaluate_test.load_year(year, n_classes, stats, "reanalysis"),
            config.AIRS_HOURS)
        probs = np.asarray(model.predict(x.astype(np.float32),
                                         batch_size=128, verbose=0))
        out.append((probs, y, times))
        print(f"  {year}: {len(x)} steps", flush=True)
    return out


def score(predicted, n_classes, none_scale: float) -> pd.DataFrame:
    """CSI/POD/FAR/FB per class with the none likelihood scaled before argmax."""
    from dl_front import dataset, evaluate

    mask = dataset.analysis_domain()
    counts = []
    for probs, y, times in predicted:
        scaled = probs.copy()
        scaled[..., -1] *= none_scale         # 'none' is the last class
        counts.append(evaluate.csi_counts(scaled.argmax(-1), y, times,
                                          n_classes, mask=mask))
    scores = evaluate.csi_scores(pd.concat(counts, ignore_index=True))
    return scores.reset_index()               # (front, dilation) -> columns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", choices=("old", "new"), default="new")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--none-scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    set_labels(a.labels)
    from dl_front import config, dataset, predict

    model = predict.load_model(Path(a.ckpt))
    stats = dataset.load_norm_stats()
    predicted = predict_years(model, config.EVAL_YEARS_6, a.classes, stats)
    frames = []
    for f in a.none_scale:
        print(f"none_scale = {f}", flush=True)
        df = score(predicted, a.classes, f)
        df = df.assign(none_scale=f, labels=a.labels,
                       ckpt=Path(a.ckpt).stem)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(out[np.isclose(out["km"], 111.2)][
            ["front", "none_scale", "csi", "pod", "far", "fb"]].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
