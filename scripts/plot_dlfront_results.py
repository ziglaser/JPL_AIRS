#!/usr/bin/env python3
"""Local, no-TF summary plots for a pulled results/dl_front/ tree.

Reads only pandas/matplotlib-friendly artifacts (CSVs, history logs) --
nothing here touches TensorFlow or the cluster's data root, so it runs
against a plain ``rsync``'d copy of ``results/dl_front/{models,test_eval}``
in any Python env with matplotlib+pandas (the repo's ``.venv`` is fine; the
fronts-tf conda env works too).

Two outputs, written to ``<results-dir>/summary_plots/``:

* ``training_curves.png``  -- loss/val_loss vs epoch, one panel per
  (fold, stage) model dir matching ``D6[ABC]-f<n>``, laid out fold x stage.
* ``csi_comparison_f<fold>.png`` -- per-front CSI with day-block-bootstrap
  CIs (``csi_lo``/``csi_hi`` columns), one panel per class-dilation front,
  bars grouped by leg (checkpoint x input source, plus bk19), for one fold.

Usage::

    python scripts/plot_dlfront_results.py --results-dir results/dl_front \\
        --fold 0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "0.2"
MODEL_DIR_RE = re.compile(r"^(D6[ABC])-f(\d+)$")
LEG_FOLD_RE = re.compile(r"-f(\d+)_")

#: dataviz reference categorical theme (references/palette.md), fixed order
#: -- used here for LEG identity (checkpoint x source), a different
#: categorical dimension from six_panel.py's front-type colors.
LEG_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948")


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #

def _discover_model_dirs(models_dir: Path) -> dict:
    """{(stage, fold): history.csv path} for every D6[ABC]-f<n> model dir
    that has a history.csv (older/unrelated dirs, e.g. smoke runs, are
    silently skipped -- not everything under models/ is part of the
    curriculum grid this plot lays out)."""
    found = {}
    for d in sorted(models_dir.iterdir()) if models_dir.is_dir() else []:
        m = MODEL_DIR_RE.match(d.name)
        hist = d / "history.csv"
        if m and hist.exists():
            found[(m.group(1), int(m.group(2)))] = hist
    return found


def plot_training_curves(results_dir: Path, out_dir: Path) -> Path | None:
    found = _discover_model_dirs(results_dir / "models")
    if not found:
        print(f"no D6[ABC]-f<n> model dirs with history.csv under "
              f"{results_dir / 'models'}; skipping training_curves.png")
        return None

    stages = sorted({s for s, _ in found})
    folds = sorted({f for _, f in found})
    fig, axes = plt.subplots(len(folds), len(stages),
                             figsize=(4.2 * len(stages), 3.2 * len(folds)),
                             squeeze=False, sharex=False)
    for i, fold in enumerate(folds):
        for j, stage in enumerate(stages):
            ax = axes[i][j]
            path = found.get((stage, fold))
            if path is None:
                ax.axis("off")
                continue
            hist = pd.read_csv(path)
            epoch = hist["epoch"] if "epoch" in hist else np.arange(len(hist))
            ax.plot(epoch, hist["loss"], color=LEG_PALETTE[0], lw=1.4,
                   label="train")
            if "val_loss" in hist:
                ax.plot(epoch, hist["val_loss"], color=LEG_PALETTE[7],
                       lw=1.4, label="val")
            ax.set_title(f"{stage}-f{fold}", color=INK, fontsize=9)
            ax.tick_params(labelsize=7, colors=INK)
            for spine in ax.spines.values():
                spine.set_color("0.7")
            if i == len(folds) - 1:
                ax.set_xlabel("epoch", color=INK, fontsize=8)
            if j == 0:
                ax.set_ylabel("loss", color=INK, fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle("dl_front training curves (loss / val_loss)", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# CSI comparison with bootstrap CIs
# --------------------------------------------------------------------------- #

def _leg_fold(leg: str) -> int | None:
    m = LEG_FOLD_RE.search(leg)
    return int(m.group(1)) if m else None


def _load_legs(test_eval_dir: Path, fold: int) -> dict:
    """{leg_name: DataFrame} for every leg CSV matching ``fold`` (bk19 has
    no fold and is included in every view)."""
    legs = {}
    for path in sorted(test_eval_dir.glob("*.csv")):
        if path.stem in ("comparison", "comparison_MISMATCHED_SAMPLE"):
            continue
        df = pd.read_csv(path)
        if not {"front", "km", "csi"} <= set(df.columns):
            continue
        leg_fold = _leg_fold(path.stem)
        if leg_fold is not None and leg_fold != fold:
            continue
        legs[path.stem] = df
    return legs


def plot_csi_comparison(results_dir: Path, out_dir: Path,
                        fold: int) -> Path | None:
    test_eval_dir = results_dir / "test_eval"
    mismatched = (test_eval_dir / "comparison_MISMATCHED_SAMPLE.csv").exists()
    legs = _load_legs(test_eval_dir, fold)
    if not legs:
        print(f"no leg CSVs for fold {fold} under {test_eval_dir}; "
              f"skipping csi_comparison_f{fold}.png")
        return None
    if len(legs) > len(LEG_PALETTE):
        print(f"WARNING: {len(legs)} legs but only {len(LEG_PALETTE)} "
              f"validated palette slots -- some legs will repeat a color")

    fronts = sorted({f for df in legs.values() for f in df["front"].unique()
                    if f != "none"})
    leg_names = sorted(legs)                  # fixed, deterministic order
    colors = {name: LEG_PALETTE[i % len(LEG_PALETTE)]
             for i, name in enumerate(leg_names)}

    fig, axes = plt.subplots(1, len(fronts), figsize=(3.2 * len(fronts), 4.2),
                             squeeze=False, sharey=True)
    width = 0.8 / max(len(leg_names), 1)
    for j, front in enumerate(fronts):
        ax = axes[0][j]
        dils = None
        for i, name in enumerate(leg_names):
            sub = legs[name][legs[name]["front"] == front].sort_values("km")
            if dils is None:
                dils = sub["km"].tolist()
            x = np.arange(len(sub)) + (i - (len(leg_names) - 1) / 2) * width
            lo = (sub["csi"] - sub.get("csi_lo", sub["csi"])).clip(lower=0)
            hi = (sub.get("csi_hi", sub["csi"]) - sub["csi"]).clip(lower=0)
            ax.bar(x, sub["csi"], width=width * 0.9, color=colors[name],
                  label=name, yerr=[lo, hi], capsize=2,
                  error_kw={"elinewidth": 0.8, "ecolor": "0.3"})
        ax.set_xticks(np.arange(len(dils)))
        ax.set_xticklabels([f"{d:g}" for d in dils], fontsize=8, color=INK)
        ax.set_title(front, color=INK, fontsize=10)
        ax.set_xlabel("neighborhood (km)", color=INK, fontsize=8)
        ax.tick_params(colors=INK, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("0.7")
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("CSI (bootstrap 95% CI)", color=INK, fontsize=9)
    fig.legend(leg_names, loc="lower center", ncol=min(len(leg_names), 4),
              frameon=False, fontsize=8, labelcolor=INK,
              bbox_to_anchor=(0.5, -0.05 - 0.04 * (len(leg_names) > 4)))
    title = f"CSI comparison, fold {fold}"
    if mismatched:
        title += "  [WARNING: legs did not score identical samples]"
    fig.suptitle(title, color=("crimson" if mismatched else INK), fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"csi_comparison_f{fold}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/dl_front")
    ap.add_argument("--fold", type=int, default=0,
                    help="fold to plot the CSI comparison for (default 0)")
    ap.add_argument("--all-folds", action="store_true",
                    help="also plot csi_comparison for folds 1 and 2")
    a = ap.parse_args(argv)

    results_dir = Path(a.results_dir)
    out_dir = results_dir / "summary_plots"

    written = [plot_training_curves(results_dir, out_dir)]
    folds = [a.fold] + ([1, 2] if a.all_folds and a.fold == 0 else [])
    for fold in dict.fromkeys(folds):           # dedupe, keep order
        written.append(plot_csi_comparison(results_dir, out_dir, fold))

    for path in filter(None, written):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
