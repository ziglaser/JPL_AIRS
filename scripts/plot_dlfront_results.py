#!/usr/bin/env python3
"""Local, no-TF summary plots for a pulled results/dl_front/ tree.

Reads only pandas/matplotlib-friendly artifacts (CSVs, history logs, small
jsons) -- nothing here touches TensorFlow or the cluster's data root, so it
runs against a plain ``rsync``'d copy of ``results/dl_front/`` in any Python
env with matplotlib+pandas (the repo's ``.venv`` is fine; the fronts-tf
conda env works too).  Every figure auto-skips with a printed note when its
input directory is empty or missing: a partially pulled results tree must
never crash the script.

Outputs, written to ``<results-dir>/summary_plots/``:

* ``training_curves.png``       loss/val_loss vs epoch, one panel per
                                (fold, stage) model dir matching
                                ``D6[ABC]-f<n>``, laid out fold x stage.
* ``csi_comparison.png``        FOLD-POOLED metric grid: rows CSI/POD/FAR x
                                one column per front type (bars grouped by
                                pooled leg, CI whiskers on the CSI row),
                                plus one panel of all-categories accuracy
                                from the ``*_paper.json`` files.
* ``csi_comparison_f<k>.png``   the pre-pooling per-fold figure, only under
                                ``--per-fold`` (debugging aid).
* ``permutation_importance.png``  CSI cost (baseline - shuffled) per input
                                channel from ``permutation/*.csv``, bars
                                grouped by pooled checkpoint, one panel per
                                source.
* ``ablation_ladder.png``       CSI vs channel-ladder rung (5ch -> 3ch ->
                                2ch) from ``ablation_eval/*.csv``, bars
                                grouped by source with pooled CI whiskers,
                                one panel per front type.

FOLD POOLING RULE (user decision, shared with evaluate_test.compare): a leg
named ``<stage>-f<k>_<source>`` pools with its sibling folds under the name
with the ``-f<k>`` stripped (``D6C-f0_kriged-airs`` -> ``D6C_kriged-airs``).
The pool is the UNWEIGHTED mean across folds -- exact-weight because every
fold's leg scores the IDENTICAL time steps (the same-sample guarantee
``compare()`` enforces via ``times_sha1``), so each fold contributes the
same denominator.  The pooled CI is the mean of the fold ``csi_lo`` /
``csi_hi`` bounds, an APPROXIMATION of the pooled sampling CI (averaging
bounds ignores the across-fold spread of the point estimates; good enough
for whiskers, not for a significance test).  ``bk19`` has no fold and
passes through unchanged.  This script pools from the PER-LEG CSVs itself
rather than reading ``comparison.csv``, so it works on a results dir where
only some legs have been pulled.

Usage::

    python scripts/plot_dlfront_results.py --results-dir results/dl_front
    python scripts/plot_dlfront_results.py --selftest   # synthetic smoke
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "0.2"
MODEL_DIR_RE = re.compile(r"^(D6[ABC]\d*)-f(\d+)$")
LEG_FOLD_RE = re.compile(r"-f(\d+)_")
#: ``-f<k>`` fold tag inside a leg/checkpoint stem (before the ``_<source>``
#: suffix or at the end of a bare checkpoint name).  Stripping it yields the
#: pooled leg name per the FOLD POOLING RULE above.
FOLD_TAG_RE = re.compile(r"-f\d+(?=_|$)")
#: Ladder checkpoint names encode their rung (input channel count) as the
#: trailing digits: D6A5 = 5ch, D6A3 = 3ch, D6A2 = 2ch (dlfront_ablation_
#: chain.sh's CHANNEL_SETS).
RUNG_RE = re.compile(r"^D6A(\d+)$")
#: Sentinel channel name on the un-permuted reference rows of a permutation
#: CSV (dl_front.permutation.BASELINE -- not imported: that module pulls in
#: the TF-adjacent dl_front package and this script must stay plain-python).
PERM_BASELINE = "<baseline>"

#: dataviz reference categorical theme (references/palette.md), fixed order
#: -- used here for LEG identity (checkpoint x source), a different
#: categorical dimension from six_panel.py's front-type colors.
LEG_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def _style_ax(ax) -> None:
    """The file's shared recessive-axes look (gray spines, ink ticks)."""
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("0.7")


def pooled_name(stem: str) -> str:
    """``D6C-f0_kriged-airs`` -> ``D6C_kriged-airs``; bk19 etc. unchanged."""
    return FOLD_TAG_RE.sub("", stem)


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #

def _discover_model_dirs(models_dir: Path) -> dict:
    """{(stage, fold): history.csv path} for every D6[ABC]*-f<n> model dir
    that has a history.csv (older/unrelated dirs, e.g. smoke runs, are
    silently skipped -- not everything under models/ is part of the
    curriculum grid this plot lays out).  Ladder stages (D6A5/D6A3/D6A2)
    match too and land in their own columns."""
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
        print(f"no D6[ABC]*-f<n> model dirs with history.csv under "
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
# Fold pooling over per-leg eval CSVs (shared by csi_comparison + ablation)
# --------------------------------------------------------------------------- #

#: Metric columns of a per-leg eval CSV that pool as an unweighted mean.
#: csi_lo/csi_hi pool the same way (mean of fold bounds) -- the documented
#: approximation of the pooled sampling CI, see the module docstring.
_POOL_COLS = ("csi", "csi_lo", "csi_hi", "pod", "far", "fb")


def _pool_eval_frames(frames: list) -> pd.DataFrame:
    """Unweighted mean over fold frames, per (front, dilation).

    Exact-weight because every fold's leg scores the identical time steps
    (same-sample guarantee); ``km`` is a label, not data, so it is carried
    by first() rather than averaged.  bk19 arrives as a single frame and
    passes through this unchanged (a mean of one).  Older CSVs without
    csi_lo/csi_hi (pre-CI bk19) simply have no CI columns in the pool.
    """
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in _POOL_COLS if c in df.columns]
    return (df.groupby(["front", "dilation"], as_index=False)
              .agg({"km": "first", **{c: "mean" for c in cols}}))


def _load_pooled_legs(test_eval_dir: Path) -> tuple:
    """(legs, accuracy): fold-pooled per-leg metric frames + accuracies.

    ``legs`` is {pooled leg name: pooled DataFrame}; ``accuracy`` is
    {pooled leg name: mean all-categories accuracy or None} -- None marks a
    leg with no ``*_paper.json`` (bk19: hard binary predictions make the
    paper's ROC sweep meaningless, so evaluate_test never writes one), which
    the accuracy panel shows as ABSENT, not zero.
    """
    groups: dict = {}
    accs: dict = {}
    for path in sorted(test_eval_dir.glob("*.csv")):
        if path.stem.startswith("comparison"):
            continue
        df = pd.read_csv(path)
        if not {"front", "km", "csi"} <= set(df.columns):
            continue                       # not a per-leg eval CSV
        name = pooled_name(path.stem)
        groups.setdefault(name, []).append(df)
        paper = path.with_name(path.stem + "_paper.json")
        if paper.exists():
            acc = json.loads(paper.read_text())["accuracy"]["all_categories"]
            accs.setdefault(name, []).append(float(acc))
    legs = {name: _pool_eval_frames(frames) for name, frames in groups.items()}
    accuracy = {name: (float(np.mean(accs[name])) if name in accs else None)
               for name in legs}
    return legs, accuracy


# --------------------------------------------------------------------------- #
# CSI/POD/FAR + accuracy comparison (fold-pooled)
# --------------------------------------------------------------------------- #

def plot_csi_comparison(results_dir: Path, out_dir: Path) -> Path | None:
    """Metric-row x front-column grid over the FOLD-POOLED legs.

    Rows CSI / POD / FAR, one column per front type, bars grouped by pooled
    leg.  CI whiskers on the CSI row only -- POD/FAR carry no CIs in the
    per-leg CSVs.  The rightmost column is a single extra panel:
    all-categories accuracy per pooled leg from the ``*_paper.json`` files
    (bk19 has none and is shown as absent).
    """
    test_eval_dir = results_dir / "test_eval"
    mismatched = (test_eval_dir / "comparison_MISMATCHED_SAMPLE.csv").exists()
    legs, accuracy = _load_pooled_legs(test_eval_dir)
    if not legs:
        print(f"no per-leg eval CSVs under {test_eval_dir}; "
              f"skipping csi_comparison.png")
        return None
    if len(legs) > len(LEG_PALETTE):
        print(f"WARNING: {len(legs)} pooled legs but only {len(LEG_PALETTE)} "
              f"validated palette slots -- some legs will repeat a color")

    fronts = sorted({f for df in legs.values() for f in df["front"].unique()
                    if f != "none"})
    leg_names = sorted(legs)                  # fixed, deterministic order
    colors = {name: LEG_PALETTE[i % len(LEG_PALETTE)]
             for i, name in enumerate(leg_names)}
    metrics = ("csi", "pod", "far")

    # 3 metric rows x (front columns + 1 accuracy column).  The accuracy
    # panel only occupies the top cell of its column; the two below are off.
    ncols = len(fronts) + 1
    fig, axes = plt.subplots(len(metrics), ncols,
                             figsize=(3.0 * ncols, 8.6), squeeze=False)
    width = 0.8 / max(len(leg_names), 1)
    for r, metric in enumerate(metrics):
        for j, front in enumerate(fronts):
            ax = axes[r][j]
            dils = None
            for i, name in enumerate(leg_names):
                sub = legs[name][legs[name]["front"] == front] \
                    .sort_values("km")
                if dils is None:
                    dils = sub["km"].tolist()
                x = (np.arange(len(sub))
                     + (i - (len(leg_names) - 1) / 2) * width)
                if metric == "csi":         # only row with CIs in the CSVs
                    lo = (sub["csi"] - sub.get("csi_lo", sub["csi"])) \
                        .clip(lower=0)
                    hi = (sub.get("csi_hi", sub["csi"]) - sub["csi"]) \
                        .clip(lower=0)
                    err = {"yerr": [lo, hi], "capsize": 2,
                           "error_kw": {"elinewidth": 0.8, "ecolor": "0.3"}}
                else:
                    err = {}
                ax.bar(x, sub[metric], width=width * 0.9,
                      color=colors[name], label=name, **err)
            ax.set_xticks(np.arange(len(dils)))
            ax.set_xticklabels([f"{d:g}" for d in dils], fontsize=8,
                              color=INK)
            _style_ax(ax)
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(front, color=INK, fontsize=10)
            if r == len(metrics) - 1:
                ax.set_xlabel("neighborhood (km)", color=INK, fontsize=8)
            if j == 0:
                label = {"csi": "CSI (pooled, mean-of-fold 95% CI)",
                         "pod": "POD", "far": "FAR"}[metric]
                ax.set_ylabel(label, color=INK, fontsize=9)

    # Accuracy panel (top-right); the rest of that column stays empty.
    ax = axes[0][ncols - 1]
    for r in range(1, len(metrics)):
        axes[r][ncols - 1].axis("off")
    for i, name in enumerate(leg_names):
        if accuracy[name] is None:          # absent (bk19), NOT zero
            ax.text(i, 0.02, "n/a", ha="center", va="bottom",
                   fontsize=8, color="0.45")
        else:
            ax.bar(i, accuracy[name], width=0.72, color=colors[name])
    ax.set_xlim(-0.6, len(leg_names) - 0.4)
    ax.set_xticks(np.arange(len(leg_names)))
    # Named ticks (not just the legend): an absent bar must be attributable
    # to its leg, and "n/a" alone carries no identity.
    ax.set_xticklabels(leg_names, fontsize=6, rotation=30, ha="right")
    _style_ax(ax)
    ax.set_ylim(0, 1)
    ax.set_title("accuracy (all categories)", color=INK, fontsize=10)

    fig.legend(*axes[0][0].get_legend_handles_labels(),
              loc="lower center", ncol=min(len(leg_names), 4),
              frameon=False, fontsize=8, labelcolor=INK,
              bbox_to_anchor=(0.5, -0.02 - 0.03 * (len(leg_names) > 4)))
    title = "fold-pooled CSI / POD / FAR + accuracy by leg"
    if mismatched:
        title += "  [WARNING: legs did not score identical samples]"
    fig.suptitle(title, color=("crimson" if mismatched else INK), fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "csi_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Per-fold CSI comparison (pre-pooling behavior, kept under --per-fold)
# --------------------------------------------------------------------------- #

def _leg_fold(leg: str) -> int | None:
    m = LEG_FOLD_RE.search(leg)
    return int(m.group(1)) if m else None


def _load_legs(test_eval_dir: Path, fold: int) -> dict:
    """{leg_name: DataFrame} for every leg CSV matching ``fold`` (bk19 has
    no fold and is included in every view)."""
    legs = {}
    for path in sorted(test_eval_dir.glob("*.csv")):
        if path.stem.startswith("comparison"):
            continue
        df = pd.read_csv(path)
        if not {"front", "km", "csi"} <= set(df.columns):
            continue
        leg_fold = _leg_fold(path.stem)
        if leg_fold is not None and leg_fold != fold:
            continue
        legs[path.stem] = df
    return legs


def plot_csi_comparison_fold(results_dir: Path, out_dir: Path,
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
        _style_ax(ax)
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


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #

def _load_permutation_costs(perm_dir: Path) -> pd.DataFrame | None:
    """Tidy (source, ckpt, channel, csi_delta) frame, fold-pooled.

    Column contract pinned to ``dl_front.permutation.CSV_COLUMNS``
    (channel, repeat, stat, front, dilation, km, csi, csi_lo, csi_hi, pod,
    far, fb, csi_delta, pod_delta); ``csi_delta`` is already
    baseline - shuffled, i.e. the CSI COST, positive = channel matters.

    Reduction per file: the "mean"-stat aggregate rows when the run had
    repeats > 1, else the single "raw" rows; the COARSEST dilation (the
    number the module's own headline and the ablation decision are made
    on); mean over front types (per-front deltas share a currency -- CSI --
    so the unweighted mean is the natural one-bar summary; per-front detail
    stays in the CSVs).  Then folds pool by unweighted mean per the
    module-level rule.  Channel order is order of first appearance, i.e.
    the model's config.INPUT_CHANNELS order as written by the CSVs.
    """
    rows, channel_order = [], []
    for path in sorted(perm_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if not {"channel", "stat", "dilation", "csi_delta"} <= set(df.columns):
            continue                        # not a permutation table
        ckpt, _, source = path.stem.rpartition("_")
        if not ckpt:
            continue
        sub = df[df["channel"] != PERM_BASELINE]
        stat = "mean" if (sub["stat"] == "mean").any() else "raw"
        sub = sub[(sub["stat"] == stat)
                  & (sub["dilation"] == sub["dilation"].max())
                  & (sub["front"] != "none")]
        for ch in sub["channel"]:
            if ch not in channel_order:
                channel_order.append(ch)
        cost = sub.groupby("channel", as_index=False)["csi_delta"].mean()
        rows.append(cost.assign(ckpt=pooled_name(ckpt), source=source))
    if not rows:
        return None
    tidy = (pd.concat(rows, ignore_index=True)
            .groupby(["source", "ckpt", "channel"], as_index=False)
            ["csi_delta"].mean())           # fold pooling
    tidy["channel"] = pd.Categorical(tidy["channel"], channel_order,
                                     ordered=True)
    return tidy


def plot_permutation_importance(results_dir: Path,
                                out_dir: Path) -> Path | None:
    perm_dir = results_dir / "permutation"
    tidy = _load_permutation_costs(perm_dir) if perm_dir.is_dir() else None
    if tidy is None:
        print(f"no permutation CSVs under {perm_dir}; "
              f"skipping permutation_importance.png")
        return None

    sources = sorted(tidy["source"].unique())
    # Fixed color per pooled checkpoint ACROSS panels (color follows the
    # entity): ladder rungs (D6A5/D6A3/D6A2) get slots automatically.
    ckpts = sorted(tidy["ckpt"].unique())
    colors = {c: LEG_PALETTE[i % len(LEG_PALETTE)]
             for i, c in enumerate(ckpts)}

    fig, axes = plt.subplots(1, len(sources),
                             figsize=(4.6 * len(sources), 4.2),
                             squeeze=False, sharey=True)
    for j, source in enumerate(sources):
        ax = axes[0][j]
        panel = tidy[tidy["source"] == source]
        chans = [c for c in tidy["channel"].cat.categories
                if c in set(panel["channel"].astype(str))]
        present = [c for c in ckpts if (panel["ckpt"] == c).any()]
        width = 0.8 / max(len(present), 1)
        for i, ckpt in enumerate(present):
            sub = panel[panel["ckpt"] == ckpt].set_index(
                panel[panel["ckpt"] == ckpt]["channel"].astype(str))
            # A ladder rung consumes a channel SUBSET; absent channels are
            # simply not drawn (no zero-height fakes).
            xs = [k for k, c in enumerate(chans) if c in sub.index]
            vals = [sub.loc[c, "csi_delta"] for c in chans if c in sub.index]
            x = np.array(xs) + (i - (len(present) - 1) / 2) * width
            ax.bar(x, vals, width=width * 0.9, color=colors[ckpt],
                  label=ckpt)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(np.arange(len(chans)))
        ax.set_xticklabels(chans, fontsize=8, color=INK)
        ax.set_title(source, color=INK, fontsize=10)
        _style_ax(ax)
        if j == 0:
            ax.set_ylabel("CSI cost (baseline - shuffled)",
                         color=INK, fontsize=9)
    # Handles built from the full ckpt->color map, not one panel's bars: a
    # checkpoint probed only under one source must still appear in the
    # legend (e.g. ladder rungs run reanalysis-only).
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in ckpts]
    fig.legend(handles, ckpts,
              loc="lower center", ncol=min(len(ckpts), 4),
              frameon=False, fontsize=8, labelcolor=INK,
              bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("permutation importance (fold-pooled; coarsest "
                 "neighborhood, mean over front types)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "permutation_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Channel-ablation ladder
# --------------------------------------------------------------------------- #

def plot_ablation_ladder(results_dir: Path, out_dir: Path) -> Path | None:
    """CSI vs ladder rung (5ch -> 3ch -> 2ch), pooled over folds.

    ``ablation_eval/*.csv`` share the per-leg eval schema; the stem is
    ``<D6A<n>-f<k>>_<source>``, and <n> IS the rung (channel count).  One
    panel per front type; grouped bars per source at the COARSEST dilation
    (the currency the ladder decision is made on, matching the permutation
    figure), with the pooled mean-of-fold CI whiskers.
    """
    abl_dir = results_dir / "ablation_eval"
    groups: dict = {}                       # (rung, source) -> [DataFrame]
    for path in sorted(abl_dir.glob("*.csv")) if abl_dir.is_dir() else []:
        df = pd.read_csv(path)
        if not {"front", "km", "csi"} <= set(df.columns):
            continue
        ckpt, _, source = path.stem.rpartition("_")
        m = RUNG_RE.match(pooled_name(ckpt))
        if not m:
            print(f"note: {path.name} is not a D6A<n> ladder leg; skipped")
            continue
        groups.setdefault((int(m.group(1)), source), []).append(df)
    if not groups:
        print(f"no ladder eval CSVs under {abl_dir}; "
              f"skipping ablation_ladder.png")
        return None

    pooled = {key: _pool_eval_frames(frames)
             for key, frames in groups.items()}
    rungs = sorted({r for r, _ in pooled}, reverse=True)   # 5 -> 3 -> 2
    sources = sorted({s for _, s in pooled})
    fronts = sorted({f for df in pooled.values()
                    for f in df["front"].unique() if f != "none"})
    colors = {s: LEG_PALETTE[i % len(LEG_PALETTE)]
             for i, s in enumerate(sources)}

    fig, axes = plt.subplots(1, len(fronts),
                             figsize=(3.2 * len(fronts), 4.2),
                             squeeze=False, sharey=True)
    width = 0.8 / max(len(sources), 1)
    km_max = None
    for j, front in enumerate(fronts):
        ax = axes[0][j]
        for i, source in enumerate(sources):
            xs, vals, los, his = [], [], [], []
            for k, rung in enumerate(rungs):
                df = pooled.get((rung, source))
                if df is None:
                    continue                # rung not pulled for this source
                sub = df[(df["front"] == front)
                         & (df["dilation"] == df["dilation"].max())]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                km_max = row["km"]
                xs.append(k)
                vals.append(row["csi"])
                los.append(max(row["csi"] - row.get("csi_lo", row["csi"]), 0))
                his.append(max(row.get("csi_hi", row["csi"]) - row["csi"], 0))
            x = np.array(xs) + (i - (len(sources) - 1) / 2) * width
            ax.bar(x, vals, width=width * 0.9, color=colors[source],
                  label=source, yerr=[los, his], capsize=2,
                  error_kw={"elinewidth": 0.8, "ecolor": "0.3"})
        ax.set_xticks(np.arange(len(rungs)))
        ax.set_xticklabels([f"{r}ch" for r in rungs], fontsize=9, color=INK)
        ax.set_title(front, color=INK, fontsize=10)
        ax.set_xlabel("ladder rung", color=INK, fontsize=8)
        _style_ax(ax)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("CSI (pooled, mean-of-fold 95% CI)",
                         color=INK, fontsize=9)
    fig.legend(*axes[0][0].get_legend_handles_labels(),
              loc="lower center", ncol=len(sources), frameon=False,
              fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, -0.04))
    km_note = f", {km_max:g} km neighborhood" if km_max is not None else ""
    fig.suptitle(f"channel-ablation ladder (fold-pooled{km_note})",
                color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_ladder.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Selftest (synthetic fixtures; no results tree or TF needed)
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    """Render every figure from tiny synthetic fixtures in a tmp dir.

    Two assertions per figure family: (1) it renders a nonempty PNG from a
    populated fixture tree, and (2) it auto-skips (returns None, no crash)
    on an empty tree -- the partial-pull guarantee.
    """
    import tempfile
    rng = np.random.default_rng(0)

    def eval_frame(seed: float, with_ci: bool = True) -> pd.DataFrame:
        rows = []
        for front in ("cold", "warm", "dryline"):
            for dil, km in ((0, 0.0), (1, 111.2), (2, 222.4)):
                csi = min(0.2 + 0.1 * dil + seed, 0.95)
                row = {"front": front, "dilation": dil, "km": km,
                       "csi": csi, "pod": min(csi + 0.1, 1.0),
                       "far": max(0.5 - 0.1 * dil, 0.0), "fb": 0.9}
                if with_ci:
                    row["csi_lo"], row["csi_hi"] = csi - 0.05, csi + 0.05
                rows.append(row)
        return pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dl_front"
        out = root / "summary_plots"

        # -- empty tree: every plotter must skip, not crash ------------- #
        for fn in (plot_training_curves, plot_csi_comparison,
                   plot_permutation_importance, plot_ablation_ladder):
            assert fn(root, out) is None, f"{fn.__name__} on empty tree"
        assert plot_csi_comparison_fold(root, out, 0) is None

        # -- models/ for the training curves ---------------------------- #
        mdir = root / "models" / "D6C-f0"
        mdir.mkdir(parents=True)
        pd.DataFrame({"epoch": range(5), "loss": np.linspace(1, .2, 5),
                      "val_loss": np.linspace(1.1, .4, 5)}) \
            .to_csv(mdir / "history.csv", index=False)

        # -- test_eval/: two folds of one leg (pool!), bk19 without CI
        #    or paper json (accuracy panel must show it absent) ---------- #
        tdir = root / "test_eval"
        tdir.mkdir()
        for k, seed in ((0, 0.00), (1, 0.10)):
            stem = f"D6C-f{k}_kriged-airs"
            eval_frame(seed).to_csv(tdir / f"{stem}.csv", index=False)
            (tdir / f"{stem}_paper.json").write_text(json.dumps(
                {"accuracy": {"all_categories": 0.9 + 0.02 * k,
                              "front_no_front": 0.95},
                 "auc": 0.9}))
        eval_frame(0.05, with_ci=False).to_csv(tdir / "bk19.csv", index=False)

        # -- permutation/: one main-chain and one ladder checkpoint ----- #
        pdir = root / "permutation"
        pdir.mkdir()
        perm_cols = ["channel", "repeat", "stat", "front", "dilation", "km",
                     "csi", "csi_lo", "csi_hi", "pod", "far", "fb",
                     "csi_delta", "pod_delta"]
        for stem, chans in (("D6C-f0_kriged-airs",
                             ("T2M", "QV2M", "SLP", "U10M", "V10M")),
                            ("D6A5-f0_reanalysis",
                             ("T2M", "QV2M", "SLP", "U10M", "V10M"))):
            rows = []
            for front in ("cold", "warm"):
                for dil, km in ((0, 0.0), (2, 222.4)):
                    base = 0.5 + 0.1 * dil
                    rows.append(["<baseline>", -1, "raw", front, dil, km,
                                 base, base - .05, base + .05, .6, .3, .9,
                                 0.0, 0.0])
                    for ch in chans:
                        d = float(rng.uniform(0, 0.15))
                        rows.append([ch, 0, "raw", front, dil, km,
                                     base - d, base - d - .05,
                                     base - d + .05, .6, .3, .9, d, d])
            pd.DataFrame(rows, columns=perm_cols) \
                .to_csv(pdir / f"{stem}.csv", index=False)

        # -- ablation_eval/: full ladder x two sources, two folds of one
        #    rung to exercise the pooling ------------------------------- #
        adir = root / "ablation_eval"
        adir.mkdir()
        for rung, seed in ((5, 0.15), (3, 0.08), (2, 0.0)):
            for source in ("reanalysis", "kriged-airs"):
                eval_frame(seed).to_csv(
                    adir / f"D6A{rung}-f0_{source}.csv", index=False)
        eval_frame(0.20).to_csv(adir / "D6A5-f1_reanalysis.csv", index=False)

        # -- render + assert -------------------------------------------- #
        made = {"training_curves.png": plot_training_curves(root, out),
                "csi_comparison.png": plot_csi_comparison(root, out),
                "csi_comparison_f0.png":
                    plot_csi_comparison_fold(root, out, 0),
                "permutation_importance.png":
                    plot_permutation_importance(root, out),
                "ablation_ladder.png": plot_ablation_ladder(root, out)}
        for name, path in made.items():
            assert path is not None, f"{name} was skipped on full fixtures"
            assert path.name == name, f"{path.name} != {name}"
            assert path.stat().st_size > 0, f"{name} is empty"

        # Pooling spot-checks (the rule, not just "a PNG exists"):
        legs, accuracy = _load_pooled_legs(tdir)
        assert set(legs) == {"D6C_kriged-airs", "bk19"}, sorted(legs)
        cold0 = legs["D6C_kriged-airs"].query("front=='cold' & dilation==0")
        assert np.isclose(cold0["csi"].iloc[0], 0.25), cold0  # mean(.2, .3)
        assert np.isclose(accuracy["D6C_kriged-airs"], 0.91)
        assert accuracy["bk19"] is None      # absent, not zero
        costs = _load_permutation_costs(pdir)
        assert set(costs["ckpt"]) == {"D6C", "D6A5"}
        assert list(costs["channel"].cat.categories) == \
            ["T2M", "QV2M", "SLP", "U10M", "V10M"]

    print("selftest OK: all five figures rendered from synthetic fixtures, "
          "empty-tree auto-skip verified, fold pooling spot-checked")
    return 0


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/dl_front")
    ap.add_argument("--per-fold", action="store_true",
                    help="ALSO write the pre-pooling per-fold CSI figure(s) "
                         "(debugging aid; the pooled figure is the default)")
    ap.add_argument("--fold", type=int, default=0,
                    help="fold for --per-fold (default 0)")
    ap.add_argument("--all-folds", action="store_true",
                    help="with --per-fold: also plot folds 1 and 2")
    ap.add_argument("--selftest", action="store_true",
                    help="render every figure from synthetic fixtures in a "
                         "tmp dir and exit (no results tree needed)")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()

    results_dir = Path(a.results_dir)
    out_dir = results_dir / "summary_plots"

    written = [plot_training_curves(results_dir, out_dir),
               plot_csi_comparison(results_dir, out_dir),
               plot_permutation_importance(results_dir, out_dir),
               plot_ablation_ladder(results_dir, out_dir)]
    if a.per_fold:
        folds = [a.fold] + ([1, 2] if a.all_folds and a.fold == 0 else [])
        for fold in dict.fromkeys(folds):       # dedupe, keep order
            written.append(plot_csi_comparison_fold(results_dir, out_dir,
                                                    fold))

    for path in filter(None, written):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
