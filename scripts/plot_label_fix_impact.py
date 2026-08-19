"""How much of the reported skill was the antimeridian label bug?

Re-scores the shipped D6A/D6B/D6C fold-0 checkpoints on the reanalysis test
leg twice -- against the shipped (contaminated) labels and against the
regenerated ones -- and plots the two side by side.  Nothing about the models
changes between the bars; only the truth they are scored against.

Input: the CSVs written by ``scripts/eval_decision_rule.py``.

Usage::

    PYTHONPATH=src python scripts/plot_label_fix_impact.py \\
        --sweep-dir <dir with sweep_{old,new}_D6?-f0.csv> \\
        --out results/dl_front/quicklook/label_fix_impact.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

#: dataviz skill categorical slots 1 and 2 (validated default palette).
OLD_COLOR, NEW_COLOR = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
CLASSES = ("cold", "warm", "stationary", "occluded", "dryline")
STAGES = (("D6A-f0", "D6A  reanalysis-trained"),
          ("D6B-f0", "D6B  + degraded-krige"),
          ("D6C-f0", "D6C  + AIRS fine-tune"))


def load(sweep_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in glob.glob(str(sweep_dir / "sweep_*_D6?-f0.csv"))]
    if not frames:
        raise SystemExit(f"no sweep_*_D6?-f0.csv under {sweep_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df[np.isclose(df.km, 111.2) & np.isclose(df.none_scale, 1.0)]


def render(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    xs = np.arange(len(CLASSES))
    w = 0.36
    for ax, (ckpt, title) in zip(axes, STAGES):
        d = df[df.ckpt == ckpt].set_index(["labels", "front"])["csi"]
        old = [d.get(("old", c), np.nan) for c in CLASSES]
        new = [d.get(("new", c), np.nan) for c in CLASSES]
        # 2 px surface gap between adjacent bars: the 0.02 inset on each side
        ax.bar(xs - w / 2 - 0.01, old, w, color=OLD_COLOR, label="shipped labels")
        ax.bar(xs + w / 2 + 0.01, new, w, color=NEW_COLOR, label="regenerated labels")
        for x, o, n in zip(xs, old, new):
            ax.text(x + w / 2 + 0.01, n + 0.012, f"{n - o:+.02f}", ha="center",
                    va="bottom", fontsize=7.5, color=MUTED)
        ax.set_title(title, color=INK, fontsize=10, pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(CLASSES, color=MUTED, fontsize=8.5, rotation=20,
                           ha="right")
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("0.75")
    axes[0].set_ylabel("CSI @ 111 km", color=MUTED, fontsize=9)
    axes[0].set_ylim(0, 0.66)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=MUTED,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Same checkpoints, same test steps — only the truth changed\n"
                 "reanalysis leg, 2016-2018, fold 0",
                 color=INK, fontsize=11.5, y=1.03, ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out", default="results/dl_front/quicklook/label_fix_impact.png")
    a = ap.parse_args(argv)
    print(f"wrote {render(load(Path(a.sweep_dir)), Path(a.out))}")


if __name__ == "__main__":
    main()
