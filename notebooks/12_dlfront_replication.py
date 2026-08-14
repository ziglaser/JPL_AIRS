# %% [markdown]
# # DL-FRONT replication (Biard & Kunkel 2019) + dryline extension
#
# Driver for the `dl_front` package: evaluates a trained checkpoint against
# the paper's published numbers (Tables 1-4, ROC AUC 0.90) and reports
# line-vs-line CSI at explicit km scales.  Training itself runs from the CLI
# (`python -m dl_front.train`); this notebook only reads results.
#
# Set CHECKPOINT/N_CLASSES below, run cells top to bottom.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_front import config, dataset, evaluate, predict
from front_finder import evaluate as fd_evaluate

RESULTS = config.RESULTS_DIR
CHECKPOINT = RESULTS / "models/R5-fold0/R5-fold0.h5"
N_CLASSES = 5
EVAL_YEARS = {5: config.EVAL_YEARS_5, 6: config.EVAL_YEARS_6}[N_CLASSES]

#: Paper values for side-by-side printing (Tables 1, 2; section 4.2.4).
PAPER_TABLE1_PRED = {"cold": 3.75, "warm": 1.17, "stationary": 5.20,
                     "occluded": 1.58, "any": 11.70, "none": 88.30}
PAPER_ACCURACY = {"all_categories": 0.8800, "front_no_front": 0.8985}
PAPER_AUC = 0.90


# %%
def q1_paper_metrics(checkpoint, n_classes, years):
    """Confusion/accuracy/fractions/ROC on the validation years, streamed."""
    model = predict.load_model(checkpoint)
    pm, counts = predict.evaluate_years(model, years, n_classes)
    return pm, counts


def q2_report_tables(pm, out_dir):
    """Print paper-vs-ours tables and save CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frac = pm.cell_fractions()
    frac["paper_predicted"] = pd.Series(PAPER_TABLE1_PRED)
    print("Table 1 (percent of masked cells):\n", frac.round(2), "\n")
    acc = pm.accuracy()
    print("Table 2 accuracy: ours", {k: round(v, 4) for k, v in acc.items()},
          " paper", PAPER_ACCURACY, "\n")
    conf = pm.confusion_table()
    print("Table 3 confusion (percent of cells):\n", conf.round(2), "\n")
    print(f"ROC AUC: ours {pm.auc():.3f}  paper {PAPER_AUC}")
    frac.to_csv(out_dir / "table1_fractions.csv")
    conf.to_csv(out_dir / "table3_confusion.csv")
    pm.roc_pr().to_csv(out_dir / "roc_pr.csv", index=False)
    return frac, conf


def q3_roc_plot(pm, out_dir):
    pts = pm.roc_pr()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(pts["fpr"], pts["tpr"], "o-")
    axes[0].plot([0, 1], [0, 1], "b--", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                title=f"front/no-front ROC (AUC {pm.auc():.3f})")
    axes[1].plot(pts["recall"], pts["precision"], "o-")
    axes[1].set(xlabel="recall", ylabel="precision", title="precision-recall")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=150)
    return fig


def q4_neighborhood_csi(counts, out_dir):
    """Line-vs-line skill with explicit km scales + day-block bootstrap CIs."""
    scores = evaluate.csi_scores(counts)
    boot = fd_evaluate.block_bootstrap(counts)
    table = scores.join(boot.lo.add_suffix("_lo")).join(boot.hi.add_suffix("_hi"))
    print(table.round(3))
    table.to_csv(out_dir / "neighborhood_csi.csv")
    return table


# %%
if __name__ == "__main__":
    out_dir = RESULTS / CHECKPOINT.parent.name
    pm, counts = q1_paper_metrics(CHECKPOINT, N_CLASSES, EVAL_YEARS)
    q2_report_tables(pm, out_dir)
    q3_roc_plot(pm, out_dir)
    q4_neighborhood_csi(counts, out_dir)
