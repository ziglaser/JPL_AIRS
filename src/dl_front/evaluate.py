"""DL-FRONT evaluation: the paper's metrics + neighborhood CSI.

Paper metrics (section 4.2), all restricted to the Fig. 2 region mask:
  - per-class grid-cell fractions (Table 1),
  - categorical accuracy, full and front/no-front (Table 2),
  - confusion matrices as % of total cells (Tables 3-4),
  - ROC and precision-recall curves for front/no-front produced by scaling
    the none-category likelihood by a factor before the argmax (section
    4.2.4), with trapezoidal AUC.

Line-vs-line skill (CSI/POD/FAR at explicit km scales) reuses
``front_finder.evaluate`` symmetric neighborhood matching unchanged.

Everything here is numpy-pure and streaming: ``PaperMetrics.update`` is fed
one year of predictions at a time, so the 8-year validation span never has
to sit in memory at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from front_finder import evaluate as fd_evaluate

from . import dataset

#: none-likelihood scaling factors for the ROC/PR sweep (section 4.2.4:
#: "multiplying the no-front category likelihood values by a factor that
#: varied from 0 to large enough that all grid cells were labeled no-front");
#: values in configs/dl_front.yaml (evaluation: roc_factors).
ROC_FACTORS = dataset.config.ROC_FACTORS


class PaperMetrics:
    """Streaming accumulator for the paper's confusion/ROC statistics.

    ``mask``: the (68, 141) bool scoring region; defaults to the paper's
    Fig. 2 region mask (the 5-class replication path).  The 6-class
    dryline/AIRS track passes ``dataset.analysis_domain()`` instead (user
    decision 2026-08-13).
    """

    def __init__(self, n_classes: int, mask: np.ndarray | None = None):
        self.n = n_classes
        self.names = dataset.class_names(n_classes)
        self.confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        # per ROC factor: TP, FP, FN, TN of the front/no-front split
        self.roc = np.zeros((len(ROC_FACTORS), 4), dtype=np.int64)
        self.mask = (dataset.region_mask() if mask is None
                     else np.asarray(mask)).astype(bool)

    def update(self, probs: np.ndarray, y_cls: np.ndarray) -> None:
        """probs (t, 68, 141, n_cls) softmax outputs; y_cls (t, 68, 141)."""
        m = np.broadcast_to(self.mask, y_cls.shape)
        p = probs[m]                                  # (pix, n_cls)
        t = y_cls[m].astype(np.int64)                 # (pix,)
        pred = p.argmax(-1)
        np.add.at(self.confusion, (t, pred), 1)

        none = self.n - 1
        truth_front = t != none
        front_max = p[:, :none].max(-1)
        for i, f in enumerate(ROC_FACTORS):
            pred_front = front_max > f * p[:, none]
            self.roc[i] += (
                (pred_front & truth_front).sum(),      # TP
                (pred_front & ~truth_front).sum(),     # FP
                (~pred_front & truth_front).sum(),     # FN
                (~pred_front & ~truth_front).sum())    # TN

    # ---- paper tables ---------------------------------------------------- #

    def cell_fractions(self) -> pd.DataFrame:
        """Table 1: % of masked cells per class, truth vs predicted."""
        total = self.confusion.sum()
        rows = {"truth": self.confusion.sum(1) / total * 100,
                "predicted": self.confusion.sum(0) / total * 100}
        df = pd.DataFrame(rows, index=list(self.names))
        any_front = df.iloc[:-1].sum()
        df.loc["any"] = any_front
        return df

    def accuracy(self) -> dict:
        """Table 2: all-category and front/no-front categorical accuracy."""
        total = self.confusion.sum()
        none = self.n - 1
        front_hit = self.confusion[:none, :none].sum()
        return {"all_categories": self.confusion.trace() / total,
                "front_no_front": (front_hit + self.confusion[none, none])
                                  / total}

    def confusion_table(self, percent: bool = True) -> pd.DataFrame:
        """Table 3: confusion matrix (actual rows x predicted columns)."""
        c = self.confusion / self.confusion.sum() * 100 if percent \
            else self.confusion
        return pd.DataFrame(c, index=list(self.names), columns=list(self.names))

    def roc_pr(self) -> pd.DataFrame:
        """ROC + precision-recall points over the none-scaling sweep."""
        tp, fp, fn, tn = self.roc.T.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            df = pd.DataFrame({
                "factor": ROC_FACTORS,
                "tpr": tp / (tp + fn), "fpr": fp / (fp + tn),
                "precision": tp / (tp + fp), "recall": tp / (tp + fn)})
        return df

    def auc(self) -> float:
        """Trapezoidal area under the ROC curve (paper: 0.90)."""
        pts = self.roc_pr().sort_values("fpr")
        x = np.concatenate([[0.0], pts["fpr"].values, [1.0]])
        y = np.concatenate([[0.0], pts["tpr"].values, [1.0]])
        # np.trapz was renamed np.trapezoid in numpy 2.0; the fronts-tf env
        # (numpy < 2) only has the old name, .venv (numpy >= 2) deprecates it.
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return float(trapezoid(y, x))


# --------------------------------------------------------------------------- #
# Neighborhood CSI (front_finder convention, explicit km scales)
# --------------------------------------------------------------------------- #

def onehot_da(cls: np.ndarray, times, n_classes: int,
              from_probs: np.ndarray | None = None) -> xr.DataArray:
    """(time, front, lat, lon) boolean DataArray of the front classes.

    ``cls`` is (t, 68, 141) class indices (argmax of the probabilities if
    ``from_probs`` given); the none class is dropped -- CSI is scored per
    front type.
    """
    if from_probs is not None:
        cls = from_probs.argmax(-1)
    names = dataset.class_names(n_classes)[:-1]
    hot = np.stack([cls == k for k in range(len(names))], axis=1)
    lat = np.asarray(dataset.config.LABEL_LATS)
    lon = np.asarray(dataset.config.LABEL_LONS)
    return xr.DataArray(hot, dims=("time", "front", "lat", "lon"),
                        coords={"time": np.asarray(times), "front": list(names),
                                "lat": lat, "lon": lon})


def csi_counts(pred_cls: np.ndarray, y_cls: np.ndarray, times,
               n_classes: int, mask: np.ndarray | None = None
               ) -> pd.DataFrame:
    """Per-day symmetric-neighborhood contingency counts inside the mask.

    ``mask``: (68, 141) bool scoring region; default = the Fig. 2 region
    mask (5-class paper path).  The 6-class track passes
    ``dataset.analysis_domain()`` (user decision 2026-08-13).
    """
    pred = onehot_da(pred_cls, times, n_classes)
    truth = onehot_da(y_cls, times, n_classes)
    mask = dataset.region_mask() if mask is None else mask
    valid = xr.DataArray(
        np.broadcast_to(np.asarray(mask).astype(bool), y_cls.shape),
        dims=("time", "lat", "lon"), coords=truth.drop_vars("front").coords)
    return fd_evaluate.contingency_by_day(pred, truth, valid=valid)


def csi_scores(counts: pd.DataFrame) -> pd.DataFrame:
    return fd_evaluate.scores_from_counts(counts)
