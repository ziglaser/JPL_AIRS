"""Neighborhood verification of binary front maps on the 1 deg label grid.

Scoring convention copied from ``fronts/generate_performance_stats.py`` so our
numbers are comparable with the paper's:

- TP and FP are scored against the truth DILATED by ``n`` 8-connected
  iterations (the "neighborhood"); FN is scored once against the UNDILATED
  truth for every neighborhood (deliberate upstream choice).
- Every pixel is weighted by cos(latitude) so a 1 deg cell counts by area.

Uncertainty: circular moving-block bootstrap over DAYS (house norm --
convection_skill audit 2026-07-23: iid resampling understates CIs 2-3x on
daily synoptic data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .labels import dilate

#: pred_hit/pred_miss partition the PREDICTED pixels against the dilated
#: truth; truth_hit/truth_miss partition the TRUTH pixels against the dilated
#: prediction.  Symmetric neighborhood matching for line-vs-line comparison
#: (Niebler et al. 2022; Lagerquist et al. 2019): a prediction offset by <= n
#: pixels is a hit in BOTH directions.  (The upstream FrontFinder convention
#: -- TP/FP vs dilated truth, FN vs exact truth -- assumes thick probability
#: blobs; on thin binary lines it caps POD near 0.5 for a 1-px offset.)
COUNT_COLS = ("pred_hit", "pred_miss", "truth_hit", "truth_miss")


# --------------------------------------------------------------------------- #
# Contingency counts
# --------------------------------------------------------------------------- #

def contingency_by_day(pred: xr.DataArray, truth: xr.DataArray,
                       valid: xr.DataArray | None = None,
                       dilations: tuple = config.EVAL_DILATIONS) -> pd.DataFrame:
    """Area-weighted symmetric neighborhood counts per (day, class, dilation).

    pred, truth: boolean (time, front, lat, lon) on the label grid.
    valid: optional boolean (time, lat, lon); False pixels are excluded
    (fill-value pixels, off-swath pixels).  Returns a tidy DataFrame keyed by
    (date, front, dilation) -- day granularity is what the block bootstrap
    resamples.
    """
    w = np.cos(np.deg2rad(truth["lat"])).broadcast_like(truth.isel(time=0, front=0))
    w = w.values[np.newaxis, np.newaxis]              # (1, 1, lat, lon)
    p = pred.transpose("time", "front", "lat", "lon").values
    t = truth.transpose("time", "front", "lat", "lon").values
    if valid is not None:
        v = valid.transpose("time", "lat", "lon").values[:, np.newaxis]
        w = w * v                                     # (time, 1, lat, lon)
    dates = pd.DatetimeIndex(truth["time"].values).normalize()
    fronts = list(truth["front"].values)

    rows = []
    for n in dilations:
        flat = lambda a: dilate(a.reshape(-1, *a.shape[-2:]), n).reshape(a.shape)
        t_n, p_n = flat(t), flat(p)
        fields = {"pred_hit": (p & t_n) * w, "pred_miss": (p & ~t_n) * w,
                  "truth_hit": (t & p_n) * w, "truth_miss": (t & ~p_n) * w}
        for k, front in enumerate(fronts):
            df = pd.DataFrame({"date": dates, "front": front, "dilation": n,
                               **{c: a[:, k].sum(axis=(1, 2))
                                  for c, a in fields.items()}})
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return (out.groupby(["date", "front", "dilation"], sort=False)[list(COUNT_COLS)]
               .sum().reset_index())


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #

def _metrics(ph, pm, th, tm):
    """POD/FAR from the two directions; CSI via the paper's Eq. (4) identity."""
    pod = th / (th + tm)
    far = pm / (ph + pm)
    csi = 1.0 / (1.0 / pod + 1.0 / (1.0 - far) - 1.0)
    fb = (ph + pm) / (th + tm)
    return csi, pod, far, fb


def scores_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """CSI/POD/FAR/FB from pooled counts, per (front, dilation).

    POD is measured on truth pixels (found within n px), FAR on predicted
    pixels (unmatched within n px); CSI combines them with paper Eq. (4).
    """
    pooled = counts.groupby(["front", "dilation"])[list(COUNT_COLS)].sum()
    csi, pod, far, fb = _metrics(*(pooled[c] for c in COUNT_COLS))
    out = pd.DataFrame({"csi": csi, "pod": pod, "far": far, "fb": fb})
    out["km"] = [d * config.KM_PER_ITERATION for _, d in out.index]
    return out


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #

def threshold_sweep(prob: xr.DataArray, truth: xr.DataArray,
                    valid: xr.DataArray | None = None,
                    thresholds=np.arange(0.01, 1.0, 0.01),
                    dilations: tuple = config.EVAL_DILATIONS) -> pd.DataFrame:
    """CSI/POD/FAR/FB at every probability threshold, per (front, dilation).

    ``prob`` is a probability DataArray (time, front, lat, lon) as produced
    by ``predict.predict_year``'s ``"probabilities"`` variable; ``truth`` is
    boolean, same dims.  For each threshold the prediction is re-binarized
    (``prob >= thr``) and scored with the same ``contingency_by_day`` /
    ``scores_from_counts`` machinery as a single fixed prediction.
    Default granularity is 0.01 (paper section 2d; 2026-08-10 -- 0.05 could
    miss the CSI peak by several points on a steep curve).

    Cost note: ``contingency_by_day`` dilates BOTH the prediction and the
    truth internally; the truth dilation is therefore recomputed once per
    threshold even though truth itself never changes. This is kept simple
    (not cached) because the prediction DOES change every iteration -- a
    correct cache would have to special-case the truth-only half of the
    dilation loop, which isn't worth the complexity here.  Runtime scales as
    O(len(thresholds) * len(dilations)).
    """
    rows = []
    for thr in thresholds:
        pred = prob >= thr
        counts = contingency_by_day(pred, truth, valid=valid, dilations=dilations)
        scores = scores_from_counts(counts).reset_index()
        scores.insert(0, "threshold", float(thr))
        rows.append(scores)
    out = pd.concat(rows, ignore_index=True)
    return out[["threshold", "front", "dilation", "csi", "pod", "far", "fb", "km"]]


def best_csi(sweep: pd.DataFrame) -> pd.DataFrame:
    """The paper's "CSI at best probability threshold": per (front,
    dilation), the sweep row with the highest CSI.
    """
    idx = sweep.groupby(["front", "dilation"])["csi"].idxmax()
    cols = ["front", "dilation", "threshold", "csi", "pod", "far", "fb", "km"]
    return sweep.loc[idx, cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Day-block bootstrap
# --------------------------------------------------------------------------- #

@dataclass
class BootstrapResult:
    scores: pd.DataFrame        # point estimates (front, dilation) x metrics
    lo: pd.DataFrame            # lower CI bound, same shape
    hi: pd.DataFrame            # upper CI bound


def block_bootstrap(counts: pd.DataFrame, block_days: int = config.BLOCK_DAYS,
                    n_reps: int = config.N_BOOT_REPS,
                    confidence: float = config.CONFIDENCE_LEVEL,
                    seed: int = config.BOOT_SEED) -> BootstrapResult:
    """Circular moving-block bootstrap over days of the pooled scores.

    Days are resampled in contiguous blocks of ``block_days`` (wrapping at the
    year edge -- Politis & Romano 1992 circular convention) to respect
    synoptic autocorrelation; scores are recomputed from pooled counts per
    replicate and CIs taken as percentiles.
    """
    days = np.sort(counts["date"].unique())
    n_days = len(days)
    day_idx = {pd.Timestamp(d): i for i, d in enumerate(days)}
    # counts cube: (day, front, dilation, 3) for fast replicate pooling
    fronts = list(dict.fromkeys(counts["front"]))
    dils = sorted(counts["dilation"].unique())
    cube = np.zeros((n_days, len(fronts), len(dils), len(COUNT_COLS)))
    fi = {f: i for i, f in enumerate(fronts)}
    di = {d: i for i, d in enumerate(dils)}
    for row in counts.itertuples(index=False):
        cube[day_idx[row.date], fi[row.front], di[row.dilation]] += (
            row.pred_hit, row.pred_miss, row.truth_hit, row.truth_miss)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_days))
    reps = np.empty((n_reps, len(fronts), len(dils), 4))
    for r in range(n_reps):
        starts = rng.integers(0, n_days, n_blocks)
        idx = (starts[:, None] + np.arange(block_days)[None]) % n_days
        pooled = cube[idx.ravel()[:n_days]].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            reps[r] = np.stack(_metrics(*(pooled[..., i] for i in range(4))),
                               axis=-1)

    alpha = (100.0 - confidence) / 2.0
    qlo = np.nanpercentile(reps, alpha, axis=0)
    qhi = np.nanpercentile(reps, 100.0 - alpha, axis=0)
    index = pd.MultiIndex.from_product([fronts, dils], names=["front", "dilation"])
    cols = ["csi", "pod", "far", "fb"]
    point = scores_from_counts(counts)
    # reindex to the point-estimate row order (groupby sorts alphabetically)
    to_df = lambda a: (pd.DataFrame(a.reshape(-1, 4), index=index, columns=cols)
                       .reindex(point.index))
    return BootstrapResult(scores=point, lo=to_df(qlo), hi=to_df(qhi))
