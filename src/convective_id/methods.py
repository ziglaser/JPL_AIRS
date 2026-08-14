"""Four convective-identification methods, one output contract.

Every classifier takes the featured table (:func:`features.add_structure_features`)
and returns a DataFrame aligned to it with two columns:

- ``label`` : bool, convective (False everywhere outside the precip domain)
- ``score`` : float, method-native confidence (NaN outside the precip domain)

Inputs are strictly the flag-free :data:`features.FEATURES`; the PrecipFlag
seed columns enter only as weak-supervision labels (forest) or object seeds
(storm objects), as documented per method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .features import FEATURES, _GRID_INDEX, _from_grid, _to_grid

RANDOM_STATE: int = 20260722

#: Thresholding defaults (tunable): a sub-pixel peak at genuinely convective
#: rain rates, OR a moderate peak that towers over its own wet-area mean
#: (an embedded core). Values in mm/h on the 1-h sub-pixel QPE.
PEAK_MM: float = 20.0
EMBEDDED_PEAK_MM: float = 10.0
EMBEDDED_RATIO: float = 5.0


def _empty(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"label": np.zeros(len(df), dtype=bool),
         "score": np.full(len(df), np.nan)}, index=df.index)


def threshold_classify(df: pd.DataFrame, peak_mm: float = PEAK_MM,
                       embedded_peak_mm: float = EMBEDDED_PEAK_MM,
                       embedded_ratio: float = EMBEDDED_RATIO) -> pd.DataFrame:
    """Physical cuts on sub-pixel structure (no fitting, fully transparent).

    Convective if the sub-pixel peak reaches convective rain rates
    (``peak_mm``), or a moderate peak (``embedded_peak_mm``) is ``embedded_ratio``
    times its wet-area mean -- a small intense core embedded in broad rain,
    which is exactly the flag-diluted MCS signature. Score = the peak rate.
    """
    out = _empty(df)
    precip = df["is_precip"].to_numpy()
    qpe_max = np.expm1(df["qpe_max_log"].to_numpy())
    ratio = np.expm1(df["peak_ratio_log"].to_numpy())
    hit = (qpe_max >= peak_mm) | ((qpe_max >= embedded_peak_mm)
                                  & (ratio >= embedded_ratio))
    out["label"] = precip & hit
    out.loc[precip, "score"] = qpe_max[precip]
    return out


def cluster_classify(df: pd.DataFrame, n_components: int = 4,
                     random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Unsupervised GMM over the flag-free feature space.

    Precipitating rows are standardized and fit with a ``n_components``-part
    Gaussian mixture; the CONVECTIVE component is chosen by its MRMS profile
    alone (largest mean sub-pixel peak) -- CAPE is never consulted, so the
    validation stays independent. Score = posterior probability of that
    component; label = score > 0.5.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    out = _empty(df)
    precip = df["is_precip"].to_numpy()
    X = df.loc[precip, list(FEATURES)].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(np.nan_to_num(X))

    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=random_state, n_init=2)
    gmm.fit(X)
    peak_col = list(FEATURES).index("qpe_max_log")
    convective_component = int(np.argmax(gmm.means_[:, peak_col]))
    prob = gmm.predict_proba(X)[:, convective_component]

    out.loc[precip, "score"] = prob
    out.loc[precip, "label"] = prob > 0.5
    return out


def forest_classify(df: pd.DataFrame, n_estimators: int = 300,
                    min_samples_leaf: int = 50,
                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Weak supervision: learn what flagged cores LOOK like, then generalize.

    Labels (from PrecipFlag, never features): positives = confident flag-level
    cores (``core_confident``); negatives = precipitating cells with zero
    convective share AND no core anywhere in their 3x3 neighborhood (clean
    stratiform, away from any storm core). The ambiguous middle -- which
    includes the anvil/MCS cells this package exists for -- is left unlabeled
    and receives a prediction. Because the forest sees only flag-free
    STRUCTURE, a stratiform-flagged anvil cell whose structure matches the
    cores is recovered. Score = out-of-model predicted probability;
    label = score > 0.5.
    """
    from sklearn.ensemble import RandomForestClassifier

    out = _empty(df)
    precip = df["is_precip"].to_numpy()
    sub = df.loc[precip]

    positive = sub["core_confident"].to_numpy()
    negative = (~sub["core_any"].to_numpy()) & (~sub["core_in_neighborhood"].to_numpy())
    labeled = positive | negative
    if positive.sum() < 50 or negative.sum() < 50:
        raise ValueError("not enough weak labels to train "
                         f"(pos={int(positive.sum())}, neg={int(negative.sum())})")

    X = np.nan_to_num(sub[list(FEATURES)].to_numpy(dtype=float))
    forest = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    forest.fit(X[labeled], positive[labeled])
    prob = forest.predict_proba(X)[:, 1]

    out.loc[precip, "score"] = prob
    out.loc[precip, "label"] = prob > 0.5
    out.attrs["feature_importances"] = dict(
        zip(FEATURES, forest.feature_importances_.round(4)))
    return out


def object_classify(df: pd.DataFrame, seed: str = "flag_or_threshold",
                    peak_mm: float = PEAK_MM,
                    max_distance_cells: int = 2) -> pd.DataFrame:
    """Storm objects: the anvil is attached to its core, so label the STORM.

    Per (date, slot), contiguous precipitating cells (8-connectivity on the
    1-deg grid) form one storm object. Cells within ``max_distance_cells`` of a
    core seed ALONG the object are convective -- stratiform-flagged anvil cells
    inherit the label from the core they are physically attached to, but the
    label cannot run away down an entire synoptic rain shield (unlimited
    spreading marked 84% of 2019 precip convective; ~2 cells ~ 200-300 km is
    the anvil scale).

    ``seed``: ``"flag"`` (a SUPPORTED flag core: share >= 0.2 with >= 10 wet
    sub-pixels -- any-core seeds are so ubiquitous in large systems that the
    label swallowed 70% of precip even distance-limited), ``"threshold"``
    (sub-pixel peak >= ``peak_mm``; fully flag-free), or the default union.
    Score = fraction of the cell's object that is seeded (core density;
    shared by every member of the object).
    """
    if seed not in ("flag", "threshold", "flag_or_threshold"):
        raise ValueError(f"unknown seed {seed!r}")
    out = _empty(df)

    grid_precip, gidx = _to_grid(
        df.assign(_p=df["is_precip"].astype(float)), "_p")
    seed_flag = df["core_confident"].to_numpy()
    seed_thresh = np.expm1(df["qpe_max_log"].to_numpy()) >= peak_mm
    seeds = {"flag": seed_flag, "threshold": seed_thresh,
             "flag_or_threshold": seed_flag | seed_thresh}[seed]
    grid_seed, _ = _to_grid(df.assign(_s=(seeds & df["is_precip"]).astype(float)), "_s")

    connect8 = np.ones((3, 3), dtype=int)
    label_grid = np.zeros_like(grid_precip, dtype=bool)
    score_grid = np.full_like(grid_precip, np.nan, dtype=float)
    n_date, n_slot = grid_precip.shape[:2]
    for d in range(n_date):
        for s in range(n_slot):
            wet = grid_precip[d, s] > 0
            if not wet.any():
                continue
            seeded = grid_seed[d, s] > 0
            # spread each seed up to max_distance_cells, constrained to the
            # wet mask (the anvil is connected to its core THROUGH the storm)
            reach = seeded & wet
            for _ in range(max_distance_cells):
                reach = ndimage.binary_dilation(reach, structure=connect8,
                                                mask=wet)
            objects, n_obj = ndimage.label(wet, structure=connect8)
            has_seed = ndimage.sum_labels(seeded.astype(float), objects,
                                          index=np.arange(1, n_obj + 1))
            sizes = ndimage.sum_labels(wet.astype(float), objects,
                                       index=np.arange(1, n_obj + 1))
            density = np.concatenate([[np.nan], has_seed / sizes])
            label_grid[d, s] = reach
            score_grid[d, s] = np.where(wet, density[objects], np.nan)

    precip = df["is_precip"].to_numpy()
    out["label"] = _from_grid(label_grid.astype(float), gidx, df).astype(bool) & precip
    out.loc[precip, "score"] = _from_grid(score_grid, gidx, df)[precip]
    return out


#: All methods under their display names (the demo/validation loop).
ALL_METHODS = {
    "threshold": threshold_classify,
    "gmm_cluster": cluster_classify,
    "random_forest": forest_classify,
    "storm_object": object_classify,
}


def classify_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every method; returns {name: label/score frame aligned to df}."""
    return {name: fn(df) for name, fn in ALL_METHODS.items()}
