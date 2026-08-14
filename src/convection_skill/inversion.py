"""Event-curve inversion: turn a NON-MONOTONE predictor into a rate-ordered
index whose Gini is apples-to-apples with monotone predictors.

The problem
-----------
The Gini/CAP coefficient is a RANK statistic: it measures how well sorting on
the predictor concentrates events at one end. A predictor whose event-rate
curve is non-monotone (A2: heavy-rain probability peaks at weak nonzero CIN
and drops at zero cap; A4: humidity) can carry real information yet score
Gini ~ 0, because high event rates sit in the MIDDLE of the ranking. Comparing
its raw Gini against CAPE's is not apples-to-apples.

The mathematically right inversion
----------------------------------
For a binary event Y and predictor X, consider any transformed index T(X).
The detection/CAP curve of T depends only on the ordering T induces, and the
ordering that maximizes the whole curve (hence the Gini, and the AUC =
(Gini+1)/2) is the one induced by the conditional event probability

    eta(x) = P(Y = 1 | X = x),

equivalently by the likelihood ratio f(x|Y=1)/f(x|Y=0), which eta orders
monotonically (Bayes). This is the Neyman-Pearson lemma applied pointwise:
the risk score is the optimal univariate screening index (McIntosh & Pepe
2002, Biometrics 58:657-664, "the risk score is the optimal combination";
Pepe 2003, *The Statistical Evaluation of Medical Tests*, ch. 4; Engelmann,
Hayden & Tasche 2003 connect CAP/Gini and AUC in exactly this construction).
Two consequences anchor the comparison:

1. If X is already monotone in eta (CAPE), Gini(T(X)) == Gini(X) -- the
   inversion is a no-op for monotone predictors (rank invariance), so
   inverted and raw Ginis live on the same scale.
2. For any X, Gini(eta(X)) >= Gini(m(X)) for every measurable m: the inverted
   Gini is the predictor's FULL ordering skill, an upper bound achieved by
   the true eta.

Two estimators of eta (both kept, per the design decision)
----------------------------------------------------------
- :func:`quadratic_logistic_index` (THEORETICAL): logit eta(x) = b0 + b1 x
  + b2 x^2. This is not an arbitrary curve: if the class-conditional
  densities f(x|Y=0), f(x|Y=1) are Gaussian with different means AND
  variances, the exact log-likelihood-ratio -- the Bayes-optimal index -- is
  quadratic in x (the 1-D case of quadratic discriminant analysis; Anderson
  1958). A quadratic logit is therefore the minimal parametric family that
  can represent a single-peaked ("optimal cap" / conditional-instability
  window) response while nesting the monotone case (b2 = 0). Three
  parameters, essentially no overfitting.
- :func:`binned_rate_index` (NUMERICAL): the empirical event-rate curve
  itself -- equal-count bins of X, per-bin event rate, piecewise-linear
  interpolation between bin centers. Nonparametric, tracks any shape the
  quadratic cannot (plateaus, asymmetry), at the price of estimation noise
  in rare-event bins.

Honesty: cross-fitting
----------------------
Estimating eta and scoring its Gini on the SAME rows is in-sample selection:
the binned estimator would partly fit noise and inflate the Gini. Both
estimators are therefore CROSS-FITTED (:func:`crossfit_index`): days are
split into ``n_folds`` CONTIGUOUS blocks (respecting the same synoptic
autocorrelation the day-block bootstrap does), and each row's index comes
from a fit that never saw its fold (out-of-fold prediction pooling, as in
cvAUC / double-ML cross-fitting, Chernozhukov et al. 2018). A pure-noise
predictor then scores ~0 by construction (tested). The downstream bootstrap
treats the cross-fitted index as a fixed column -- CI width does not include
transform-estimation noise; the fold-stability diagnostic returned alongside
makes that visible instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

INVERSION_METHODS: tuple[str, ...] = ("quadratic", "binned")


# --------------------------------------------------------------------------- #
# The two eta estimators (fit on train, evaluate anywhere)
# --------------------------------------------------------------------------- #
#: A value holding at least this fraction of the training sample is treated as
#: a point mass (atom) and gets its own likelihood-ratio offset (see below).
ATOM_MIN_FRACTION: float = 0.05


def quadratic_logistic_index(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
) -> np.ndarray:
    """THEORETICAL inversion: MLE of logit eta = b0 + b1 z + b2 z^2 [+ b3 at
    the atom].

    z is x standardized by the training moments (conditioning only). Exact
    Bayes index under Gaussian class-conditionals with unequal variances;
    nests the monotone case at b2 = 0.

    Zero inflation: meteorological predictors like CIN carry a large point
    mass (CIN == 0 for uncapped columns) at which the true eta is
    DISCONTINUOUS -- no smooth curve can represent it (on 2019 data the plain
    quadratic scored 0.02 n.s. against the binned estimator's 0.16 for this
    reason). Under MIXED class-conditionals, f(x|Y) = w_Y * delta(x - m) +
    (1 - w_Y) * g_Y(x), the exact log-likelihood-ratio is still quadratic on
    the continuous part but jumps by log(w_1/w_0) at the atom -- an indicator
    column 1[x == m], added whenever one value holds >= ATOM_MIN_FRACTION of
    the training sample. Degenerate training data (one class) returns a
    constant (no ordering claimed).
    """
    from sklearn.linear_model import LogisticRegression

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=bool)
    if y_train.all() or not y_train.any():
        return np.full(np.asarray(x_eval, dtype=float).shape, float(y_train.mean()))
    mu, sd = float(np.mean(x_train)), float(np.std(x_train))
    sd = sd if sd > 0 else 1.0
    values, freq = np.unique(x_train, return_counts=True)
    top = np.argmax(freq)
    atom = values[top] if freq[top] >= ATOM_MIN_FRACTION * x_train.size else None

    def design(x):
        x = np.asarray(x, dtype=float)
        z = (x - mu) / sd
        cols = [z, z * z]
        if atom is not None:
            cols.append((x == atom).astype(float))
        return np.column_stack(cols)

    model = LogisticRegression(penalty=None, max_iter=2000)
    model.fit(design(x_train), y_train)
    return model.predict_proba(design(x_eval))[:, 1]


def binned_rate_index(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """NUMERICAL inversion: the empirical event-rate curve as the index.

    Equal-count bins from the training quantiles (duplicate edges from point
    masses -- e.g. CIN == 0 -- are collapsed), per-bin event rate, then
    piecewise-linear interpolation between bin centers (medians), clamped at
    the edge rates. Nonparametric; noisier than the quadratic where events
    are rare.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    edges = np.unique(np.quantile(x_train, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:  # a (near-)constant predictor carries no ordering
        return np.full(np.asarray(x_eval, dtype=float).shape, float(y_train.mean()))
    which = np.clip(np.searchsorted(edges, x_train, side="right") - 1,
                    0, edges.size - 2)
    centers = np.array([np.median(x_train[which == b])
                        for b in range(edges.size - 1)])
    rates = np.array([y_train[which == b].mean() if (which == b).any() else np.nan
                      for b in range(edges.size - 1)])
    ok = np.isfinite(centers) & np.isfinite(rates)
    return np.interp(np.asarray(x_eval, dtype=float), centers[ok], rates[ok])


# --------------------------------------------------------------------------- #
# Cross-fitting over contiguous day blocks
# --------------------------------------------------------------------------- #
def _contiguous_day_folds(day: np.ndarray, n_folds: int) -> np.ndarray:
    """Fold id per row: unique days, time-ordered, cut into contiguous chunks."""
    days = np.unique(day)
    chunk_of_day = (np.arange(days.size) * n_folds) // days.size
    lookup = dict(zip(days, chunk_of_day))
    return np.array([lookup[d] for d in day])


def crossfit_index(
    predictor: np.ndarray,
    flags: np.ndarray,
    day: np.ndarray,
    method: str = "quadratic",
    n_folds: int = 5,
    n_bins: int = 20,
    return_diagnostics: bool = False,
):
    """Out-of-fold eta(X) for every row: the honest rate-ordered index.

    Rows with non-finite predictor/flags get NaN (dropped pairwise downstream,
    like any predictor column). ``method``: "quadratic" (theoretical) or
    "binned" (numerical). With ``return_diagnostics``, also returns a dict
    with the per-fold transforms evaluated on a common x-grid and their
    pairwise Spearman fold stability (low stability = the transform is
    chasing noise; treat its Gini accordingly).
    """
    if method not in INVERSION_METHODS:
        raise ValueError(f"method must be one of {INVERSION_METHODS}, got {method!r}")
    fit = (quadratic_logistic_index if method == "quadratic"
           else lambda xt, yt, xe: binned_rate_index(xt, yt, xe, n_bins=n_bins))

    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags, dtype=float)
    valid = np.isfinite(predictor) & np.isfinite(flags)
    index = np.full(predictor.shape, np.nan)

    x, y = predictor[valid], flags[valid] > 0
    fold = _contiguous_day_folds(np.asarray(day)[valid], n_folds)
    out = np.full(x.shape, np.nan)
    grid = np.quantile(x, np.linspace(0.005, 0.995, 101))
    fold_curves = []
    for f in range(n_folds):
        hold = fold == f
        if not hold.any() or hold.all():
            continue
        out[hold] = fit(x[~hold], y[~hold], x[hold])
        fold_curves.append(fit(x[~hold], y[~hold], grid))
    index[valid] = out

    if not return_diagnostics:
        return index
    from scipy.stats import spearmanr

    stability = [spearmanr(a, b).statistic
                 for i, a in enumerate(fold_curves) for b in fold_curves[i + 1:]]
    return index, {"grid": grid, "fold_curves": np.array(fold_curves),
                   "fold_stability_spearman": float(np.mean(stability))
                                              if stability else np.nan}


def plot_inversion(predictor, flags, day, save_path, name: str = "",
                   n_folds: int = 5) -> None:
    """Diagnostic: empirical event-rate bins + both cross-fitted transforms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, method in zip(axes, INVERSION_METHODS):
        _, diag = crossfit_index(predictor, flags, day, method=method,
                                 n_folds=n_folds, return_diagnostics=True)
        for i, curve in enumerate(diag["fold_curves"]):
            ax.plot(diag["grid"], curve, lw=1.0, alpha=0.7,
                    label="per-fold transform" if i == 0 else None)
        ax.set_title(f"{method} (fold stability rho = "
                     f"{diag['fold_stability_spearman']:.2f})")
        ax.set_xlabel(name or "predictor")
        ax.set_ylabel("estimated P(event | x)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Event-curve inversion diagnostics: {name}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
