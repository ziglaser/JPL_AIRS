"""Single-pass permutation importance (paper section 2e).

Pixel-based, POD-only, no neighborhood: "we... permute one input variable
at a time... and measure the resulting drop in POD... without any
neighborhood approaches" (paper section 2e). Model interaction is limited to
``predict.predict_batch`` so tests can substitute a fake model that never
touches TensorFlow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, dataset
from .predict import _unpad, predict_batch


def collect_arrays(sample_iter, n_max: int | None = None):
    """Stack an (x, y) sample iterator (e.g. ``dataset.year_samples``) into
    ``(X, Y)`` arrays with a leading sample axis."""
    xs, ys = [], []
    for i, (x, y) in enumerate(sample_iter):
        if n_max is not None and i >= n_max:
            break
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def pod_at(model, X: np.ndarray, Y: np.ndarray, threshold: float = 0.5,
          batch_size: int = 8) -> np.ndarray:
    """Pixel-based POD per front class: hits / (hits + misses).

    Truth is ``Y``'s one-hot front channels (index 1:5, "none" excluded),
    unpadded to match ``predict_batch``'s output grid; only pixels with
    positive weight (``Y``'s trailing channel) are scoreable.
    """
    n_front = len(config.FRONT_TYPES)
    hits = np.zeros(n_front)
    misses = np.zeros(n_front)
    Yu = _unpad(Y)
    truth = Yu[..., 1:1 + n_front] > 0.5
    scoreable = Yu[..., -1] > 0
    for start in range(0, X.shape[0], batch_size):
        batch = X[start:start + batch_size]
        probs = predict_batch(model, batch)                    # (b, lat, lon, n_front)
        t = truth[start:start + batch_size]
        w = scoreable[start:start + batch_size][..., None]
        hit = (probs >= threshold) & t & w
        miss = (probs < threshold) & t & w
        hits += hit.sum(axis=(0, 1, 2))
        misses += miss.sum(axis=(0, 1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return hits / (hits + misses)


def single_pass(model, X: np.ndarray, Y: np.ndarray, winds: bool, rng,
                threshold: float = 0.5) -> pd.DataFrame:
    """Single-pass permutation importance: baseline POD, then per-channel,
    per-variable, and per-level POD after permuting the sample axis.

    ``kind == "single"`` rows permute one (level, channel) pair at a time
    (INCLUDING the mask channel); ``kind == "variable"`` rows permute one
    channel across ALL levels at once (one shared permutation); ``kind ==
    "level"`` rows permute ALL channels at one level at once. Each shuffle
    uses exactly one fixed permutation drawn from ``rng``.
    ``delta_pod = baseline - permuted`` (positive = important).
    """
    names = dataset.channel_names(winds)
    baseline = pod_at(model, X, Y, threshold)
    n = X.shape[0]
    rows = []

    def _rows(kind, variable, level, permuted_pod):
        for fi, front in enumerate(config.FRONT_TYPES):
            rows.append({"kind": kind, "variable": variable, "level": level,
                        "front": front,
                        "delta_pod": float(baseline[fi] - permuted_pod[fi])})

    # single (level, channel) permutations
    for c, name in enumerate(names):
        for k, lev in enumerate(config.TARGET_LEVELS_HPA):
            Xp = X.copy()
            perm = rng.permutation(n)
            Xp[..., k, c] = Xp[perm][..., k, c]
            _rows("single", name, lev, pod_at(model, Xp, Y, threshold))

    # grouped: one variable, all levels together
    for c, name in enumerate(names):
        Xp = X.copy()
        perm = rng.permutation(n)
        Xp[..., :, c] = Xp[perm][..., :, c]
        _rows("variable", name, None, pod_at(model, Xp, Y, threshold))

    # grouped: one level, all variables together
    for k, lev in enumerate(config.TARGET_LEVELS_HPA):
        Xp = X.copy()
        perm = rng.permutation(n)
        Xp[..., k, :] = Xp[perm][..., k, :]
        _rows("level", None, lev, pod_at(model, Xp, Y, threshold))

    return pd.DataFrame(rows)[["kind", "variable", "level", "front", "delta_pod"]]
