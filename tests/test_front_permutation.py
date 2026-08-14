"""Tests for ``front_finder.permutation`` against a deterministic fake
model (no TensorFlow).

``permutation`` imports ``front_finder.dataset`` (for ``channel_names``),
which imports ``derive``, which imports the vendored ``fronts/utils``
modules that unconditionally ``import tensorflow`` at module scope (see
``test_front_dataset.py``'s docstring for the full story). We reuse the same
``tests/_stubs/tensorflow.py`` stub so this file never needs a real
TensorFlow install.

The fake model's predicted probability for the "cold" class is a
deterministic function of exactly ONE (level, channel) input slot:
``clip(X[..., 0, 0], 0, 1)`` (level index 0 = 1000 hPa, channel index 0 =
the first thermo variable). All other classes are always 0.

- Permuting that one slot destroys the (perfect, by construction) predictive
  relationship, so POD for "cold" must drop: ``delta_pod > 0``.
- Permuting any OTHER single (level, channel) slot leaves the fake model's
  output byte-for-byte unchanged (it never reads that slot), so
  ``delta_pod == 0`` EXACTLY -- including the trailing "mask" channel.
- The "variable" and "level" grouped-permutation kinds each produce rows
  (including a "variable" row for the important channel that also degrades
  POD, since permuting all levels together still touches level 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_STUBS = Path(__file__).resolve().parent / "_stubs"
if str(_STUBS) not in sys.path:                    # see module docstring
    sys.path.insert(0, str(_STUBS))

from front_finder import config, dataset, permutation  # noqa: E402

N_SAMPLES = 24
N_LEV = len(config.TARGET_LEVELS_HPA)
N_CH = len(dataset.channel_names(winds=True))
COLD_CLASS_IDX = 1   # CLASS_NAMES = ("none", "cold", "warm", "stationary", "occluded")


class FakeFrontModel:
    """predict() mimics deep-supervision output: a LIST whose first element
    is the primary head, shape (batch, 72, 144, 5).

    ``input_shape`` mimics a Keras model's attribute of the same name (only
    the last element -- the channel count -- is inspected by
    ``predict.predict_batch``'s channel-count sanity check).
    """

    input_shape = (None, *config.PADDED_SHAPE, len(config.TARGET_LEVELS_HPA), N_CH)

    def predict(self, x, verbose=0):
        batch = x.shape[0]
        out = np.zeros((batch, *config.PADDED_SHAPE, 5), dtype=np.float32)
        out[..., COLD_CLASS_IDX] = np.clip(x[..., 0, 0], 0.0, 1.0)
        out[..., 0] = 1.0 - out[..., COLD_CLASS_IDX]
        return [out]


def _build_xy(rng):
    """X (n, 72, 144, 5, C); Y (n, 72, 144, 6) with a perfect X[...,0,0] <->
    "cold" relationship baked in."""
    shape2d = config.PADDED_SHAPE
    is_cold = rng.integers(0, 2, size=(N_SAMPLES, *shape2d)).astype(np.float32)

    X = rng.uniform(0.0, 1.0, size=(N_SAMPLES, *shape2d, N_LEV, N_CH)).astype(np.float32)
    X[..., 0, 0] = is_cold  # the one slot the fake model reads

    n_classes = len(dataset.CLASS_NAMES)
    Y = np.zeros((N_SAMPLES, *shape2d, n_classes + 1), dtype=np.float32)
    Y[..., COLD_CLASS_IDX] = is_cold
    Y[..., 0] = 1.0 - is_cold
    # A handful of fixed truth pixels for the OTHER front classes (unrelated
    # to X) so their baseline POD is a well-defined 0 (the fake model never
    # predicts them) rather than 0/0 -- keeps delta_pod well-defined too.
    # (interior pixels only -- predict._unpad crops away the padded border,
    # so fixed truth pixels must land inside the unpadded 68x141 region).
    for class_idx, (li, lj) in zip((2, 3, 4), ((10, 10), (11, 11), (12, 12))):
        Y[:, li, lj, class_idx] = 1.0
        Y[:, li, lj, COLD_CLASS_IDX] = 0.0
        Y[:, li, lj, 0] = 0.0
    Y[..., -1] = 1.0  # weight: every pixel scoreable
    return X, Y


# --------------------------------------------------------------------------- #
# collect_arrays
# --------------------------------------------------------------------------- #

def test_collect_arrays_stacks_iterator_with_n_max():
    rng = np.random.default_rng(0)
    X, Y = _build_xy(rng)
    sample_iter = ((X[i], Y[i]) for i in range(N_SAMPLES))

    Xc, Yc = permutation.collect_arrays(sample_iter, n_max=5)
    assert Xc.shape == (5, *X.shape[1:])
    assert Yc.shape == (5, *Y.shape[1:])
    np.testing.assert_array_equal(Xc, X[:5])


# --------------------------------------------------------------------------- #
# pod_at
# --------------------------------------------------------------------------- #

def test_pod_at_is_perfect_for_the_baked_in_relationship():
    rng = np.random.default_rng(1)
    X, Y = _build_xy(rng)
    model = FakeFrontModel()

    pod = permutation.pod_at(model, X, Y, threshold=0.5)
    assert pod.shape == (len(config.FRONT_TYPES),)
    cold_i = config.FRONT_TYPES.index("cold")
    assert pod[cold_i] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# single_pass
# --------------------------------------------------------------------------- #

def test_single_pass_important_channel_degrades_pod():
    rng = np.random.default_rng(2)
    X, Y = _build_xy(rng)
    model = FakeFrontModel()
    names = dataset.channel_names(winds=True)

    df = permutation.single_pass(model, X, Y, winds=True,
                                 rng=np.random.default_rng(42), threshold=0.5)

    important_row = df[(df["kind"] == "single") & (df["variable"] == names[0])
                       & (df["level"] == config.TARGET_LEVELS_HPA[0])
                       & (df["front"] == "cold")]
    assert len(important_row) == 1
    assert important_row["delta_pod"].iloc[0] > 0.0


def test_single_pass_unused_channel_gives_exact_zero_delta():
    rng = np.random.default_rng(3)
    X, Y = _build_xy(rng)
    model = FakeFrontModel()
    names = dataset.channel_names(winds=True)
    unused = names[1]  # anything other than names[0] is never read by the fake model

    df = permutation.single_pass(model, X, Y, winds=True,
                                 rng=np.random.default_rng(7), threshold=0.5)

    unused_rows = df[(df["kind"] == "single") & (df["variable"] == unused)]
    assert len(unused_rows) == len(config.FRONT_TYPES) * N_LEV
    assert (unused_rows["delta_pod"] == 0.0).all()


def test_single_pass_includes_mask_channel_row():
    rng = np.random.default_rng(4)
    X, Y = _build_xy(rng)
    model = FakeFrontModel()
    names = dataset.channel_names(winds=True)
    assert names[-1] == "mask"

    df = permutation.single_pass(model, X, Y, winds=True,
                                 rng=np.random.default_rng(11), threshold=0.5)

    mask_rows = df[(df["kind"] == "single") & (df["variable"] == "mask")]
    assert len(mask_rows) == len(config.FRONT_TYPES) * N_LEV
    assert (mask_rows["delta_pod"] == 0.0).all()


def test_single_pass_has_grouped_variable_and_level_rows():
    rng = np.random.default_rng(5)
    X, Y = _build_xy(rng)
    model = FakeFrontModel()
    names = dataset.channel_names(winds=True)

    df = permutation.single_pass(model, X, Y, winds=True,
                                 rng=np.random.default_rng(13), threshold=0.5)

    variable_rows = df[df["kind"] == "variable"]
    level_rows = df[df["kind"] == "level"]
    assert len(variable_rows) == len(names) * len(config.FRONT_TYPES)
    assert len(level_rows) == N_LEV * len(config.FRONT_TYPES)
    assert variable_rows["level"].isna().all()
    assert level_rows["variable"].isna().all()

    # permuting the important variable's channel (across all levels, which
    # includes level 0) must still degrade "cold" POD.
    important = variable_rows[(variable_rows["variable"] == names[0])
                              & (variable_rows["front"] == "cold")]
    assert important["delta_pod"].iloc[0] > 0.0
