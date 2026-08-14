"""Tests for front_finder.predict (fake model; no TensorFlow needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "_stubs"))

from front_finder import config, predict  # noqa: E402


class FakeModel:
    """Deep-supervision-shaped model: predict returns a LIST of outputs."""

    def __init__(self, n_ch):
        self.input_shape = (None, 72, 144, 5, n_ch)

    def predict(self, x, verbose=0):
        n = x.shape[0]
        out = np.zeros((n, 72, 144, 5), dtype=np.float32)
        out[..., 0] = 0.6                       # "none"
        out[..., 1] = 0.4                       # "cold"
        return [out, out * 0.5]                 # primary head first


def test_predict_batch_unpads_and_drops_none():
    m = FakeModel(8)
    x = np.zeros((2, 72, 144, 5, 8), dtype=np.float32)
    p = predict.predict_batch(m, x)
    assert p.shape == (2, *config.GRID_SHAPE, len(config.FRONT_TYPES))
    # primary head used (0.4, not 0.2), none-class dropped
    assert np.allclose(p[..., 0], 0.4)
    assert np.allclose(p[..., 1:], 0.0)


def test_predict_batch_channel_mismatch_is_informative():
    m = FakeModel(8)
    x = np.zeros((1, 72, 144, 5, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="winds"):
        predict.predict_batch(m, x)


def test_unpad_inverts_pad():
    from front_finder.dataset import _pad

    a = np.arange(68 * 141, dtype=float).reshape(68, 141)
    padded = _pad(a[..., None])[None]           # (1, 72, 144, 1)
    back = predict._unpad(padded)[0, ..., 0]
    assert np.array_equal(back, a)
