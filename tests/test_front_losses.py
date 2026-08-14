"""Tests for the masked fractions skill score (FSS) loss.

These check the masked FSS loss (fronts/models/custom_losses.py:masked_fractions_skill_score)
against the existing (unmasked) FSS loss and against hand-computed answers so a reader can
confirm the implementation matches the formula without trusting any real training run:

- w == 1 everywhere      -> masked loss matches the stock fss_loss (after Keras-style reduction)
- hand-computable 4x4 case with mask_size=1 (no pooling) -> masked loss matches a plain-numpy
  computation of the formula
- all-masked batch (w == 0 everywhere) -> loss is finite (no NaN/Inf)
- masking out a disagreeing region strictly lowers the loss relative to leaving it unmasked
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
# test_front_dataset.py may have injected the numpy-only TF stub from
# tests/_stubs -- these tests execute real ops, so skip under the stub too.
if "_stubs" in getattr(tf, "__file__", "") or not hasattr(tf, "reduce_sum"):
    pytest.skip("requires a real TensorFlow (run in the fronts-tf env)",
                allow_module_level=True)

FRONTS = Path(__file__).resolve().parents[1] / "fronts"
if str(FRONTS) not in sys.path:
    sys.path.insert(0, str(FRONTS))

from models.custom_losses import fractions_skill_score, masked_fractions_skill_score  # noqa: E402


def _sigmoid(x, alpha=1.0, beta=0.5):
    return 1.0 / (1.0 + np.exp(-alpha * (x - beta)))


def test_unit_weight_matches_stock_fss():
    """w == 1 everywhere should reproduce the stock FSS loss (after Keras-style mean reduction).

    The stock ``fss_loss`` returns a per-pixel tensor (mean_squared_error only reduces the
    class axis), which Keras would average over when used as a real loss. The masked loss is a
    global weighted mean by construction, so we compare against ``tf.reduce_mean`` of the stock
    loss's un-reduced output -- these are algebraically equal when w == 1 and class_weights is
    None, since averaging a per-pixel mean-over-classes is the same as a global mean over every
    element.
    """
    rng = np.random.default_rng(0)
    n_classes = 3
    shape = (2, 8, 8, n_classes)

    logits_true = rng.random(shape).astype(np.float32)
    y_true_onehot = tf.one_hot(np.argmax(logits_true, axis=-1), depth=n_classes).numpy().astype(np.float32)
    y_pred = rng.random(shape).astype(np.float32)

    weight_channel = np.ones(shape[:-1] + (1,), dtype=np.float32)
    y_true_masked = np.concatenate([y_true_onehot, weight_channel], axis=-1)

    stock_loss_fn = fractions_skill_score(mask_size=(3, 3))
    masked_loss_fn = masked_fractions_skill_score(mask_size=(3, 3))

    stock_loss = tf.reduce_mean(stock_loss_fn(tf.constant(y_true_onehot), tf.constant(y_pred)))
    masked_loss = masked_loss_fn(tf.constant(y_true_masked), tf.constant(y_pred))

    assert np.allclose(stock_loss.numpy(), masked_loss.numpy(), atol=1e-6)


def test_hand_computed_quadrant_mask():
    """4x4, 1-class, mask_size=1 (no pooling): fractions equal the discretized values.

    Layout is a 4x4 grid split into four 2x2 quadrants. y_true and y_pred agree everywhere
    except in the bottom-right quadrant. Setting w=0 there should exactly match a plain-numpy
    evaluation of the masked-FSS formula restricted to the unmasked (agreeing) pixels.
    """
    alpha, beta = 1.0, 0.5

    y_true_raw = np.zeros((1, 4, 4, 1), dtype=np.float32)
    y_true_raw[0, 2:, 2:, 0] = 1.0  # bottom-right quadrant is class 1, rest is class 0

    y_pred_raw = y_true_raw.copy()
    y_pred_raw[0, 2:, 2:, 0] = 0.0  # model disagrees only in the bottom-right quadrant

    w = np.ones((1, 4, 4, 1), dtype=np.float32)
    w[0, 2:, 2:, 0] = 0.0  # mask out the disagreeing quadrant

    y_true_masked = np.concatenate([y_true_raw, w], axis=-1)

    masked_loss_fn = masked_fractions_skill_score(mask_size=(1, 1))
    masked_loss = masked_loss_fn(tf.constant(y_true_masked), tf.constant(y_pred_raw)).numpy()

    # Hand computation with plain numpy, mirroring the formula exactly.
    O_n = _sigmoid(y_true_raw, alpha, beta)
    M_n = _sigmoid(y_pred_raw, alpha, beta)
    weight = w  # cw == 1 (no class_weights)

    weight_sum = weight.sum() + tf.keras.backend.epsilon()
    MSE_n = np.sum(weight * (O_n - M_n) ** 2) / weight_sum
    MSE_ref = np.sum(weight * (O_n ** 2 + M_n ** 2)) / weight_sum
    FSS = 1 - MSE_n / (MSE_ref + tf.keras.backend.epsilon())
    expected_loss = 1 - FSS

    assert masked_loss == pytest.approx(expected_loss, abs=1e-6)

    # Since every unmasked pixel is a perfect agreement (O_n == M_n there), the expected loss
    # should be exactly 0 up to floating-point epsilon.
    assert expected_loss == pytest.approx(0.0, abs=1e-5)


def test_all_masked_batch_is_finite():
    """w == 0 everywhere must still return a finite loss (no NaN/Inf)."""
    rng = np.random.default_rng(1)
    n_classes = 3
    shape = (2, 8, 8, n_classes)

    y_true_onehot = tf.one_hot(rng.integers(0, n_classes, size=shape[:-1]), depth=n_classes).numpy().astype(np.float32)
    y_pred = rng.random(shape).astype(np.float32)
    w = np.zeros(shape[:-1] + (1,), dtype=np.float32)

    y_true_masked = np.concatenate([y_true_onehot, w], axis=-1)

    masked_loss_fn = masked_fractions_skill_score(mask_size=(3, 3))
    loss = masked_loss_fn(tf.constant(y_true_masked), tf.constant(y_pred)).numpy()

    assert np.isfinite(loss)


def test_masking_disagreement_lowers_loss():
    """Masking a region of disagreement should strictly lower the loss vs. leaving it unmasked."""
    y_true_raw = np.zeros((1, 8, 8, 1), dtype=np.float32)
    y_true_raw[0, 4:, 4:, 0] = 1.0

    y_pred_raw = y_true_raw.copy()
    y_pred_raw[0, 4:, 4:, 0] = 0.0  # disagreement confined to the bottom-right quadrant

    w_masked = np.ones((1, 8, 8, 1), dtype=np.float32)
    w_masked[0, 4:, 4:, 0] = 0.0

    w_unmasked = np.ones((1, 8, 8, 1), dtype=np.float32)

    y_true_masked = np.concatenate([y_true_raw, w_masked], axis=-1)
    y_true_unmasked = np.concatenate([y_true_raw, w_unmasked], axis=-1)

    masked_loss_fn = masked_fractions_skill_score(mask_size=(3, 3))

    loss_masked = masked_loss_fn(tf.constant(y_true_masked), tf.constant(y_pred_raw)).numpy()
    loss_unmasked = masked_loss_fn(tf.constant(y_true_unmasked), tf.constant(y_pred_raw)).numpy()

    assert loss_masked < loss_unmasked
