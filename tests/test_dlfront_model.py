"""DL-FRONT model/loss tests (analytic; requires real TF -> fronts-tf env)."""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
if "_stubs" in getattr(tf, "__file__", "") or not hasattr(tf, "reduce_sum"):
    pytest.skip("requires a real TensorFlow (run in the fronts-tf env)",
                allow_module_level=True)

from dl_front import config, model as model_mod  # noqa: E402


def test_architecture_matches_paper():
    """Section 3.1: 3x (5x5 conv, 80 filters, ReLU, spatial dropout) +
    (5x5 conv, n_cls filters, softmax); parameter counts are analytic."""
    m = model_mod.build(5)
    conv_params = [(5 * 5 * 5 + 1) * 80,      # (k*k*in + bias) * filters
                   (5 * 5 * 80 + 1) * 80,
                   (5 * 5 * 80 + 1) * 80,
                   (5 * 5 * 80 + 1) * 5]
    assert m.count_params() == sum(conv_params)
    convs = [l for l in m.layers if "conv" in l.name]
    assert [l.count_params() for l in convs] == conv_params
    assert convs[-1].activation.__name__ == "softmax"
    drops = [l for l in m.layers if "drop" in l.name]
    assert len(drops) == 3 and all(l.rate == 0.5 for l in drops)


def test_output_shape_and_simplex():
    m = model_mod.build(6)
    x = np.random.default_rng(0).normal(size=(2, 68, 141, 5)).astype("f4")
    p = m.predict(x, verbose=0)
    assert p.shape == (2, 68, 141, 6)
    np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-5)


def test_weighted_cce_analytic():
    """Hand-computed Eq. 4 on a 2-pixel, 2-class toy."""
    loss = model_mod.make_loss([1.0, 0.35])
    # pixel A: class 0, p0 = 0.8, weight 1; pixel B: class 1, p1 = 0.5, weight 1
    y_true = np.array([[[[1, 0, 1.0], [0, 1, 1.0]]]], dtype="f4")
    y_pred = np.array([[[[0.8, 0.2], [0.5, 0.5]]]], dtype="f4")
    expected = (1.0 * -np.log(0.8) + 0.35 * -np.log(0.5)) / 2.0
    np.testing.assert_allclose(float(loss(tf.constant(y_true),
                                          tf.constant(y_pred))),
                               expected, rtol=1e-6)


def test_weighted_cce_ignores_zero_weight_pixels():
    loss = model_mod.make_loss([1.0, 1.0])
    y_true = np.array([[[[1, 0, 1.0], [0, 1, 0.0]]]], dtype="f4")  # B masked
    y_pred = np.array([[[[0.9, 0.1], [0.01, 0.99]]]], dtype="f4")
    np.testing.assert_allclose(float(loss(tf.constant(y_true),
                                          tf.constant(y_pred))),
                               -np.log(0.9), rtol=1e-5)


def test_masked_accuracy():
    y_true = np.array([[[[1, 0, 1.0], [0, 1, 1.0], [0, 1, 0.0]]]], dtype="f4")
    y_pred = np.array([[[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]]], dtype="f4")
    # inside mask: 1 hit of 2; the masked-out wrong pixel must not count
    np.testing.assert_allclose(
        float(model_mod.masked_accuracy(tf.constant(y_true),
                                        tf.constant(y_pred))), 0.5)


def test_observed_min_fraction_matches_ingest():
    """degrade_sfc duplicates the constant to avoid the fronts/ TF chain."""
    import sys

    from dl_front import degrade_sfc
    from front_finder import config as fd_config

    sys.path.insert(0, str(fd_config.FRONTS_REPO))
    from front_finder.ingest_hysplit import OBSERVED_MIN_FRACTION

    assert degrade_sfc.OBSERVED_MIN_FRACTION == OBSERVED_MIN_FRACTION


def test_class_weight_vector():
    assert model_mod.class_weight_vector(5) == [1, 1, 1, 1, 0.35]
    assert model_mod.class_weight_vector(6) == [1, 1, 1, 1, 1, 0.35]
