"""dl_front evaluation + degradation tests (numpy-only, analytic)."""
from __future__ import annotations

import numpy as np
import pytest

from dl_front import config, dataset, degrade_sfc, evaluate

#: ``dataset.analysis_domain()``/``crop_domain()``/``region_mask()``
#: interpolate the land-fraction mask off disk, so even synthetic tests that
#: reach them need the data root's mask file.  Skipped, never failed, on
#: checkouts without a populated data root.
needs_land_mask = pytest.mark.skipif(
    not config.LAND_MASK_PATH.exists(),
    reason=f"land mask {config.LAND_MASK_PATH} not on disk "
           f"(set JPL_AIRS_DATA to a populated data root)")


def _uniform_metrics(n_classes=5):
    pm = evaluate.PaperMetrics(n_classes)
    pm.mask = np.ones(config.GRID_SHAPE, dtype=bool)   # analytic denominator
    return pm


@needs_land_mask
def test_confusion_and_accuracy_analytic():
    pm = _uniform_metrics(5)
    # truth: all none except one cold pixel; prediction: perfect
    y = np.full((1, *config.GRID_SHAPE), 4, dtype=np.uint8)
    y[0, 0, 0] = 0
    probs = np.zeros((1, *config.GRID_SHAPE, 5), dtype=np.float32)
    probs[..., 4] = 0.9
    probs[0, 0, 0] = [0.9, 0.025, 0.025, 0.025, 0.025]
    pm.update(probs, y)
    assert pm.accuracy()["all_categories"] == 1.0
    assert pm.accuracy()["front_no_front"] == 1.0
    total = 68 * 141
    frac = pm.cell_fractions()
    np.testing.assert_allclose(frac.loc["cold", "truth"], 100 / total)
    np.testing.assert_allclose(frac.loc["any", "predicted"], 100 / total)
    conf = pm.confusion_table(percent=False)
    assert conf.loc["cold", "cold"] == 1
    assert conf.loc["none", "none"] == total - 1


@needs_land_mask
def test_roc_endpoints():
    """factor 0 -> everything front (TPR=FPR=1); huge factor -> nothing."""
    pm = _uniform_metrics(5)
    rng = np.random.default_rng(1)
    y = (rng.random((2, *config.GRID_SHAPE)) < 0.1).astype(np.uint8) * 2
    y[y == 0] = 4
    y[y == 2] = 2
    probs = rng.dirichlet(np.ones(5), size=(2, *config.GRID_SHAPE)).astype("f4")
    pm.update(probs, y)
    pts = pm.roc_pr().set_index("factor")
    assert pts.loc[0.0, "tpr"] == 1.0 and pts.loc[0.0, "fpr"] == 1.0
    assert pts.loc[4096.0, "tpr"] < 0.05
    assert 0.0 <= pm.auc() <= 1.0


@needs_land_mask
def test_roc_auc_for_perfect_predictor():
    pm = _uniform_metrics(5)
    y = np.full((1, *config.GRID_SHAPE), 4, dtype=np.uint8)
    y[0, :10] = 0
    probs = np.zeros((1, *config.GRID_SHAPE, 5), dtype=np.float32)
    probs[..., 4] = 1.0
    probs[0, :10] = [1.0, 0, 0, 0, 0]
    pm.update(probs, y)
    np.testing.assert_allclose(pm.auc(), 1.0, atol=1e-6)


@needs_land_mask
def test_csi_counts_perfect_prediction():
    times = np.array([np.datetime64("2010-01-01T00")])
    cls = np.full((1, *config.GRID_SHAPE), 4, dtype=np.uint8)
    cls[0, 30, 60:70] = 0                              # a cold front line
    counts = evaluate.csi_counts(cls, cls, times, 5)
    scores = evaluate.csi_scores(counts)
    np.testing.assert_allclose(scores.loc[("cold", 0), "csi"], 1.0)
    assert np.isnan(scores.loc[("warm", 0), "csi"])    # no warm pixels anywhere


def test_degrade_severity_zero_is_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, *config.GRID_SHAPE, 5)).astype(np.float32)
    stats = {v: [2.0, 3.0] for v in config.SFC_VARS}
    out = degrade_sfc.degrade_x(x, rng, stats, severity=0.0)
    np.testing.assert_allclose(out, x, atol=1e-5)


def test_degrade_noise_statistics():
    """Additive T noise has the requested sigma in physical units; winds,
    SLP untouched; gap pixels imputed to 0 on T/q channels only."""
    rng = np.random.default_rng(7)
    x = np.zeros((40, *config.GRID_SHAPE, 5), dtype=np.float32)
    stats = {v: [0.0, 2.0] for v in config.SFC_VARS}
    out = degrade_sfc.degrade_x(x, rng, stats, severity=1.0)
    it = config.SFC_VARS.index("T2M")
    sigma_std = out[..., it].std() * 2.0               # back to physical K
    np.testing.assert_allclose(sigma_std, degrade_sfc.T2M_NOISE_SIGMA_K,
                               rtol=0.05)
    for v in ("SLP", "U10M", "V10M"):
        assert np.all(out[..., config.SFC_VARS.index(v)] == 0.0)

    vf = np.ones(config.GRID_SHAPE, dtype=np.float32)
    vf[:10] = 0.0                                      # hard gap band
    out = degrade_sfc.degrade_x(x, rng, stats, severity=1.0, vf=vf)
    assert np.all(out[:, :10, :, it] == 0.0)
    assert not np.all(out[:, 10:, :, it] == 0.0)


def test_q_noise_mean_preserving():
    rng = np.random.default_rng(11)
    x = np.zeros((200, *config.GRID_SHAPE, 5), dtype=np.float32)
    stats = {v: [1.0, 1.0] for v in config.SFC_VARS}   # physical q == 1
    out = degrade_sfc.degrade_x(x, rng, stats, severity=1.0)
    iq = config.SFC_VARS.index("QV2M")
    q_phys = out[..., iq] * 1.0 + 1.0
    np.testing.assert_allclose(q_phys.mean(), 1.0, rtol=1e-3)
