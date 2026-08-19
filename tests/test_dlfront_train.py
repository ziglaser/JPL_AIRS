"""dl_front.train loss-mask routing (user decision 2026-08-13).

The routing helper is pure numpy and runs everywhere; the end-to-end check
that ``run`` hands the routed mask to the tf.data pipeline needs a real
TensorFlow and skips cleanly in the numpy-only .venv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_front import config, dataset, train

#: ``dataset.analysis_domain()``/``crop_domain()``/``region_mask()``
#: interpolate the land-fraction mask off disk, so even synthetic tests that
#: reach them need the data root's mask file.  Skipped, never failed, on
#: checkouts without a populated data root.
needs_land_mask = pytest.mark.skipif(
    not config.LAND_MASK_PATH.exists(),
    reason=f"land mask {config.LAND_MASK_PATH} not on disk "
           f"(set JPL_AIRS_DATA to a populated data root)")


@needs_land_mask
def test_loss_mask_routing_6class():
    """Stage A trains on box+halo (crop); every gap-degraded stage on the
    analysis domain only (user decision 2026-08-13)."""
    name, m = train.loss_mask_for(6, "reanalysis")
    assert name == "crop_domain" and m.dtype == np.float32
    np.testing.assert_array_equal(m, dataset.crop_domain().astype(np.float32))

    for source in ("kriged-degraded", "kriged-airs"):
        name, m = train.loss_mask_for(6, source)
        assert name == "analysis_domain" and m.dtype == np.float32
        np.testing.assert_array_equal(
            m, dataset.analysis_domain().astype(np.float32))

    # legacy on-the-fly stage B (--degraded) is a gap-degraded stage too
    name, m = train.loss_mask_for(6, "reanalysis", degraded=True)
    assert name == "analysis_domain"


@needs_land_mask
def test_loss_mask_routing_5class_unchanged():
    """The 5-class paper replication keeps the Fig. 2 region mask for every
    source -- the domain decision applies to the 6-class track only."""
    for source in ("reanalysis", "kriged-degraded", "kriged-airs"):
        for degraded in (False, True):
            name, m = train.loss_mask_for(5, source, degraded)
            assert name == "region_mask"
            np.testing.assert_array_equal(m, dataset.region_mask())


def test_run_passes_routed_mask_and_records_it(tmp_path, monkeypatch):
    """``run`` must hand the routed mask to make_tf_dataset as the loss
    weights AND record its name in run_config.yaml (TF only)."""
    tf = pytest.importorskip("tensorflow")
    if not hasattr(tf, "keras"):  # tests/_stubs/tensorflow.py may be loaded
        pytest.skip("tensorflow is the tests/_stubs stand-in, not the real TF")
    import yaml

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(dataset, "load_norm_stats",
                        lambda: {v: [0.0, 1.0] for v in config.SFC_VARS})
    x = np.zeros((6, *config.GRID_SHAPE, 5), np.float16)
    y = np.zeros((6, *config.GRID_SHAPE), np.uint8)
    times = pd.date_range("2007-01-01", periods=6, freq="3h")
    monkeypatch.setattr(dataset, "stack_years",
                        lambda years, n_classes, stats: (x, y, times))

    class Sentinel(Exception):
        """Abort run() right after the dataset call we inspect."""

    captured = {}

    def fake_make_tf_dataset(x_, y_, n_classes, batch_size,
                             shuffle=True, weights=None):
        captured["weights"] = weights
        raise Sentinel

    monkeypatch.setattr(dataset, "make_tf_dataset", fake_make_tf_dataset)
    with pytest.raises(Sentinel):
        train.run("mask-probe", 6, smoke=True)

    np.testing.assert_array_equal(captured["weights"],
                                  dataset.crop_domain().astype(np.float32))
    rc = yaml.safe_load(
        (tmp_path / "models/mask-probe/run_config.yaml").read_text())
    assert rc["loss_mask"] == "crop_domain"
