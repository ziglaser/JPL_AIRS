"""Tests for ``front_finder.materialize`` (per-year sample shards).

Reuses the synthetic MERRA-2/CODSUS builders from test_front_dataset (same
tf stub caveat -- nothing here touches real TensorFlow; the tf.data reader
over shards is exercised by the training smoke run instead).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from test_front_dataset import (  # noqa: F401  (patched_dirs is a fixture)
    _write_codsus_year, _write_merra2_day, patched_dirs)

from front_finder import config, dataset, materialize


@pytest.fixture
def shard_setup(patched_dirs, tmp_path, monkeypatch):
    """One synthetic year (2 days, 1 missing day) + patched SHARD_DIR."""
    monkeypatch.setattr(config, "SHARD_DIR", tmp_path / "shards")
    year = 2003
    rng = np.random.default_rng(11)
    day1, day2 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-01-02")
    times = (pd.date_range(day1, periods=8, freq="3h")
             .append(pd.date_range(day2, periods=1, freq="3h")))
    _write_merra2_day(day1, rng)                    # day2 file deliberately absent
    _write_codsus_year(year, times, masked=True)
    stats = dataset.compute_norm_stats(years=[year], step_days=1,
                                       path=tmp_path / "stats.json")
    return year, stats


def test_count_matches_generator(shard_setup):
    year, stats = shard_setup
    n_gen = sum(1 for _ in dataset.year_samples(year, stats, winds=True))
    assert materialize.count_samples(year) == n_gen == 8


def test_materialize_roundtrip_equals_generator(shard_setup):
    year, stats = shard_setup
    n = materialize.materialize_year(year, stats=stats)
    assert n == 8 and materialize.year_done(year)

    p = materialize._year_paths(year)
    x = np.load(p["x"], mmap_mode="r")
    y = np.load(p["y"], mmap_mode="r")
    t = np.load(p["times"])
    meta = json.loads(p["meta"].read_text())
    assert meta["n"] == n and meta["channels"] == list(
        dataset.channel_names(winds=True))
    assert x.shape == (n, *config.PADDED_SHAPE, 5,
                       len(dataset.channel_names(True)))

    for i, (xg, yg, tg) in enumerate(
            dataset.year_samples(year, stats, winds=True, return_times=True)):
        np.testing.assert_array_equal(x[i], xg)
        np.testing.assert_array_equal(y[i], yg)
        assert t[i] == np.datetime64(tg)


def test_materialize_skips_finished_year(shard_setup):
    year, stats = shard_setup
    materialize.materialize_year(year, stats=stats)
    mtime = materialize._year_paths(year)["x"].stat().st_mtime_ns
    materialize.materialize_year(year, stats=stats)   # no-op: already done
    assert materialize._year_paths(year)["x"].stat().st_mtime_ns == mtime
    assert dataset.shards_exist([year])
    assert not dataset.shards_exist([year, year + 1])


def test_thermo_channel_selection_matches_thermo_generator(shard_setup):
    """Slicing thermo channels out of wind shards == generating winds=False."""
    year, stats = shard_setup
    materialize.materialize_year(year, stats=stats)
    x_wind = np.load(materialize._year_paths(year)["x"], mmap_mode="r")
    idx = dataset._thermo_channel_idx()
    for i, (xg, _) in enumerate(dataset.year_samples(year, stats, winds=False)):
        np.testing.assert_array_equal(x_wind[i][..., idx], xg)
