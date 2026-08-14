"""Tests for the NOAA-XML (dryline) label path in ``front_finder``.

Synthetic-file based, like test_front_labels.py: a tiny NOAA-schema netCDF
(NaN fill, six front_type channels incl. dryline) is written to tmp_path and
loaded through labels.load_noaa / load_fronts.  The class wiring (make_y with
five front types, model class count) is exercised by monkeypatching
config.FRONT_TYPES + dataset.CLASS_NAMES exactly as JPL_FRONT_LABELS=noaa
would set them at import; the env-var plumbing itself is checked in a
subprocess so this test run's own imports stay untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from front_finder import config, dataset, labels

NOAA_TYPES = ("cold", "warm", "stationary", "occluded", "dryline")
NOAA_CLASS_NAMES = ("none",) + NOAA_TYPES


def _write_noaa_year(dirpath: Path, year: int, n_time: int = 2) -> np.ndarray:
    """A minimal NOAA-schema file: full label grid, NaN outside a box.

    Returns the raw (time, front, lat, lon) float array for reference.
    Dryline pixel at (5, 5); cold at (5, 6); NaN column at lon 0.
    """
    front_type = list(NOAA_TYPES) + ["none"]
    fr = np.zeros((n_time, len(front_type), *config.GRID_SHAPE),
                  dtype=np.float32)
    fr[:, front_type.index("none")] = 1.0
    fr[:, front_type.index("dryline"), 5, 5] = 1.0
    fr[:, front_type.index("none"), 5, 5] = 0.0
    fr[:, front_type.index("cold"), 5, 6] = 1.0
    fr[:, front_type.index("none"), 5, 6] = 0.0
    fr[:, :, :, 0] = np.nan                       # rasterizer domain edge
    ds = xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fr)},
        coords={
            "time": pd.date_range(f"{year}-01-01", periods=n_time, freq="3h"),
            "front_type": ("front", front_type),
            "lat": np.arange(10.0, 78.0),
            "lon": np.arange(-171.0, -30.0),
        })
    dirpath.mkdir(parents=True, exist_ok=True)
    # Post-reorg layout: year files live in a {width}wide/ subdirectory.
    (dirpath / "1wide").mkdir(exist_ok=True)
    ds.to_netcdf(dirpath / "1wide" / f"noaa_fronts_merra2-1deg_1wide_{year}.nc")
    return fr


def _write_region_mask(path: Path) -> np.ndarray:
    """A mask that is 1 everywhere except the last longitude column."""
    mask = np.ones(config.GRID_SHAPE, dtype=np.float32)
    mask[:, -1] = 0.0
    xr.Dataset({"codsus_mask": (("lat", "lon"), mask)}).to_netcdf(path)
    return mask


@pytest.fixture()
def noaa_env(tmp_path, monkeypatch):
    """Synthetic NOAA labels + region mask, config paths pointed at them."""
    _write_noaa_year(tmp_path, 2010)
    _write_region_mask(tmp_path / "mask.nc")
    monkeypatch.setattr(config, "NOAA_LABELS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGION_MASK_PATH", tmp_path / "mask.nc")
    return tmp_path


# --------------------------------------------------------------------------- #
# load_noaa / load_fronts
# --------------------------------------------------------------------------- #

def test_load_noaa_converts_nan_to_fill(noaa_env):
    with labels.load_noaa(2010) as ds:
        fr = ds["fronts"].values
        assert fr.dtype == np.uint8, "fill conversion must produce bytes"
        assert not np.isnan(fr.astype(float)).any()
        assert (fr[:, :, :, 0] == config.LABEL_FILL).all(), \
            "NaN pixels must become LABEL_FILL"
        # valid_mask now sees CODSUS semantics: NaN column invalid, rest valid
        vm = labels.valid_mask(ds).values
        assert not vm[:, :, 0].any() and vm[:, :, 1:].all()


def test_load_noaa_front_stack_has_dryline(noaa_env, monkeypatch):
    monkeypatch.setattr(config, "FRONT_TYPES", NOAA_TYPES)
    with labels.load_noaa(2010) as ds:
        fs = labels.front_stack(ds)
        assert list(fs["front"].values) == list(NOAA_TYPES)
        dl = fs.sel(front="dryline").values
        assert dl[:, 5, 5].all() and dl.sum() == dl.shape[0]


def test_load_noaa_masked_applies_region_mask(noaa_env):
    with labels.load_noaa(2010, masked=True) as ds:
        fr = ds["fronts"].values
        assert (fr[:, :, :, -1] == config.LABEL_FILL).all(), \
            "mask==0 pixels must become fill"
        assert not labels.valid_mask(ds).values[:, :, -1].any()


def test_load_fronts_dispatches_on_label_source(noaa_env, monkeypatch):
    monkeypatch.setattr(config, "LABEL_SOURCE", "noaa")
    with labels.load_fronts(2010) as ds:
        assert "dryline" in [str(s) for s in ds["front_type"].values]


# --------------------------------------------------------------------------- #
# five-type class wiring (make_y / model bookkeeping)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def noaa_classes(monkeypatch):
    monkeypatch.setattr(config, "LABEL_SOURCE", "noaa")
    monkeypatch.setattr(config, "FRONT_TYPES", NOAA_TYPES)
    monkeypatch.setattr(dataset, "CLASS_NAMES", NOAA_CLASS_NAMES)


def test_make_y_dryline_channel(noaa_classes):
    fronts_t = np.zeros((len(NOAA_TYPES), *config.GRID_SHAPE), dtype=bool)
    fronts_t[NOAA_TYPES.index("dryline"), 5, 5] = True
    fronts_t[NOAA_TYPES.index("cold"), 5, 6] = True
    valid_t = np.ones(config.GRID_SHAPE, dtype=bool)
    y = dataset.make_y(fronts_t, valid_t)
    assert y.shape == (*config.PADDED_SHAPE, len(NOAA_CLASS_NAMES) + 1)
    # _pad offsets: lat +2, lon +1
    onehot = y[..., :-1]
    assert onehot[5 + 2, 5 + 1].argmax() == NOAA_CLASS_NAMES.index("dryline")
    assert onehot[5 + 2, 6 + 1].argmax() == NOAA_CLASS_NAMES.index("cold")
    assert onehot[0, 0].argmax() == 0                      # none elsewhere


def test_make_y_overlap_priority_unchanged(noaa_classes):
    """Overlaps still resolve to the FIRST True in FRONT_TYPES order."""
    fronts_t = np.zeros((len(NOAA_TYPES), *config.GRID_SHAPE), dtype=bool)
    fronts_t[NOAA_TYPES.index("cold"), 8, 8] = True
    fronts_t[NOAA_TYPES.index("dryline"), 8, 8] = True     # loses to cold
    y = dataset.make_y(fronts_t, np.ones(config.GRID_SHAPE, dtype=bool))
    assert y[8 + 2, 8 + 1, :-1].argmax() == NOAA_CLASS_NAMES.index("cold")


# --------------------------------------------------------------------------- #
# env-var plumbing (real import path, isolated in a subprocess)
# --------------------------------------------------------------------------- #

def test_env_var_switches_config():
    src = Path(__file__).resolve().parents[1] / "src"
    code = ("from front_finder import config, dataset\n"
            "assert config.LABEL_SOURCE == 'noaa'\n"
            "assert config.FRONT_TYPES[-1] == 'dryline'\n"
            "assert dataset.CLASS_NAMES == "
            "('none', 'cold', 'warm', 'stationary', 'occluded', 'dryline')\n"
            "assert config.SHARD_DIR.name == 'shards_noaa'\n"
            "assert config.PRETRAIN_TRAIN_YEARS[0] == 2006\n"
            "print('ok')\n")
    env = dict(os.environ, JPL_FRONT_LABELS="noaa",
               PYTHONPATH=str(src) + os.pathsep + str(Path(__file__).parent / "_stubs"))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_env_var_rejects_unknown_source():
    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, JPL_FRONT_LABELS="frontfinder", PYTHONPATH=str(src))
    out = subprocess.run([sys.executable, "-c", "import front_finder.config"],
                         env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "JPL_FRONT_LABELS" in out.stderr
