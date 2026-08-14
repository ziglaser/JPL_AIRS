"""Tests for ``front_finder.nfa_baseline`` (the classical 1965 NFA baseline,
workplan section 4.2).

Follows the house style of ``test_front_ingest.py``/``test_front_dataset.py``:
the ``tests/_stubs`` trick puts a minimal ``tensorflow`` stand-in on
``sys.path`` before anything imports ``front_finder.nfa_baseline`` (which
transitively imports ``fronts/nfa/methods.py`` -> ``fronts/utils/data_utils.py``
-> real ``tensorflow``), so the whole suite runs on a plain ``python3`` with
no TensorFlow installed. Fullgrid/CODSUS fixtures are reused from
``test_front_ingest`` (same synthetic-fullgrid geometry, same file-naming/
schema conventions) rather than re-invented here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_STUBS = Path(__file__).resolve().parent / "_stubs"
if str(_STUBS) not in sys.path:                    # see test_front_dataset.py
    sys.path.insert(0, str(_STUBS))

from front_finder import config, nfa_baseline as nb  # noqa: E402

from test_front_ingest import (  # noqa: E402  (reuse the synthetic fullgrid fixtures)
    _write_codsus_2018,
    _write_fullgrid,
)
from front_finder import ingest_hysplit as ih  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. front_mask on a synthetic single-ridge field
# --------------------------------------------------------------------------- #

LAT = np.arange(10.0, 78.0, 1.0)   # 68: the label-grid latitudes
LON = np.arange(-171.0, -30.0, 1.0)  # 141: the label-grid longitudes
assert (LAT.size, LON.size) == config.GRID_SHAPE
BAND_CENTER = 34  # row index of the gradient maximum


def _ridge_theta() -> np.ndarray:
    """A sharp, purely-meridional gradient band: constant along lon, a single
    logistic-shaped ridge in |d theta/d lat| centered on row ``BAND_CENTER``.
    """
    row = np.arange(config.GRID_SHAPE[0])[:, None]
    theta_row = 300.0 + 5.0 * np.tanh((row - BAND_CENTER) / 1.5)
    return np.broadcast_to(theta_row, config.GRID_SHAPE).copy()


def test_front_mask_marks_band_near_center_and_covers_most_of_its_length():
    theta = _ridge_theta()
    with warnings.catch_warnings():
        # far-field rows have ~0 gradient magnitude -> 0/0 in the vendored
        # unit-gradient-vector division; harmless (min_gradient screens the
        # result out), but noisy under pytest's default warning filters.
        warnings.simplefilter("ignore", RuntimeWarning)
        mask = nb.front_mask(theta, LAT, LON)

    assert mask.dtype == bool
    assert mask.shape == config.GRID_SHAPE

    detected_rows = np.flatnonzero(mask.any(axis=1))
    assert detected_rows.size > 0
    # analytic: TFP's zero crossing (the ridge of |grad theta|) sits within a
    # couple of grid rows of the true center, given the forward-difference
    # stencil's half-cell location shift.
    assert np.all(np.abs(detected_rows - BAND_CENTER) <= 3)

    # far from the band (many e-folding lengths away) nothing is detected.
    far_rows = np.r_[0:BAND_CENTER - 10, BAND_CENTER + 10:config.GRID_SHAPE[0]]
    assert not mask[far_rows].any()

    # the detected band covers a reasonable fraction of its zonal length
    # (all but the last, structurally-excluded, longitude column).
    band_row = detected_rows[0]
    assert mask[band_row].sum() >= 0.9 * (config.GRID_SHAPE[1] - 1)


def test_front_mask_nan_block_yields_no_front_inside_or_adjacent():
    theta = _ridge_theta()
    theta[50:61, :] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mask = nb.front_mask(theta, LAT, LON)

    # rows 49..61 include the NaN block plus its 8-connected border
    assert not mask[49:62].any()


def test_front_mask_nan_does_not_raise():
    theta = _ridge_theta()
    theta[:, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mask = nb.front_mask(theta, LAT, LON)
    assert not mask.any()


# --------------------------------------------------------------------------- #
# 2. tfp_field
# --------------------------------------------------------------------------- #

def test_tfp_field_shape_and_nan_propagation():
    # a field with a nonzero gradient everywhere (unlike the saturating
    # tanh ridge, whose far tails have a genuinely ~0 gradient and so
    # legitimately divide-by-zero inside the vendored unit-gradient-vector
    # step -- that is a property of the field, not of NaN propagation).
    row = np.arange(config.GRID_SHAPE[0])[:, None]
    theta = np.broadcast_to(300.0 + 0.1 * row, config.GRID_SHAPE).copy()

    tfp = nb.tfp_field(theta, LAT, LON)
    assert tfp.shape == theta.shape
    assert np.isfinite(tfp).all()   # no NaN input here

    theta_nan = theta.copy()
    theta_nan[10, 10] = np.nan
    tfp_nan = nb.tfp_field(theta_nan, LAT, LON)
    assert np.isnan(tfp_nan[10, 10])


# --------------------------------------------------------------------------- #
# 3. baseline_binary on a synthetic channel dataset
# --------------------------------------------------------------------------- #

def _synthetic_channel_dataset() -> xr.Dataset:
    """A (lat, lon, lev) channel dataset with a ridge in theta_e at every
    level, built directly (not via ``ingest_hysplit``) since only theta_e and
    the coordinate/dim layout matter for ``baseline_binary``.
    """
    theta_row = 300.0 + 5.0 * np.tanh(
        (np.arange(config.GRID_SHAPE[0])[:, None] - BAND_CENTER) / 1.5)
    theta_e = np.broadcast_to(
        theta_row[:, :, None], (*config.GRID_SHAPE, len(config.TARGET_LEVELS_HPA))
    ).copy()
    # a swath edge: NaN out the last few lon columns at every level
    theta_e[:, -3:, :] = np.nan

    return xr.Dataset(
        {"theta_e": (("lat", "lon", "lev"), theta_e)},
        coords={"lat": LAT, "lon": LON, "lev": list(config.TARGET_LEVELS_HPA)},
    )


def test_baseline_binary_shape_dtype_and_nan_masking():
    ch = _synthetic_channel_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = nb.baseline_binary(ch, level_hpa=850)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("lat", "lon")
    assert out.shape == config.GRID_SHAPE
    assert out.dtype == bool

    # the NaN'd swath edge (and its 8-connected border) never contains a front
    assert not out.values[:, -4:].any()


# --------------------------------------------------------------------------- #
# 4. baseline_vs_labels
# --------------------------------------------------------------------------- #

@pytest.fixture
def patched_codsus_dir(tmp_path, monkeypatch):
    codsus_dir = tmp_path / "CODSUS"
    monkeypatch.setattr(config, "CODSUS_DIR", codsus_dir)
    return codsus_dir


def test_baseline_vs_labels_nonempty_with_paired_2018_file(
        tmp_path, patched_codsus_dir):
    fullgrid_path = _write_fullgrid(tmp_path)
    bulletin = ih.nearest_bulletin(ih.overpass_time(fullgrid_path))
    _write_codsus_2018(patched_codsus_dir, bulletin)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = nb.baseline_vs_labels([fullgrid_path])

    assert not scores.empty
    assert list(scores.columns) == list(nb.SCORE_COLUMNS)
    assert set(scores["dilation"]) <= set(config.EVAL_DILATIONS)
    assert (scores["front"] == "any").all()


def test_baseline_vs_labels_unpaired_year_warns_and_returns_empty(
        tmp_path, patched_codsus_dir):
    fullgrid_path = _write_fullgrid(tmp_path, date="20190605")

    with pytest.warns(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = nb.baseline_vs_labels([fullgrid_path])

    assert scores.empty
    assert list(scores.columns) == list(nb.SCORE_COLUMNS)
