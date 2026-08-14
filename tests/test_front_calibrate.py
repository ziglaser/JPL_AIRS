"""Analytic-answer tests for ``front_finder.calibrate``.

No TensorFlow import chain here: ``calibrate`` only touches ``config`` and
``labels`` (pure numpy/scipy/xarray/sklearn), so no ``_stubs`` shim is
needed (contrast ``test_front_dataset.py``, which imports ``dataset`` and
therefore ``derive``).

- A perfectly-calibrated synthetic probability field (bernoulli draws at
  known probe probabilities) fits an isotonic map that reproduces those
  probe probabilities almost exactly (law of large numbers, many samples
  per probe value).
- A miscalibrated field (raw probability = half the true event frequency)
  fits a map whose ``apply()`` output is close to double the raw
  probability, at every probe point.
- ``reliability``'s per-bin ``n`` sums to the number of valid pixels.
- ``save``/``load`` round-trip preserves predictions exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from front_finder import calibrate

PROBES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
N_PER_PROBE = 6000


def _single_pixel_dataarrays(values, rng_seed, freq=None):
    """(time, front=["cold"], lat=[0.], lon=[0.]) prob/truth pair.

    ``values`` gives the per-timestep raw probability; the truth is a
    bernoulli draw with success probability ``freq`` (defaults to
    ``values`` itself, i.e. perfectly calibrated).
    """
    if freq is None:
        freq = values
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    truth_1d = rng.random(n) < freq
    time = pd.date_range("2018-01-01", periods=n, freq="3h")
    coords = {"time": time, "front": ["cold"], "lat": [0.0], "lon": [0.0]}
    prob = xr.DataArray(values.reshape(n, 1, 1, 1), dims=("time", "front", "lat", "lon"),
                        coords=coords)
    truth = xr.DataArray(truth_1d.reshape(n, 1, 1, 1), dims=("time", "front", "lat", "lon"),
                         coords=coords)
    return prob, truth


def _probe_values():
    return np.repeat(PROBES, N_PER_PROBE)


# --------------------------------------------------------------------------- #
# fit: perfectly calibrated probabilities recover ~identity
# --------------------------------------------------------------------------- #

def test_fit_perfectly_calibrated_probabilities_recovers_identity():
    values = _probe_values()
    prob, truth = _single_pixel_dataarrays(values, rng_seed=1)

    models = calibrate.fit(prob, truth, dilation=0)
    assert set(models) == {"cold"}

    predicted = models["cold"].predict(PROBES)
    np.testing.assert_allclose(predicted, PROBES, atol=0.03)


# --------------------------------------------------------------------------- #
# fit + apply: miscalibrated probabilities (raw = freq / 2) roughly doubled
# --------------------------------------------------------------------------- #

def test_apply_miscalibrated_probabilities_roughly_doubles():
    freq = _probe_values()
    raw = freq / 2.0
    prob, truth = _single_pixel_dataarrays(raw, rng_seed=2, freq=freq)

    models = calibrate.fit(prob, truth, dilation=0)
    calibrated = calibrate.apply(prob, models)

    # compare, per probe block, the calibrated mean to the true frequency
    calibrated_flat = calibrated.values.reshape(-1)
    for p, f in zip(PROBES, PROBES):  # PROBES themselves are freq values here
        block = calibrated_flat[np.isclose(raw, p / 2.0)]
        assert block.mean() == pytest.approx(f, abs=0.05)

    assert calibrated.shape == prob.shape
    assert calibrated.dims == prob.dims


# --------------------------------------------------------------------------- #
# reliability: per-bin n sums to the number of valid pixels
# --------------------------------------------------------------------------- #

def test_reliability_bin_counts_sum_to_valid_pixel_count():
    values = _probe_values()
    prob, truth = _single_pixel_dataarrays(values, rng_seed=3)

    table = calibrate.reliability(prob, truth, dilation=0)
    assert set(table["front"]) == {"cold"}
    assert table["n"].sum() == prob.sizes["time"]

    # excluding some pixels via `valid` removes exactly that many from n
    valid = xr.DataArray(np.ones((prob.sizes["time"], 1, 1), dtype=bool),
                         dims=("time", "lat", "lon"),
                         coords={"time": prob["time"], "lat": prob["lat"],
                                 "lon": prob["lon"]})
    valid[:10] = False
    table_masked = calibrate.reliability(prob, truth, valid=valid, dilation=0)
    assert table_masked["n"].sum() == prob.sizes["time"] - 10


# --------------------------------------------------------------------------- #
# save / load round trip
# --------------------------------------------------------------------------- #

def test_save_load_round_trip_preserves_predictions(tmp_path):
    values = _probe_values()
    prob, truth = _single_pixel_dataarrays(values, rng_seed=4)
    models = calibrate.fit(prob, truth, dilation=0)

    path = tmp_path / "calibration.pkl"
    calibrate.save(models, path)
    loaded = calibrate.load(path)

    assert set(loaded) == set(models)
    np.testing.assert_array_equal(
        loaded["cold"].predict(PROBES), models["cold"].predict(PROBES))
