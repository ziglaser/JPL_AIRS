"""Inference: trained checkpoint -> per-class probability fields on the label grid.

Deep supervision produces multiple outputs; the FIRST is the primary
full-resolution head (convention copied from fronts/evaluation/predict_tf.py,
which takes side output 0).  Padding is stripped so downstream evaluation
sees plain (time, front, lat 68, lon 141) probability arrays; the "none"
class is dropped (paper convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import config, dataset

_LAT = np.arange(10.0, 77.1, 1.0)
_LON = np.arange(-171.0, -30.9, 1.0)


def load_checkpoint(path):
    """Load a saved model for inference (no compile needed)."""
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def _unpad(a: np.ndarray) -> np.ndarray:
    """(batch, 72, 144, C) -> (batch, 68, 141, C): inverse of dataset._pad."""
    return a[:, 2:2 + config.GRID_SHAPE[0], 1:1 + config.GRID_SHAPE[1]]


def predict_batch(model, x: np.ndarray) -> np.ndarray:
    """(batch, 72, 144, 5, C) inputs -> (batch, 68, 141, n_front) probs."""
    want = model.input_shape[-1]
    if x.shape[-1] != want:
        raise ValueError(
            f"checkpoint expects {want} channels but inputs have "
            f"{x.shape[-1]} -- winds flag mismatch? (thermo-only = "
            f"{len(config.THERMO_VARS) + 1}, with winds = "
            f"{len(config.THERMO_VARS) + len(config.WIND_VARS) + 1})")
    out = model.predict(x, verbose=0)
    if isinstance(out, list):
        out = out[0]                        # primary head (predict_tf.py)
    return _unpad(out)[..., 1:]             # drop the "none" class


def _to_dataset(probs: np.ndarray, times) -> xr.Dataset:
    da = xr.DataArray(
        probs.transpose(0, 3, 1, 2).astype(np.float32),
        dims=("time", "front", "lat", "lon"),
        coords={"time": list(times), "front": list(config.FRONT_TYPES),
                "lat": _LAT, "lon": _LON},
        name="probabilities")
    return da.to_dataset()


def predict_year(model, year: int, winds: bool, stats: dict | None = None,
                 batch_size: int = 8, limit: int | None = None) -> xr.Dataset:
    """Predict every labeled MERRA-2 timestep of a year (E1/E2 evaluation).

    Iterates dataset.year_samples so inputs are IDENTICAL to training ones;
    the paired y is discarded but its timestep bookkeeping is reused.
    ``limit`` caps the number of timesteps (mechanics checks / quick looks).
    """
    import itertools

    stats = stats or dataset.load_norm_stats()
    probs, times, xs = [], [], []
    samples = dataset.year_samples(year, stats, winds, return_times=True)
    if limit:
        samples = itertools.islice(samples, limit)
    for x, _y, t in samples:
        xs.append(x)
        times.append(t)
        if len(xs) == batch_size:
            probs.append(predict_batch(model, np.stack(xs)))
            xs = []
    if xs:
        probs.append(predict_batch(model, np.stack(xs)))
    return _to_dataset(np.concatenate(probs), times)


def predict_airs(model, paths, winds: bool, stats: dict | None = None,
                 slot: int = 0) -> xr.Dataset:
    """Predict fullgrid AIRS files -> probabilities + the swath mask.

    The returned ``observed`` variable must be passed to evaluation so every
    baseline is scored on identical pixels (workplan section 4.1).
    """
    stats = stats or dataset.load_norm_stats()
    probs, times, observed = [], [], []
    for p in paths:
        x, obs, t = dataset.airs_x(p, stats, winds, slot)
        probs.append(predict_batch(model, x[None])[0])
        observed.append(obs)
        times.append(t)
    ds = _to_dataset(np.stack(probs), times)
    ds["observed"] = xr.DataArray(np.stack(observed), dims=("time", "lat", "lon"),
                                  coords={"time": ds["time"], "lat": _LAT,
                                          "lon": _LON})
    return ds
