"""Per-class probability calibration (paper/fronts ``calibrate_model.py``).

Isotonic regression maps a model's raw predicted probability for a front
class to the observed frequency of that class (DILATED by the eval
neighborhood, since the raw network is trained/evaluated against dilated
truth) among pixels sharing that probability. This is the standard
reliability-diagram calibration used by the paper: fit once on a held-out
year, then apply before computing any reported scores.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.isotonic import IsotonicRegression

from . import config
from .labels import dilate

DEFAULT_PATH = config.RESULTS_DIR / "calibration.pkl"
#: Cap on pixels handed to IsotonicRegression.fit per class (tractability);
#: a deterministic stride is used instead of random subsampling so runs are
#: reproducible without needing a seed.
MAX_FIT_POINTS = 2_000_000


def _dilated_truth(truth: xr.DataArray, dilation: int) -> np.ndarray:
    """(time, front, lat, lon) truth, dilated per timestep+front independently."""
    t = truth.transpose("time", "front", "lat", "lon").values
    return dilate(t.reshape(-1, *t.shape[-2:]), dilation).reshape(t.shape)


def _valid_array(prob: xr.DataArray, valid: xr.DataArray | None) -> np.ndarray:
    if valid is None:
        return np.ones((prob.sizes["time"], prob.sizes["lat"], prob.sizes["lon"]),
                       dtype=bool)
    return valid.transpose("time", "lat", "lon").values


def fit(prob: xr.DataArray, truth: xr.DataArray,
       valid: xr.DataArray | None = None,
       dilation: int = config.LABEL_DILATION) -> dict:
    """Per-front-class ``IsotonicRegression`` mapping probability -> frequency.

    ``truth`` is dilated by ``dilation`` 8-connected iterations before being
    used as the calibration target (matches the eval neighborhood the model
    is judged against). Invalid pixels (``valid`` False) are excluded.
    Pixels are subsampled with a fixed stride so at most ``MAX_FIT_POINTS``
    land in each class's fit.
    """
    prob_v = prob.transpose("time", "front", "lat", "lon").values
    prob_fronts = list(prob["front"].values)
    truth_fronts = list(truth["front"].values)
    t_dilated = _dilated_truth(truth, dilation)
    valid_v = _valid_array(prob, valid)

    models = {}
    for name in prob_fronts:
        pi = prob_fronts.index(name)
        if name not in truth_fronts:
            continue
        ti = truth_fronts.index(name)
        p = prob_v[:, pi][valid_v]
        y = t_dilated[:, ti][valid_v].astype(np.float64)
        n = p.size
        stride = max(1, n // MAX_FIT_POINTS)
        p_s, y_s = p[::stride], y[::stride]
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(p_s, y_s)
        models[name] = model
    return models


def apply(prob: xr.DataArray, models: dict) -> xr.DataArray:
    """Map every class of ``prob`` through its isotonic model (identity if
    the class has no model)."""
    out = prob.copy(deep=True)
    for name in prob["front"].values:
        name = str(name)
        if name not in models:
            continue
        vals = out.sel(front=name).values
        calibrated = models[name].predict(vals.ravel()).reshape(vals.shape)
        out.loc[dict(front=name)] = calibrated.astype(vals.dtype)
    return out


def save(models: dict, path=DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load(path=DEFAULT_PATH) -> dict:
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def reliability(prob: xr.DataArray, truth: xr.DataArray,
                valid: xr.DataArray | None = None,
                bins=np.linspace(0, 1, 11),
                dilation: int = config.LABEL_DILATION) -> pd.DataFrame:
    """Reliability-diagram table: mean forecast probability vs. observed
    frequency per probability bin, per front class.

    Columns: front, bin_mid, mean_prob, obs_freq, n.
    """
    prob_v = prob.transpose("time", "front", "lat", "lon").values
    prob_fronts = list(prob["front"].values)
    truth_fronts = list(truth["front"].values)
    t_dilated = _dilated_truth(truth, dilation)
    valid_v = _valid_array(prob, valid)
    bin_mid = 0.5 * (np.asarray(bins)[:-1] + np.asarray(bins)[1:])

    rows = []
    for name in prob_fronts:
        pi = prob_fronts.index(name)
        if name not in truth_fronts:
            continue
        ti = truth_fronts.index(name)
        p = prob_v[:, pi][valid_v]
        y = t_dilated[:, ti][valid_v].astype(np.float64)
        idx = np.clip(np.digitize(p, bins) - 1, 0, len(bin_mid) - 1)
        for b in range(len(bin_mid)):
            sel = idx == b
            n = int(sel.sum())
            rows.append({
                "front": name,
                "bin_mid": float(bin_mid[b]),
                "mean_prob": float(p[sel].mean()) if n else np.nan,
                "obs_freq": float(y[sel].mean()) if n else np.nan,
                "n": n,
            })
    return pd.DataFrame(rows)
