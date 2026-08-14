"""Mark's reference random-forest code (email 2026-07-21), VERBATIM, wired to
the suite's cell-day samples.

The functions in the verbatim block are copied unchanged from
``src/email_zach_models.txt`` (@author: markr) -- same policy as
:mod:`convection_skill.mark_screens`: do not edit their bodies. Everything
they need that our pipeline expresses differently lives in the adapters:

- :func:`samples_from_cell_days` turns the suite's one-row-per-cell-day wide
  table (:func:`convection_skill.dataset.to_cell_days`, columns ``qpe_h1`` ...)
  back into the ``(time, sample)`` stacked ``xr.Dataset`` his
  ``train_random_forest`` consumes (his ``extract_samples`` output form).
- :func:`compare_with_fronts` fits his RF twice on the SAME finite sample --
  baseline features vs baseline + CODSUS front flags -- so the front columns'
  added value is read directly off test skill and feature importances.

His walkthrough's example predicts the 6-hour QPE series from the CAPE/CIN
series plus overpass soil moisture; our default mirrors that with the
timing-guarded ``sm_anom`` (pre-window L4 mean) as the SM feature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .fronts import front_columns

#: Mirrors Mark's walkthrough example (CAPE + CIN series, one antecedent SM
#: value), with our timing-guarded SM anomaly in place of SMAP_smsfc_av0.
DEFAULT_BASE_FEATURES: tuple[str, ...] = ("mu_cape", "mu_cin", "sm_anom")
DEFAULT_TARGET: str = "qpe"
#: Mark's walkthrough rfr_kwargs.
DEFAULT_RFR_KWARGS: dict = {"max_depth": 14, "n_estimators": 75}


# =========================================================================== #
# VERBATIM from email_zach_models.txt -- do not edit
# =========================================================================== #
import scipy.stats as s
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def train_random_forest(ds_samples, X_keys, Y_keys, hourly=False,
                        train_fraction=0.25, seed=42, rfr_kwargs=None):
    """
    Extracts features/targets from ds_samples and fits a Random Forest Regressor.

    Parameters:
    - ds_samples: xr.Dataset stacked along 'sample' (and optionally 'time').
    - X_keys: List of predictor variable names.
    - Y_keys: Target variable name (str) or list of names.
    - hourly: If True, reshapes data to predict hour-by-hour (N*hours, n_features).
              If False, preserves all hours as separate feature columns (N, n_features*hours).
    - train_fraction: Float for train/test split ratio.
    - seed: Integer random seed for reproducibility.
    - rfr_kwargs: Dict of arguments for RandomForestRegressor.

    Returns:
    - Dict containing: 'model', 'X_train', 'X_test', 'Y_train', 'Y_test', 'X_keys', 'Y_keys', 'hourly'
    """
    if rfr_kwargs is None:
        rfr_kwargs = {'max_depth': 12, 'n_estimators': 75}

    np.random.seed(seed)

    # Identify time dimension length if present in any 2D variable
    n_hours = None
    for k in X_keys + (Y_keys if isinstance(Y_keys, list) else [Y_keys]):
        if 'time' in ds_samples[k].dims:
            n_hours = ds_samples[k].sizes['time']
            break

    # -------------------------------------------------------------
    # 1. Build Predictor Matrix X
    # -------------------------------------------------------------
    X_list = []
    for var in X_keys:
        da = ds_samples[var]

        if 'time' in da.dims:
            arr = da.values.T  # Shape: (sample, time)
            if hourly:
                # Flatten across time -> (sample * time, 1)
                arr = arr.ravel()[:, np.newaxis]
        else:
            arr = da.values  # Shape: (sample,)
            if hourly and n_hours is not None:
                # Duplicate static/initial variable for every hour of that sample
                arr = np.repeat(arr, n_hours)[:, np.newaxis]
            else:
                arr = arr[:, np.newaxis]

        X_list.append(arr)

    X = np.hstack(X_list)

    # -------------------------------------------------------------
    # 2. Build Target Matrix/Vector Y
    # -------------------------------------------------------------
    if isinstance(Y_keys, str):
        Y_keys = [Y_keys]

    Y_list = []
    for var in Y_keys:
        da = ds_samples[var]
        if 'time' in da.dims:
            arr = da.values.T
            if hourly:
                arr = arr.ravel()[:, np.newaxis]
        else:
            arr = da.values
            if hourly and n_hours is not None:
                arr = np.repeat(arr, n_hours)[:, np.newaxis]
            else:
                arr = arr[:, np.newaxis]
        Y_list.append(arr)

    Y = np.hstack(Y_list)

    # Flatten single-target 2D array to 1D vector
    if Y.shape[1] == 1:
        Y = Y.ravel()

    # -------------------------------------------------------------
    # 3. Train / Test Split
    # -------------------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True
    )

    # -------------------------------------------------------------
    # 4. Fit Random Forest
    # -------------------------------------------------------------
    print(f"Training RF [hourly={hourly}] on X shape {X_train.shape} -> Y shape {Y_train.shape}...")
    rfr = RandomForestRegressor(random_state=seed, **rfr_kwargs)
    rfr.fit(X_train, Y_train)

    # -------------------------------------------------------------
    # 5. Pack Results
    # -------------------------------------------------------------
    return {
        'model': rfr,
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'X_keys': X_keys,
        'Y_keys': Y_keys,
        'hourly': hourly
    }


# some quick functions to get scores, hists etc


def make_binned_freqs(predY,obsY,obs_pct=99,thresh=None,cum=True,seed=1,
                      bin_pcts=None):
    np.random.seed(seed)

    if thresh is None:
        thresh = np.percentile(obsY,obs_pct)

    if bin_pcts is None:
        bin_pcts = np.arange(101)

    bins = np.percentile(predY,bin_pcts)

    # add small perturbations to get lower bins - they are often zero
    # but need to be nonzero for this quick plot
    if len(set(bins))!=len(bins):
        eps = np.random.randn(len(predY)) * predY.std() * 1.e-12
        predY = predY + eps

    bins = np.percentile(predY,bin_pcts)

    binned_freqs = s.binned_statistic(predY,obsY>thresh,bins=bins)[0]

    if cum:
        binned_freqs = np.cumsum(binned_freqs) / binned_freqs.sum()

    return binned_freqs
# =========================================================================== #
# end verbatim block
# =========================================================================== #


def samples_from_cell_days(cell_days: pd.DataFrame) -> xr.Dataset:
    """The suite's cell-day wide table -> Mark's ``(time, sample)`` stacked form.

    Every ``<var>_h<slot>`` column family becomes one ``(time, sample)``
    variable named ``<var>`` (time = the slot numbers, ascending); every other
    numeric column becomes a ``(sample,)`` variable under its own name. The
    (day, lat, lon) identity rides along as sample coordinates.
    """
    hourly: dict[str, dict[int, str]] = {}
    for col in cell_days.columns:
        stem, _, suffix = col.rpartition("_h")
        if stem and suffix.isdigit():
            hourly.setdefault(stem, {})[int(suffix)] = col

    slots = sorted({s for cols in hourly.values() for s in cols})
    n = len(cell_days)
    out = xr.Dataset(coords={"sample": np.arange(n),
                             "time": np.array(slots)})
    for key in ("day", "lat", "lon"):
        if key in cell_days:
            out.coords[key] = ("sample", cell_days[key].to_numpy())

    for var, cols in hourly.items():
        arr = np.stack([cell_days[cols[s]].to_numpy(dtype=float) if s in cols
                        else np.full(n, np.nan) for s in slots])
        out[var] = (("time", "sample"), arr)
    for col in cell_days.columns:
        if col in ("day", "lat", "lon") or col in {c for m in hourly.values()
                                                   for c in m.values()}:
            continue
        if pd.api.types.is_numeric_dtype(cell_days[col]):
            out[col] = ("sample", cell_days[col].to_numpy(dtype=float))
    return out


def feature_names(ds_samples: xr.Dataset, X_keys: list[str],
                  hourly: bool = False) -> list[str]:
    """Column names of the X matrix his ``train_random_forest`` builds."""
    names = []
    for var in X_keys:
        if not hourly and "time" in ds_samples[var].dims:
            names += [f"{var}_h{int(t)}" for t in ds_samples["time"].values]
        else:
            names.append(var)
    return names


def finite_samples(ds_samples: xr.Dataset, keys: list[str]) -> xr.Dataset:
    """Samples where every listed variable is finite at every hour.

    The RF cannot take NaN; restricting BOTH the baseline and the with-fronts
    fit to this common sample keeps the comparison apples-to-apples (with
    front features this is effectively the 2016-2018 overlap).
    """
    ok = np.ones(ds_samples.sizes["sample"], dtype=bool)
    for k in keys:
        vals = ds_samples[k].values
        finite = np.isfinite(vals)
        ok &= finite.all(axis=0) if vals.ndim == 2 else finite
    return ds_samples.isel(sample=ok)


def importance_table(rfr: dict, ds_samples: xr.Dataset) -> pd.Series:
    """Named, sorted feature importances for one train_random_forest result."""
    names = feature_names(ds_samples, rfr["X_keys"], hourly=rfr["hourly"])
    return pd.Series(rfr["model"].feature_importances_, index=names,
                     name="importance").sort_values(ascending=False)


def compare_with_fronts(
    cell_days: pd.DataFrame,
    base_features: tuple[str, ...] = DEFAULT_BASE_FEATURES,
    front_features: tuple[str, ...] = front_columns(),
    target: str = DEFAULT_TARGET,
    hourly: bool = False,
    obs_pct: float = 99.5,
    seed: int = 42,
    rfr_kwargs: Optional[dict] = None,
) -> dict:
    """Mark's RF with vs without the CODSUS front flags, on one common sample.

    Returns a dict with both fitted results (``base``/``fronts``), their named
    importances, test R^2, and the tail-capture CDF at ``obs_pct`` (his
    ``make_binned_freqs`` on the per-sample max, as in the walkthrough).
    """
    ds = samples_from_cell_days(cell_days)
    base_keys, front_keys = list(base_features), list(front_features)
    ds = finite_samples(ds, base_keys + front_keys + [target])

    out = {"n_samples": ds.sizes["sample"]}
    for label, keys in (("base", base_keys), ("fronts", base_keys + front_keys)):
        rfr = train_random_forest(
            ds, keys, target, hourly=hourly, seed=seed,
            rfr_kwargs=dict(rfr_kwargs or DEFAULT_RFR_KWARGS))
        pred, obs = rfr["model"].predict(rfr["X_test"]), rfr["Y_test"]
        pred_max = pred if pred.ndim == 1 else pred.max(axis=1)
        obs_max = obs if obs.ndim == 1 else obs.max(axis=1)
        out[label] = {
            "rfr": rfr,
            "r2_test": float(rfr["model"].score(rfr["X_test"], obs)),
            "importances": importance_table(rfr, ds),
            "tail_cdf": make_binned_freqs(pred_max, obs_max, obs_pct=obs_pct),
        }
    return out
