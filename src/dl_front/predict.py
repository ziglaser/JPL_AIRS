"""Checkpoint -> per-year front-likelihood grids (and streamed evaluation).

``predict_year`` keeps inference inputs byte-identical to training inputs by
reusing ``dataset.year_arrays``.  ``evaluate_years`` is the whole paper-
metrics evaluation loop; ``save_probabilities`` writes netCDF mirroring the
authors' published merra2_fronts likelihood files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, dataset, evaluate


def load_model(path):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def predict_year(model, year: int, n_classes: int, stats: dict,
                 batch_size: int = 64):
    """One year -> (probs (t, 68, 141, n_cls), y_cls, times)."""
    x, y, times = dataset.year_arrays(year, n_classes, stats)
    probs = model.predict(x.astype(np.float32), batch_size=batch_size,
                          verbose=0)
    return probs.astype(np.float32), y, times


def evaluate_years(model, years, n_classes: int, stats: dict | None = None
                   ) -> tuple[evaluate.PaperMetrics, pd.DataFrame]:
    """Streamed evaluation: paper metrics + neighborhood-CSI counts.

    Scoring mask follows the track (user decision 2026-08-13): ALL 6-class
    scoring is restricted to ``dataset.analysis_domain()``; the 5-class
    paper replication keeps the Fig. 2 region mask.
    """
    stats = stats or dataset.load_norm_stats()
    mask = (dataset.analysis_domain() if n_classes == 6
            else dataset.region_mask().astype(bool))
    pm = evaluate.PaperMetrics(n_classes, mask=mask)
    counts = []
    for year in years:
        probs, y, times = predict_year(model, year, n_classes, stats)
        pm.update(probs, y)
        counts.append(evaluate.csi_counts(probs.argmax(-1), y, times,
                                          n_classes, mask=mask))
    return pm, pd.concat(counts, ignore_index=True)


def save_probabilities(model, year: int, n_classes: int, out_dir,
                       stats: dict | None = None) -> Path:
    """Write one year of likelihood grids (time, front, lat, lon) netCDF."""
    stats = stats or dataset.load_norm_stats()
    probs, _, times = predict_year(model, year, n_classes, stats)
    names = list(dataset.class_names(n_classes))
    ds = xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"),
                    probs.transpose(0, 3, 1, 2))},
        coords={"time": np.asarray(times), "front": names,
                "lat": np.asarray(config.LABEL_LATS),
                "lon": np.asarray(config.LABEL_LONS)})
    ds["fronts"].attrs["long_name"] = "DL-FRONT category likelihoods"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dlfront_{n_classes}cls_likelihoods_{year}.nc"
    enc = {"fronts": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(path, encoding=enc)
    return path
