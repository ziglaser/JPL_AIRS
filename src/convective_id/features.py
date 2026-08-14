"""Flag-free MRMS structure features, plus the flag-derived seeds/labels.

STRICT SEPARATION, because the PrecipFlag categories are the thing being
corrected:

- :data:`FEATURES` (what every classifier is allowed to see) come from MRMS
  QPE structure only -- sub-pixel peak/skewness/wet fraction and their spatial
  (3x3-neighbor) and temporal (previous slot, cell-day) context. No PrecipFlag,
  no AIRS, no SMAP.
- The PrecipFlag-derived columns built here (``core_any``, ``core_confident``,
  ``core_in_neighborhood``) are ONLY for weak supervision (forest positives/
  negatives) and object seeds -- never model inputs.

Everything is computed on the full (date, slot, lat, lon) grid of the unified
base table (:func:`convection_skill.dataset.build_base_table`), then returned
as columns on the row table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from convection_skill import config as cs_config

#: Classification domain: precipitating cell-hours (the suite's any-precip bar).
PRECIP_MIN_MM: float = cs_config.ANY_PRECIP_MM

#: The feature vector every classifier sees (flag-free MRMS structure).
FEATURES: tuple[str, ...] = (
    "qpe_wet_log",       # log1p conditional (wet-area) mean rate  [mm/h]
    "qpe_max_log",       # log1p sub-pixel peak rate               [mm/h]
    "qpe_sk",            # sub-pixel skewness (concentration of intensity)
    "wet_frac",          # precipitating fraction of the cell (cnt/81)
    "peak_ratio_log",    # log1p (peak / wet-mean): core prominence
    "nbr_qpe_max_log",   # log1p max sub-pixel peak in the 3x3 neighborhood
    "nbr_wet_cells",     # precipitating neighbors (storm-size proxy, 0-8)
    "dqpe_max",          # peak growth since the previous slot     [mm/h]
    "daymax_qpe_max_log",  # log1p cell-day max peak (storm history in window)
)

_GRID_INDEX = ["date", "slot", "lat", "lon"]


def _to_grid(df: pd.DataFrame, col: str):
    """One column as a dense (date, slot, lat, lon) array + the axis index.

    The base table carries every grid row (ocean and dry included), so a
    reshape after sorting is exact; missing combinations would break the size
    check and raise, never silently misalign.
    """
    sorted_df = df.sort_values(_GRID_INDEX, kind="mergesort")
    axes = [np.unique(sorted_df[k].to_numpy()) for k in _GRID_INDEX]
    shape = tuple(len(a) for a in axes)
    if np.prod(shape) != len(sorted_df):
        raise ValueError(
            f"table is not a complete grid: {len(sorted_df):,} rows != "
            f"{'x'.join(str(s) for s in shape)}; build features on the BASE table")
    return (sorted_df[col].to_numpy(dtype=float).reshape(shape),
            sorted_df.index.to_numpy())


def _from_grid(values: np.ndarray, original_index: np.ndarray,
               df: pd.DataFrame) -> np.ndarray:
    """Back from grid order to the row order of ``df``."""
    out = pd.Series(values.ravel(), index=original_index)
    return out.reindex(df.index).to_numpy()


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the base table with FEATURES + seed columns added.

    ``is_precip`` marks the classification domain (cell-mean QPE above the
    suite's any-precip bar). Neighborhood operators treat NaN/dry as 0 rain --
    an honest floor at the domain edge.
    """
    out = df.copy()

    qpe = out["qpe"].to_numpy(dtype=float)
    qpe_wet = out["qpe_wet"].to_numpy(dtype=float)
    qpe_max = out["qpe_max"].to_numpy(dtype=float)

    out["is_precip"] = qpe > PRECIP_MIN_MM
    with np.errstate(invalid="ignore", divide="ignore"):
        wet_frac = np.where(qpe_wet > 0, qpe / qpe_wet, 0.0)
        peak_ratio = np.where(qpe_wet > 0, qpe_max / qpe_wet, 0.0)
    out["wet_frac"] = np.clip(wet_frac, 0.0, 1.0)
    out["qpe_wet_log"] = np.log1p(np.nan_to_num(qpe_wet))
    out["qpe_max_log"] = np.log1p(np.nan_to_num(qpe_max))
    out["peak_ratio_log"] = np.log1p(peak_ratio)
    out["qpe_sk"] = np.nan_to_num(out["qpe_sk"].to_numpy(dtype=float))

    # ---- spatial context (3x3 on the 1-deg grid, per date/slot) ----------------
    grid_max, gidx = _to_grid(out.assign(_m=np.nan_to_num(qpe_max)), "_m")
    nbr_max = ndimage.maximum_filter(grid_max, size=(1, 1, 3, 3), mode="constant")
    out["nbr_qpe_max_log"] = np.log1p(_from_grid(nbr_max, gidx, out))

    grid_wet, _ = _to_grid(out.assign(_w=out["is_precip"].astype(float)), "_w")
    # neighbors only: box sum minus the center cell
    nbr_wet = (ndimage.uniform_filter(grid_wet, size=(1, 1, 3, 3), mode="constant")
               * 9.0 - grid_wet)
    out["nbr_wet_cells"] = _from_grid(nbr_wet, gidx, out)

    # ---- temporal context (within the 21-02 UTC window, per cell-day) ----------
    # grid axis 1 is the slot axis, ascending -- shift gives the previous slot
    prev = np.full_like(grid_max, 0.0)
    prev[:, 1:] = grid_max[:, :-1]
    out["dqpe_max"] = _from_grid(grid_max - prev, gidx, out)
    out["daymax_qpe_max_log"] = np.log1p(
        _from_grid(np.broadcast_to(grid_max.max(axis=1, keepdims=True),
                                   grid_max.shape).copy(), gidx, out))

    # ---- PrecipFlag-derived seeds/labels (NEVER features) ----------------------
    share = out["convective_share"].to_numpy(dtype=float)
    core_any = np.isfinite(share) & (share > 0)
    out["core_any"] = core_any
    # "confident core" needs pixel SUPPORT as well as share: with 2 wet pixels,
    # 1 flagged pixel already gives share 0.5, so an unsupported share cut
    # selects tiny showers (2019: only 54 land cells at share>=0.5) rather
    # than real convective cells. >=10 wet sub-pixels (~12% of the cell) and
    # the audit-informed share bar of 0.2 give a usable, honest core class.
    wet_px = out["wet_frac"].to_numpy() * cs_config.WET_CELL_MAX_CNT
    out["core_confident"] = (np.isfinite(share) & (share >= 0.2)
                             & (wet_px >= 10))
    grid_core, _ = _to_grid(out.assign(_c=core_any.astype(float)), "_c")
    nbr_core = ndimage.maximum_filter(grid_core, size=(1, 1, 3, 3), mode="constant")
    out["core_in_neighborhood"] = _from_grid(nbr_core, gidx, out) > 0

    return out
