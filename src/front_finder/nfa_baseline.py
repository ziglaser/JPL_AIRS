"""Classical numerical-frontal-analysis (NFA) baseline: "does deep learning
beat 1965?" (workplan section 4.2).

Wraps the vendored ``fronts/nfa/methods.py`` (Renard & Clarke 1965 thermal
front parameter; Huber-Pock & Kress 1981 thermal front locator) so it can be
scored against CODSUS with exactly the same pairing/evaluation machinery used
for the neural models (:mod:`front_finder.evaluate`, :mod:`.labels`).

Import path
-----------
Same trick as :mod:`.derive`: ``fronts/`` is not a Python package on
``sys.path`` by default, so we insert ``config.FRONTS_REPO`` before importing
it.  Importing ``nfa.methods`` transitively imports
``fronts/utils/data_utils.py``, which does an unconditional
``import tensorflow as tf`` at module scope (plus ``shapely``/``regionmask``)
-- fine in the real fronts-tf environment, but the system ``python3`` used by
this repo's test suite has no TensorFlow installed. Tests import this module
only after putting ``tests/_stubs`` (a minimal ``tensorflow`` stand-in) on
``sys.path``, exactly as ``tests/test_front_ingest.py`` /
``tests/test_front_dataset.py`` do for :mod:`.derive`.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import config, evaluate
from .labels import dilate

sys.path.insert(0, str(config.FRONTS_REPO))
from nfa import methods as nfa_methods  # noqa: E402  (fronts/nfa/methods.py)
from utils import data_utils  # noqa: E402  (fronts/utils/data_utils.py; TF chain)

#: Standard columns of the scores frame returned by :func:`baseline_vs_labels`
#: (matches ``evaluate.scores_from_counts`` with the (front, dilation) index
#: reset to columns).
SCORE_COLUMNS = ("front", "dilation", "csi", "pod", "far", "fb", "km")


# --------------------------------------------------------------------------- #
# Thin wrappers over fronts/nfa/methods.py
# --------------------------------------------------------------------------- #

def tfp_field(theta: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Thermal front parameter (Renard & Clarke 1965) for a theta-like field.

    ``theta`` is any thermodynamic field of shape (..., lat, lon) -- here
    used for theta_e in K, but ``thermal_front_parameter`` itself is generic
    (it was designed for temperature in degC; the units only rescale the
    output). Thin pass-through to ``nfa.methods.thermal_front_parameter``.

    NaN-safe by inaction, not by design: NaNs propagate through the
    finite-difference/sqrt arithmetic exactly like IEEE-754 NaN propagation
    everywhere else in numpy -- any NaN in ``theta`` contaminates the TFP at
    that pixel and (via the ``np.diff`` stencil) its immediate neighbor along
    each axis. Callers who need NaN-free output (e.g. :func:`front_mask`) are
    responsible for filling/masking themselves.
    """
    theta = np.asarray(theta, dtype=float)
    return np.asarray(nfa_methods.thermal_front_parameter(theta, lats, lons))


def front_mask(theta: np.ndarray, lats: np.ndarray, lons: np.ndarray,
               min_gradient: float = 1.0, **locator_kwargs) -> np.ndarray:
    """Boolean front mask from the thermal front locator (Huber-Pock & Kress 1981).

    ``nfa.methods.thermal_front_locator`` does NOT return a boolean grid or a
    list of front positions -- it returns a continuous field TFL of the same
    shape as the input, expressed in degC/(100km)^4 (verified by reading
    ``fronts/nfa/methods.py``: it is literally
    ``sum(d(TFP)/ds * unit_gradient_vector)``, i.e. the directional
    derivative of the thermal front parameter (TFP) along the local
    temperature-gradient direction, where TFP itself is defined there with a
    leading minus sign, ``TFP = sum(-d(|grad theta|)/ds * unit_gradient)``).
    We convert it to a boolean grid using the classical Hewson (1998)
    construction that TFP/TFL were designed for, with the sign empirically
    pinned to THIS implementation's convention (verified against a synthetic
    single-ridge field, see ``tests/test_front_nfa.py``):

    * fronts sit on the zero contour of TFP (TFP is the negative directional
      derivative of the gradient magnitude along the gradient direction, so
      it changes sign exactly where |grad theta| is locally extremal);
    * at a MAXIMUM of |grad theta| (a front), |grad theta| is increasing
      then decreasing along the direction of increasing theta, so TFP goes
      negative -> positive through the crossing, i.e. TFL = d(TFP)/ds > 0;
      at a MINIMUM (a trough between fronts) the transition -- and TFL's
      sign -- is reversed. So TFL > 0 selects fronts and rejects troughs;
    * ``min_gradient`` (degC per 100 km) rejects zero-crossings in
      near-uniform regions, where the TFP unit-gradient-vector division by a
      tiny |grad theta| makes TFP numerically noisy and would otherwise flag
      spurious "fronts" from roundoff.

    A pixel is marked True iff it (or an axis-neighbor, since the zero
    crossing falls *between* grid points) has a TFP sign change, TFL > 0
    there, and the local gradient magnitude exceeds ``min_gradient``.

    ``**locator_kwargs`` is accepted for forward compatibility with
    ``nfa.methods.thermal_front_locator`` (which today takes no keyword
    arguments) and is not otherwise used.

    NaN handling: NaN regions (and swath-edge pixels immediately adjacent to
    them, 8-connected) always come back False rather than raising. We
    temporarily fill NaNs with ``np.nanmean(theta)`` before calling the
    locator (a flat fill has zero gradient, so it cannot manufacture a real
    front on its own), then zero out any front pixel that is NaN itself or
    has a NaN among its 8 neighbors in the ORIGINAL field -- this is what
    keeps swath-edge gradient artifacts from counting as fronts.
    """
    theta = np.asarray(theta, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    nan_mask = np.isnan(theta)
    fill_value = 0.0 if nan_mask.all() else np.nanmean(theta)
    filled = np.where(nan_mask, fill_value, theta)

    tfp = np.asarray(nfa_methods.thermal_front_parameter(filled, lats, lons))
    tfl = np.asarray(nfa_methods.thermal_front_locator(filled, lats, lons))

    # gradient magnitude in degC/(100km), same cartesian projection the
    # vendored methods use internally, for the min_gradient screen.
    Lons, Lats = np.meshgrid(lons, lats)
    x, y = data_utils.haversine(Lons, Lats)
    dFdx = np.diff(filled, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    dFdy = np.diff(filled, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    grad_mag = np.sqrt(dFdx ** 2 + dFdy ** 2) * 100.0  # degC/km -> degC/(100km)

    # TFP zero-crossing: sign change with either axis-neighbor (0 counts as
    # positive so a literal zero doesn't suppress the crossing it sits on).
    sign = np.where(tfp == 0, 1.0, np.sign(tfp))
    crossing = np.zeros_like(tfp, dtype=bool)
    dy_change = sign[..., :-1, :] != sign[..., 1:, :]
    crossing[..., :-1, :] |= dy_change
    crossing[..., 1:, :] |= dy_change
    dx_change = sign[..., :, :-1] != sign[..., :, 1:]
    crossing[..., :, :-1] |= dx_change
    crossing[..., :, 1:] |= dx_change

    mask = crossing & (tfl > 0) & (grad_mag > min_gradient)

    # NaN / swath-edge screen: no front inside a NaN region or on its
    # 8-connected border.
    if nan_mask.any():
        nan_or_neighbor = dilate(nan_mask, 1)
        mask = mask & ~nan_or_neighbor

    # Domain-boundary artifact: the vendored methods take np.diff(..., 1,
    # append=0) along both axes, so the LAST row and LAST column always see a
    # finite difference against a fabricated appended zero rather than a real
    # neighbor. That manufactures a huge, meaningless gradient/TFP-sign-flip
    # on that single border, independent of any real data (confirmed
    # empirically: a spatially uniform-along-lon field still "detects" a
    # front on every row of the last longitude column, and on the entire
    # last latitude row). These two edges are therefore never front pixels.
    mask[..., -1, :] = False
    mask[..., :, -1] = False

    return mask


# --------------------------------------------------------------------------- #
# AIRS channel-dataset baseline
# --------------------------------------------------------------------------- #

def baseline_binary(ch: xr.Dataset, level_hpa: int = 850) -> xr.DataArray:
    """Any-front boolean (lat, lon) baseline from an ``ingest_hysplit``-style
    channel dataset, using theta_e at ``level_hpa`` as the frontal-analysis
    thermodynamic field.

    theta_e substitutes for the classical wet-bulb potential temperature
    theta_w: Renard & Clarke (1965) and follow-on numerical-frontal-analysis
    work used various thermodynamic fields (temperature, theta, theta_w),
    and Hewson (1998) specifically recommends theta_w for TFP-based frontal
    location. AIRS retrievals give us T and q directly, from which theta_e
    (already one of our seven derived thermo channels, ``derive.py``) is the
    natural analog; theta_e is monotonic in theta_w for the moist,
    tropospheric conditions this dataset covers, so a theta_e-based analysis
    identifies the same frontal zones theta_w would for our purposes.
    """
    theta_e = ch["theta_e"].sel(lev=level_hpa)
    mask = front_mask(theta_e.values, ch["lat"].values, ch["lon"].values)
    return xr.DataArray(mask, dims=("lat", "lon"),
                        coords={"lat": ch["lat"], "lon": ch["lon"]},
                        name="front_baseline")


# --------------------------------------------------------------------------- #
# E3 comparison table: baseline vs CODSUS
# --------------------------------------------------------------------------- #

def baseline_vs_labels(paths, dilations: tuple = config.EVAL_DILATIONS,
                       level_hpa: int = 850, label_width: int = 1,
                       masked_labels: bool = True) -> pd.DataFrame:
    """Score the 1965 NFA baseline against CODSUS on a list of fullgrid files.

    Builds ``baseline_binary`` per file, pairs it with the CODSUS bulletin
    exactly as ``dataset.airs_samples`` does (``ingest_hysplit.nearest_bulletin``),
    and scores any-front presence (truth = ``labels.front_stack(...).any``
    over the front dimension) with ``evaluate.contingency_by_day`` +
    ``evaluate.scores_from_counts``, restricted to observed-AND-valid pixels
    (AIRS swath mask AND CODSUS label validity). Files whose bulletin has no
    CODSUS year (e.g. 2019, pending the CSB extension) are skipped with a
    warning, mirroring ``dataset.airs_samples``. Returns
    ``evaluate.scores_from_counts`` with the (front, dilation) index reset to
    columns (``SCORE_COLUMNS``); an empty frame (same columns, zero rows) if
    no file could be paired with a label year.
    """
    from . import ingest_hysplit as ih
    from . import labels as L

    label_cache: dict = {}
    times, preds, truths, valids = [], [], [], []
    lat = lon = None

    for path in paths:
        ch = ih.to_label_grid(ih.load_fullgrid(path), slot=0, winds=False)
        t = ih.nearest_bulletin(ih.overpass_time(path))
        year = t.year

        if year not in label_cache:
            try:
                truth_ds = L.load_fronts(year, width=label_width, masked=masked_labels)
            except FileNotFoundError:
                label_cache[year] = None
            else:
                fr_any = L.front_stack(truth_ds).any("front")
                valid = L.valid_mask(truth_ds)
                label_cache[year] = (pd.DatetimeIndex(truth_ds["time"].values),
                                     fr_any.values, valid.values)
                truth_ds.close()

        if label_cache[year] is None:
            warnings.warn(f"{path}: no CODSUS labels for {year} "
                          "(CSB extension pending); skipped")
            continue

        bulletin_times, fr_any_vals, valid_vals = label_cache[year]
        i = np.flatnonzero(bulletin_times == t)
        if len(i) == 0:
            warnings.warn(f"{path}: bulletin {t} missing from CODSUS; skipped")
            continue

        pred = baseline_binary(ch, level_hpa=level_hpa)
        if lat is None:
            lat, lon = pred["lat"].values, pred["lon"].values

        times.append(t)
        preds.append(pred.values)
        truths.append(fr_any_vals[i[0]])
        valids.append(valid_vals[i[0]] & ch["observed"].values)

    if not times:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))

    coords = {"time": times, "front": ["any"], "lat": lat, "lon": lon}
    pred_da = xr.DataArray(np.stack(preds)[:, np.newaxis],
                           dims=("time", "front", "lat", "lon"), coords=coords)
    truth_da = xr.DataArray(np.stack(truths)[:, np.newaxis],
                            dims=("time", "front", "lat", "lon"), coords=coords)
    valid_da = xr.DataArray(np.stack(valids), dims=("time", "lat", "lon"),
                            coords={"time": times, "lat": lat, "lon": lon})

    counts = evaluate.contingency_by_day(pred_da, truth_da, valid=valid_da,
                                         dilations=dilations)
    scores = evaluate.scores_from_counts(counts)
    return scores.reset_index()[list(SCORE_COLUMNS)]
