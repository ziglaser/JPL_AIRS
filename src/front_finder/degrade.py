"""Stage-B "AIRS simulator" degradation operators (workplan section 3.6).

Pretraining (stage A) uses clean MERRA-2 profiles; stage B bridges to real
AIRS retrievals by degrading those same profiles with two independent
error models, both estimated from AIRS/AMSU validation literature:

1. **Vertical smoothing.** AIRS's IR weighting functions have ~1-2 km
   effective vertical resolution (Susskind et al. 2003; Maddy & Barnet
   2008) -- far coarser than a native profile. We approximate this as a
   Gaussian convolution in log-pressure (height) space. The workplan note
   of 2026-08-04 records that the pretraining corpus stores only the 5
   ``config.TARGET_LEVELS_HPA`` levels (bandwidth), not the full profile,
   so the convolution degenerates to a small dense **mixing matrix** on
   those 5 levels rather than a true full-profile convolution -- still
   applied to T/q *before* any derived-variable computation (derive.py),
   matching how derive.py treats profile-resolution physics.
2. **Retrieval noise.** AIRS validation finds T errors of order 1 K per
   1-km layer and q errors of order 15-20% per 2-km layer (Divakarla et
   al. 2006), with errors vertically correlated because overlapping
   weighting functions cause adjacent levels' retrieval errors to move
   together. We add additive Gaussian T noise and multiplicative
   lognormal q noise, both driven by a shared-form AR(1) process along
   the level axis whose innovations are horizontally correlated
   (e-folding ``NOISE_CORR_KM``) -- real retrieval errors are
   scene-correlated, and horizontally white noise is trivially averaged
   away by a CNN (2026-08-10).

Synthetic AIRS gap/swath fields (contiguous swaths, cloud-correlated
holes) live in :mod:`.synth_gaps`; this module degrades only the values,
not the coverage.

Pure NumPy (and xarray for the dataset-level entry point) -- no TensorFlow
import here, and no dependency on ``fronts/`` -- so this module is cheap to
unit test and safe to import from any stage of the pipeline.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from . import config

# --------------------------------------------------------------------------- #
# Constants (degrade.py-local per the workplan instruction to avoid touching
# config.py while other work is in flight there; cited inline).
# --------------------------------------------------------------------------- #

#: Hydrostatic scale height used to convert pressure levels to an
#: approximate geometric height z = -H * ln(p / 1000 hPa) for the vertical
#: smoothing kernel. 7.4 km is the mid-troposphere US Standard Atmosphere
#: value (workplan section 3.6 / AIRS averaging-kernel discussion).
SCALE_HEIGHT_KM = 7.4

#: AIRS effective vertical resolution (FWHM) for the log-p Gaussian mixing
#: matrix, SPLIT BY VARIABLE (2026-08-10): temperature retrievals resolve
#: ~1 km layers in the lower troposphere while water vapor resolves only
#: ~2-3 km layers (Susskind et al. 2003; Maddy & Barnet 2008) -- one shared
#: 2-km FWHM under-degraded q and over-degraded T.
T_FWHM_KM = 1.5
Q_FWHM_KM = 2.5

#: Additive temperature retrieval noise, ~1 K per ~1-km layer
#: (Divakarla et al. 2006 AIRS/AMSU validation).
T_NOISE_SIGMA_K = 1.0

#: Multiplicative water-vapor retrieval noise, ~15-20% per ~2-km layer
#: (Divakarla et al. 2006); 0.18 is the workplan's chosen point estimate.
Q_NOISE_FRAC_SIGMA = 0.18

#: Vertical AR(1) correlation of retrieval error between adjacent levels
#: -- AIRS weighting functions for adjacent levels overlap substantially,
#: so their retrieval errors are not independent (workplan section 3.6).
LEVEL_NOISE_RHO = 0.7

#: Horizontal e-folding length of the retrieval-error field (2026-08-10).
#: Real AIRS errors are scene-correlated -- cloud-contaminated retrievals
#: fail in coherent mesoscale patches, not pixel-by-pixel -- and white noise
#: is the easiest possible noise for a CNN to average away, so stage B was
#: overstating robustness.  ~300 km is a mesoscale cloud-system scale;
#: heuristic point estimate, revisit against real E3 residuals.
NOISE_CORR_KM = 300.0

#: Number of epochs over which stage B severity ramps 0 -> 1 linearly.
#: Raised 10 -> 41 (2026-08-10): stage B trains on overpass hours only
#: (~2 samples/day x 12 yr ~= 8,800 samples), so one full pass is ~14
#: epochs at 640 samples/epoch.  10 epochs (<1 pass) was a shock, not a
#: curriculum, for a model leaving a converged stage-A optimum; 41 epochs
#: = 3 round passes -- every sample is seen at low, medium and high
#: severity before full severity locks in.
DEFAULT_RAMP_EPOCHS = 41


# --------------------------------------------------------------------------- #
# 1. Vertical mixing matrix
# --------------------------------------------------------------------------- #

def vertical_mixing_matrix(levels_hpa=config.TARGET_LEVELS_HPA,
                           fwhm_km: float = None, *,
                           scale_height_km: float = SCALE_HEIGHT_KM
                           ) -> np.ndarray:
    """5x5 (in general len(levels_hpa)^2) row-stochastic Gaussian mixing matrix.

    Approximates the AIRS averaging kernel's vertical smoothing as a
    Gaussian in log-pressure height z = -H * ln(p / 1000 hPa)
    (H = ``scale_height_km``). Entry w_ij = exp(-0.5 * ((z_i - z_j) /
    sigma)^2) with sigma = fwhm_km / 2.355 (FWHM -> Gaussian sigma), each
    row renormalized to sum to 1 so applying the matrix to a profile is a
    weighted average (workplan section 3.6, "5-level mixing matrix" note,
    2026-08-04).

    ``fwhm_km`` is REQUIRED: pass ``T_FWHM_KM`` or ``Q_FWHM_KM`` -- the two
    variables resolve differently, so there is deliberately no shared
    default (2026-08-10).
    """
    if fwhm_km is None:
        raise TypeError("vertical_mixing_matrix: fwhm_km is required -- "
                        "pass T_FWHM_KM or Q_FWHM_KM")
    p = np.asarray(levels_hpa, dtype=np.float64)
    z = -scale_height_km * np.log(p / 1000.0)
    dz = z[:, None] - z[None, :]
    if fwhm_km <= 0.0:
        return np.eye(len(p))
    sigma = fwhm_km / 2.3548200450309493  # 2*sqrt(2*ln2): FWHM -> Gaussian sigma
    w = np.exp(-0.5 * (dz / sigma) ** 2)
    return w / w.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 2. Profile smoothing (NaN-aware application of the mixing matrix)
# --------------------------------------------------------------------------- #

def _apply_matrix_nan_aware(arr: np.ndarray, matrix: np.ndarray,
                            axis: int) -> np.ndarray:
    """Apply ``matrix`` along ``axis`` of ``arr``, renormalizing each output
    level's weights over the finite entries of that point's profile.

    A level that is itself NaN in the input (below-ground fill) is left
    NaN in the output -- we never invent a value there, only use finite
    inputs to smooth *other, finite* output levels.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = matrix.shape[0]
    arr_m = np.moveaxis(arr, axis, -1)                # (..., n)
    finite = np.isfinite(arr_m)
    filled = np.where(finite, arr_m, 0.0)
    out = np.full_like(arr_m, np.nan)
    for i in range(n):
        w = np.broadcast_to(matrix[i], arr_m.shape)
        w_eff = np.where(finite, w, 0.0)
        denom = w_eff.sum(axis=-1)
        numer = (filled * w_eff).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., i] = np.where(denom > 0, numer / denom, np.nan)
    out = np.where(finite, out, np.nan)                # below-ground stays NaN
    return np.moveaxis(out, -1, axis)


def smooth_profiles(t: np.ndarray, q: np.ndarray, axis: int,
                    matrix: np.ndarray | None = None,
                    t_matrix: np.ndarray | None = None,
                    q_matrix: np.ndarray | None = None):
    """Apply vertical mixing matrices to T and q profiles along ``axis``.

    T and q get SEPARATE matrices (T_FWHM_KM / Q_FWHM_KM) because AIRS
    resolves temperature more finely than water vapor.  ``matrix`` applies
    one shared matrix to both (legacy callers/tests); ``t_matrix``/
    ``q_matrix`` override per variable and default to the per-variable
    FWHM constants.  NaN-aware per :func:`_apply_matrix_nan_aware`.
    """
    if matrix is not None:
        t_matrix = q_matrix = matrix
    if t_matrix is None:
        t_matrix = vertical_mixing_matrix(fwhm_km=T_FWHM_KM)
    if q_matrix is None:
        q_matrix = vertical_mixing_matrix(fwhm_km=Q_FWHM_KM)
    t_s = _apply_matrix_nan_aware(t, t_matrix, axis)
    q_s = _apply_matrix_nan_aware(q, q_matrix, axis)
    return t_s, q_s


# --------------------------------------------------------------------------- #
# 3. Retrieval noise
# --------------------------------------------------------------------------- #

def horizontally_correlated_noise(shape: tuple, rng: np.random.Generator,
                                  corr_px: float,
                                  horizontal_axes: tuple = (-2, -1)
                                  ) -> np.ndarray:
    """Unit-variance Gaussian noise with ~``corr_px`` horizontal correlation.

    White noise is Gaussian-filtered over ``horizontal_axes`` (all other
    axes stay independent) and renormalized to unit sample variance.  The
    filter sigma is ``corr_px / 2`` because the autocorrelation of
    Gaussian-filtered white noise is exp(-d^2 / (4 sigma^2)), whose
    e-folding distance is d = 2 sigma -- so ``corr_px`` IS the e-folding
    length in pixels.  ``corr_px <= 0`` returns plain white noise.
    """
    from scipy import ndimage

    e = rng.standard_normal(shape)
    if corr_px <= 0:
        return e
    sigma = [0.0] * len(shape)
    for ax in horizontal_axes:
        sigma[ax % len(shape)] = corr_px / 2.0
    e = ndimage.gaussian_filter(e, sigma=sigma, mode="nearest")
    sd = e.std()
    return e / sd if sd > 0 else e


def _ar1_unit_variance(shape: tuple, axis: int, rho: float,
                       rng: np.random.Generator,
                       corr_px: float = 0.0,
                       horizontal_axes: tuple = (-2, -1)) -> np.ndarray:
    """Stationary unit-variance AR(1) noise along ``axis`` of ``shape``.

    e_0 ~ N(0, 1); e_k = rho * e_{k-1} + sqrt(1 - rho^2) * iid_k, iid_k ~
    N(0, 1). This recursion keeps Var(e_k) == 1 for all k (the classic
    AR(1) stationary-variance identity), so scaling e by a target sigma
    afterwards gives noise with exactly that marginal sigma.

    ``corr_px > 0`` makes each innovation field horizontally correlated
    (:func:`horizontally_correlated_noise`) BEFORE the vertical AR(1)
    chaining -- the result is correlated both along levels (rho) and
    across the map (corr_px), still with ~unit marginal variance.
    """
    n = shape[axis]
    iid = horizontally_correlated_noise(shape, rng, corr_px, horizontal_axes)
    iid_m = np.moveaxis(iid, axis, -1)                 # (..., n)
    e = np.empty_like(iid_m)
    e[..., 0] = iid_m[..., 0]
    scale = np.sqrt(1.0 - rho ** 2)
    for k in range(1, n):
        e[..., k] = rho * e[..., k - 1] + scale * iid_m[..., k]
    return np.moveaxis(e, -1, axis)


def add_noise(t: np.ndarray, q: np.ndarray, rng: np.random.Generator,
             t_sigma_k: float = T_NOISE_SIGMA_K,
             q_frac_sigma: float = Q_NOISE_FRAC_SIGMA,
             level_rho: float = LEVEL_NOISE_RHO, axis: int = -1,
             corr_px: float = 0.0, horizontal_axes: tuple = (-2, -1)):
    """Additive Gaussian T noise + multiplicative lognormal q noise, both
    AR(1)-correlated along the level axis.

    T: ``t + t_sigma_k * e_t``, e_t unit-variance AR(1) (Divakarla et al.
    2006: ~1 K/1-km layer).

    q: ``q * exp(sigma_ln * e_q - sigma_ln^2 / 2)`` with
    ``sigma_ln = sqrt(ln(1 + q_frac_sigma^2))`` so that, marginally
    (e_q ~ N(0, 1)), the multiplicative factor has mean 1 (mean-preserving)
    and the log of the factor has std sigma_ln, i.e. an approximately
    ``q_frac_sigma``-fraction multiplicative std (Divakarla et al. 2006:
    ~15-20 %/2-km layer). e_t and e_q are independent AR(1) draws sharing
    ``level_rho`` -- retrieval errors are vertically correlated because
    AIRS weighting functions for adjacent levels overlap.

    ``corr_px > 0`` additionally correlates both error fields over
    ``horizontal_axes`` (e-folding ``corr_px`` pixels) -- scene-correlated
    retrieval error, see ``NOISE_CORR_KM``.  The caller must ensure
    ``horizontal_axes`` does not include ``axis`` (the level axis).
    """
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    e_t = _ar1_unit_variance(t.shape, axis, level_rho, rng, corr_px,
                             horizontal_axes)
    e_q = _ar1_unit_variance(q.shape, axis, level_rho, rng, corr_px,
                             horizontal_axes)
    t_noisy = t + t_sigma_k * e_t
    if q_frac_sigma > 0:
        sigma_ln = np.sqrt(np.log(1.0 + q_frac_sigma ** 2))
        q_noisy = q * np.exp(sigma_ln * e_q - 0.5 * sigma_ln ** 2)
    else:
        q_noisy = q.copy()
    return t_noisy, q_noisy


# --------------------------------------------------------------------------- #
# 4. Severity curriculum
# --------------------------------------------------------------------------- #

def severity_ramp(epoch: int, ramp_epochs: int = DEFAULT_RAMP_EPOCHS) -> float:
    """Linear stage-B severity in [0, 1]: 0 at epoch 0, 1 at/after
    ``ramp_epochs`` (workplan section 3.6; 40-epoch default, see
    ``DEFAULT_RAMP_EPOCHS``).
    """
    if ramp_epochs <= 0:
        return 1.0
    return float(np.clip(epoch / ramp_epochs, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 5. Dataset-level entry point
# --------------------------------------------------------------------------- #

def degrade_day(day: xr.Dataset, rng: np.random.Generator,
                severity: float = 1.0) -> xr.Dataset:
    """Apply stage-B smoothing + noise to one MERRA-2 daily file's T/QV.

    ``day`` follows the schema of ``acquire_merra2.download_day`` /
    ``tests/test_front_dataset.py``: T, QV, U, V on dims
    (time, lev[hPa], lat, lon) plus PS(time, lat, lon), lev matching
    ``config.TARGET_LEVELS_HPA``. Winds (U, V) and surface pressure (PS)
    are returned untouched -- stage B degrades only the profile variables
    AIRS actually retrieves (workplan section 0, finding 1).

    Smoothing blends identity -> the full mixing matrix by ``severity``
    (a linear blend of two row-stochastic matrices is itself row-stochastic,
    so this is a valid partial smoothing, not just an approximation); T and
    q use separate matrices (T_FWHM_KM / Q_FWHM_KM).  Noise sigmas (T
    additive sigma, q multiplicative sigma) are scaled by ``severity``
    directly; the level-to-level correlation ``LEVEL_NOISE_RHO`` and the
    horizontal e-folding ``NOISE_CORR_KM`` are structural properties of the
    error process and are not ramped.  At ``severity == 0`` this is the
    identity map to floating-point precision.
    """
    dims = day["T"].dims
    lev_axis = dims.index("lev")
    horiz = (dims.index("lat"), dims.index("lon"))
    identity = np.eye(len(config.TARGET_LEVELS_HPA))
    blend = lambda full: (1.0 - severity) * identity + severity * full
    t_matrix = blend(vertical_mixing_matrix(fwhm_km=T_FWHM_KM))
    q_matrix = blend(vertical_mixing_matrix(fwhm_km=Q_FWHM_KM))

    t_smooth, q_smooth = smooth_profiles(
        day["T"].values, day["QV"].values, axis=lev_axis,
        t_matrix=t_matrix, q_matrix=q_matrix)
    t_noisy, q_noisy = add_noise(
        t_smooth, q_smooth, rng,
        t_sigma_k=severity * T_NOISE_SIGMA_K,
        q_frac_sigma=severity * Q_NOISE_FRAC_SIGMA,
        level_rho=LEVEL_NOISE_RHO, axis=lev_axis,
        corr_px=NOISE_CORR_KM / config.KM_PER_ITERATION,
        horizontal_axes=horiz)

    out = day.copy()
    out["T"] = (day["T"].dims, t_noisy.astype(day["T"].dtype))
    out["QV"] = (day["QV"].dims, q_noisy.astype(day["QV"].dtype))
    return out
