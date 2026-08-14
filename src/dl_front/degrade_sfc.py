"""Stage-B "AIRS simulator" degradation for the DL-FRONT surface inputs.

The AIRS-deployment premise (front_finder workplan section 0 + JPL data
audit): AIRS retrieves near-surface temperature and humidity but NOT winds
or sea-level pressure -- on the JPL laptop those channels come from the
paired forecast fields (FCST files), so stage B degrades only T2M and QV2M:

1. **Retrieval noise** -- reuses ``front_finder.degrade.add_noise``
   (additive Gaussian T, mean-preserving lognormal q) with ``level_rho=0``
   (single level, no vertical correlation to model).  Near-surface AIRS
   errors are the worst of the profile: T ~1.5-2 K, q ~20 %+ in the lowest
   2 km (Divakarla et al. 2006 AIRS/AMSU validation).
2. **Real retrieval-gap masks** -- sampled from the harvested AIRS
   valid-fraction bank (``front_finder.mask_bank``); the lowest stored
   level (1000 hPa) proxies the surface gap field.  Gap pixels are imputed
   to the standardized mean (0.0) and the loss still scores them
   (workplan 3.4: analyst fronts exist under cloud).

Severity in [0, 1] scales the noise sigmas and blends the gap field toward
all-valid, exactly as the UNET3+ stage B does.
"""
from __future__ import annotations

import numpy as np

from front_finder.degrade import add_noise

from . import config

# All three degradation knobs live in configs/dl_front.yaml (degradation:
# section) with their sources; aliased here because this module is their
# only consumer.  OBSERVED_MIN_FRACTION must equal
# front_finder.ingest_hysplit's (asserted in tests/test_dlfront_model.py).
OBSERVED_MIN_FRACTION = config.OBSERVED_MIN_FRACTION
T2M_NOISE_SIGMA_K = config.T2M_NOISE_SIGMA_K
Q2M_NOISE_FRAC_SIGMA = config.Q2M_NOISE_FRAC_SIGMA

_IT, _IQ = (config.SFC_VARS.index("T2M"), config.SFC_VARS.index("QV2M"))


def degrade_x(x: np.ndarray, rng: np.random.Generator, stats: dict,
              severity: float = 1.0, vf: np.ndarray | None = None
              ) -> np.ndarray:
    """Degrade standardized inputs x (..., 68, 141, 5) at ``severity``.

    Noise is applied in physical units (unstandardize -> perturb ->
    restandardize) because the q noise is multiplicative.  ``vf`` is a real
    (68, 141) valid-fraction field; pixels below OBSERVED_MIN_FRACTION are
    imputed to 0.0 (the standardized mean) on the T2M/QV2M channels only.
    """
    out = np.array(x, dtype=np.float32, copy=True)
    mt, st = stats["T2M"]
    mq, sq = stats["QV2M"]
    t_phys = out[..., _IT] * st + mt
    q_phys = out[..., _IQ] * sq + mq
    t_noisy, q_noisy = add_noise(
        t_phys, q_phys, rng,
        t_sigma_k=severity * T2M_NOISE_SIGMA_K,
        q_frac_sigma=severity * Q2M_NOISE_FRAC_SIGMA,
        level_rho=0.0, axis=-1)
    out[..., _IT] = (t_noisy - mt) / st
    out[..., _IQ] = (q_noisy - mq) / sq
    if vf is not None:
        vf_s = 1.0 - severity * (1.0 - vf)           # blend toward all-valid
        gap = vf_s < OBSERVED_MIN_FRACTION
        out[..., _IT] = np.where(gap, 0.0, out[..., _IT])
        out[..., _IQ] = np.where(gap, 0.0, out[..., _IQ])
    return out


def surface_gap_field(bank_vf: np.ndarray, rng: np.random.Generator,
                      month: int | None = None,
                      dates: np.ndarray | None = None) -> np.ndarray:
    """One real (68, 141) surface gap field: the lowest bank level.

    The bank is harvested by front_finder.ingest_hysplit, whose lowest
    target level (1000 hPa) sits BETWEEN the fullgrid's 985-hPa bin and the
    always-empty 1015-hPa bin: the linear vertical interp of the observed
    indicator is then 0.5 * obs(985), capping the stored valid fraction at
    0.5 -- exactly the OBSERVED_MIN_FRACTION threshold, which would punch
    out ~99 % of every field.  A bank-wide max <= 0.5 at the surface level
    is the unambiguous signature of that halving (a healthy bank always
    contains fully-surrounded pixels with vf == 1.0 somewhere), so the draw
    is rescaled by 2 to recover the true 985-hPa availability; a bank
    harvested after an ingest fix passes through untouched.
    """
    from front_finder import mask_bank

    vf = mask_bank.sample_mask(bank_vf, rng, month=month, dates=dates)[..., 0]
    if float(bank_vf[..., 0].max()) <= 0.5:
        vf = np.clip(vf * 2.0, 0.0, 1.0)
    return vf
