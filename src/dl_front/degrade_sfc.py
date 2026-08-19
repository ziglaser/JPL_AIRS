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
2. **Real retrieval-gap masks** -- sampled from the terrain-following
   surface gap bank (``swath.sample_gap_field``, harvested alongside the
   swath bank; user decision 2026-08-16).  Gap pixels are imputed to the
   standardized mean (0.0) and the loss still scores them (workplan 3.4:
   analyst fronts exist under cloud).

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

def _tq_indices() -> tuple[int, int]:
    """Positions of T2M/QV2M on the x channel axis, resolved per call.

    These used to be module-level constants derived from
    ``config.SFC_VARS``.  That was correct only while the x channel axis WAS
    ``SFC_VARS``; since the channel-subsetting work the axis is
    ``config.INPUT_CHANNELS`` (integration 2026-08-18), so a frozen
    ``SFC_VARS`` index silently addresses the WRONG channel under
    ``--channels`` -- e.g. ``--channels QV2M,SLP`` would degrade SLP using
    T2M's noise sigma and QV2M's norm stats, producing plausible-looking
    but meaningless inputs.  Resolving at call time also means a
    ``--channels`` applied after import (which is how the CLIs do it) is
    honoured rather than baked in at module-import time.
    """
    ch = config.INPUT_CHANNELS
    missing = [v for v in ("T2M", "QV2M") if v not in ch]
    if missing:
        raise ValueError(
            f"AIRS-simulator degradation needs {missing} on the model's "
            f"input channels, but config.INPUT_CHANNELS is "
            f"{list(ch)} -- degrade_sfc perturbs the retrieved "
            "temperature/humidity pair and has no meaning without them.  "
            "Either drop --degraded (and any 'kriged-degraded' source) or "
            "widen --channels to include T2M,QV2M (or the 'inputs: "
            "channels:' list in configs/dl_front.yaml).")
    return ch.index("T2M"), ch.index("QV2M")


def degrade_x(x: np.ndarray, rng: np.random.Generator, stats: dict,
              severity: float = 1.0, vf: np.ndarray | None = None
              ) -> np.ndarray:
    """Degrade standardized inputs x (..., 68, 141, n_ch) at ``severity``.

    ``n_ch`` is ``len(config.INPUT_CHANNELS)`` -- the channels the MODEL
    consumes, not the five on-disk ``config.SFC_VARS``.

    Noise is applied in physical units (unstandardize -> perturb ->
    restandardize) because the q noise is multiplicative.  ``vf`` is a real
    (68, 141) valid-fraction field; pixels below OBSERVED_MIN_FRACTION are
    imputed to 0.0 (the standardized mean) on the T2M/QV2M channels only.
    """
    _IT, _IQ = _tq_indices()
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


def surface_gap_field(rng: np.random.Generator, month: int | None = None,
                      hour: int | None = None) -> np.ndarray:
    """One real (68, 141) surface gap field from the terrain-following bank.

    Draws from ``swath``'s sfc_gap_bank (harvested alongside the swath bank
    by the SAME terrain-following extraction stage C uses, user decision
    2026-08-16) -- the retired front_finder gap_bank stored fixed-level
    (1000 hPa) fields that punched out ALL elevated terrain, so stage B
    simulated a permanent western void stage C never had.  Season- and
    hour-conditional sampling lives in :func:`swath.sample_gap_field`.
    """
    from . import swath

    return swath.sample_gap_field(rng, month=month, hour=hour)
