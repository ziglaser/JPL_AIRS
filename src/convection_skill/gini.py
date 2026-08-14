from __future__ import annotations
from typing import Optional
import numpy as np

from . import config

def exceedance_flags(target: np.ndarray, percentile: float) -> np.ndarray:
    threshold = float(np.nanpercentile(target, percentile))
    return target > threshold


def break_zero_ties(
    predictor: np.ndarray,
    eps: float = config.TIEBREAK_EPS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)
    out = np.asarray(predictor, dtype=float).copy()
    is_zero = (out == 0.0)
    n_zero = int(is_zero.sum())
    if n_zero:
        out[is_zero] += rng.uniform(-eps, eps, size=n_zero)
    return out


# --------------------------------------------------------------------------- #
# Detection CDF (the "event-capture curve")
# --------------------------------------------------------------------------- #
def detection_cdf(
    predictor: np.ndarray,
    flags: np.ndarray,
    n_bins: int = config.N_BINS,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:

    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags, dtype=float)
    valid = np.isfinite(predictor) & np.isfinite(flags)
    predictor = predictor[valid]
    flags = flags[valid]

    n = predictor.size
    if n == 0:
        raise ValueError("detection_cdf received no valid samples.")
    total_events = flags.sum()
    if total_events == 0:
        raise ValueError(
            "detection_cdf received no events (all flags are 0); "
            "the event fraction is undefined."
        )

    # Sort ascending by the tie-broken predictor.
    perturbed = break_zero_ties(predictor, rng=rng)
    order = np.argsort(perturbed, kind="mergesort")  # stable; ties already broken
    flags_sorted = flags[order]

    # Assign each rank to one of n_bins near-equal-count bins.
    ranks = np.arange(n)
    bin_index = (ranks * n_bins) // n  # values 0 .. n_bins-1

    # Events per bin, then cumulative fraction of all events.
    events_per_bin = np.bincount(bin_index, weights=flags_sorted, minlength=n_bins)
    cumulative_events = np.concatenate([[0.0], np.cumsum(events_per_bin)])
    event_fraction = cumulative_events / total_events

    sample_fraction = np.arange(n_bins + 1) / n_bins
    return sample_fraction, event_fraction


# --------------------------------------------------------------------------- #
# Gini coefficient
# --------------------------------------------------------------------------- #
def gini_from_cdf(sample_fraction: np.ndarray, event_fraction: np.ndarray) -> float:
    """Gini = 2 x (area between the 1:1 diagonal and the detection curve).

    With the predictor sorted ascending, an informative predictor keeps events out
    of the low bins, so the curve lies *below* the diagonal and the area is
    positive. A uniform predictor lies on the diagonal (Gini = 0); a predictor that
    perfectly isolates rare events approaches Gini = 1.
    """
    area_between = np.trapz(sample_fraction - event_fraction, sample_fraction)
    return float(2.0 * area_between)


def gini(
    predictor: np.ndarray,
    flags: np.ndarray,
    n_bins: int = config.N_BINS,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Convenience wrapper: detection CDF -> Gini coefficient.

    This is the single call the analysis and the SMAP extension use::

        g = gini(table["mu_cape"].to_numpy(), flags)   # CAPE (paper)
        g = gini(table["smap_smsfc_av"].to_numpy(), flags)  # SMAP (extension)
    """
    x, y = detection_cdf(predictor, flags, n_bins=n_bins, rng=rng)
    return gini_from_cdf(x, y)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # Runnable demo so the module can be inspected standalone.
    rng = np.random.default_rng(0)
    n = 100_000
    pred = rng.random(n)
    # Events concentrated at high predictor: probability rises with the predictor.
    flags_demo = rng.random(n) < (pred ** 3) * 0.05
    print(f"events: {flags_demo.sum()} of {n}")
    print(f"informative predictor  Gini = {gini(pred, flags_demo):.3f}")
    print(f"shuffled  predictor    Gini = {gini(rng.permutation(pred), flags_demo):.3f}")
