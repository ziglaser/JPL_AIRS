"""Significance tests for Gini coefficients (paper Fig. 3 claims).

Two procedures, both transcribed from Richardson et al. (2024), Methods > "Gini
and significance calculations":

1. Bootstrap standard error (:func:`bootstrap_gini_se`): resample -- with
   replacement -- from the pooled six-forecast-hour predictor down to a one-hour
   sample size, recompute the Gini 500 times, and take the SD as the 1 sigma SE.
   Because hours are treated as independent, an hour-to-hour *difference* is
   significant at p<0.05 when it exceeds ``2*sqrt(2)*sigma`` (~0.08 in the paper).

2. Hourly trend (:func:`hourly_trend_test`): OLS of Gini vs forecast hour; the
   trend is significant at p<0.05 when ``|slope| > 2 * SE(slope)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from . import config
from .gini import gini


@dataclass
class BootstrapResult:
    """Outcome of the pooled-resample bootstrap for one predictor."""

    mean_gini: float      # mean Gini across bootstrap replicates
    se: float             # 1 sigma standard error (SD of replicate Ginis)
    reps: int             # number of replicates actually computed
    sample_size: int      # one-hour sample size drawn each replicate

    @property
    def diff_significance_threshold(self) -> float:
        """Smallest hour-to-hour Gini difference significant at p<0.05.

        Methods: hours independent => SE(difference) = sqrt(2)*sigma, and the p<0.05
        level is twice that, 2*sqrt(2)*sigma.
        """
        return config.DIFF_SIGNIFICANCE_MULTIPLIER * self.se


def bootstrap_gini_se(
    predictor: np.ndarray,
    flags: np.ndarray,
    sample_size: int,
    reps: int = config.BOOTSTRAP_REPS,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapResult:
    """Bootstrap the 1 sigma standard error of a Gini coefficient.

    Methods: "randomly sampling (with replacement) all values of the predictor from
    all six forecast hours down to a sample size equal to one hour. The Gini
    coefficient is then calculated for this new sample, and the resampling
    procedure is repeated 500 times and the standard deviation of the 500 Gini
    coefficients is assumed to be the 1 sigma standard error."

    Parameters
    ----------
    predictor, flags
        The *pooled* predictor values and event flags across all forecast hours.
    sample_size
        The one-hour sample size to draw each replicate (e.g. rows in one hour).
    reps
        Number of bootstrap replicates (default 500).
    rng
        Random generator (seeded from config by default for reproducibility).
    """
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)
    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags)
    n_pool = predictor.size

    ginis = np.empty(reps, dtype=float)
    computed = 0
    for i in range(reps):
        idx = rng.integers(0, n_pool, size=sample_size)
        sample_flags = flags[idx]
        if sample_flags.sum() == 0:
            # A resample with zero events has an undefined Gini; skip it. Rare
            # unless events are extremely sparse relative to sample_size.
            continue
        ginis[computed] = gini(predictor[idx], sample_flags, rng=rng)
        computed += 1

    ginis = ginis[:computed]
    return BootstrapResult(
        mean_gini=float(np.mean(ginis)),
        se=float(np.std(ginis, ddof=1)),
        reps=computed,
        sample_size=sample_size,
    )


@dataclass
class TrendResult:
    """Outcome of the OLS Gini-vs-hour trend test."""

    slope: float          # Gini change per forecast hour
    slope_se: float       # standard error of the slope
    intercept: float
    r_value: float
    significant: bool     # |slope| > 2 * slope_se

    def __str__(self) -> str:
        flag = "significant" if self.significant else "not significant"
        return (f"slope = {self.slope:+.4f} +/- {self.slope_se:.4f} per hour "
                f"({flag} at p<0.05)")


def hourly_trend_test(hours: np.ndarray, ginis: np.ndarray) -> TrendResult:
    """OLS trend of Gini vs forecast hour, significant if |slope| > 2*SE(slope).

    Methods: "ordinary least squares was used to calculate the hourly trend in Gini
    coefficient ... and twice the slope standard error was treated as significant
    at p<0.05."

    ``hours`` should be a monotonic forecast-hour index (e.g. 0..5 for the six
    forecast steps), NOT wall-clock UTC, since 21,22,23,0,1,2 wraps past midnight.
    """
    hours = np.asarray(hours, dtype=float)
    ginis = np.asarray(ginis, dtype=float)
    reg = stats.linregress(hours, ginis)
    significant = abs(reg.slope) > config.TREND_SIGNIFICANCE_MULTIPLIER * reg.stderr
    return TrendResult(
        slope=float(reg.slope),
        slope_se=float(reg.stderr),
        intercept=float(reg.intercept),
        r_value=float(reg.rvalue),
        significant=bool(significant),
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    rng = np.random.default_rng(0)
    n = 120_000
    pred = rng.random(n)
    flags = rng.random(n) < pred ** 3 * 0.02
    res = bootstrap_gini_se(pred, flags, sample_size=n // 6, rng=rng)
    print(f"Gini = {res.mean_gini:.3f} +/- {res.se:.3f} "
          f"(p<0.05 hour-diff threshold {res.diff_significance_threshold:.3f})")
