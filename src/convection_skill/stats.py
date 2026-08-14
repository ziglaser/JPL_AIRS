"""Gini inference for the unified suite -- BOTH uncertainty conventions.

Every entry point takes ``method``:

- ``"iid"``  -- the paper's construction (Richardson Methods > "Gini and
  significance"): resample rows iid with replacement, SD of the replicate
  Ginis is the SE. Default (decision 2026-07-22).
- ``"block"`` -- circular moving-block bootstrap over DAYS: whole days resample
  together in blocks of ``block_days``, preserving within-day spatial and
  synoptic temporal correlation. The audit showed iid SEs are 2-3x
  overconfident on this data (median inflation 2.35), so this is the honest
  alternative. Requires ``day_index``.

One convention drives the CIs/p-values of a run; both SEs are always reported
(``se_naive`` / ``se_block``) so the inflation stays visible.

Speed comes from one factorization: rows are sorted by predictor once and
aggregated into (rank-bin x day) count/event matrices, after which a block
replicate is a matvec and an iid replicate a multinomial/binomial draw.

Also here: :func:`conditional_gini` (skill beyond a control, rank-based),
:func:`event_rate_curve` (for non-monotonic hypotheses), :func:`fdr_bh`
(Benjamini-Hochberg field significance, Wilks 2016), and
:func:`weighted_gini` (exact Lorenz-area Gini, equals the paper's at unit
weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .gini import break_zero_ties

#: Rank bins for the binned backend. Binning error is O(1/bins) -- negligible
#: against the CI widths here (the paper's own construction uses 100 bins).
N_RANK_BINS: int = 512

METHODS = config.INFERENCE_METHODS  # ("iid", "block")


# --------------------------------------------------------------------------- #
# Gini on sorted / binned samples
# --------------------------------------------------------------------------- #
def _gini_from_sorted(flags_sorted: np.ndarray, weights_sorted: np.ndarray) -> float:
    """Exact Lorenz-area Gini for rows already sorted ascending by predictor."""
    total_weight = weights_sorted.sum()
    total_event_weight = float(np.dot(weights_sorted, flags_sorted))
    if total_weight <= 0 or total_event_weight <= 0:
        return np.nan
    x = np.concatenate([[0.0], np.cumsum(weights_sorted) / total_weight])
    y = np.concatenate([[0.0], np.cumsum(weights_sorted * flags_sorted) / total_event_weight])
    return float(2.0 * np.trapz(x - y, x))


def _gini_from_binned(counts: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Gini per replicate from per-bin (counts, events) of shape (bins, reps).

    Same Lorenz construction as :func:`_gini_from_sorted`, on binned capture
    curves, vectorized over replicate columns. No events -> NaN.
    """
    count_totals = counts.sum(axis=0)
    event_totals = events.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.vstack([np.zeros_like(count_totals), np.cumsum(counts, axis=0) / count_totals])
        y = np.vstack([np.zeros_like(event_totals), np.cumsum(events, axis=0) / event_totals])
    dx = np.diff(x, axis=0)
    mid_2x_minus_2y = (x[1:] + x[:-1]) - (y[1:] + y[:-1])
    gini = np.sum(mid_2x_minus_2y * dx, axis=0)
    return np.where((count_totals > 0) & (event_totals > 0), gini, np.nan)


def weighted_gini(
    predictor: np.ndarray,
    flags: np.ndarray,
    weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Exact (unbinned) Gini with optional per-row weights.

    Equals the paper's Gini at unit weights and full resolution. Zero-valued
    predictors get the paper's +/-1e-10 tiebreak.
    """
    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags, dtype=float)
    if weights is None:
        weights = np.ones_like(flags)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(predictor) & np.isfinite(flags) & np.isfinite(weights)
    predictor, flags, weights = predictor[valid], flags[valid], weights[valid]
    if predictor.size == 0:
        return np.nan
    order = np.argsort(break_zero_ties(predictor, rng=rng), kind="mergesort")
    return _gini_from_sorted(flags[order], weights[order])


# --------------------------------------------------------------------------- #
# The binned (rank-bin x day) backend, shared by both conventions
# --------------------------------------------------------------------------- #
def _bin_day_matrices(
    flags_sorted: np.ndarray,
    day_codes_sorted: np.ndarray,
    n_days: int,
    n_bins: int = N_RANK_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate predictor-sorted rows into (rank-bin x day) count/event matrices."""
    n = flags_sorted.size
    n_bins = min(n_bins, n)
    bin_index = (np.arange(n) * n_bins) // n
    counts = np.zeros((n_bins, n_days))
    events = np.zeros((n_bins, n_days))
    np.add.at(counts, (bin_index, day_codes_sorted), 1.0)
    np.add.at(events, (bin_index, day_codes_sorted), flags_sorted)
    return counts, events


def _block_weights(n_days: int, block_days: int, n_reps: int,
                   rng: np.random.Generator) -> np.ndarray:
    """(n_days x n_reps) day-multiplicity matrix from circular moving blocks."""
    n_blocks = int(np.ceil(n_days / block_days))
    weights = np.zeros((n_days, n_reps))
    for r in range(n_reps):
        starts = rng.integers(0, n_days, size=n_blocks)
        picked = (starts[:, None] + np.arange(block_days)[None, :]).ravel()
        picked = picked[:n_days] % n_days
        weights[:, r] = np.bincount(picked, minlength=n_days)
    return weights


def _iid_replicates(counts_per_bin: np.ndarray, events_per_bin: np.ndarray,
                    n_reps: int, rng: np.random.Generator,
                    sample_size: Optional[int] = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(counts, events) of shape (bins, reps) under the paper's iid row bootstrap.

    An iid row bootstrap is a multinomial over rank bins (probabilities = bin
    counts / n); events among a bin's drawn rows follow that bin's event
    fraction. ``sample_size`` defaults to the full n (Richardson's Fig. 3
    variant resamples down to a one-hour size; pass it explicitly there).
    """
    n = int(counts_per_bin.sum())
    if sample_size is None:
        sample_size = n
    with np.errstate(invalid="ignore", divide="ignore"):
        event_frac = np.where(counts_per_bin > 0, events_per_bin / counts_per_bin, 0.0)
    drawn = rng.multinomial(sample_size, counts_per_bin / n, size=n_reps).T
    drawn_events = rng.binomial(drawn, event_frac[:, None]).astype(float)
    return drawn.astype(float), drawn_events


def _two_sided_p(boot: np.ndarray, point: float) -> float:
    """Two-sided bootstrap p for H0: Gini = 0, from the centered distribution."""
    centered = boot - boot.mean()
    p = 2.0 * min((centered >= point).mean(), (centered <= point).mean())
    return float(min(max(p, 1.0 / (boot.size + 1)), 1.0))


def _check_method(method: str, day_index) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "block" and day_index is None:
        raise ValueError("method='block' requires day_index")


# --------------------------------------------------------------------------- #
# The one bootstrap entry point
# --------------------------------------------------------------------------- #
@dataclass
class GiniResult:
    """One Gini estimate; CIs/p from the chosen ``method``, both SEs reported."""

    gini: float
    n: int
    n_events: int
    ci_lo: float          # 2.5% of the chosen method's distribution
    ci_hi: float          # 97.5%
    se_block: float       # day-block bootstrap SE (NaN without day_index)
    se_naive: float       # iid-row bootstrap SE (the paper's construction)
    p_value: float        # two-sided p for H0: Gini = 0, chosen method
    n_days: int
    block_days: int
    method: str = "iid"

    @property
    def inflation(self) -> float:
        """How much wider honest (block) errors are than naive iid errors."""
        if self.se_naive > 0:
            return self.se_block / self.se_naive
        return np.nan


def bootstrap_gini(
    predictor: np.ndarray,
    flags: np.ndarray,
    day_index: Optional[np.ndarray] = None,
    method: str = "iid",
    n_reps: int = config.N_BOOT_REPS,
    block_days: int = config.BLOCK_DAYS,
    rng: Optional[np.random.Generator] = None,
) -> GiniResult:
    """Gini with bootstrap uncertainty under the chosen convention.

    ``method="iid"`` needs no day structure; ``method="block"`` resamples whole
    days in circular moving blocks (pass ``day_index``). When ``day_index`` is
    given, BOTH SEs are computed regardless of method (the off-method
    distribution runs at reduced reps) so the inflation column stays populated.
    """
    _check_method(method, day_index)
    rng = rng or np.random.default_rng(0)
    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags, dtype=float)

    valid = np.isfinite(predictor) & np.isfinite(flags)
    predictor, flags = predictor[valid], flags[valid]
    if day_index is not None:
        day_index = np.asarray(day_index)[valid]
    if predictor.size == 0 or flags.sum() == 0:
        return GiniResult(gini=np.nan, n=int(predictor.size), n_events=int(flags.sum()),
                          ci_lo=np.nan, ci_hi=np.nan, se_block=np.nan, se_naive=np.nan,
                          p_value=np.nan, n_days=0, block_days=block_days, method=method)

    order = np.argsort(break_zero_ties(predictor, rng=rng), kind="mergesort")
    flags_sorted = flags[order]
    point = _gini_from_sorted(flags_sorted, np.ones_like(flags_sorted))

    if day_index is not None:
        days, day_codes = np.unique(day_index, return_inverse=True)
        n_days = days.size
        counts, events = _bin_day_matrices(flags_sorted, day_codes[order], n_days)
    else:
        n_days = 0
        counts, events = _bin_day_matrices(
            flags_sorted, np.zeros(flags_sorted.size, dtype=int), 1)

    side_reps = min(n_reps, 200)  # off-method distribution, for the SE column only

    naive_reps = n_reps if method == "iid" else side_reps
    iid_boot = _gini_from_binned(*_iid_replicates(
        counts.sum(axis=1), events.sum(axis=1), naive_reps, rng))
    iid_boot = iid_boot[np.isfinite(iid_boot)]

    block_len = max(1, min(block_days, n_days)) if n_days else 0
    if day_index is not None:
        block_reps = n_reps if method == "block" else side_reps
        day_weights = _block_weights(n_days, block_len, block_reps, rng)
        block_boot = _gini_from_binned(counts @ day_weights, events @ day_weights)
        block_boot = block_boot[np.isfinite(block_boot)]
    else:
        block_boot = np.array([])

    chosen = iid_boot if method == "iid" else block_boot
    return GiniResult(
        gini=point, n=int(flags_sorted.size), n_events=int(flags_sorted.sum()),
        ci_lo=float(np.percentile(chosen, 2.5)) if chosen.size else np.nan,
        ci_hi=float(np.percentile(chosen, 97.5)) if chosen.size else np.nan,
        se_block=float(block_boot.std(ddof=1)) if block_boot.size > 1 else np.nan,
        se_naive=float(iid_boot.std(ddof=1)) if iid_boot.size > 1 else np.nan,
        p_value=_two_sided_p(chosen, point) if chosen.size else np.nan,
        n_days=int(n_days), block_days=int(block_len), method=method,
    )


def block_bootstrap_gini(predictor, flags, day_index, **kw) -> GiniResult:
    """Back-compat alias: :func:`bootstrap_gini` with ``method="block"``."""
    kw.pop("method", None)
    return bootstrap_gini(predictor, flags, day_index, method="block", **kw)


# --------------------------------------------------------------------------- #
# Conditional (within-control-bins) Gini
# --------------------------------------------------------------------------- #
def _control_bins(control: np.ndarray, n_bins: int, rng) -> np.ndarray:
    """Equal-count bin index of each row, ranked by the control variable."""
    n = control.size
    order = np.argsort(break_zero_ties(control, rng=rng), kind="mergesort")
    bin_of = np.empty(n, dtype=int)
    bin_of[order] = (np.arange(n) * n_bins) // n
    return bin_of


def _pooled_gini_point(bins: list[dict]) -> tuple[float, list[tuple[float, float]]]:
    """Event-weighted mean of per-bin point Ginis, plus the per-bin values."""
    per_bin, num, den = [], 0.0, 0.0
    for b in bins:
        g = float(_gini_from_binned(b["counts"].sum(axis=1, keepdims=True),
                                    b["events"].sum(axis=1, keepdims=True))[0])
        per_bin.append((g, b["n_events"]))
        if np.isfinite(g):
            num += b["n_events"] * g
            den += b["n_events"]
    return (num / den if den > 0 else np.nan), per_bin


def _pooled_gini_replicates(bins: list[dict], method: str, n_reps: int,
                            n_days: int, block_days: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Pooled conditional Gini per replicate under the chosen convention."""
    if method == "block":
        day_weights = _block_weights(n_days, block_days, n_reps, rng)
    numerator = np.zeros(n_reps)
    denominator = np.zeros(n_reps)
    for b in bins:
        if method == "block":
            counts = b["counts"] @ day_weights
            events = b["events"] @ day_weights
        else:  # iid rows within each control bin
            counts, events = _iid_replicates(
                b["counts"].sum(axis=1), b["events"].sum(axis=1), n_reps, rng)
        gini = _gini_from_binned(counts, events)
        event_weight = events.sum(axis=0)
        ok = np.isfinite(gini) & (event_weight > 0)
        numerator[ok] += event_weight[ok] * gini[ok]
        denominator[ok] += event_weight[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def conditional_gini(
    predictor: np.ndarray,
    flags: np.ndarray,
    control: np.ndarray,
    day_index: Optional[np.ndarray] = None,
    method: str = "iid",
    n_control_bins: int = config.N_CONTROL_BINS,
    min_events_per_bin: int = config.MIN_EVENTS_PER_BIN,
    n_reps: int = config.N_BOOT_REPS,
    block_days: int = config.BLOCK_DAYS,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Gini of ``predictor`` within equal-count bins of ``control``, pooled.

    A predictor that merely proxies the control scores ~0 (no residual ordering
    skill within a control bin); an independent signal keeps its marginal Gini.
    Pooling is event-weighted across bins with at least ``min_events_per_bin``
    events. CI/p come from the ``method`` convention.
    """
    _check_method(method, day_index)
    rng = rng or np.random.default_rng(0)
    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags, dtype=float)
    control = np.asarray(control, dtype=float)

    empty = {"conditional_gini": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
             "p_value": np.nan, "per_bin": [], "n_bins_used": 0}
    valid = np.isfinite(predictor) & np.isfinite(flags) & np.isfinite(control)
    if day_index is not None:
        day_index = np.asarray(day_index)[valid]
    predictor, flags, control = predictor[valid], flags[valid], control[valid]
    if predictor.size == 0 or flags.sum() == 0:
        return empty

    if day_index is None:
        day_index = np.zeros(predictor.size, dtype=int)  # single pseudo-day
    days, day_codes = np.unique(day_index, return_inverse=True)
    n_days = days.size

    bin_of = _control_bins(control, n_control_bins, rng)
    bins = []
    for b in range(n_control_bins):
        in_bin = bin_of == b
        if flags[in_bin].sum() < min_events_per_bin:
            continue
        order = np.argsort(break_zero_ties(predictor[in_bin], rng=rng), kind="mergesort")
        counts, events = _bin_day_matrices(
            flags[in_bin][order], day_codes[in_bin][order], n_days)
        bins.append({"counts": counts, "events": events,
                     "n_events": float(flags[in_bin].sum())})
    if not bins:
        return empty

    point, per_bin = _pooled_gini_point(bins)
    block_len = max(1, min(block_days, n_days))
    boot = _pooled_gini_replicates(bins, method, n_reps, n_days, block_len, rng)
    boot = boot[np.isfinite(boot)]

    ci_lo = ci_hi = p = np.nan
    if boot.size:
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        p = _two_sided_p(boot, point)
    return {"conditional_gini": float(point), "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi), "p_value": p, "per_bin": per_bin,
            "n_bins_used": len(bins)}


# --------------------------------------------------------------------------- #
# Event-rate curve (for non-monotonic hypotheses, e.g. A2 CIN)
# --------------------------------------------------------------------------- #
def event_rate_curve(
    predictor: np.ndarray,
    flags: np.ndarray,
    day_index: Optional[np.ndarray] = None,
    method: str = "iid",
    n_bins: int = 20,
    n_reps: int = 300,
    block_days: int = config.BLOCK_DAYS,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """P(event) per equal-count predictor bin, with bootstrap bands.

    Returns bin centers (median predictor per bin), rates, and 2.5/97.5% bands
    under the chosen convention.
    """
    _check_method(method, day_index)
    rng = rng or np.random.default_rng(0)
    predictor = np.asarray(predictor, dtype=float)
    flags = np.asarray(flags, dtype=float)
    valid = np.isfinite(predictor) & np.isfinite(flags)
    if day_index is not None:
        day_index = np.asarray(day_index)[valid]
    predictor, flags = predictor[valid], flags[valid]

    bin_of = _control_bins(predictor, n_bins, rng)
    centers = np.array([np.median(predictor[bin_of == b]) for b in range(n_bins)])
    rate = np.array([flags[bin_of == b].mean() for b in range(n_bins)])

    if method == "block":
        days, day_codes = np.unique(day_index, return_inverse=True)
        n_days = days.size
        counts = np.zeros((n_bins, n_days))
        events = np.zeros((n_bins, n_days))
        np.add.at(counts, (bin_of, day_codes), 1.0)
        np.add.at(events, (bin_of, day_codes), flags)
        day_weights = _block_weights(n_days, max(1, min(block_days, n_days)),
                                     n_reps, rng)
        drawn_counts = counts @ day_weights
        drawn_events = events @ day_weights
    else:  # iid rows: multinomial over bins, binomial events per bin
        counts_per_bin = np.bincount(bin_of, minlength=n_bins).astype(float)
        events_per_bin = np.bincount(bin_of, weights=flags, minlength=n_bins)
        drawn_counts, drawn_events = _iid_replicates(
            counts_per_bin, events_per_bin, n_reps, rng)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_rates = np.where(drawn_counts > 0, drawn_events / drawn_counts, np.nan)
    lo = np.nanpercentile(boot_rates, 2.5, axis=1)
    hi = np.nanpercentile(boot_rates, 97.5, axis=1)
    return {"centers": centers, "rate": rate, "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Battery-wide multiplicity control
# --------------------------------------------------------------------------- #
def fdr_bh(p_values: np.ndarray, alpha: float = config.FDR_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg 'significant' mask controlling FDR at ``alpha``.

    The field-significance guard for a battery of many tests (Wilks 2016).
    NaN p-values are never significant.
    """
    p = np.asarray(p_values, dtype=float)
    significant = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return significant

    pv = p[finite]
    m = pv.size
    order = np.argsort(pv)
    passed = pv[order] <= alpha * (np.arange(1, m + 1) / m)
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        sig_sorted = np.arange(m) <= cutoff
        sig = np.zeros(m, dtype=bool)
        sig[order] = sig_sorted
        significant[finite] = sig
    return significant
