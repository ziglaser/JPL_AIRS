"""Reusable analysis computations shared by the figure scripts.

These sit one level above the predictor-agnostic :mod:`convection_skill.gini`
core: they know the *shape* of the analysis table (a ``qpe`` target column, an
``hour_utc`` column, one predictor column per candidate), but still take the
predictor column *by name*, so every routine works unchanged for a SMAP predictor.

Keeping these here means Phases 3-6 all call the same, tested implementation of
"Gini across thresholds" and "Gini per forecast hour + significance" rather than
each re-deriving it.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .gini import detection_cdf, exceedance_flags, gini_from_cdf
from .significance import BootstrapResult, TrendResult, bootstrap_gini_se, hourly_trend_test


def pooled_event_flags(
    table: pd.DataFrame,
    percentile: float,
    target_col: str = "qpe",
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Event flags for ``target_col > QPE_X`` with the threshold from the *pooled*
    sample -- the paper's rule that thresholds use "all locations, all seasons,
    both wet and dry hours."

    Pass ``threshold`` (in target units) to use a threshold derived from a
    *broader* base sample than ``table`` itself -- the paper derives QPE_X from
    ALL in-domain land rows including those without valid AIRS data (see
    :func:`convection_skill.quality_control.threshold_base`), then evaluates
    skill on the screened rows. If ``threshold`` is None the percentile of
    ``table``'s own target column is used.
    """
    if threshold is None:
        return exceedance_flags(table[target_col].to_numpy(), percentile)
    return table[target_col].to_numpy() > threshold


def gini_for(
    predictor: np.ndarray, flags: np.ndarray, rng_seed: int = config.RANDOM_SEED
) -> float:
    """Gini for one predictor/flags pair, dropping NaNs pairwise via detection_cdf."""
    x, y = detection_cdf(predictor, flags, rng=np.random.default_rng(rng_seed))
    return gini_from_cdf(x, y)


def gini_by_percentile(
    table: pd.DataFrame,
    predictor_cols: Sequence[str],
    percentiles: Sequence[float] = config.QPE_PERCENTILES,
    target_col: str = "qpe",
    rng_seed: int = config.RANDOM_SEED,
    match_valid: bool = True,
    thresholds: Optional[Mapping[float, float]] = None,
) -> pd.DataFrame:
    """Gini for each predictor across QPE-threshold percentiles (Fig. 2b).

    Returns a DataFrame indexed by percentile with one column per predictor.

    If ``match_valid`` (default), all predictors are evaluated on the identical
    sample where *every* predictor column is finite -- a fair head-to-head, needed
    e.g. when comparing forecast CAPE against the sometimes-missing overpass CAPE
    (or against a SMAP field with its own gaps). Pass ``match_valid=False`` to
    score each predictor on its own valid rows instead (the reading that matches
    the paper's overpass-baseline curves; see tests/test_paper_benchmarks.py).

    ``thresholds`` optionally maps percentile -> absolute threshold derived from
    a broader base sample (paper: "thresholds are based on all data"); if None,
    thresholds come from ``table``'s own pooled target column.
    """
    mask = np.ones(len(table), dtype=bool)
    if match_valid:
        for col in predictor_cols:
            mask &= np.isfinite(table[col].to_numpy())
    sub = table[mask]
    target = sub[target_col].to_numpy()

    out = {}
    for col in predictor_cols:
        pred = sub[col].to_numpy()
        valid = np.isfinite(pred)
        col_gini = []
        for p in percentiles:
            flags = (target > thresholds[p]) if thresholds is not None \
                else exceedance_flags(target, p)
            col_gini.append(gini_for(pred[valid], flags[valid], rng_seed))
        out[col] = col_gini
    return pd.DataFrame(out, index=pd.Index(percentiles, name="percentile"))


def hourly_gini(
    table: pd.DataFrame,
    predictor_col: str,
    percentile: float = config.HEADLINE_PERCENTILE,
    target_col: str = "qpe",
    rng_seed: int = config.RANDOM_SEED,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Gini per forecast hour for one predictor (Fig. 3), pooled-sample threshold.

    Returns a DataFrame with columns [step, hour_utc, gini], ordered by forecast
    step (0..5 for 21,22,23,0,1,2 UTC). ``threshold`` as in
    :func:`pooled_event_flags`.
    """
    flags = pooled_event_flags(table, percentile, target_col, threshold)
    tbl = table.assign(_event=flags)
    rows = []
    for step, hour in enumerate(config.FORECAST_HOURS_UTC):
        sub = tbl[tbl["hour_utc"] == hour]
        pred = sub[predictor_col].to_numpy()
        ev = sub["_event"].to_numpy()
        valid = np.isfinite(pred)
        rows.append(dict(step=step, hour_utc=hour,
                         gini=gini_for(pred[valid], ev[valid], rng_seed)))
    return pd.DataFrame(rows)


def stratified_gini(
    table: pd.DataFrame,
    predictor_col: str,
    strata,
    percentile: float = config.HEADLINE_PERCENTILE,
    target_col: str = "qpe",
    rng_seed: int = config.RANDOM_SEED,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Gini of ``predictor_col`` computed *within* each stratum.

    This is the key primitive for the SMAP extension: it answers "does CAPE's
    skill at sorting extreme precipitation differ across soil-moisture regimes (or
    regions)?" -- the regime-dependence the literature review flags (Tuttle &
    Salvucci 2016; Guillod et al. 2015).

    Parameters
    ----------
    predictor_col
        Column to score (e.g. ``"mu_cape"``, or any SMAP field).
    strata
        Array-like of stratum labels aligned to ``table`` rows (e.g. soil-moisture
        terciles from ``pd.qcut``, or an east/plains region label). Kept as a plain
        label array so this function stays agnostic to *how* strata were defined.
    percentile
        Event threshold; computed once on the *whole* table (not per stratum) so the
        strata are compared against the same globally-extreme events.

    Returns
    -------
    DataFrame with columns [stratum, n, n_events, gini], one row per stratum.
    """
    flags = pooled_event_flags(table, percentile, target_col, threshold)
    df = pd.DataFrame({
        "pred": table[predictor_col].to_numpy(),
        "flag": flags,
        "stratum": np.asarray(strata),
    })
    rows = []
    for name, grp in df.groupby("stratum", observed=True):
        valid = np.isfinite(grp["pred"].to_numpy())
        pred = grp["pred"].to_numpy()[valid]
        ev = grp["flag"].to_numpy()[valid]
        n_events = int(ev.sum())
        g = gini_for(pred, ev, rng_seed) if n_events > 0 else float("nan")
        rows.append(dict(stratum=name, n=int(valid.sum()), n_events=n_events, gini=g))
    return pd.DataFrame(rows)


def hourly_significance(
    table: pd.DataFrame,
    predictor_col: str,
    percentile: float = config.HEADLINE_PERCENTILE,
    target_col: str = "qpe",
    rng_seed: int = config.RANDOM_SEED,
    threshold: Optional[float] = None,
) -> tuple[pd.DataFrame, BootstrapResult, TrendResult]:
    """Per-hour Gini + bootstrap SE + OLS trend for one predictor (Fig. 3 stats).

    Returns ``(hourly_df, bootstrap, trend)`` where:
    - ``hourly_df`` is :func:`hourly_gini`'s table with an added ``se`` column;
    - ``bootstrap`` is the pooled-resample-to-one-hour :class:`BootstrapResult`;
    - ``trend`` is the OLS Gini-vs-step :class:`TrendResult`.
    """
    hourly = hourly_gini(table, predictor_col, percentile, target_col, rng_seed, threshold)
    flags = pooled_event_flags(table, percentile, target_col, threshold)
    pred = table[predictor_col].to_numpy()
    valid = np.isfinite(pred)
    one_hour_size = int(valid.sum()) // len(config.FORECAST_HOURS_UTC)
    boot = bootstrap_gini_se(
        pred[valid], flags[valid], sample_size=one_hour_size,
        rng=np.random.default_rng(rng_seed),
    )
    hourly = hourly.assign(se=boot.se)
    trend = hourly_trend_test(hourly["step"].to_numpy(), hourly["gini"].to_numpy())
    return hourly, boot, trend
