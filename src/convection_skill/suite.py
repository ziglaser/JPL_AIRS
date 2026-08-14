"""The runner: one ``test_hypothesis`` function applied to every spec over the
same prepared dataset, producing an easily comparable result suite.

- :func:`test_hypothesis` -- one hypothesis over one :class:`Prepared` sample:
  overall Gini, conditional Ginis within each control, per-stratum Ginis, and
  (for ``curve`` specs) the event-rate curve. All under the run's ONE inference
  convention (``cfg.inference``; decision 2026-07-22: never mix conventions
  within a run).
- :func:`run_suite` -- the config's selected specs over one config. Returns a tidy
  results DataFrame (one row per spec x scope) carrying ``run`` (the config
  label) and ``inference`` columns, plus the curves dict. BH-FDR runs across
  the whole run (Wilks field significance). Results from different configs
  concatenate directly for comparison -- same event definitions by
  construction (absolute base-sample thresholds).

Shared work is computed ONCE per run and reused by every spec: the flags dict,
the stratifier labels, and the prepared table itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import stats as S
from .config import AnalysisConfig
from .dataset import Prepared, prepare
from .hypotheses import STRATIFIERS, HypothesisSpec, select_specs


# --------------------------------------------------------------------------- #
# Per-run shared context
# --------------------------------------------------------------------------- #
def _strata_labels(frame: pd.DataFrame, names=None) -> dict[str, pd.Series]:
    """The requested stratifiers' labels for one frame, computed once and
    reused (``names=None`` computes all of them)."""
    return {name: pd.Series(STRATIFIERS[name](frame), index=frame.index)
            for name in (STRATIFIERS if names is None else names)}


def resolved_strata(spec: HypothesisSpec, cfg: AnalysisConfig) -> tuple[str, ...]:
    """The stratifiers one spec runs under one config: the config's
    per-hypothesis override (falling back to the spec's own strata) plus the
    config's always-on defaults, deduplicated."""
    if not cfg.run_strata:
        return ()
    own = tuple(cfg.strata.get(spec.id, spec.strata))
    strata = tuple(dict.fromkeys(own + tuple(cfg.default_strata)))
    unknown = [s for s in strata if s not in STRATIFIERS]
    if unknown:
        raise ValueError(f"unknown stratifiers {unknown} for {spec.id}; "
                         f"choose from {sorted(STRATIFIERS)}")
    return strata


def _row(spec, scope, r: "S.GiniResult") -> dict:
    """One tidy results row from a GiniResult."""
    return {"id": spec.id, "scope": scope, "n": r.n, "n_events": r.n_events,
            "gini": r.gini, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p": r.p_value,
            "se_naive": r.se_naive, "se_block": r.se_block,
            "inflation": r.inflation, "expected": spec.expected,
            "testability": spec.testability}


def _placeholder_row(spec, scope="—") -> dict:
    return {"id": spec.id, "scope": scope, "n": 0, "n_events": 0,
            "gini": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
            "se_naive": np.nan, "se_block": np.nan, "inflation": np.nan,
            "expected": spec.expected, "testability": spec.testability}


# --------------------------------------------------------------------------- #
# The one test function
# --------------------------------------------------------------------------- #
def test_hypothesis(
    spec: HypothesisSpec,
    prepared: Prepared,
    strata_labels: Optional[dict[str, pd.Series]] = None,
    onset_strata_labels: Optional[dict[str, pd.Series]] = None,
    n_reps: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[list[dict], Optional[dict]]:
    """Run one hypothesis over a prepared sample; return (rows, curve or None).

    Rows: 'overall', one 'ctrl:<control>' per control (conditional Gini =
    skill beyond that control), and '<stratifier>=<level>' per resolved
    stratum. Which controls/strata/curves run -- and the inference convention,
    rep count and seed -- all come from the prepared config (per-hypothesis
    ``cfg.controls`` / ``cfg.strata`` overrides beat the spec's own).
    """
    cfg = prepared.cfg
    method = cfg.inference
    n_reps = cfg.n_boot_reps if n_reps is None else n_reps
    seed = cfg.seed if seed is None else seed

    if spec.testability == "untestable":
        return [_placeholder_row(spec)], None

    controls = (tuple(cfg.controls.get(spec.id, spec.controls))
                if cfg.run_controls else ())
    strata = resolved_strata(spec, cfg)
    if spec.sample == "onset":
        strata = tuple(s for s in strata if s != "slot_phase")  # no slots there

    if spec.sample == "onset":
        if prepared.onset is None or not len(prepared.onset):
            return [], None
        frame = prepared.onset
        flags = frame["early_onset"].to_numpy().astype(float)
        labels = onset_strata_labels or _strata_labels(frame, strata)
    else:
        frame = prepared.table
        flags = prepared.flags[spec.target].astype(float)
        labels = strata_labels or _strata_labels(frame, strata)

    pred = frame[spec.predictor].to_numpy(dtype=float)
    day = frame["day"].to_numpy()
    if spec.invert:
        # replace the raw predictor by its cross-fitted rate-ordered index
        # eta(x) = P(event|x): the apples-to-apples Gini for non-monotone
        # responses (see inversion.py; out-of-fold, so no in-sample leakage)
        from .inversion import crossfit_index
        pred = crossfit_index(pred, flags, day, method=spec.invert)
    rows = []

    # overall
    r = S.bootstrap_gini(pred, flags, day, method=method, n_reps=n_reps,
                         block_days=cfg.block_days,
                         rng=np.random.default_rng(seed))
    rows.append(_row(spec, "overall", r))

    # conditional Gini within bins of each control (skill beyond the control)
    for ctrl in controls:
        ctrl_vals = frame[ctrl].to_numpy(dtype=float)
        c = S.conditional_gini(
            pred, flags, ctrl_vals, day_index=day,
            method=method,
            n_control_bins=cfg.n_control_bins,
            min_events_per_bin=cfg.min_events_per_bin,
            n_reps=n_reps, block_days=cfg.block_days,
            rng=np.random.default_rng(seed + 1))
        # report the rows conditional_gini actually used (it drops any row
        # with a non-finite predictor/flag/control -- e.g. no-front-data years)
        used = np.isfinite(pred) & np.isfinite(flags) & np.isfinite(ctrl_vals)
        rows.append({"id": spec.id, "scope": f"ctrl:{ctrl}", "n": int(used.sum()),
                     "n_events": int(np.nansum(flags[used])),
                     "gini": c["conditional_gini"], "ci_lo": c["ci_lo"],
                     "ci_hi": c["ci_hi"], "p": c["p_value"],
                     "se_naive": np.nan, "se_block": np.nan, "inflation": np.nan,
                     "expected": spec.expected, "testability": spec.testability})

    # per-stratum Ginis
    for strat in strata:
        lab = labels[strat]
        for level in pd.unique(lab.dropna()):
            in_level = (lab == level).to_numpy()
            r = S.bootstrap_gini(
                pred[in_level], flags[in_level], day[in_level], method=method,
                n_reps=n_reps, block_days=cfg.block_days,
                rng=np.random.default_rng(seed + 2))
            rows.append(_row(spec, f"{strat}={level}", r))

    curve = None
    if spec.curve and cfg.wants_curve(spec.id):
        curve = S.event_rate_curve(
            pred, flags, day, method=method, n_bins=20,
            n_reps=min(n_reps, 300), block_days=cfg.block_days,
            rng=np.random.default_rng(seed + 3))
    return rows, curve


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #
def run_suite(
    cfg_or_prepared,
    specs: Optional[tuple[HypothesisSpec, ...]] = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """The config's specs over one config; return (results table, curves dict).

    Accepts an :class:`AnalysisConfig` (dataset gets prepared, cached base
    reused) or an already-:func:`prepare`-d sample. Which specs run comes from
    ``cfg.hypotheses`` ("all" | "topline" | explicit ids) unless ``specs``
    overrides it. One results row per (spec, scope); columns include ``run``
    (= cfg.label()) and ``inference`` so suites from different configs
    concatenate into a directly comparable table. FDR (BH,
    alpha=``cfg.fdr_alpha``) runs across all rows of the run.
    """
    prepared = (cfg_or_prepared if isinstance(cfg_or_prepared, Prepared)
                else prepare(cfg_or_prepared, use_cache=use_cache))
    cfg = prepared.cfg
    if specs is None:
        specs = select_specs(cfg.hypotheses)

    # shared per-run work: the labels of every stratifier any spec needs
    needed: set = set()
    for spec in specs:
        needed.update(resolved_strata(spec, cfg))
    strata_labels = _strata_labels(prepared.table, needed)
    onset_labels = (_strata_labels(prepared.onset, needed)
                    if prepared.onset is not None and len(prepared.onset) else None)

    rows, curves = [], {}
    for spec in specs:
        spec_rows, curve = test_hypothesis(
            spec, prepared,
            strata_labels=strata_labels, onset_strata_labels=onset_labels)
        rows.extend(spec_rows)
        if curve is not None:
            curves[spec.id] = curve

    results = pd.DataFrame(rows)
    results["fdr_significant"] = S.fdr_bh(results["p"].to_numpy(), cfg.fdr_alpha)
    results.insert(0, "run", cfg.label())
    results.insert(1, "inference", cfg.inference)
    return results, curves
