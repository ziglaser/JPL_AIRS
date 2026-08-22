"""Mark's random forest over a 36-experiment feature-set grid on the suite's
cell-day table.

Mark Richardson's RF walkthrough lives VERBATIM in
:mod:`convection_skill.models` (``train_random_forest`` /
``make_binned_freqs`` plus the adapters ``samples_from_cell_days`` /
``finite_samples`` / ``importance_table``).  This module NEVER touches those
functions -- it only assembles the enriched cell-day sample, drives his RF
over every cell of the experiment grid below, and writes comparable metrics,
importances, tail-capture curves, a report and figures.

Experiment grid (36 = 3 x 3 x 2 x 2)
------------------------------------
``base`` in {airs, smap, both}  x  ``fronts`` in {none, met, pred}  x
``smidx`` in {0,1}  x  ``pbl`` in {0,1}.  The fronts axis is a single 3-way
choice, so analyst-drawn (met) and model-predicted (pred) front flags are
STRUCTURALLY never in the same experiment -- the whole point is a symmetric
"what if we only had the model's fronts" comparison.  Experiment ids:
``base-{airs|smap|both}_fronts-{none|met|pred}_smidx-{0|1}_pbl-{0|1}``.

Feature blocks -- every inclusion choice, spelled out
-----------------------------------------------------
Hourly stems expand to 6 features each (columns ``<stem>_h1..h6``) through
Mark's time-dim handling in ``samples_from_cell_days`` + ``hourly=False``.

AIRS (hourly): ``mu_cape, mu_cin, mml_cape, mml_cin, mu_el, mu_lcl, mml_lcl,
fcst_q, fcst_t`` -- the AIRS-FCST thermodynamic set: both parcel definitions'
CAPE/CIN/LCL, the MU equilibrium level, and the forecast-hour surface-layer
q/t that drive them.  ``qpe_max``/``qpe_sk`` are NEVER included: they are
MRMS observations of the target hour (target-adjacent leakage, not
predictors).  ``mu_cape_overpass`` is excluded too -- it is a degraded copy
of ``mu_cape`` (slot 0 replicated), redundant inside a set that already has
the full hourly series.

SMAP (daily): ``sm_raw, wind, pflux_prewindow, pflux_ante, qlay1_anom,
tlay1_anom`` -- the raw-ish timing-guarded SMAP L4 fields.  All same-day
fields are pre-window means (16:30/19:30 UTC L4 slots only, the Tuttle &
Salvucci guard baked into ``dataset.build_daily``); ``pflux_ante`` is the
prior-1-5-day precip control, ``pflux_prewindow`` the same-day pre-window
precip control.

SM_IDX (daily): ``sm_anom, sm_local, sm_nonlocal, smsd_anom, absgrad_anom,
wegrad_anom, sngrad_anom, sm_anom_lag1, sm_anom_lag3, sm_anom_lag7,
sm_anom_lag14, sm_anom_lag30, sm_cell_clim`` -- Zach's derived TOP-LEVEL
(surface-layer) soil-moisture indexes: the standardized harmonic anomaly,
its Guillod local/non-local split, the sub-grid SD and gradient anomalies,
the antecedent lag ladder, and the per-cell aridity climatology.

FRONTS_MET (hourly): ``met_front_{cold,warm,stationary,occluded,dryline,
any}_3w`` -- ANALYST fronts from the NOAA-XML source (the only met source
carrying DRYLINES), read from the enriched year files written by
``scripts/add_front_flags.py`` and renamed with a ``met_`` prefix here so
they can NEVER be confused with the base table's built-in WPC ``front_*``
columns (a different label source).

FRONTS_PRED (hourly): ``pred_front_{cold,warm,stationary,occluded,dryline,
any}_3w`` -- OUR MODEL's predicted fronts from the same enriched files,
names kept as written.  3-WIDE FLAGS ONLY throughout: every model we trained
saw 3wide labels, so 3wide is the only width that gives a symmetric
met-vs-pred comparison.

PBL (hourly): ``UPW_pblh, UPW_pblh_anom`` from
``results/upwind_features/UPWIND_FEATURES_{year}.nc``.  ``UPW_gamma_gap_*``
is DELIBERATELY EXCLUDED: it needs the FCST LFC and is finite on only ~15%
of rows, so including it would collapse the one common sample every
experiment must share.

One common sample
-----------------
``models.samples_from_cell_days`` runs ONCE on the enriched table and
``models.finite_samples`` ONCE with the union of ALL block features (+ the
target), so all 36 experiments fit and score on the IDENTICAL sample.
Consequences logged into the report: the PBL product starts 2017 (drops
2016) and predicted fronts are NaN outside the model's trained analysis
domain (spatial restriction).

Three model modes (``--mode``), one grid each
---------------------------------------------
All three run the SAME 36-experiment grid on the SAME common sample and the
SAME cell-day train/test partition, so any (experiment, hour) cell is directly
comparable across modes.  They differ only in how the forest sees the hour
axis (Zach's ask, 2026-08-21):

``wide`` (default; the original runs)
    Mark's ``hourly=False``: one row per cell-day, every hour of every
    predictor is its own column, one multi-output forest predicts the whole
    qpe h1..h6 series jointly.  No temporal ordering is enforced (hour-6 CAPE
    is available when "predicting" hour-1 rain).

``pooled`` (one hour-blind forest, hour-matched features)
    One row per (cell-day, hour): each hourly predictor contributes ONE
    column holding the CURRENT hour's value, daily predictors repeat across
    the hours, and a single forest predicts the current hour's qpe.  A split
    like "PBL depth > 1000 m" therefore reads hour-1 PBLH on hour-1 rows and
    hour-5 PBLH on hour-5 rows.  The forecast hour itself is deliberately NOT
    a feature -- the forest is hour-blind by construction, which is the point
    of the contrast with ``perhour``.  Feature layout is exactly Mark's
    ``hourly=True`` construction; the ONE deviation (documented, deliberate)
    is the split: his raveled iid split would scatter hours of the same
    cell-day across train and test (within-storm leakage), so rows are
    grouped by cell-day using the SAME cell-day partition as ``wide``.

``perhour`` (six independent forests)
    One forest per forecast hour h: hour-h predictor values -> hour-h qpe,
    on the same cell-day rows and split.  Systematic differences across the
    six fits (skill, importances) reveal hour structure the pooled forest is
    forced to average over.  Implemented by slicing the stacked sample to a
    single hour and calling Mark's verbatim function unchanged.

Headline metric (Zach, 2026-08-22)
----------------------------------
The **Gini coefficient** of the RF prediction as a risk score for
qpe-exceedance events at two set points, **P95** and **P99.5**
(``gini_p95`` / ``gini_p99_5`` columns) -- the same
``convection_skill.gini`` coefficient the hypothesis battery reports.
Thresholds are ABSOLUTE mm/h values from the suite's base table (all
in-domain land rows in the training window, pre-screen -- the paper's
"thresholds are based on all data" convention), so every experiment scores
against identical event definitions.  R^2 stays as a SECONDARY column (see
the iid-optimism caveat in the report).  Per-experiment capture curves are
persisted for BOTH set points (``tail_cdf/<id>_p95.csv`` /
``<id>_p99_5.csv``) and drive the figure family.

Training window (Zach, 2026-08-22)
----------------------------------
**March-November, 2017-2021** (``--months 3-11 --years 2017-2021``
defaults).  2016 was previously dropped only implicitly by PBL-product
availability; the years default now states the intent.  The window is part
of the sample-consistency contract: ``sample_info.json`` records both years
and months, and :func:`check_window_consistency` refuses to mix results
produced under a different window.

CLI
---
``python -m convection_skill.rf_experiments --years 2017-2021 --months 3-11
[--mode wide|pooled|perhour] [--subset REGEX] [--out-dir DIR] [--force]
[--no-figures]``.  ``--subset`` filters experiment ids by regex for batching;
the common sample depends only on ``--years``/``--months``, so batches are
apples-to-apples by construction.  Each mode writes to its own directory
(``rf_experiments/``, ``rf_experiments/pooled_hourly/``,
``rf_experiments/per_hour/``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import r2_score

from . import config, dataset, models
from .config import AnalysisConfig
from .gini import gini as _gini

# --------------------------------------------------------------------------- #
# Feature blocks (see module docstring for every inclusion choice)
# --------------------------------------------------------------------------- #
FRONT_TYPES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded",
                                "dryline", "any")

AIRS_STEMS: tuple[str, ...] = (
    "mu_cape", "mu_cin", "mml_cape", "mml_cin", "mu_el", "mu_lcl", "mml_lcl",
    "fcst_q", "fcst_t",
)
SMAP_STEMS: tuple[str, ...] = (
    "sm_raw", "wind", "pflux_prewindow", "pflux_ante", "qlay1_anom",
    "tlay1_anom",
)
SM_IDX_STEMS: tuple[str, ...] = (
    "sm_anom", "sm_local", "sm_nonlocal", "smsd_anom", "absgrad_anom",
    "wegrad_anom", "sngrad_anom", "sm_anom_lag1", "sm_anom_lag3",
    "sm_anom_lag7", "sm_anom_lag14", "sm_anom_lag30", "sm_cell_clim",
)
FRONTS_MET_STEMS: tuple[str, ...] = tuple(
    f"met_front_{t}_3w" for t in FRONT_TYPES)
FRONTS_PRED_STEMS: tuple[str, ...] = tuple(
    f"pred_front_{t}_3w" for t in FRONT_TYPES)
PBL_STEMS: tuple[str, ...] = ("UPW_pblh", "UPW_pblh_anom")

BLOCKS: dict[str, tuple[str, ...]] = {
    "AIRS": AIRS_STEMS,
    "SMAP": SMAP_STEMS,
    "SM_IDX": SM_IDX_STEMS,
    "FRONTS_MET": FRONTS_MET_STEMS,
    "FRONTS_PRED": FRONTS_PRED_STEMS,
    "PBL": PBL_STEMS,
}
#: Union of every block stem: the finite-sample screen set (+ TARGET).
ALL_STEMS: tuple[str, ...] = tuple(
    itertools.chain.from_iterable(BLOCKS.values()))

TARGET: str = "qpe"
#: The two headline Gini set points (Zach 2026-08-22): label -> percentile.
#: Labels are the results.csv column suffixes (gini_p95 / gini_p99_5) AND the
#: tail_cdf file-name suffixes (<id>_p95.csv / <id>_p99_5.csv), so metrics,
#: persisted curves and figures can never drift apart.
SET_POINTS: dict[str, float] = {"p95": 95.0, "p99_5": 99.5}
#: The single ranking/selection metric for reports and curated figures.
#: P99.5 is the harder, rarer event bar -- the one the project cares about.
HEADLINE_METRIC: str = "gini_p99_5"
SEED: int = 42          # Mark's walkthrough seed
TRAIN_FRACTION: float = 0.25
#: Bin index whose cumulative value covers everything BELOW the 90th
#: predicted percentile: make_binned_freqs bins on percentile edges 0..100,
#: so bin k spans [P_k, P_k+1) and cumsum index 89 is the mass in P0..P90.
TOP10_BIN_INDEX: int = 89

# --------------------------------------------------------------------------- #
# Enrichment sources
# --------------------------------------------------------------------------- #
#: Enriched COPIES of the year files carrying the front-flag variables
#: (analyst + model), produced by ``scripts/add_front_flags.py``.
FRONTS_ENRICHED_DIR_NAME: str = "FCST_SMAP_MRMS_fronts"
FRONTS_PRODUCER: str = "scripts/add_front_flags.py"
#: In-file (source) names of the analyst flags; renamed with ``met_`` on read.
MET_SRC_VARS: tuple[str, ...] = tuple(f"front_{t}_3w" for t in FRONT_TYPES)
PRED_SRC_VARS: tuple[str, ...] = FRONTS_PRED_STEMS  # names kept as-is

UPWIND_DIR_NAME: str = "upwind_features"
UPWIND_PRODUCER: str = "scripts/merge_upwind_features.py"


def fronts_year_path(year: int) -> Path:
    return config.DATA_DIR / FRONTS_ENRICHED_DIR_NAME / f"FCST_SMAP_MRMS_{year}.nc"


def upwind_year_path(year: int) -> Path:
    return config.RESULTS_DIR / UPWIND_DIR_NAME / f"UPWIND_FEATURES_{year}.nc"


# --------------------------------------------------------------------------- #
# The experiment grid
# --------------------------------------------------------------------------- #
#: base axis -> which feature blocks it contributes.
BASES: dict[str, tuple[str, ...]] = {
    "airs": ("AIRS",),
    "smap": ("SMAP",),
    "both": ("AIRS", "SMAP"),
}
#: fronts axis: ONE 3-way choice, so met and pred are never co-present.
FRONTS_AXIS: dict[str, tuple[str, ...]] = {
    "none": (),
    "met": ("FRONTS_MET",),
    "pred": ("FRONTS_PRED",),
}


@dataclass(frozen=True)
class Experiment:
    """One cell of the grid: an id, its axis levels, and its feature stems."""

    id: str
    base: str
    fronts: str
    smidx: int
    pbl: int

    @property
    def blocks(self) -> tuple[str, ...]:
        blocks = BASES[self.base] + FRONTS_AXIS[self.fronts]
        if self.smidx:
            blocks += ("SM_IDX",)
        if self.pbl:
            blocks += ("PBL",)
        return blocks

    @property
    def features(self) -> list[str]:
        """Ordered stem list for train_random_forest's X_keys."""
        return list(itertools.chain.from_iterable(
            BLOCKS[b] for b in self.blocks))


def experiment_grid() -> list[Experiment]:
    """All 36 experiments, in a fixed deterministic order."""
    grid = []
    for base, fronts, smidx, pbl in itertools.product(
            BASES, FRONTS_AXIS, (0, 1), (0, 1)):
        grid.append(Experiment(
            id=f"base-{base}_fronts-{fronts}_smidx-{smidx}_pbl-{pbl}",
            base=base, fronts=fronts, smidx=smidx, pbl=pbl))
    return grid


# --------------------------------------------------------------------------- #
# Slot pivot + cell-day merge (the one alignment idiom, used by both sources)
# --------------------------------------------------------------------------- #
def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Merge keys to one canonical dtype so float32/float64 and datetime64
    unit mismatches can never silently produce an empty merge.

    The base table stores lat/lon as float32 and ``day`` as whatever
    datetime64 unit pandas chose; the netCDF sources give float64 and ns.
    The grid is half-degree-centered, so round(3) is exact for real cells and
    only guards against representation noise.
    """
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"]).values.astype("datetime64[ns]")
    for k in ("lat", "lon"):
        df[k] = df[k].astype(np.float64).round(3)
    return df


def slot_wide(ds: xr.Dataset, rename: dict[str, str] | None = None,
              slots: tuple[int, ...] = config.FORECAST_SLOTS) -> pd.DataFrame:
    """(date, time, lat, lon) variables -> one row per (day, lat, lon) with
    forecast slots widened to ``<name>_h<slot>`` columns.

    Exactly the shape ``dataset.to_cell_days`` gives the base table's slot
    variables, so the merged columns behave identically downstream (the same
    ``_h`` convention ``models.samples_from_cell_days`` re-stacks on).
    ``rename`` maps source variable names to output stems (e.g. the ``met_``
    prefix for analyst front flags).
    """
    rename = rename or {}
    names = list(ds.data_vars)
    df = (ds[names].sel(time=list(slots))
          .to_dataframe().reset_index())
    df["day"] = pd.to_datetime(df["date"]).values.astype("datetime64[ns]")
    wide = df.pivot(index=["day", "lat", "lon"], columns="time", values=names)
    wide.columns = [f"{rename.get(var, var)}_h{int(slot)}"
                    for var, slot in wide.columns]
    return wide.reset_index()


def merge_on_cell_day(cell_days: pd.DataFrame,
                      wide: pd.DataFrame) -> pd.DataFrame:
    """Left-merge widened enrichment columns onto the cell-day table.

    Left join: enrichment can never add or drop cell-days, only attach
    columns (missing years/cells come through as NaN and fall to the common
    finite-sample screen, which is where sample decisions belong).  A column
    collision raises -- the ``met_`` prefix exists precisely so enrichment
    can never overwrite a base-table column.
    """
    new_cols = [c for c in wide.columns if c not in ("day", "lat", "lon")]
    clash = sorted(set(new_cols) & set(cell_days.columns))
    if clash:
        raise ValueError(f"enrichment would overwrite existing cell-day "
                         f"columns {clash}; rename the source variables")
    left = _normalize_keys(cell_days)
    right = _normalize_keys(wide)
    return left.merge(right, on=["day", "lat", "lon"], how="left",
                      validate="one_to_one")


# --------------------------------------------------------------------------- #
# Enrichment: front flags (analyst + model) and PBL depth
# --------------------------------------------------------------------------- #
def load_front_flags_wide(years: tuple[int, ...]) -> pd.DataFrame:
    """Analyst (``met_``-prefixed) + predicted 3wide front flags, widened.

    Reads the ENRICHED year-file copies under
    ``$JPL_AIRS_DATA/FCST_SMAP_MRMS_fronts/`` -- ordinary netCDF variables
    ``front_{type}_3w`` (analyst; NOAA source, includes drylines) and
    ``pred_front_{type}_3w`` (our model) on (date, time, lat, lon), written
    by ``scripts/add_front_flags.py``.  A missing file or variable raises
    with the producing script named: silently proceeding would make the
    fronts axis of the whole grid meaningless.
    """
    src_vars = list(MET_SRC_VARS) + list(PRED_SRC_VARS)
    rename = {src: f"met_{src}" for src in MET_SRC_VARS}
    frames = []
    for year in years:
        path = fronts_year_path(year)
        if not path.exists():
            raise FileNotFoundError(
                f"enriched front-flag file {path} does not exist; produce it "
                f"with `python {FRONTS_PRODUCER} --years {year} "
                f"--label-source noaa --pred-dir <bk19-schema tree>` first")
        with xr.open_dataset(path, drop_variables=["FCST_parceltime"]) as f:
            missing = [v for v in src_vars if v not in f.data_vars]
            if missing:
                raise KeyError(
                    f"{path} is missing front-flag variables {missing}; "
                    f"re-run {FRONTS_PRODUCER} (analyst flags need "
                    f"--label-source noaa for the dryline channel; "
                    f"predicted flags need --pred-dir)")
            sub = f[src_vars].load()
        frames.append(slot_wide(sub, rename=rename))
    return pd.concat(frames, ignore_index=True)


def load_pbl_wide(years: tuple[int, ...]) -> pd.DataFrame:
    """``UPW_pblh`` / ``UPW_pblh_anom`` from the upwind-features files, widened.

    A missing YEAR file only warns (the PBL product starts 2017, so e.g. a
    2016 file may legitimately never exist): its cell-days come through as
    NaN and drop out at the common finite-sample screen, which is exactly the
    "common sample drops 2016" behaviour the report documents.  If NO year
    resolves, this raises -- the PBL axis would be meaningless.
    """
    frames = []
    for year in years:
        path = upwind_year_path(year)
        if not path.exists():
            print(f"NOTE: {path} missing (PBL product starts 2017); "
                  f"produce with {UPWIND_PRODUCER}; year {year} PBL -> NaN")
            continue
        with xr.open_dataset(path) as f:
            missing = [v for v in PBL_STEMS if v not in f.data_vars]
            if missing:
                raise KeyError(f"{path} is missing {missing}; "
                               f"re-run {UPWIND_PRODUCER}")
            sub = f[list(PBL_STEMS)].load()
        frames.append(slot_wide(sub))
    if not frames:
        raise FileNotFoundError(
            f"no UPWIND_FEATURES_<year>.nc found for years {years} under "
            f"{config.RESULTS_DIR / UPWIND_DIR_NAME}; produce them with "
            f"{UPWIND_PRODUCER}")
    return pd.concat(frames, ignore_index=True)


def enrich_cell_days(cell_days: pd.DataFrame,
                     years: tuple[int, ...]) -> pd.DataFrame:
    """Base cell-day table + front flags + PBL depth, ready for sampling."""
    out = merge_on_cell_day(cell_days, load_front_flags_wide(years))
    out = merge_on_cell_day(out, load_pbl_wide(years))
    assert_stems_present(out, ALL_STEMS + (TARGET,))
    return out


def assert_stems_present(cell_days: pd.DataFrame,
                         stems: tuple[str, ...]) -> None:
    """Fail LOUDLY, naming the absent stems, before any RF is fit."""
    missing = missing_stems(cell_days, stems)
    if missing:
        raise AssertionError(
            f"assembled cell-day table is missing feature stems {missing} "
            f"(neither a daily column nor any <stem>_h<slot> family)")


def missing_stems(cell_days: pd.DataFrame,
                  stems: tuple[str, ...]) -> list[str]:
    """Stems with neither a daily column nor an ``_h<slot>`` column family."""
    cols = set(cell_days.columns)
    hourly = set()
    for c in cols:
        stem, _, suffix = c.rpartition("_h")
        if stem and suffix.isdigit():
            hourly.add(stem)
    return [s for s in stems if s not in cols and s not in hourly]


# --------------------------------------------------------------------------- #
# The ONE common sample
# --------------------------------------------------------------------------- #
def filter_month_window(cell_days: pd.DataFrame,
                        months: tuple[int, int]) -> pd.DataFrame:
    """Keep only cell-days whose month lies in [months[0], months[1]]
    (inclusive, same (lo, hi) convention as ``AnalysisConfig.months``).

    The base-table builder already restricts to ``cfg.months``, so with the
    default plumbing this is a no-op -- it exists to make the RF training
    window EXPLICIT and independent of upstream defaults: whatever cell-day
    table arrives here (a cached base table built under other settings, the
    compiled cluster superset, a synthetic test frame), the sample the
    forests see is provably Mar-Nov (or whatever --months says).
    """
    lo, hi = months
    month = pd.to_datetime(cell_days["day"]).dt.month
    return cell_days[(month >= lo) & (month <= hi)].reset_index(drop=True)


def build_common_sample(enriched: pd.DataFrame) -> xr.Dataset:
    """Stack ONCE, screen ONCE on the union of all block features + target.

    Every experiment then fits and scores on this identical sample, so R^2
    differences are feature-set differences, never sample differences.
    """
    ds = models.samples_from_cell_days(enriched)
    return models.finite_samples(ds, list(ALL_STEMS) + [TARGET])


def sample_facts(ds: xr.Dataset, enriched: pd.DataFrame) -> dict:
    """The common-sample provenance the report records.

    Two known screens shape it (log them, don't hide them): the PBL product
    starts 2017 (2016 drops entirely) and the predicted front flags are NaN
    outside the model's trained analysis domain (spatial restriction).
    """
    years = pd.DatetimeIndex(np.asarray(ds["day"].values)).year
    per_year = {int(y): int(n) for y, n in
                pd.Series(years).value_counts().sort_index().items()}
    cells_before = len(enriched[["lat", "lon"]].drop_duplicates())
    cells_after = len(pd.DataFrame({
        "lat": np.asarray(ds["lat"].values),
        "lon": np.asarray(ds["lon"].values)}).drop_duplicates())
    return {
        "n_samples": int(ds.sizes["sample"]),
        "n_cell_days_before_screen": int(len(enriched)),
        "per_year_counts": per_year,
        "n_cells_before_screen": cells_before,
        "n_cells_in_common_sample": cells_after,
    }


# --------------------------------------------------------------------------- #
# Fit + metrics for one experiment
# --------------------------------------------------------------------------- #
def tail_top10_capture(cdf: np.ndarray) -> float:
    """Fraction of observed tail events captured by the TOP 10% of predictions.

    ``cdf`` is make_binned_freqs' cumulative curve over 100 predicted-
    percentile bins; its value at ``TOP10_BIN_INDEX`` (=89) is the event mass
    at predictions BELOW the 90th predicted percentile, so the top-decile
    capture is one minus that.  A perfect ranker scores 1.0; an uninformative
    one ~0.10 (events land uniformly over prediction bins).
    """
    cdf = np.asarray(cdf)
    if len(cdf) != 100:
        raise ValueError(f"expected the 100-bin cumulative curve, got "
                         f"{len(cdf)} bins")
    return float(1.0 - cdf[TOP10_BIN_INDEX])


def gini_metrics(obs_rows: np.ndarray, pred_rows: np.ndarray,
                 thresholds: dict[str, float] | None) -> dict[str, float]:
    """Gini of the RF prediction as a risk score for exceedance events.

    ``thresholds`` maps a label (e.g. ``"p95"``) to an ABSOLUTE qpe threshold
    (mm/h); events are ``obs > threshold`` at the hour-row level, the
    predictor is the corresponding hour-row prediction, and the Gini is the
    suite's own (``convection_skill.gini.gini``) -- the same coefficient the
    hypothesis battery reports, so RF experiments and battery results read on
    one scale.  Empty when ``thresholds`` is None (synthetic/unit-test runs
    only; both the local CLI and the cluster runner always pass them).
    """
    if not thresholds:
        return {}
    obs, pred = np.ravel(obs_rows), np.ravel(pred_rows)
    return {f"gini_{label}": float(_gini(pred, obs > thr))
            for label, thr in thresholds.items()}


def capture_curves(obs_rows: np.ndarray, pred_rows: np.ndarray,
                   thresholds: dict[str, float] | None = None,
                   ) -> dict[str, np.ndarray]:
    """One tail-capture (Lorenz-style) curve per headline set point.

    The curve for label L is Mark's ``make_binned_freqs`` cumulative curve
    over 100 predicted-percentile bins, on the SAME hour-row (obs, pred)
    pairs and the SAME event definition as ``gini_metrics``: when an absolute
    ``thresholds[L]`` is given (both real runners always pass one) events are
    ``obs > thresholds[L]``, so the persisted curve and the reported
    ``gini_<L>`` describe one and the same ranking.  Without thresholds
    (unit tests on synthetic data) events fall back to the SET_POINTS
    percentile of the test-leg obs -- Mark's original ``obs_pct`` convention.
    """
    obs, pred = np.ravel(obs_rows), np.ravel(pred_rows)
    out: dict[str, np.ndarray] = {}
    for label, pct in SET_POINTS.items():
        thr = (thresholds or {}).get(label)
        cdf = np.asarray(models.make_binned_freqs(pred, obs, obs_pct=pct,
                                                  thresh=thr), dtype=float)
        if len(cdf) != 100:
            raise ValueError(f"expected the 100-bin cumulative curve for "
                             f"{label}, got {len(cdf)} bins")
        out[label] = cdf
    return out


def write_tail_curves(out_dir: Path, exp_id: str,
                      curves: dict[str, np.ndarray]) -> None:
    """Persist one CSV per set point: ``tail_cdf/<id>_<label>.csv``.

    The per-threshold suffix replaces the legacy single-file ``<id>.csv``
    naming; figures read ONLY the suffixed names and only for ids present in
    the current results.csv, so a legacy or orphaned curve can never be
    plotted (and the window-consistency guard deletes all tail_cdf CSVs,
    legacy names included, whenever the sample changes).
    """
    tail_dir = Path(out_dir) / "tail_cdf"
    tail_dir.mkdir(parents=True, exist_ok=True)
    for label, cdf in curves.items():
        pd.DataFrame({"bin": np.arange(1, 101), "cum_freq": cdf}).to_csv(
            tail_dir / f"{exp_id}_{label}.csv", index=False)


def run_experiment(ds: xr.Dataset, exp: Experiment,
                   thresholds: dict[str, float] | None = None) -> dict:
    """Fit Mark's RF for one experiment on the common sample; all metrics.

    Fit spec is Mark's walkthrough exactly (seed 42, train_fraction 0.25,
    max_depth 14, n_estimators 75); ``n_jobs`` is passed through
    ``rfr_kwargs`` so the verbatim code stays untouched.
    """
    rfr = models.train_random_forest(
        ds, exp.features, TARGET, hourly=False, seed=SEED,
        train_fraction=TRAIN_FRACTION,
        rfr_kwargs={**models.DEFAULT_RFR_KWARGS, "n_jobs": -1})
    obs = rfr["Y_test"]
    pred = rfr["model"].predict(rfr["X_test"])

    hours = [int(t) for t in ds["time"].values]
    per_hour = r2_score(obs, pred, multioutput="raw_values")
    # curves + capture on the same hour-row pairs the Ginis score, so every
    # tail number in the row describes one ranking (see capture_curves)
    curves = capture_curves(obs, pred, thresholds)
    return {
        "r2_test": float(rfr["model"].score(rfr["X_test"], obs)),
        "r2_per_hour": {f"r2_h{h}": float(v)
                        for h, v in zip(hours, per_hour)},
        "tail_cdfs": curves,
        "tail_top10_capture": tail_top10_capture(curves["p99_5"]),
        "importances": models.importance_table(rfr, ds),
        **gini_metrics(obs, pred, thresholds),
    }


# --------------------------------------------------------------------------- #
# Hour-matched modes: one shared cell-day split, two fit paths
# --------------------------------------------------------------------------- #
def cell_day_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """The ONE cell-day train/test partition, shared by all three modes.

    ``sklearn.train_test_split``'s row permutation depends only on the number
    of rows and ``random_state``, so splitting index positions here with
    Mark's (train_fraction, seed, shuffle) reproduces EXACTLY the row
    partition his verbatim ``train_random_forest(hourly=False)`` draws
    internally on the same sample (unit-tested).  ``pooled`` and ``perhour``
    reuse it so every mode trains and tests on the same cell-days -- hours of
    one cell-day never straddle the split in any mode.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(n_samples),
                            train_size=TRAIN_FRACTION,
                            random_state=SEED, shuffle=True)


def hour_matched_matrices(
        ds: xr.Dataset, X_keys: list[str], target: str = TARGET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mark's ``hourly=True`` layout: (X, y, sample_of_row, hour_of_row).

    Mirrors his construction exactly (``models.train_random_forest``,
    hourly branch): time-dim variables are transposed to (sample, time) and
    raveled SAMPLE-MAJOR (sample 0's h1..h6 rows, then sample 1's, ...);
    static/daily variables are ``np.repeat``-ed across the hours.  Each
    hourly stem is therefore ONE column holding the current hour's value.
    Returned alongside are each row's originating sample index and forecast
    hour, so the caller can group the split by cell-day and score by hour.
    """
    n_hours, n = ds.sizes["time"], ds.sizes["sample"]
    cols = []
    for var in X_keys:
        da = ds[var]
        if "time" in da.dims:
            cols.append(da.values.T.ravel()[:, np.newaxis])
        else:
            cols.append(np.repeat(da.values, n_hours)[:, np.newaxis])
    X = np.hstack(cols)
    y = ds[target].values.T.ravel()
    sample_of_row = np.repeat(np.arange(n), n_hours)
    hour_of_row = np.tile(np.asarray(ds["time"].values, dtype=int), n)
    return X, y, sample_of_row, hour_of_row


def _per_hour_r2(obs_rows: np.ndarray, pred_rows: np.ndarray,
                 hours_rows: np.ndarray, hours: list[int]) -> dict:
    """Per-hour R^2 from row-level (obs, pred, hour) triples."""
    return {f"r2_h{h}": float(r2_score(obs_rows[hours_rows == h],
                                       pred_rows[hours_rows == h]))
            for h in hours}


def run_experiment_pooled(ds: xr.Dataset, exp: Experiment,
                          thresholds: dict[str, float] | None = None) -> dict:
    """One hour-blind forest on hour-matched rows (mode ``pooled``).

    Fit spec (trees, depth, seed) is Mark's; the feature layout is his
    ``hourly=True`` construction via :func:`hour_matched_matrices`; the split
    is the shared cell-day partition (see module docstring for why that
    replaces his raveled iid split).  ``importances`` come back as ONE value
    per stem -- "how much does PBL depth matter", full stop -- which is the
    interpretability payoff of this mode.
    """
    from sklearn.ensemble import RandomForestRegressor

    X, y, sample_of_row, hour_of_row = hour_matched_matrices(ds, exp.features)
    train_idx, test_idx = cell_day_split(ds.sizes["sample"])
    in_train = np.zeros(ds.sizes["sample"], dtype=bool)
    in_train[train_idx] = True
    row_train = in_train[sample_of_row]

    np.random.seed(SEED)  # Mark's global-seed convention
    rfr = RandomForestRegressor(random_state=SEED,
                                **{**models.DEFAULT_RFR_KWARGS, "n_jobs": -1})
    print(f"Training pooled hour-matched RF on X shape "
          f"{X[row_train].shape} ...")
    rfr.fit(X[row_train], y[row_train])

    obs, pred = y[~row_train], rfr.predict(X[~row_train])
    hours = [int(t) for t in ds["time"].values]
    per_hour = _per_hour_r2(obs, pred, hour_of_row[~row_train], hours)
    curves = capture_curves(obs, pred, thresholds)
    names = models.feature_names(ds, exp.features, hourly=True)
    return {
        "r2_test": float(r2_score(obs, pred)),
        "r2_per_hour": per_hour,
        "tail_cdfs": curves,
        "tail_top10_capture": tail_top10_capture(curves["p99_5"]),
        "importances": pd.Series(rfr.feature_importances_, index=names,
                                 name="importance").sort_values(
                                     ascending=False),
        **gini_metrics(obs, pred, thresholds),
    }


def run_experiment_perhour(ds: xr.Dataset, exp: Experiment,
                           thresholds: dict[str, float] | None = None
                           ) -> dict:
    """Six independent hour-h forests (mode ``perhour``).

    Each hour's fit IS Mark's verbatim ``train_random_forest`` on the sample
    sliced to that single hour (a length-1 time dim, so every hourly stem
    contributes exactly its hour-h column and the target is hour-h qpe).
    The internal iid split is leakage-safe here -- rows are whole cell-days
    -- and identical to :func:`cell_day_split` (same n, same seed), so all
    six fits and the other two modes share one test set.  The pooled-across-
    hours R^2 / tail statistics concatenate the six test legs so they are
    comparable to the other modes' overall numbers; ``importances`` is a
    (stem x hour) frame -- the "systematic differences by hour" readout.
    """
    hours = [int(t) for t in ds["time"].values]
    obs_cols, pred_cols, imp_cols = [], [], {}
    per_hour = {}
    for i, h in enumerate(hours):
        sub = ds.isel(time=[i])
        rfr = models.train_random_forest(
            sub, exp.features, TARGET, hourly=False, seed=SEED,
            train_fraction=TRAIN_FRACTION,
            rfr_kwargs={**models.DEFAULT_RFR_KWARGS, "n_jobs": -1})
        obs, pred = rfr["Y_test"], rfr["model"].predict(rfr["X_test"])
        per_hour[f"r2_h{h}"] = float(rfr["model"].score(rfr["X_test"], obs))
        obs_cols.append(obs)
        pred_cols.append(pred)
        # hour-sliced names are <stem>_h<h>; index by the bare stem so the
        # six hours line up row-for-row in one frame
        named = models.importance_table(rfr, sub)
        named.index = [n.rsplit("_h", 1)[0] for n in named.index]
        imp_cols[f"h{h}"] = named
    obs_mat = np.column_stack(obs_cols)   # (n_test, n_hours), shared split
    pred_mat = np.column_stack(pred_cols)
    curves = capture_curves(obs_mat, pred_mat, thresholds)
    importances = pd.DataFrame(imp_cols)
    importances.index.name = "feature"
    return {
        "r2_test": float(r2_score(obs_mat.ravel(), pred_mat.ravel())),
        "r2_per_hour": per_hour,
        "tail_cdfs": curves,
        "tail_top10_capture": tail_top10_capture(curves["p99_5"]),
        "importances": importances.loc[
            importances.mean(axis=1).sort_values(ascending=False).index],
        **gini_metrics(obs_mat, pred_mat, thresholds),
    }


#: mode name -> (fit function, output subdirectory under rf_experiments/).
#: ``wide`` keeps the top-level directory so the original runs stay in place.
MODES: dict[str, tuple] = {
    "wide": (run_experiment, ""),
    "pooled": (run_experiment_pooled, "pooled_hourly"),
    "perhour": (run_experiment_perhour, "per_hour"),
}


# --------------------------------------------------------------------------- #
# Results bookkeeping (idempotent over batches)
# --------------------------------------------------------------------------- #
RESULT_COLUMNS: tuple[str, ...] = (
    "id", "base", "fronts", "smidx", "pbl", "n_samples",
    "gini_p95", "gini_p99_5",   # headline set-point Ginis (P99.5 leads)
    "r2_test",                  # secondary (iid-split optimism, caveat a)
    "r2_h1", "r2_h2", "r2_h3", "r2_h4", "r2_h5", "r2_h6",
    "tail_top10_capture",
)


def load_results(out_dir: Path) -> pd.DataFrame:
    path = Path(out_dir) / "results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=list(RESULT_COLUMNS))


def _discard_stale_artifacts(out_dir: Path) -> None:
    """Remove every per-experiment CSV so a fresh run cannot inherit curves
    or importances that were fit on a DIFFERENT common sample."""
    for sub in ("importances", "tail_cdf"):
        for f in sorted((Path(out_dir) / sub).glob("*.csv")):
            f.unlink()


def check_window_consistency(out_dir: Path, years: tuple[int, ...],
                             months: tuple[int, int],
                             force: bool = False,
                             columns: tuple[str, ...] = RESULT_COLUMNS,
                             ) -> pd.DataFrame:
    """Enforce the apples-to-apples contract across resumed / batched runs.

    (Renamed from ``check_years_consistency`` when the training WINDOW grew a
    months dimension, 2026-08-22: the common sample depends on --years AND
    --months, so both are part of the contract.)

    The idempotent skip in :func:`experiments_to_run` is only sound if every
    retained row of ``results.csv`` was fit on the SAME common sample.
    ``sample_info.json`` records the (years, months) window of the run that
    produced the existing rows; if either disagrees with the requested window
    (or the record is missing while rows exist -- including pre-months-era
    files that recorded only years), the retained rows would silently be
    ranked in REPORT.md against rows from a different sample -- so we refuse.
    ``--force`` re-runs everything anyway, so under ``--force`` we just
    discard ALL stale artifacts (rows + per-experiment importances /
    tail-cdf CSVs, which figures read by filename) and start clean.

    ``columns`` lets rf_cluster reuse the guard with its own results schema.
    Returns the results frame to resume from (existing rows when consistent,
    empty otherwise).
    """
    out_dir = Path(out_dir)
    results = load_results(out_dir)
    facts_path = out_dir / "sample_info.json"
    rec_years: tuple[int, ...] | None = None
    rec_months: tuple[int, ...] | None = None
    if facts_path.exists():
        recorded = json.loads(facts_path.read_text())
        rec_years = tuple(recorded.get("years", ()))
        # pre-months-era files have no "months" key -> () -> never matches:
        # results of unknown month window are as untrustworthy as unknown years
        rec_months = tuple(recorded.get("months", ()))

    if results.empty:
        return results  # nothing retained, nothing to contaminate
    if rec_years == tuple(years) and rec_months == tuple(months):
        return results  # same common sample by construction: safe to resume

    why = (f"was produced with --years {list(rec_years)} --months "
           f"{list(rec_months)}" if rec_years or rec_months
           else f"has no {facts_path.name} recording its --years/--months")
    if not force:
        raise RuntimeError(
            f"{out_dir / 'results.csv'} holds {len(results)} experiment "
            f"row(s) but {why}; the requested window --years {list(years)} "
            f"--months {list(months)} would build a DIFFERENT common sample, "
            f"so mixing them breaks the apples-to-apples guarantee. Point "
            f"--out-dir at a fresh directory, rerun with the matching "
            f"window, or pass --force to discard the stale results and "
            f"start over.")
    print(f"WARNING: --force with a mismatched window ({why}); discarding "
          f"{len(results)} stale result row(s) and all per-experiment CSVs")
    _discard_stale_artifacts(out_dir)
    (out_dir / "results.csv").unlink(missing_ok=True)
    facts_path.unlink(missing_ok=True)
    return pd.DataFrame(columns=list(columns))


def experiments_to_run(grid: list[Experiment], existing_ids: set[str],
                       subset: str | None = None,
                       force: bool = False) -> list[Experiment]:
    """Subset regex + idempotent skip: already-scored ids rerun only --force."""
    todo = [e for e in grid if subset is None or re.search(subset, e.id)]
    if not force:
        todo = [e for e in todo if e.id not in existing_ids]
    return todo


def upsert_result(results: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Replace-or-append one experiment row, keeping grid order stable."""
    results = results[results["id"] != row["id"]]
    new = pd.DataFrame([row])
    results = new if results.empty else pd.concat([results, new],
                                                  ignore_index=True)
    order = {e.id: i for i, e in enumerate(experiment_grid())}
    return (results.assign(_o=results["id"].map(order))
            .sort_values("_o").drop(columns="_o").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
CAVEATS: str = """\
## Caveats

- **(a) iid split optimism.** Mark's train/test split is an iid shuffle over
  cell-days; spatial and temporal autocorrelation put near-duplicates of
  training rows in the test set, so absolute R^2 is OPTIMISTIC. Comparisons
  ACROSS feature sets remain fair (identical sample, identical split). This
  is one reason R^2 is a SECONDARY column: the headline set-point Ginis are
  computed on the same split (so also nominally optimistic) but rank tail
  discrimination, which is the question being asked.
- **(b) fronts are concurrent, not antecedent.** Front flags are CONCURRENT
  with the 21-02 UTC target window (Zach 2026-08-05: concurrent flags only)
  -- a synoptic-environment covariate like CAPE, not a timing-guarded
  antecedent predictor. Front skill here is "knowing the synoptic setting",
  not lead-time forecast skill.
- **(c) predicted-front provenance.** Predicted fronts come from the
  pre-label-fix D6C 3-fold ensemble driven by REANALYSIS surface fields (the
  kriged-AIRS caches are cluster-only). All prediction years 2016-2021 are
  OUTSIDE the model's 2007-2015 training span, so they are honest
  out-of-sample predictions of a model with known label-era caveats.
- **(d) PBL coverage starts 2017.** The PBL depth product has no 2016, so
  the common finite sample (shared by ALL experiments, including pbl=0 ones)
  drops 2016 entirely.
- **(e) predicted-front spatial domain.** Predicted flags are NaN outside
  the model's trained analysis domain, so the common sample is also
  spatially restricted to that domain (again for ALL experiments).
"""


def _md_table(df: pd.DataFrame, cols: list[str], floatfmt: str = ".4f") -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in df.iterrows():
        cells = [f"{r[c]:{floatfmt}}" if isinstance(r[c], (float, np.floating))
                 else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rank_metric(results: pd.DataFrame) -> str:
    """The column reports/figures rank on: the headline Gini when present.

    Falls back to ``r2_test`` only for results frames from before the Gini
    columns existed (which the window guard normally refuses anyway) so a
    partially-migrated directory still renders instead of crashing.
    """
    if (HEADLINE_METRIC in results.columns
            and results[HEADLINE_METRIC].notna().any()):
        return HEADLINE_METRIC
    return "r2_test"


def _axis_deltas(results: pd.DataFrame, metric: str) -> list[str]:
    """Human-readable added-value deltas along each grid axis, on ``metric``
    (the headline Gini in current runs).

    Each delta is a mean of PAIRED differences (same levels on every other
    axis), so it is a clean main effect on the identical sample.
    """
    lines = []
    r = results.set_index(["base", "fronts", "smidx", "pbl"])[metric]

    def paired_delta(level_from, level_to, axis):
        diffs = []
        for key in r.index:
            if key[axis] != level_from:
                continue
            other = list(key)
            other[axis] = level_to
            other = tuple(other)
            if other in r.index:
                diffs.append(r[other] - r[key])
        return (float(np.mean(diffs)), len(diffs)) if diffs else (np.nan, 0)

    lines.append(f"### Fronts added value (delta {metric} vs fronts=none, "
                 "paired over smidx/pbl)")
    for base in BASES:
        for choice in ("met", "pred"):
            diffs = []
            for smidx in (0, 1):
                for pbl in (0, 1):
                    a = (base, choice, smidx, pbl)
                    b = (base, "none", smidx, pbl)
                    if a in r.index and b in r.index:
                        diffs.append(r[a] - r[b])
            if diffs:
                lines.append(f"- base={base}: fronts={choice}: "
                             f"{np.mean(diffs):+.4f} (n={len(diffs)} pairs)")
    d, n = paired_delta(0, 1, axis=2)
    lines.append(f"\n### SM_IDX added value: {d:+.4f} "
                 f"(mean paired delta smidx 0->1, n={n})")
    d, n = paired_delta(0, 1, axis=3)
    lines.append(f"### PBL added value: {d:+.4f} "
                 f"(mean paired delta pbl 0->1, n={n})")
    return lines


MODE_BLURBS: dict[str, str] = {
    "wide": ("Mode **wide**: Mark's hourly=False layout -- one row per "
             "cell-day, every hour a separate feature column, one "
             "multi-output forest for the whole qpe h1..h6 series."),
    "pooled": ("Mode **pooled**: ONE hour-blind forest on hour-matched rows "
               "-- each hourly predictor is a single 'current hour' column, "
               "daily predictors repeat across hours, forecast hour is NOT "
               "a feature. Split grouped by cell-day (same partition as "
               "wide), replacing Mark's raveled iid split to keep hours of "
               "one cell-day on one side."),
    "perhour": ("Mode **perhour**: SIX independent forests, one per "
                "forecast hour (hour-h features -> hour-h qpe), Mark's "
                "verbatim fit on the hour-sliced sample; same cell-day "
                "split in every hour and in the other modes. r2_test pools "
                "the six test legs."),
}


def write_report(out_dir: Path, results: pd.DataFrame,
                 facts: dict | None, mode: str = "wide") -> Path:
    """REPORT.md: sample provenance, ranked topline, axis deltas, caveats.

    Rows rank on the headline set-point Gini (gini_p99_5); R^2 stays a
    column because it answers a different (variance-explained) question, but
    it is SECONDARY -- see caveat (a) on iid-split optimism.
    """
    out = ["# RF feature-set experiments (Mark's RF, 36-cell grid)", ""]
    out.append("Target: hourly grid-mean QPE (h1..h6); fit spec: Mark's "
               f"walkthrough (seed {SEED}, train_fraction {TRAIN_FRACTION}, "
               f"{models.DEFAULT_RFR_KWARGS}).")
    out.append("")
    out.append("Headline metric: **Gini of the RF prediction as a risk "
               "score for qpe exceedance at the P95 / P99.5 set points** "
               "(gini_p95, gini_p99_5; absolute thresholds from the base "
               "sample). R^2 is retained as a secondary column only -- its "
               "absolute level is inflated by the iid split (caveat a), and "
               "the set-point Ginis are what the project ranks on.")
    out.append("")
    out.append(MODE_BLURBS[mode])
    out.append("")
    if facts:
        out.append("## Common sample")
        if "years" in facts or "months" in facts:
            out.append(f"- training window: years {facts.get('years')}, "
                       f"months {facts.get('months')} (inclusive; "
                       f"results under a different window are refused by "
                       f"the consistency guard)")
        if facts.get("thresholds"):
            out.append("- set-point thresholds (mm/h, base sample): "
                       + ", ".join(f"{k}={v:.4f}" for k, v in
                                   facts["thresholds"].items()))
        out.append(f"- n_samples (one common finite sample for ALL "
                   f"experiments): **{facts['n_samples']}** of "
                   f"{facts['n_cell_days_before_screen']} screened cell-days")
        yrs = facts["per_year_counts"]
        out.append("- per-year counts: "
                   + ", ".join(f"{y}: {n}" for y, n in sorted(yrs.items())))
        out.append(f"- surviving year span: "
                   f"{min(yrs)}-{max(yrs)}" if yrs else "- EMPTY sample")
        out.append(f"- cells: {facts['n_cells_in_common_sample']} of "
                   f"{facts['n_cells_before_screen']} (predicted-front "
                   f"domain restriction, caveat e)")
        out.append("")
    if len(results):
        metric = _rank_metric(results)
        ranked = results.sort_values(metric, ascending=False)
        cols = ["id", "n_samples"]
        cols += [c for c in ("gini_p95", "gini_p99_5") if c in ranked.columns]
        cols += ["r2_test", "tail_top10_capture"]
        out.append(f"## Ranked results ({len(results)}/36 experiments, "
                   f"by {metric})")
        out.append(_md_table(ranked, cols))
        out.append("")
        out.append(f"## Key deltas (on {metric})")
        out += _axis_deltas(results, metric)
        out.append("")
    out.append(CAVEATS)
    path = Path(out_dir) / "REPORT.md"
    path.write_text("\n".join(out))
    return path


# --------------------------------------------------------------------------- #
# Figures (visual conventions of scripts/plot_dlfront_results.py:
# dataviz-validated categorical palette, ink text tokens, recessive spines,
# one file per figure)
# --------------------------------------------------------------------------- #
INK = "0.2"
#: dataviz reference categorical theme, fixed order (same constant as
#: scripts/plot_dlfront_results.py). Assigned to the FRONTS choice (the
#: figure's categorical identity), never cycled.
LEG_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")
FRONTS_COLORS = {"none": LEG_PALETTE[0], "met": LEG_PALETTE[1],
                 "pred": LEG_PALETTE[2]}
#: smidx/pbl variants: texture (hatch) + lightness, a secondary encoding on
#: top of the fronts hue so the four variants stay tellable in print/CVD.
VARIANT_STYLE = {(0, 0): dict(alpha=1.00, hatch=None),
                 (1, 0): dict(alpha=0.75, hatch="///"),
                 (0, 1): dict(alpha=0.75, hatch="..."),
                 (1, 1): dict(alpha=0.55, hatch="xxx")}


def _style_ax(ax) -> None:
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("0.7")


#: An experiment within this margin of the best headline Gini is ALWAYS shown
#: in the curated grid figure, curated or not -- the data-driven catch-all
#: that keeps curation from ever hiding a surprise winner.
NEAR_BEST_MARGIN: float = 0.005


def worth_showing(results: pd.DataFrame) -> list[str]:
    """The curated, EXPLICIT selection of experiments for the headline grid
    figure -- figures are VIEWS on results.csv, never silently truncated
    data, so every inclusion has a stated reason:

    - each base's bare baseline (fronts-none, smidx-0, pbl-0): the floor
      every add-on is judged against;
    - each base's +pbl variant: PBL is the established biggest single add-on,
      so its lift per base is the first question anyone asks;
    - the fronts contrast at both+pbl (none/met/pred): the head-to-head the
      fronts axis exists for, on the strongest base;
    - the full both+met+smidx+pbl: everything on, the ceiling;
    - the NEAR-BEST CATCH-ALL: any scored experiment within
      NEAR_BEST_MARGIN of the global best headline Gini that curation
      missed -- a surprise winner can never be hidden by the hand-picked
      list above.

    Only ids present in ``results`` are returned (partial runs stay valid);
    order is curation order, catch-alls appended.  The caller prints how many
    scored experiments the figure omits.
    """
    curated: list[str] = []
    for base in BASES:  # bare baseline, then +pbl, per base
        curated.append(f"base-{base}_fronts-none_smidx-0_pbl-0")
        curated.append(f"base-{base}_fronts-none_smidx-0_pbl-1")
    for fronts in FRONTS_AXIS:  # fronts contrast on the strongest base
        curated.append(f"base-both_fronts-{fronts}_smidx-0_pbl-1")
    curated.append("base-both_fronts-met_smidx-1_pbl-1")  # everything on

    scored = set(results["id"])
    picked = [i for i in dict.fromkeys(curated) if i in scored]

    metric = _rank_metric(results)
    if metric in results.columns:
        vals = results.set_index("id")[metric].dropna()
        if len(vals):
            near_best = vals[vals >= vals.max() - NEAR_BEST_MARGIN]
            picked += [i for i in near_best.sort_values(ascending=False).index
                       if i not in picked]
    return picked


def figure_gini_grid(results: pd.DataFrame, path: Path) -> None:
    """Curated headline-metric grid: gini_p99_5 bars for the WORTH_SHOWING
    selection, with a lighter marker for gini_p95 on each bar.  The full
    36-experiment table always lives in results.csv / REPORT.md; this figure
    is a curated view and SAYS SO (printed omission count)."""
    import matplotlib.pyplot as plt

    ids = worth_showing(results)
    if not ids:
        print("NOTE: nothing selected for the curated grid figure; skipping")
        return
    omitted = len(results) - len(ids)
    print(f"curated grid figure shows {len(ids)} of {len(results)} scored "
          f"experiments ({omitted} omitted); the full table lives in "
          f"results.csv / REPORT.md next to the figure")

    r = results.set_index("id")
    metric = _rank_metric(results)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * len(ids)), 4.6))
    xs = np.arange(len(ids))
    for x, eid in zip(xs, ids):
        fronts = r.loc[eid, "fronts"] if "fronts" in r.columns else "none"
        ax.bar(x, r.loc[eid, metric], width=0.72,
               color=FRONTS_COLORS.get(fronts, LEG_PALETTE[0]),
               edgecolor="white", linewidth=0.5)
        # secondary set point as a lighter marker (a second bar per id makes
        # the curated story harder to read; P95 details stay in the report)
        if "gini_p95" in r.columns and np.isfinite(r.loc[eid, "gini_p95"]):
            ax.plot(x, r.loc[eid, "gini_p95"], marker="D", ms=5,
                    color="0.45", mfc="white", mew=1.2, ls="none")
    ax.set_xticks(xs)
    ax.set_xticklabels([i.replace("base-", "").replace("_fronts-", "\n")
                        .replace("_smidx-", " sm").replace("_pbl-", " pbl")
                        for i in ids], color=INK, fontsize=7)
    ax.set_ylabel(f"{metric} (bars; diamonds = gini_p95)", color=INK,
                  fontsize=9)
    ax.set_title("Curated headline view: set-point Gini by experiment\n"
                 f"(explicit worth_showing selection + any run within "
                 f"{NEAR_BEST_MARGIN} of the best; hue = fronts choice)",
                 color=INK, fontsize=10)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FRONTS_COLORS[f], label=f"fronts={f}")
               for f in FRONTS_AXIS]
    ax.legend(handles=handles, fontsize=7, frameon=False, labelcolor=INK)
    _style_ax(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: Importance-trim rule (Zach 2026-08-22): keep the smallest head of the
#: ranked features covering this fraction of total importance ...
IMPORTANCE_COVERAGE: float = 0.90
#: ... capped at this many features, so near-zero clutter drops without a
#: flat arbitrary top-N.
IMPORTANCE_CAP: int = 12


def trim_importances(imp: pd.Series,
                     coverage: float = IMPORTANCE_COVERAGE,
                     cap: int = IMPORTANCE_CAP) -> pd.Series:
    """Ranked importances -> the smallest head covering ``coverage`` of the
    total, capped at ``cap`` features (always at least one).

    Replaces the old flat top-15: a forest that concentrates on 4 features
    shows 4 bars, one that spreads over 30 shows the 12 that matter most --
    the trim rule is stated in the axis label so the figure is self-
    describing.
    """
    imp = imp.sort_values(ascending=False)
    total = float(imp.sum())
    if total <= 0 or not np.isfinite(total):
        return imp.head(min(cap, len(imp)))  # degenerate: nothing to rank on
    frac = imp.cumsum() / total
    n_cover = int(np.searchsorted(frac.to_numpy(), coverage) + 1)
    return imp.head(max(1, min(n_cover, cap, len(imp))))


def _best_id_per_base(results: pd.DataFrame, base: str) -> str | None:
    """Best experiment id for one base, by the headline set-point Gini
    (falls back to r2_test only on pre-Gini-era frames)."""
    sub = results[results["base"] == base]
    metric = _rank_metric(results)
    sub = sub[sub[metric].notna()] if metric in sub.columns else sub.iloc[:0]
    if sub.empty:
        return None
    return sub.loc[sub[metric].idxmax(), "id"]


def figure_importances(results: pd.DataFrame, out_dir: Path,
                       path: Path) -> None:
    """Feature importances for the BEST (by gini_p99_5) experiment of each
    base, trimmed to 90% cumulative importance (max 12 features)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(BASES), figsize=(4.0 * len(BASES), 4.6))
    for ax, base in zip(np.atleast_1d(axes), BASES):
        best = _best_id_per_base(results, base)
        imp_path = Path(out_dir) / "importances" / f"{best}.csv"
        if best is None or not imp_path.exists():
            ax.set_axis_off()
            continue
        frame = pd.read_csv(imp_path, index_col=0)
        if "importance" not in frame.columns:
            # e.g. a perhour-shaped (stem x hour) CSV in a wide/pooled dir:
            # skip the panel rather than plot the wrong thing
            print(f"NOTE: {imp_path} has no 'importance' column "
                  f"(wrong-mode artifact?); skipping its panel")
            ax.set_axis_off()
            continue
        imp = trim_importances(frame["importance"])
        ax.barh(range(len(imp))[::-1], imp.values, height=0.72,
                color=LEG_PALETTE[0])
        ax.set_yticks(range(len(imp))[::-1])
        ax.set_yticklabels(imp.index, fontsize=7, color=INK)
        ax.set_xlabel(f"RF importance (features covering "
                      f"{IMPORTANCE_COVERAGE:.0%} of total, max "
                      f"{IMPORTANCE_CAP})", color=INK, fontsize=8)
        ax.set_title(f"best {base} by {HEADLINE_METRIC}:\n{best}",
                     color=INK, fontsize=8)
        _style_ax(ax)
    fig.suptitle(f"Feature importances, best experiment per base "
                 f"(by {HEADLINE_METRIC})", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: The curated head-to-head contrasts the Gini-curve figures draw -- each
#: panel varies EXACTLY ONE grid axis so the curves answer one question:
#: (a) fronts none/met/pred on the strongest base with PBL on;
#: (b) which base, with fronts off and PBL on;
#: (c) does PBL move the tail, at base=both fronts=met.
#: smidx is held at 0 throughout so no panel confounds two axes.
CURVE_CONTRASTS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("fronts (base=both, pbl=1)", tuple(
        (f"base-both_fronts-{f}_smidx-0_pbl-1", f"fronts={f}")
        for f in ("none", "met", "pred"))),
    ("base (fronts=none, pbl=1)", tuple(
        (f"base-{b}_fronts-none_smidx-0_pbl-1", f"base={b}")
        for b in ("airs", "smap", "both"))),
    ("pbl (base=both, fronts=met)", tuple(
        (f"base-both_fronts-met_smidx-0_pbl-{p}", f"pbl={p}")
        for p in (0, 1))),
)


def figure_gini_curves(out_dir: Path, path: Path, results: pd.DataFrame,
                       label: str) -> None:
    """Capture (Lorenz-style) curve panels for ONE set point (``label`` in
    SET_POINTS), one panel per curated contrast in CURVE_CONTRASTS, each
    curve's Gini from results.csv annotated in the legend.

    Curves are read from ``tail_cdf/<id>_<label>.csv`` by filename, and ONLY
    for ids present in the CURRENT results frame -- an orphaned or legacy CSV
    left by an interrupted or differently-scoped earlier run must never sneak
    in (same scored-ids discipline as the old single tail figure).
    """
    import matplotlib.pyplot as plt

    r = results.set_index("id")
    gini_col = f"gini_{label}"
    fig, axes = plt.subplots(1, len(CURVE_CONTRASTS),
                             figsize=(4.4 * len(CURVE_CONTRASTS), 4.0),
                             sharey=True)
    plotted_any = 0
    for ax, (title, members) in zip(np.atleast_1d(axes), CURVE_CONTRASTS):
        plotted = 0
        for ci, (eid, line_label) in enumerate(members):
            if eid not in r.index:
                print(f"NOTE: {eid} not in current results.csv; "
                      f"skipping its {label} curve")
                continue
            cdf_path = Path(out_dir) / "tail_cdf" / f"{eid}_{label}.csv"
            if not cdf_path.exists():
                print(f"NOTE: {cdf_path} missing (old-sample artifact?); "
                      f"skipping its curve")
                continue
            cdf = pd.read_csv(cdf_path)["cum_freq"].to_numpy()
            # fronts panels keep the house fronts hue; others take the
            # palette in order
            color = (FRONTS_COLORS[line_label.split("=")[1]]
                     if line_label.startswith("fronts=")
                     else LEG_PALETTE[ci])
            gini = (float(r.loc[eid, gini_col])
                    if gini_col in r.columns else np.nan)
            gtxt = f" (Gini {gini:.3f})" if np.isfinite(gini) else ""
            ax.plot(np.arange(1, 101), cdf, lw=2.0, color=color,
                    label=f"{line_label}{gtxt}")
            plotted += 1
        if not plotted:
            ax.set_axis_off()
            continue
        plotted_any += plotted
        ax.plot([0, 100], [0, 1], lw=1.0, ls="--", color="0.6",
                label="no skill")
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("predicted-value percentile bin", color=INK, fontsize=8)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
        _style_ax(ax)
    if not plotted_any:
        plt.close(fig)
        print(f"NOTE: no {label} curves available; skipping {path.name}")
        return
    np.atleast_1d(axes)[0].set_ylabel(
        f"cumulative fraction of obs > P{SET_POINTS[label]} events",
        color=INK, fontsize=8)
    fig.suptitle(f"Tail capture at the P{SET_POINTS[label]} set point "
                 "(lower curve = more events pushed to high predictions)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_perhour_heatmap(results: pd.DataFrame, out_dir: Path,
                           path: Path) -> None:
    """(stem x hour) importance heatmaps for the best (by gini_p99_5)
    experiment per base -- the ``perhour`` mode's "systematic differences by
    hour" readout.  Rows trimmed by the same 90%-coverage / max-12 rule as
    the bar figure, ranked on the stems' mean importance across hours."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(BASES), figsize=(4.6 * len(BASES), 5.2))
    for ax, base in zip(np.atleast_1d(axes), BASES):
        best = _best_id_per_base(results, base)
        imp_path = Path(out_dir) / "importances" / f"{best}.csv"
        if best is None or not imp_path.exists():
            ax.set_axis_off()
            continue
        imp = pd.read_csv(imp_path, index_col=0)
        keep = trim_importances(imp.mean(axis=1)).index
        imp = imp.loc[keep]
        im = ax.imshow(imp.to_numpy(), aspect="auto", cmap="magma_r",
                       vmin=0.0)
        ax.set_xticks(range(imp.shape[1]))
        ax.set_xticklabels(imp.columns, fontsize=7, color=INK)
        ax.set_yticks(range(imp.shape[0]))
        ax.set_yticklabels(imp.index, fontsize=6, color=INK)
        ax.set_title(f"best {base} by {HEADLINE_METRIC}:\n{best}",
                     color=INK, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=6,
                                                           colors=INK)
        _style_ax(ax)
    fig.suptitle("Per-hour RF importances (one forest per forecast hour; "
                 f"stems covering {IMPORTANCE_COVERAGE:.0%} of mean "
                 f"importance, max {IMPORTANCE_CAP})",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: Headline configuration for the cross-mode hour-profile figure.
HEADLINE_ID: str = "base-both_fronts-met_smidx-1_pbl-1"


def figure_r2_by_hour(out_dir: Path, path: Path, mode: str,
                      results: pd.DataFrame) -> None:
    """Test R^2 per forecast hour for the headline experiment, this mode's
    curve plus any sibling mode whose results.csv already exists -- the
    direct wide vs pooled vs perhour hour-profile comparison (all modes
    share the sample and the test cell-days, so the curves are paired)."""
    import matplotlib.pyplot as plt

    root = Path(out_dir)
    base_root = root if mode == "wide" else root.parent
    mode_colors = dict(zip(MODES, LEG_PALETTE))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    plotted = 0
    for m, (_, sub) in MODES.items():
        mdir = base_root / sub if sub else base_root
        frame = results if m == mode else (
            pd.read_csv(mdir / "results.csv")
            if (mdir / "results.csv").exists() else None)
        if frame is None or HEADLINE_ID not in set(frame["id"]):
            continue
        row = frame.set_index("id").loc[HEADLINE_ID]
        hours = range(1, 7)
        ax.plot(list(hours), [row[f"r2_h{h}"] for h in hours], "o-", lw=2,
                color=mode_colors[m], label=m)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("forecast hour (slot; 21..02 UTC)", color=INK, fontsize=9)
    ax.set_ylabel("test R$^2$", color=INK, fontsize=9)
    ax.set_title(f"Hour profile of test R$^2$, {HEADLINE_ID}\n"
                 "(same sample, same test cell-days in every mode)",
                 color=INK, fontsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, title="mode",
              title_fontsize=8)
    _style_ax(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figures(out_dir: Path, results: pd.DataFrame,
                 mode: str = "wide") -> None:
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        print("NOTE: no results yet; skipping figures")
        return
    figure_gini_grid(results, fig_dir / "gini_grid_curated.png")
    if mode == "perhour":
        figure_perhour_heatmap(results, out_dir,
                               fig_dir / "importances_by_hour.png")
    else:
        figure_importances(results, out_dir, fig_dir / "importances_best.png")
    # the lead figure family: one capture-curve panel figure per set point
    for label in SET_POINTS:
        figure_gini_curves(out_dir, fig_dir / f"gini_curves_{label}.png",
                           results, label)
    figure_r2_by_hour(out_dir, fig_dir / "r2_by_hour_modes.png", mode,
                      results)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_years(spec: str) -> tuple[int, ...]:
    """'2016-2021' or '2016,2018,2020' -> sorted tuple of ints."""
    years: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            years += list(range(int(lo), int(hi) + 1))
        else:
            years.append(int(part))
    return tuple(sorted(set(years)))


def parse_months(spec: str) -> tuple[int, int]:
    """'3-11' (or a single '6') -> the inclusive (lo, hi) month window,
    the same (lo, hi) convention as ``AnalysisConfig.months``."""
    lo, _, hi = spec.partition("-")
    lo, hi = int(lo), int(hi) if hi else int(lo)
    if not (1 <= lo <= hi <= 12):
        raise ValueError(f"--months {spec!r}: need 1 <= lo <= hi <= 12")
    return lo, hi


def gini_thresholds(base: pd.DataFrame, cfg: AnalysisConfig
                    ) -> dict[str, float]:
    """Absolute mm/h thresholds for the SET_POINTS labels, from the base
    table's in-domain land rows (pre-screen -- the paper's "thresholds are
    based on all data" convention, same as the hypothesis battery and the
    cluster runner), so every experiment scores identical event definitions.
    """
    ladder = dataset.qpe_percentile_thresholds(
        base, cfg, percentiles=tuple(SET_POINTS.values()))
    return {label: ladder[pct] for label, pct in SET_POINTS.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Mark's RF over the 36-experiment feature-set grid")
    # 2017-2021: 2016 was only ever dropped IMPLICITLY (the PBL product
    # starts 2017, so the common finite sample lost 2016 anyway); the default
    # now states that intent explicitly (Zach 2026-08-22).
    ap.add_argument("--years", default="2017-2021",
                    help="year range/list; part of the training window the "
                         "common sample depends on, so batched --subset runs "
                         "stay comparable")
    ap.add_argument("--months", default="3-11",
                    help="inclusive month window lo-hi (default 3-11 = "
                         "March-November, the convective season the whole "
                         "suite analyses); part of the sample-consistency "
                         "contract like --years")
    ap.add_argument("--mode", default="wide", choices=tuple(MODES),
                    help="wide = Mark's multi-output layout (original runs); "
                         "pooled = one hour-blind forest on hour-matched "
                         "rows; perhour = six independent hour-h forests")
    ap.add_argument("--subset", default=None,
                    help="regex over experiment ids (batching)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run experiments already in results.csv")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    years = parse_years(args.years)
    months = parse_months(args.months)
    fit_fn, mode_subdir = MODES[args.mode]
    out_dir = Path(args.out_dir) if args.out_dir else (
        config.RESULTS_DIR / "rf_experiments" / mode_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "importances").mkdir(exist_ok=True)
    (out_dir / "tail_cdf").mkdir(exist_ok=True)

    # Refuse (or, under --force, discard) results left by a run with a
    # different --years/--months window: they were fit on a different common
    # sample (pre-months-era directories are refused too -- their month
    # window is unknown).
    results = check_window_consistency(out_dir, years, months,
                                       force=args.force)
    grid = experiment_grid()
    todo = experiments_to_run(grid, set(results["id"]), subset=args.subset,
                              force=args.force)
    print(f"{len(todo)} of {len(grid)} experiments to run "
          f"({len(results)} already in results.csv)")

    facts = None
    facts_path = out_dir / "sample_info.json"
    if todo:
        cfg = AnalysisConfig(years=years, months=months,
                             sample_unit="cell_day")
        prepared = dataset.prepare(cfg)
        # cfg.months already restricts the base table; the explicit filter
        # makes the Mar-Nov training window provable at this layer too
        cell_days = filter_month_window(prepared.cell_days, months)
        # absolute Gini set-point thresholds from the base table (pre-screen)
        thresholds = gini_thresholds(dataset.build_base_table(cfg), cfg)
        enriched = enrich_cell_days(cell_days, years)
        ds = build_common_sample(enriched)
        facts = {"years": list(years), "months": list(months),
                 "thresholds": thresholds, "mode": args.mode,
                 **sample_facts(ds, enriched)}
        facts_path.write_text(json.dumps(facts, indent=2))
        print(f"common sample: {facts['n_samples']} cell-days; per-year "
              f"{facts['per_year_counts']}; cells "
              f"{facts['n_cells_in_common_sample']}/"
              f"{facts['n_cells_before_screen']}; set-point thresholds "
              f"{thresholds}")

        for exp in todo:
            print(f"=== [{args.mode}] {exp.id} ({len(exp.features)} stems) ===")
            metrics = fit_fn(ds, exp, thresholds=thresholds)
            metrics["importances"].to_csv(
                out_dir / "importances" / f"{exp.id}.csv",
                index_label="feature")
            write_tail_curves(out_dir, exp.id, metrics["tail_cdfs"])
            row = {"id": exp.id, "base": exp.base, "fronts": exp.fronts,
                   "smidx": exp.smidx, "pbl": exp.pbl,
                   "n_samples": facts["n_samples"],
                   "gini_p95": metrics["gini_p95"],
                   "gini_p99_5": metrics["gini_p99_5"],
                   "r2_test": metrics["r2_test"],
                   **metrics["r2_per_hour"],
                   "tail_top10_capture": metrics["tail_top10_capture"]}
            results = upsert_result(results, row)
            results.to_csv(out_dir / "results.csv", index=False)  # resumable

    if facts is None and facts_path.exists():
        facts = json.loads(facts_path.read_text())
    write_report(out_dir, results, facts, mode=args.mode)
    if not args.no_figures:
        make_figures(out_dir, results, mode=args.mode)
    print(f"done: {len(results)}/36 experiments in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
