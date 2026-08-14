"""Run the unified hypothesis suite from the two-file config layout.

    PYTHONPATH=src python src/run_suite.py [data_table.yaml] [hypothesis.yaml]

Two config files define a run: the DATA-TABLE config (which rows exist:
scope + validity; default ``configs/data_table.yaml``) and the HYPOTHESIS
config (everything governing the tests: event thresholds, post-rain screens,
which hypotheses, strata, inference; default ``configs/hypothesis_tests.yaml``).
Files may be given in either order -- each is recognized by its keys -- and
either may be omitted to use its default. A single combined file (both key
groups in one mapping) also still works. The hypothesis file may be a
comparison spec (``defaults:`` + ``runs:`` -- each entry runs as one
comparable suite over the same table).

Per run, artifacts land under ``results/suite/``:

    results_<label>.csv    tidy results (one row per hypothesis x scope)
    RESULTS_<label>.md     report (verdicts, skill increments, strata, honesty)
    curves_<label>.npz     event-rate curves
    figures/ht_*_<label>.png
    <HYP_ID>/              per-hypothesis folder (results.csv, topline.md,
                           strata.png, event_rate.png for curve specs)

With several configs, a combined ``results_combined_<labels>.csv`` is also
written -- directly comparable across runs because every run's event
definitions come from absolute base-sample thresholds.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convection_skill import dataset as D, report as R, suite  # noqa: E402
from convection_skill.config import (  # noqa: E402
    DATA_TABLE_KEYS, HYPOTHESIS_KEYS, AnalysisConfig, load_config_mapping,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CFG = REPO / "configs" / "data_table.yaml"
DEFAULT_HYP_CFG = REPO / "configs" / "hypothesis_tests.yaml"
OUT = Path("results/suite")
FIG = OUT / "figures"


def run_one(cfg):
    label = cfg.label()
    t0 = time.time()
    print(f"[{label}] preparing dataset for years {cfg.years} ...", flush=True)
    prepared = D.prepare(cfg)
    print(f"[{label}] {len(prepared.table):,} rows, "
          f"{prepared.table['day'].nunique()} days, "
          f"{len(prepared.onset):,} precipitating cell-days "
          f"({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    results, curves = suite.run_suite(prepared)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / f"results_{label}.csv", index=False)
    flat = {f"{hid}__{k}": v for hid, c in curves.items() for k, v in c.items()}
    np.savez(OUT / f"curves_{label}.npz", **flat)
    R.write_report(results, OUT / f"RESULTS_{label}.md",
                   run_label=f"{label} ({cfg.years})", cfg=cfg)
    R.plot_forest(results, FIG / f"ht_forest_{label}.png")
    R.plot_curves(curves, FIG / f"ht_curves_{label}.png")

    n_sig = int(results["fdr_significant"].sum())
    print(f"[{label}] {len(results)} tests, {n_sig} FDR-significant "
          f"({time.time() - t0:.0f}s) -> {OUT}/results_{label}.csv", flush=True)
    return label, results, curves


def _config_kind(path: Path) -> str:
    """Which config file this is, recognized by its keys."""
    keys = set(load_config_mapping(path)) - {"name"}
    if keys & {"runs", "defaults"}:
        return "hypothesis"
    if keys <= DATA_TABLE_KEYS | {"preset"}:
        return "data-table"
    if keys <= HYPOTHESIS_KEYS:
        return "hypothesis"
    return "combined"  # legacy single-file layout: both key groups in one


def load_configs(args: list[str]) -> "list[AnalysisConfig]":
    """CLI paths (0-2, any order) -> run configs, defaulting missing files."""
    if len(args) > 2:
        raise SystemExit(f"expected at most 2 config files, got {args}")
    by_kind = {}
    for arg in args:
        kind = _config_kind(Path(arg))
        if kind == "combined":
            if len(args) > 1:
                raise SystemExit(f"{arg} mixes data-table and hypothesis keys; "
                                 "a combined config must be the only file")
            return AnalysisConfig.from_file(arg)
        if kind in by_kind:
            raise SystemExit(f"both {by_kind[kind]} and {arg} look like "
                             f"{kind} configs; pass one of each kind")
        by_kind[kind] = arg
    return AnalysisConfig.from_files(
        by_kind.get("data-table", DEFAULT_DATA_CFG),
        by_kind.get("hypothesis", DEFAULT_HYP_CFG))


def main():
    configs = load_configs(sys.argv[1:])

    results_by_run, curves_by_run = {}, {}
    for cfg in configs:
        label, results, curves = run_one(cfg)
        results_by_run[label] = results
        curves_by_run[label] = curves

    # per-hypothesis folders (results.csv, topline.md, strata.png, event_rate
    # for curve specs), runs side by side when there are several
    R.write_hypothesis_folders(results_by_run, curves_by_run, OUT)
    print(f"per-hypothesis folders written -> {OUT}/<HYP_ID>/", flush=True)

    if len(results_by_run) > 1:
        combined = pd.concat(results_by_run.values(), ignore_index=True)
        labels = "_vs_".join(results_by_run)
        combined_path = OUT / f"results_combined_{labels}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"combined comparable table -> {combined_path}", flush=True)


if __name__ == "__main__":
    main()
