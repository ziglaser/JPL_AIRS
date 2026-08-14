# Unified skill suite — merge of `convection_skill` + `hypothesis_tests`

2026-07-22. Decisions confirmed with Zach (AskUserQuestion, this date):

1. **Inference**: both conventions implemented; ONE convention per run
   (`AnalysisConfig.inference`), recorded in every results row. Default is the
   paper's iid row bootstrap (`"iid"`); the day-block bootstrap (`"block"`) is
   the honest alternative (audit: iid SEs 2–3× overconfident on this data).
2. **Sample validity**: a config option, compositional —
   `valid_datasets ⊆ {"airs", "airs_fcst", "smap", "mrms"}` +
   `require_complete_days`. Zach's three named modes:
   (1) every cell-hour valid for all four products → `("airs", "airs_fcst",
   "smap", "mrms")`; (2) same but only complete cell-days →
   `+ require_complete_days=True`; (3) SMAP-specific → `("smap", "mrms")`.
   Event thresholds ALWAYS come from the base sample (all in-domain land rows,
   the paper's "thresholds are based on all data" rule) regardless of validity.
3. **Package**: everything merges INTO `convection_skill`;
   `hypothesis_tests` is deleted.
4. **Replication layer**: ported now — notebooks 01–05 and the paper-benchmark
   fixtures build their samples through the unified builder. Benchmark values
   are unchanged and must stay green.

## Layout (post-merge)

- `config.py`      — merged cited constants + `AnalysisConfig` dataclass
                     (sample filters, screens, convective filter, inference)
                     with presets for the three validity modes + paper mode.
- `data_loading.py` — stages 1–2 only (`load_raw`, `make_uniform`); the
                     overpass replication is generalized to
                     `OVERPASS_REPLICATED_VARS` (fixes the removed
                     `PREDICTOR_VAR`). `finalize_table`/`build_analysis_table`
                     are absorbed by `dataset.py`.
- `dataset.py`     — THE builder. `build_base_table(cfg)` (cached superset:
                     domain rows incl. ocean/no-AIRS), `event_thresholds(base)`,
                     `build_dataset(cfg)` (validity + screens applied),
                     `prepare(cfg) -> Prepared(table, flags, onset, thresholds)`,
                     plus `build_flags`, `rain_screen_mask`, `build_onset_table`,
                     daily SM predictor derivation.
- `quality_control.py` — unchanged cited rule functions; builder calls them.
- `gini.py`, `significance.py`, `analysis.py`, `paper_benchmarks.py` — unchanged
                     (analysis functions are already table-agnostic).
- `stats.py`       — battery inference (moved) with `method="iid"|"block"`
                     throughout (shared rank-bin×day backend).
- `predictors.py`  — anomaly engineering (moved verbatim).
- `hypotheses.py`  — `HypothesisSpec` + registry + stratifiers (moved).
- `suite.py`       — `test_hypothesis(spec, prepared, ...)` and
                     `run_suite(cfg)` → tidy comparable results
                     (columns incl. `run`, `inference`); BH-FDR per run.
- `report.py`      — battery report/figures (moved).

## Efficiency contract

- The expensive artifact is the cached BASE table (keyed by years/slots/months/
  domain + `DATASET_VERSION`); every validity mode, rain screen, and convective
  setting is a cheap in-memory row/flag operation on top — configs share cache.
- Flags, stratifier labels, and day codes are computed once per run and shared
  across all hypotheses.

## Two-file config layout (2026-07-23)

`AnalysisConfig` now carries EVERYTHING that governs a run, split across two
YAML files whose key ownership is enforced by the loader
(`AnalysisConfig.from_files`; a knob in the wrong file raises, naming where it
belongs):

- **`configs/data_table.yaml`** (`DATA_TABLE_KEYS`) — which rows exist:
  years/months/slots, domain, land fraction, `valid_datasets`,
  `require_complete_days`.
- **`configs/hypothesis_tests.yaml`** (`HYPOTHESIS_KEYS`) — everything that
  governs the tests: event definitions (`heavy_percentile` = THE QPE
  percentile the Gini events use, plus the sens/any/max/sk thresholds), the
  post-rain screens (moved here — an analysis decision, not a table
  property), the convective filter, which hypotheses run
  (`hypotheses: all | topline | [ids]`; `topline` = `tier="topline"` specs,
  the primary spec per hypothesis-table row), what stratifies what
  (`default_strata` + per-hypothesis `strata:`/`controls:` overrides),
  scope toggles (`run_controls`/`run_strata`/`run_curves`), and inference
  (`inference`, `n_boot_reps`, `block_days`, `seed`, `fdr_alpha`,
  control-bin settings).

`configs/hypothesis_topline_p99.yaml` is the minimal example: topline
hypotheses only, heavy events at P99. Driver:
`PYTHONPATH=src python src/run_suite.py [data.yaml] [hypothesis.yaml]` —
files in either order (recognized by keys), either omissible (defaults to the
two `configs/` files). `src/battery_config.yaml` (the old combined file) is
retired; a single combined mapping still loads via `AnalysisConfig.from_file`.

Follow-ups (same date):

- `run_curves` accepts `true | false | [ids]` (`cfg.wants_curve`); A4_q's
  registry curve is retired (monotone response) — event-rate curves are now
  A2-only by default.
- `run_suite.py` is THE driver (Zach's call): it now also writes the
  per-hypothesis folders (`results/suite/<HYP_ID>/`) after the runs, side by
  side when there are several. Notebook 08 stays for interactive exploration
  and the convective diagnostics; headless it uses the SAME two-file loader
  (`run_suite.load_configs`) — no more forced pooled+paper double run or CLI
  keyword flags: one run per config, years/hypotheses/strata/curves all from
  the config. Comparison runs = `defaults:`+`runs:` in the hypothesis file.
- Per-run colors in the per-hypothesis figures are assigned per label
  (`_run_colors`), not hard-coded to pooled/paper.
