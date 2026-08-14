# Work Plan: Unified Hypothesis-Test Battery (SM × Thermodynamics → Heavy Precip)

**Goal.** Implement statistical tests for the hypotheses in
`ScienceReports/hypothesis_table.md` (T1–T5, S1–S5, A1–A6) on the
`FCST_SMAP_MRMS_{2016–2021}` dataset, as **one unified, reusable battery** rather
than 16 bespoke scripts. First pass uses the existing predictor-agnostic
**Gini** machinery (`convection_skill.gini`); the architecture leaves clean seams
for logistic/quantile regression later. Local conditions only for now; every
hypothesis records how the **HYSPLIT trajectory kernels** (`trajectory_kernels/`)
extend it.

## Design principles

1. **One spec, one runner.** Each hypothesis is a declarative `HypothesisSpec`
   (predictor column/recipe, target, expected sign, controls, strata,
   kernel-extension note). A single runner executes any spec; adding a
   hypothesis = adding a spec, not code.
2. **Autocorrelation is first-class.** Significance comes from a **day/block
   bootstrap**: resample whole days (preserves within-day spatial correlation)
   in moving blocks of L days (preserves synoptic/SM temporal persistence).
   Naive iid CIs are also reported to show the inflation factor. Block length L
   from the measured e-folding of daily-anomaly autocorrelation (audit agent).
   Battery-wide multiplicity via Benjamini–Hochberg FDR (Wilks-style field
   significance).
3. **Seasonality removed at the predictor, tested at the strata.** Predictors
   enter as **day-of-year harmonic-climatology anomalies** (per cell, pooled
   over years), optionally z-scored per cell; season (MAM/JJA/SON) is also a
   stratification axis so seasonal regime differences are visible rather than
   hidden.
4. **Increments over the A1 baseline.** SM terms are evaluated as *conditional*
   Gini within CAPE bins (rank-based analog of "skill beyond CAPE"), per the
   table's mandate that Group T/S be skill increments.
5. **Pre-registered strata** (garden-of-forking-paths guard): humidity terciles
   (T3), lon east/west split + cell-climatology aridity terciles (T1/T2 regime),
   wind terciles (S2), local-solar-hour early/late (A3, S5), season. Declared in
   config before results are seen.
6. **Confounding (Tuttle & Salvucci 2017):** antecedent-precipitation
   conditioning enters as a control (conditional Gini on antecedent P) — full
   T&S persistence framework flagged as an extension.

## Package: `src/hypothesis_tests/`

| Module | Role |
|---|---|
| `config.py` | thresholds, strata definitions, block length, FDR alpha, variable map — all cited |
| `predictors.py` | harmonic deseasonalization, per-cell-month z-scores, antecedent lags, local-vs-nonlocal (Guillod) decomposition, derived predictors (MU−MML, interactions) |
| `table.py` | one tidy multi-year analysis table (extends `convection_skill.data_loading` with the extra variables + daily L4 fields) |
| `stats.py` | weighted Gini (one-sort trick), day/moving-block bootstrap CIs & p-values, conditional (within-control-bin) Gini, event-rate curves (for non-monotonic A2), BH-FDR |
| `experiments.py` | `HypothesisSpec` dataclass + the T/S/A registry |
| `report.py` | results table (markdown) + standard figures |
| `notebooks/08_hypothesis_tests.py` | thin driver |

**Tests:** synthetic-data analytic answers throughout — deseasonalization
recovers a planted sinusoid; block bootstrap widens CIs on AR(1)-by-day data but
matches naive on iid data; conditional Gini ≈ 0 for a predictor collinear with
its control and ≈ marginal for an independent one; weighted Gini == unweighted
at unit weights.

## Hypothesis → first-pass Gini operationalization (local-only)

| ID | Predictor (anomaly unless noted) | Target | Test | Kernel extension |
|----|----|----|----|----|
| T1/T2 | `SMAP_L4_smsfc_av` anom | QPE>P99.9 | signed Gini, ± by regime strata; conditional on CAPE | upstream kernel-weighted SM instead of local |
| T3 | same | same | Gini sign across humidity terciles (`FCST_q` or `qlay1`) | kernel-weighted upstream humidity as the gate |
| T4 | antecedent SM (lags 1–30 d) | QPE + mediator CIN>Pxx | lagged Gini ladder; SM→CIN mediation Gini | antecedent SM along the *inflow path* (lagged kernels) |
| T5 | SM anom | MML_CAPE>Pxx and MML_LCL<Pxx | two mediator Ginis, opposite signs | kernel-weighted SM → advected CAPE adjustment (RECEPTOR_BAND full-column mode) |
| S1 | `smsfc_sd`, `_absgrad` | QPE>P99.9 | Gini conditional on mean SM + CAPE | gradients sampled along inflow (kernel-weighted gradient) |
| S2 | S1 strata by wind | same | Gini by \|u\| terciles (⚠ u-only; flag) | wind gate becomes trajectory speed itself |
| S3 | signed `_wegrad`,`_sngrad` | QPE_max>Pxx, sk>Pxx | signed Gini (⚠ no shear vector; flag) | ∇SM·(flow direction) alignment from trajectories |
| S4 | nonlocal (domain-day) vs local (cell−neighborhood) anoms | QPE>P99.9 | two Ginis, opposite signs expected | upstream-vs-receptor decomposition via kernel |
| S5 | −SM anom | early onset among precipitating cell-days | Gini | onset timing vs upstream SM contact hour |
| A1 | `FCST_MU_CAPE` raw | QPE>P99.9/99.95 | baseline Gini (replicates paper) | (already the trajectory product) |
| A2 | `FCST_MU_CIN` | P(extreme) vs P(any precip) | event-rate curves + Gini sign flip between targets | CIN along path (cap erosion history) |
| A3 | MU−MML CAPE | QPE by local-hour strata | Gini late vs early; SM Gini weaker late (falsification) | elevated receptor band kernels |
| A4 | `FCST_q` | QPE>P99.9 | conditional Gini on CAPE + event-rate threshold shape (⚠ needs RH profile for full test) | kernel-weighted upstream column humidity |
| A5 | ⛔ (needs profile+PBL) | — | flagged only | PBL model already pluggable in kernels |
| A6 | ⛔ local | — | flagged: **the flagship trajectory-kernel application** (discount-weighted upstream SM = recycling proxy) | `apply_kernel(kernels, smap)` + rain-out discount |

## Phases

- **Phase 0** *(agents in flight)*: variable audit (existence/semantics incl.
  `MRMS_max`/`_sk`, `FCST_q`, L4 gradients, `pflux`) + methods lit
  (block length, FDR, anomaly practice, T&S recipe, Guillod decomposition).
- **Phase 1**: `predictors.py` + `stats.py` + synthetic tests (agent-independent).
- **Phase 2**: `table.py` (audit-informed variable map), build the 2016–2021 table.
- **Phase 3**: `experiments.py` registry + runner; run battery
  (primary pooled 2016–2021; sensitivity: paper years 2019–2020).
- **Phase 4**: `report.py` — results markdown (incl. per-hypothesis verdicts,
  naive-vs-block CI inflation, FDR-adjusted significance), figures, and the
  **variable-assumption corrections** the hypothesis table asked for.
