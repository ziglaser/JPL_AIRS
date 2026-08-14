# JPL_AIRS Codebase Guide

**What this repository is.** A three-stage research codebase for testing how soil
moisture and advected thermodynamics control heavy convective precipitation over
CONUS, built on the `FCST_SMAP_MRMS` dataset (AIRS soundings advected with
HYSPLIT/WRF → hourly CAPE; SMAP L4 soil moisture; MRMS hourly QPE; 2016–2021,
1°, Mar–Nov, evening window 21–02 UTC).

The three stages build on each other:

```
convection_skill/          trajectory_kernels/           hypothesis_tests/
─────────────────          ───────────────────           ─────────────────
Replicate Richardson   →   Turn HYSPLIT forward      →   Test the T/S/A hypothesis
et al. (2024): Gini of     trajectories into             table: unified Gini battery
advected CAPE vs heavy     source–receptor "influence    on LOCAL predictors, with
QPE. Establishes the       kernels": where/when the      autocorrelation-honest
predictor-agnostic         arriving air touched the      inference. Each hypothesis
statistical core and       land surface. The upgrade     records its kernel-based
the data corrections.      path from LOCAL to UPSTREAM   (upstream) extension.
                           soil moisture.
```

Design contract shared by all three (see `docs/WORKPLAN.md` §"Design constraint"):
predictor-agnostic pure functions; one function = one cited rule; all constants
in one cited `config.py` per package; explicit sequential stages usable alone;
analytic-answer unit tests; thin notebooks.

---

## Directory map

```
JPL_AIRS/
├── README.md                      project overview
├── docs/
│   ├── CODEBASE_GUIDE.md          ← this file
│   ├── WORKPLAN.md                replication plan (convection_skill)
│   ├── TRAJECTORY_KERNEL_WORKPLAN.md   kernel-tool design + data audit
│   ├── HYPOTHESIS_TESTS_WORKPLAN.md    battery design + methods citations
│   ├── convective_initiation_lit_review.md   CI mechanism catalog (A–N)
│   └── papers/                    Richardson et al. 2024 + supplement
├── data/                          DVC-tracked NetCDF (FCST_SMAP_MRMS_{year}.nc,
│                                  lsm.nc, wrf27km_20190605/ trajectories)
├── src/
│   ├── convection_skill/          stage 1: Gini replication core
│   ├── trajectory_kernels/        stage 2: influence-kernel tool
│   └── hypothesis_tests/          stage 3: hypothesis battery
├── notebooks/                     thin drivers, one deliverable each (01–08)
├── tests/                         pytest; analytic answers; ~170 tests
└── results/
    ├── replication/               stage-1 outputs (+ figures/)
    ├── trajectory_kernels/        stage-2 kernels (.nc), notes (+ figures/)
    └── hypothesis_tests/          stage-3 battery: RESULTS_{run}.md,
        ├── figures/               battery-level forest + curve figures
        └── <HYP_ID>/              per-hypothesis: results.csv, topline.md,
                                   strata.png (season/region/time), event_rate.png
```

---

## Stage 1 — `convection_skill` (the statistical core)

**Flow:** `data_loading` (3 explicit stages: `load_raw` → `make_uniform` →
`finalize_table`) → `quality_control` → `gini` → `analysis`/`significance` →
notebooks 01–05 reproduce the paper's Fig. 2/3 and the SMAP first look.

**Techniques:**
- **Gini / Lorenz-area skill.** Sort samples by predictor, accumulate the
  fraction of events captured, Gini = 2 × area between the capture curve and the
  diagonal (`gini.py`; equals `(2·AUC−1)(1−f)`, verified against the ROC
  identity in tests). Predictor-agnostic: any column + event flags.
- **QPE reconstruction.** The files' `MRMS_*_av` is a *precipitating-area
  conditional mean*; the paper's cell mean is `_av·_cnt/81`. Established by
  matching the paper's threshold ladder (see `config.py` QPE note); this is the
  single most important data correction in the repo.
- **Threshold construction.** Event flags are exceedances of pooled-sample
  percentiles (P95–P99.95), thresholds computed from the base sample.

**Regenerate:** `PYTHONPATH=src python notebooks/0{1..5}_*.py` →
`results/replication/`.

---

## Stage 2 — `trajectory_kernels` (from trajectories to footprints)

**Flow:** `trajectories` (ingest granule files → tidy `(parcel, step)`) →
`pbl` + `contact` (is the parcel surface-coupled?) → `resample` + `fuzz`
(sub-hourly path, uncertainty blur) → `footprint` (the kernel builder) → `io`
(NetCDF) → `apply` (kernel × surface field → predictor). Notebook 06 is the
end-to-end demo; 07 the scientific exploration.

**The organizing physics fact:** in these files parcel humidity `q` is a
*conserved tracer* (only reduced by condensation, logged in `q_excess`), so
soil-moisture influence CANNOT be read off the trajectories themselves. The tool
therefore builds a purely **geometric** footprint, and all soil-moisture physics
enters by convolving it with an external field — which is what makes it
predictor-agnostic.

**Techniques (each a pluggable interface, not a flag):**
- **STILT-style residence-time footprint** (Lin 2003; Fasoli 2018): for a
  receptor (cell, arrival time), each arriving parcel deposits its land- and
  PBL-contact-weighted residence time at its upstream positions, binned by lag.
  Normalized kernel sums to 1 = "fraction of the arriving air's land contact
  from here/then".
- **PBL gate** (`PBLModel`): contact only below `f_c·PBLH` (0.5 STILT / 1.5
  Sodemann presets, default 1.0) with a smooth taper. `ClimatologicalPBL` has a
  diurnal cycle whose evening collapse shuts off late-lag coupling — verified
  against the parcels themselves (notebook 07, Q2).
- **Uncertainty fuzzing** (`FuzzKernel`): each deposit is a mass-conserving 2-D
  Gaussian with σ = σ₀ + α·(distance travelled back from the receptor).
  `StohlFuzz` uses α = 0.2 (Stohl 1998); `EmpiricalFuzz.from_fullgrid()` measures
  α from the day's own sub-grid wind spread (0.191 on 2019-06-05 — independent
  confirmation of the default).
- **Rain-out discount** (`discount.py`, optional): Sodemann-style proportional
  discounting, exact here because losses-only: `w = q(arrival)/q(t)`.
- **Application** (`apply.py`): `influence(x_r,t_r) = Σ K·S` for any surface
  field S; NaN-gapped fields (SMAP is ~46% NaN) are handled by renormalizing
  over the retained kernel weight, with a `min_coverage` guard.

**Verification style:** analytic single-parcel tests (straight trajectory →
known footprint line; above-PBL → zero; uniform field → constant) plus
real-data physics checks (footprint centroid must lie upwind of the fullgrid
low-level wind).

**Regenerate:** `PYTHONPATH=src python notebooks/06_trajectory_kernels_demo.py`
(and `07_kernel_exploration.py`) → `results/trajectory_kernels/`.

---

## Stage 3 — `hypothesis_tests` (the unified battery)

**Flow:** `table` (one tidy row table: targets + slot predictors + daily SM
predictors) → `experiments` (declarative `HypothesisSpec` registry + one runner)
→ `stats` (all inference) → `report` (battery report + per-hypothesis folders).
Notebook 08 is the driver.

**Predictor engineering (`predictors.py` + `table.py`):**
- **Seasonality removal:** per-cell day-of-year climatology via 2-harmonic
  least-squares fit (annual + semiannual), anomalies z-scored per cell
  (Guillod/GLACE convention). Season additionally kept as a stratum.
- **Timing discipline (Tuttle & Salvucci endogeneity guard):** SM predictors
  use only the pre-window L4 slots (16:30/19:30 UTC). The 25.5/28.5 "hours" are
  01:30/04:30 UTC the NEXT day — after the QPE window — so same-day daily means
  would leak post-rain soil moisture into the predictor.
- **Antecedent structure:** lagged anomalies (1–30 d ladder) and the two
  precipitation controls — prior-day `pflux_ante(1–5 d)` and same-day
  `pflux_prewindow`. The latter was decisive: it collapses the naive same-day
  wet-soil signal (+0.31 → −0.04), exposing same-synoptic-system morning rain
  as the confounder.
- **Guillod local/nonlocal split:** local = cell anomaly − 3×3-neighborhood
  mean (spatial signal); nonlocal = neighborhood mean (temporal signal).

**Inference (`stats.py`) — the autocorrelation machinery:**
- **Day/moving-block bootstrap:** whole days are the exchangeable unit
  (preserves within-day spatial correlation); days are resampled in circular
  blocks of 7 (≈ 2× the measured SM-anomaly decorrelation). *Computational
  core:* rows are sorted by predictor once and aggregated into (rank-bin × day)
  count/event matrices, so all 500 replicates are matrix products (~35× faster
  than per-rep passes). Calibration-verified against true sampling SD on
  synthetic data. On real data, honest SEs are **2–3× (max 16×)** the naive
  iid-row bootstrap — the paper-style bootstrap is materially overconfident
  here.
- **Conditional Gini:** the predictor's Gini within equal-count bins of a
  control, event-weighted across bins — the rank-based "skill beyond the
  baseline" (beyond CAPE, beyond antecedent precip). Collinear predictor → ~0;
  independent signal → keeps its marginal skill.
- **Event-rate curves:** P(event) per predictor-quantile bin with block-
  bootstrap bands, for explicitly non-monotonic hypotheses (A2's CIN response —
  peak at weak-but-nonzero cap — is invisible to any rank statistic).
- **Convective-event toggle:** `convective_min` / `convective_col` on
  `build_flags` / `run_battery` / `build_onset_table` restrict precipitation
  events to convective cells, via MRMS PrecipFlag `convective_share`
  (convective / raining sub-pixels). The MRMS Convection flag marks strict
  cores — P99.9 events average only ~0.22 share — so calibrate thresholds
  against `figures/convective_frac_vs_qpe.png` (default 0.2 keeps ~55% of
  heavy events). Notebook-08 CLI: add `convective`.
- **Field significance:** Benjamini–Hochberg FDR at α = 0.10 across all
  (hypothesis × scope) tests (Wilks 2016's recommendation, with the 2α rule for
  spatial correlation).
- Strata are **pre-registered** in `config.py` (humidity/aridity terciles,
  east–west at 95°W, wind terciles, season, early/late window) — the
  forking-paths guard. Every runnable hypothesis additionally gets the
  season/region/time-of-window breakdown by default.

**Outputs:** `results/hypothesis_tests/RESULTS_{pooled,paper}.md` (verdicts,
skill increments, strata, corrections, kernel-extension map) and one folder per
hypothesis (`<HYP_ID>/`) with `results.csv`, `topline.md`, `strata.png`,
`event_rate.png`.

**Regenerate everything:** `PYTHONPATH=src python notebooks/08_hypothesis_tests.py`
(runs pooled 2016–2021 + paper-years 2019–2020, writes every folder and figure).
The expensive row table is parquet-cached in `results/hypothesis_tests/cache/`
(keyed by year set + `TABLE_VERSION` in `table.py`; bump the version or pass
`use_cache=False` after schema changes) — reruns skip straight to the statistics.

---

## Cross-cutting data facts (the ones that bite)

1. `MRMS_*_av` is conditional; cell mean = `_av·_cnt/81`; `_cnt` is
   area-weighted (non-integer); `_max` = sub-pixel max; `_sk` = sub-pixel
   skewness (0 for ~77% of rows).
2. All non-L4 `SMAP_*` variables are empty in every year; use `SMAP_L4_*`.
3. `FCST_N` is identically zero (parcel-count leftover; NOT Brunt–Väisälä).
4. Trajectory `q` is a conserved tracer; `q_excess` = condensate above
   saturation (≥0, sparse).
5. `SMAP_L4_ulay1_av` is a wind-speed magnitude, not a u-component.
6. Data cover Mar–Nov only, with day-level gaps (2016 shortest at 243 days);
   slots: 0 = AIRS overpass (~17–21 UTC), 1–6 = 21…02 UTC, constant across years.
7. L4 observation "hours" 25.5/28.5 are next-day 01:30/04:30 UTC.

## Testing philosophy

Every statistical/physical function has tests with *analytically known answers*
(synthetic data where the right answer is derivable), not golden files: Gini of
a perfect predictor = 1−f; block bootstrap calibrated against true sampling SD;
deseasonalization removes a planted sinusoid; footprint total = in-contact
hours; fuzz deposit conserves mass exactly. Real-data tests are structural
(round-trips, invariants like "footprint lies upwind") and skip when data is
absent. Run: `python -m pytest tests/` (~170 tests).
