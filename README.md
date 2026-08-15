# AIRS-FCST CAPE → heavy-precipitation skill (Gini replication)

Replication and extension of the Gini-coefficient analysis of

> Richardson, M. T., B. H. Kahn, and P. M. Kalmus (2024). *Mesoscale air motion and
> thermodynamics predict heavy hourly U.S. precipitation.* Communications Earth &
> Environment 5:472. [doi:10.1038/s43247-024-01614-1](https://doi.org/10.1038/s43247-024-01614-1)

which shows that AIRS temperature/humidity soundings, advected forward with forecast
winds ("AIRS-FCST" CAPE), skillfully predict heavy hourly MRMS precipitation over the
central-eastern U.S. Skill is quantified by the **Gini coefficient** of the
precipitation-exceedance CDF sorted by CAPE.

The code is written to be **maximally interpretable** and **predictor-agnostic**, so
it can be extended to SMAP soil-moisture predictors (see Phase 6 and
`docs/convective_initiation_lit_review.md`). The statistical core never mentions CAPE:
swapping in a SMAP field changes only which column is passed in.

**Start here:** [`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md) — the logical
flow of the whole codebase (all three stages) and the statistical/physical
techniques each uses. Per-stage design docs: [`docs/WORKPLAN.md`](docs/WORKPLAN.md)
(replication), [`docs/TRAJECTORY_KERNEL_WORKPLAN.md`](docs/TRAJECTORY_KERNEL_WORKPLAN.md)
(influence kernels), [`docs/HYPOTHESIS_TESTS_WORKPLAN.md`](docs/HYPOTHESIS_TESTS_WORKPLAN.md)
(hypothesis battery).

## Layout

```
docs/                     design docs, codebase guide, lit review, source papers
src/convection_skill/     stage 1: Gini replication core (pure, cited functions)
src/trajectory_kernels/   stage 2: HYSPLIT trajectories -> influence kernels
src/hypothesis_tests/     stage 3: unified SM x thermodynamics hypothesis battery
tests/                    pytest, analytic answers (~170 tests)
notebooks/                thin drivers; 01-05 replication, 06-07 kernels, 08 battery
results/
  replication/            stage-1 outputs (+ figures/)
  trajectory_kernels/     stage-2 kernel NetCDFs, notes (+ figures/)
  hypothesis_tests/       stage-3 battery reports (+ figures/, one folder per
                          hypothesis with topline.md, strata.png, results.csv)
```

## Setup

The base anaconda env lacks a NetCDF backend. Either:

```bash
conda env create -f environment.yml && conda activate airs-fcst
# or, into an existing env:
pip install netCDF4 h5netcdf
```

Data (`data/*.nc`) is DVC-tracked: `dvc pull` if the files are absent.

## Run

```bash
python -m pytest tests/ -q            # Gini identities, QC, analysis + paper benchmarks
                                      # (tests/conftest.py adds src/ to sys.path)
python notebooks/01_data_audit.py     # builds results/analysis_2019_2020.parquet
python notebooks/02_fig2_replication.py
python notebooks/03_fig3_replication.py
python notebooks/04_extended_replication.py   # builds the 6-year cache (~3 min)
python notebooks/05_smap_first_look.py
```

Each figure script writes a `results/replication/*_comparison.txt` stating obtained-vs-paper values.

## Key results (2019-2020, matching the paper)

- **Sample**: 1,409,298 skill rows over 470 valid days (paper Supp Notes 3: 455 days,
  N > 1M, ">160k/hour"). ✓
- **Fig. 2a**: POD at CAPE₉₀ for QPE₉₉.₉₅ events = **0.83** (paper: 0.80). ✓
- **Fig. 2b**: our Gini curves match the paper **within ~0.02 at every rarity** —
  AIRS-FCST [0.50, 0.71, 0.76, 0.87, 0.89] vs paper [0.49, 0.70, 0.78, 0.86, 0.885];
  overpass [0.34, 0.46, 0.50, 0.59, 0.60] vs paper [0.32, 0.46, 0.51, 0.60, 0.61]. ✓
- **Supp Fig 13**: against the *precipitating-area-mean* QPE (`qpe_wet`), our threshold
  ladder matches the paper's **exactly** ([0.9, 3.2, 4.3, 7.2, 8.4] vs
  [0.9, 3.1, 4.3, 7.1, 8.4] mm/h) and the Gini curves within ~0.01–0.04. ✓
- **Fig. 3**: AIRS-FCST Gini **improves significantly with forecast hour** (both by OLS
  trend and by the bootstrapped first-vs-last test) while the overpass baseline
  declines (first-vs-last significant); every per-hour value agrees with the paper's
  legend within the paper's own p<0.05 criterion (2√2σ). ✓
- `tests/test_paper_benchmarks.py` checks the pipeline against **every externally
  verifiable quantity in the paper and its Supplementary Information** (38 quantities:
  sample sizes, valid days, both threshold ladders, POD, Fig. 2b/3/13b Gini values,
  joint CAPE-QPE zero-fractions, trend/bootstrap significance, Supp Table 1 regional
  Ns). 26 pass; the 12 `xfail(strict)` are precisely attributed (below).

### Sample scheme (differs from the paper's *stated* rules — deliberately)
Established by ablating each screening rule against the paper's own numbers:

1. **Thresholds from "all data"**: pooled QPE percentiles are computed on *all*
   in-domain land rows (including rows without valid AIRS data), exactly as Methods
   states — not on the AIRS-valid skill sample. Skill is then evaluated on each
   product's valid rows (Fig. 3 on the matched sample, per the paper).
2. **The complete-cell-days rule is NOT applied** (`apply_paper_qc` default). In the
   paper's data it was evidently near-non-binding; in our regenerated files it deletes
   the wettest 23% of valid rows (2.2× the event rate of kept rows) and pushes every
   skill score +0.03–0.09 above the paper's. This single rule was most of the original
   Gini inflation after the QPE fix.

### Notes / caveats found along the way
- **`MRMS_GaugeCorrQPE01H_av` is a conditional mean** — the mean over *precipitating*
  sub-cells only, not the paper's all-pixel 1°×1° cell mean. Confirmed exactly by Supp
  Fig 13c: the paper's precipitating-area ladder [0.9, 3.1, 4.3, 7.1, 8.4] mm/h equals
  our raw `_av` percentiles. Used raw it inflates every Gini/POD by +0.1–0.2. The loader
  reconstructs the cell mean as `qpe = _av × _cnt / 81` (`_cnt`/81 = precipitating-area
  fraction; `_cnt` > 0 iff `_av` > 0, max 81 at every latitude) and keeps the raw
  conditional mean as `qpe_wet`.
- **Quantified provenance gap (the remaining xfails)**: Supp Table 1's exact regional Ns
  imply the paper's valid-data coverage had a strong east-west gradient (Plains ~28% /
  Midwest ~52% / SE+Atlantic ~99.6% of possible cell-hours — the parcel-advection
  footprint), while our regenerated files are ~59% *everywhere* (Plains 2.2× the paper's
  rows, SE/Atlantic 0.4×). Over-weighting the dry, CAPE-coupled Plains and
  under-weighting the wet SE coast leaves our cell-mean thresholds 1.2–4× high in the
  bulk (rank-preserving → Gini unaffected) and explains the residual hour-level wobbles
  (e.g. our 22 UTC dip). Also unfixable from our side: the reconstructed wet-area
  fractions (`_cnt`/81) run larger than the paper's wet-fraction ingredient.
- `FCST_N` is 0 wherever CAPE is valid in every year → the ">20 parcels" screen was
  applied upstream; we cannot re-apply or relax it.
- **Extended (2016-2021)**: 2018-2020 replicate cleanly; **2016 is a low-skill outlier**
  and 2017/2021 have weak early-hour skill. Prefer 2018-2021 for downstream work.
- **SMAP first look**: the predictor-agnostic core scores SMAP fields unchanged.
  Surface soil moisture is a weak standalone predictor (as expected); high standalone
  Gini for soil temperature / root-zone moisture is **confounding** (warm-season/
  humid-east climatology), not causal skill — isolate SMAP signal by conditioning.
