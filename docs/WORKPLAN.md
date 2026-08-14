# Work Plan: Replicating the Gini-Coefficient Analysis of Richardson et al. (2024)

**Goal.** Reproduce the core statistical result of Richardson, Kahn & Kalmus (2024, *Comms Earth & Environ* 5:472): the Gini coefficient quantifying how well AIRS-FCST CAPE predicts heavy hourly precipitation (MRMS QPE exceedances) over the central-eastern CONUS — specifically the analyses behind their Fig. 2 (Gini derivation and percentile sweep) and Fig. 3a (Gini vs. forecast hour, with significance).

**Design constraint (non-negotiable).** All code must be **maximally interpretable** — it is the foundation for extending this analysis to SMAP soil-moisture predictors. Concretely:

1. **Predictor-agnostic core.** The Gini machinery must accept *any* 1-D predictor array + event flags, never "CAPE" specifically. Swapping in `SMAP_smsfc_av` must require zero changes to the statistics code.
2. **One function = one paper sentence.** Each step of the paper's Methods maps to one small, pure function whose docstring quotes/cites the exact passage it implements (e.g. *"Methods → 'Gini and significance calculations', ¶1"*).
3. **All constants in one config file**, each with a comment citing its source in the paper (domain bounds, QC rules, percentile lists, bin count, bootstrap reps).
4. **No hidden state.** The data layer produces one tidy analysis table (rows = day × hour × grid cell; columns = predictors + QPE + masks); everything downstream consumes that table.
5. **Unit tests with analytically known answers** for the Gini function (uniform → 0, perfectly sorted → →1, half-sorted → known value).
6. **Notebooks are thin.** Each notebook reproduces one figure/claim by calling library functions; no analysis logic lives in notebooks.

---

## 1. The method being replicated (distilled from the paper)

Paper framing: AIRS temperature/humidity profiles, advected forward with HYSPLIT + WRF27km winds ("trajectory enhancement"), yield hourly 1°×1° CAPE fields (AIRS-FCST). Skill is measured as the nonuniformity of P(conv | q,T) via the Gini coefficient, with MRMS gauge-corrected hourly QPE exceedances as the convection proxy.

**Gini recipe (Methods → "Gini and significance calculations"):**

1. Choose a QPE percentile threshold X (they sweep X = 95 → 99.95; QPE₉₉.₉₅ ≈ 5.1 mm h⁻¹ in their sample). Thresholds are computed from the **entire pooled sample** — all locations, seasons, hours, wet *and* dry.
2. Flag each sample 1 if QPE > QPE_X, else 0.
3. Sort flags by the predictor value. Zero-valued predictors get a tiny random perturbation (~±1×10⁻¹⁰) so sorting is unambiguous (this makes the CDF linear over the zero-CAPE range).
4. Compute the normalized cumulative distribution of events across **100 equal-count predictor bins** (each bin = 1% of samples).
5. Gini = 2 × area between that event-capture curve and the 1:1 diagonal. (Equivalently Gini = 2·AUC − 1 of the ROC; the paper's refs 76–77 give the relationship. Implement the binned-CDF version to match the paper exactly; verify against the ROC identity as a cross-check.)
   - Gini = 0 → uninformative predictor; Gini → 1 → all events at highest predictor values.

**Significance (for the Fig. 3 hour-by-hour claims):**

- *Bootstrap SE:* pool the predictor samples from all six forecast hours; resample with replacement down to a one-hour sample size; compute Gini; repeat 500×; SD of the 500 Ginis = 1σ. Hours treated as independent → SE of an hour-to-hour difference = √2·σ; p < 0.05 threshold ≈ 2√2·σ (≈ ±0.08 in their sample).
- *Trend test:* OLS slope of Gini vs. forecast hour within a product; |slope| > 2×SE(slope) → significant at p < 0.05.
- Expected qualitative result: AIRS-FCST Gini **improves significantly with forecast hour**; overpass-time AIRS CAPE degrades.

**Sample-selection rules (Methods → "Calculation of thermodynamic indices and selection of data for analysis"):**

- Predictor = **MU_CAPE only** (their sensitivity tests found it most consistently predictive; MML/SFC parcels and CIN/EL/LCL/LFC added ~no skill).
- Analysis hours = the **six forecast hours 21–02 UTC** (not the overpass snapshot).
- Grid cells kept for a day only if valid at **all** timesteps (so the geographic sample is identical for every forecast hour): (i) > 20 AIRS-FCST parcels in the profile, and (ii) valid MU_CAPE, MU_EL, MU_LCL, MU_CIN (zero counts as valid).
- Domain: **32–53°N, 107–64°W** (data south of 32°N excluded for poor performance), **land fraction ≥ 50%**.
- Season: March–November. Paper years: **2019–2020**. Their pooled N > 1 million (>160k per forecast hour).

## 2. Mapping the paper onto our data

Local data (DVC-tracked): `data/FCST_SMAP_MRMS_{2016..2021}.nc` (six years) + `data/lsm.nc`.

Each yearly file: dims `(date: 366, nhours: 7, lat: 28, lon: 43)`; grid 25.5–52.5°N × 106.5–64.5°W at 1°; valid days ≈ day-of-year 60–334 (March–November).

| Paper quantity | Local variable | Notes |
|---|---|---|
| AIRS-FCST MU CAPE (predictor) | `FCST_MU_CAPE` | J/kg. `FCST_MML_*` also present for the sensitivity check. |
| QC companions | `FCST_MU_CIN`, `FCST_MU_EL`, `FCST_MU_LCL` | required-valid per paper QC |
| Parcel count per profile | `FCST_N` | ⚠ appears to be all zeros in 2016 where CAPE is valid — see Open Question Q2 |
| MRMS hourly QPE (target) | `MRMS_GaugeCorrQPE01H_av` | 1° grid-cell **mean**, matching the paper's "1° × 1° grid-cell mean QPE" |
| Forecast hours 21–02 UTC | `nhours` slots **1–6** | verified via `FCST_parceltime`: slot 0 = AIRS overpass (~17:45 UTC snapshot ⇒ the paper's "AIRS" proximity-sounding baseline), slots 1–6 = 21,22,23,00,01,02 UTC |
| Land fraction ≥ 50% | `data/lsm.nc` (`lsm`, 1° global) | subset to our grid; threshold at 0.5 |
| 32–53°N restriction | slice `lat ≥ 32` | our grid starts at 25.5°N, so this cut must be applied |
| SMAP extension fields | `SMAP_*` (overpass-relative hours) and `SMAP_L4_*` (fixed hours 16.5–28.5 UTC) | soil moisture av/sd, W-E & S-N & abs gradients, layer-1 T/q/u, precip flux — the predictor menu for Phase 6 |
| Precip-type context | `MRMS_PrecipFlag_cnt` | 12 flags incl. `Convection`, `Hail` — useful later for verifying "heavy QPE ≈ convection" |

**Not in our files:** ERA5 CAPE/QPE, HRRR QPE, and a separate AIRS-overpass CAPE variable (though `nhours` slot 0 *is* overpass-time CAPE, replicated per the paper's method). So the full Fig. 2b/3 multi-product comparison is out of scope; the replication targets the **AIRS-FCST rows** of those figures. ERA5/HRRR comparators could be added later from the paper's JPL Open Repository dataset if desired.

**Known differences from the paper (expect small numerical departures):**

- We have **2016–2021** (~3× the paper's sample). For validation, first run on the **2019–2020 subset** to match the paper; then re-run on all six years as the "extended replication."
- September 2021 onward is affected by the AIRS deep-space-maneuver retrieval issue the paper cites — decide whether to truncate 2021 (recommend: run 2021 both ways and compare).
- Our files come from the FCST_SMAP_MRMS pipeline cut, which may embed slightly different upstream QC than the paper's exact dataset ("clean data" per project notes).

## 3. Open questions to resolve in Phase 1 (data audit)

- **Q1 — hour-slot semantics:** confirm slot 0 = overpass for all years, and that slots 1–6 are always 21–02 UTC (check a sample of days per year via `FCST_parceltime`).
- **Q2 — `FCST_N`:** in 2016 it is 0 wherever CAPE is valid, so the paper's ">20 parcels" screen can't be applied as-is. Determine what `FCST_N` encodes (ask data provider / check other years). Working assumption: the parcel-count QC was already applied upstream in this "clean" cut; document whichever conclusion holds.
- **Q3 — threshold sanity check:** does our pooled 2019–2020 QPE₉₉.₉₅ land near the paper's 5.1 mm h⁻¹? A large discrepancy signals a units or sampling mismatch.
- **Q4 — sample size:** paper reports >160k per forecast hour for 2019–2020 after QC; check ours matches to within ~10%.
- **Q5 — 2016/2020 leap-day handling** (366-day files for non-leap years contain a padding day; confirm it is all-NaN and dropped by masking).

## 4. Repository layout & interpretability rules

```
JPL_AIRS/
├── WORKPLAN.md                     # this file
├── environment.yml                 # pinned conda env (netCDF4/xarray missing from base env!)
├── src/convection_skill/           # importable package, pure functions throughout
│   ├── config.py                   # ALL constants, each with a paper citation comment:
│   │                               #   DOMAIN_LAT = (32, 53)   # Methods: "restricted to north of 32N"
│   │                               #   N_BINS = 100            # Methods: "100 equally sized bins"
│   │                               #   PERCENTILES = [95, 99, 99.5, 99.9, 99.95]
│   │                               #   BOOTSTRAP_REPS = 500, TIEBREAK_EPS = 1e-10, ...
│   ├── data_loading.py             # open_year(), build_analysis_table()
│   │                               #   → one tidy DataFrame: [date, hour_utc, lat, lon,
│   │                               #     mu_cape, mu_cin, ..., qpe, smap_*..., land_frac]
│   ├── quality_control.py          # apply_paper_qc(table) — each rule its own named function
│   ├── gini.py                     # PREDICTOR-AGNOSTIC core:
│   │                               #   exceedance_flags(values, percentile) -> bool[]
│   │                               #   detection_cdf(predictor, flags, n_bins) -> (x, cdf)
│   │                               #   gini_from_cdf(x, cdf) -> float
│   │                               #   gini(predictor, flags) -> float   (convenience)
│   ├── significance.py             # bootstrap_gini_se(), hourly_trend_test()
│   └── plotting.py                 # fig2a_style(), fig2b_style(), fig3_style()
├── tests/
│   ├── test_gini.py                # analytic cases; gini == 2*AUC-1 cross-check vs sklearn
│   └── test_quality_control.py     # tiny synthetic table exercising each QC rule
├── notebooks/  (or scripts/ — plain .py with # %% cells also fine)
│   ├── 01_data_audit.ipynb         # Phase 1: answers Q1–Q5, coverage maps, threshold check
│   ├── 02_fig2_replication.ipynb   # CDF + Gini percentile sweep (pooled forecast hours)
│   ├── 03_fig3_replication.ipynb   # per-hour Gini + bootstrap significance + trend
│   └── 04_smap_first_look.ipynb    # Phase 6: same machinery, SMAP predictors
└── results/                        # derived tables (CSV/parquet) — DVC-track if large
```

Code style contract: NumPy/pandas only in the core (no framework magic); type hints + docstrings everywhere; docstrings state *which paper sentence* the function implements and *why* (e.g. why the ±1e-10 perturbation exists); no function longer than ~40 lines; module-level `if __name__ == "__main__"` demos on the core modules so each is runnable/inspectable standalone.

## 5. Phased plan

**Phase 0 — Environment (½ day).**
Create `environment.yml` (python, numpy, pandas, xarray, netcdf4, matplotlib, scipy, pytest, jupyter). Note: the base anaconda env lacks a netCDF backend — `h5py` works but `netcdf4` is the interpretable choice. Verify `xr.open_dataset` on one yearly file. Commit env + skeleton package.

**Phase 1 — Data audit (1 day).** `01_data_audit.ipynb`
Resolve Q1–Q5. Produce: per-year valid-sample counts, map of mean MU_CAPE and QPE, histogram of QPE with percentile thresholds marked, land-mask plot after the ≥50% cut. **Gate: do not proceed until Q1–Q3 are resolved and documented in the notebook.**

**Phase 2 — Core library + tests (1–2 days).**
Implement `gini.py`, `quality_control.py`, `data_loading.py`, `significance.py` per the layout above. Tests must pass, including the Gini-vs-ROC identity check and a shuffle test (shuffled predictor → Gini ≈ 0 ± noise).

**Phase 3 — Fig. 2 replication (1 day).** `02_fig2_replication.ipynb`, 2019–2020 subset, forecast hours pooled.
- 2a: event-capture CDF for QPE₉₉.₉₅ vs. MU_CAPE percentile, Gini annotated.
- 2b: Gini vs. threshold percentile (95 → 99.95) for AIRS-FCST CAPE (and overpass-CAPE from slot 0 as the available comparator).
- 2c: CDFs for several QPE_X on one axes.
**Acceptance:** curve shapes match the paper (linear segment over zero-CAPE range; ~80% of QPE₉₉.₉₅ events captured above CAPE₉₀); Gini values within ~±0.05 of the paper's AIRS-FCST values (bootstrap SE ~0.01, plus dataset-cut differences).

**Phase 4 — Fig. 3a replication (1 day).** `03_fig3_replication.ipynb`, 2019–2020 subset.
Per-hour (21–02 UTC) CDFs and Ginis for QPE₉₉.₉₅; bootstrap SEs; hour-difference significance at 2√2σ; OLS trend test. Also run slot-0 overpass CAPE per hour (its Gini should *decline* with hour, mirroring the paper's Fig. 3b).
**Acceptance:** significant *increase* of AIRS-FCST Gini with hour and significant *decline* for overpass CAPE, per the paper's stated result; σ ≈ 0.02–0.04 scale so that ±0.08 is the p<0.05 difference threshold.

**Phase 5 — Extended replication (½ day).**
Re-run Phases 3–4 on all of 2016–2021 (with and without post-Aug-2021). Report how Gini and its uncertainty tighten with 3× sample. This is the new-science baseline the SMAP work builds on.

**Phase 6 — SMAP extension hooks (1 day, gateway to the real project).** `04_smap_first_look.ipynb`
Demonstrate the machinery is truly predictor-agnostic:
- Gini of QPE₉₉.₉ using each `SMAP_*` field alone (expect low — soil moisture is a modulator, not a CAPE substitute).
- **Stratified Gini:** CAPE→QPE Gini computed within SMAP soil-moisture terciles (dry/mid/wet) and, per the lit review's regime-dependence warnings (Tuttle & Salvucci 2016; Guillod et al. 2015), separately for east vs. plains longitude bands. Requires one new helper: `stratified_gini(table, predictor, stratifier, n_strata)`.
- Document the two natural extension paths: (a) conditioning/stratification, (b) bivariate predictors (e.g., CAPE × soil-moisture-gradient score) — the latter needs only a column-combining step before the same `gini()` call.

## 6. Verification & bookkeeping

- Each figure notebook ends with a "Comparison to paper" cell stating target value, obtained value, and pass/fail against the acceptance criteria above.
- Derived thresholds (QPE_X in mm/h per sample definition) written to `results/thresholds.csv` so every downstream analysis uses identical, versioned cut-offs.
- Code in git; `results/` and any regenerated data in DVC, consistent with the existing repo setup.
- Random seeds fixed in `config.py` for the tie-break perturbation and bootstrap (document that results must be insensitive to the seed — the paper states the perturbation does not affect results; verify with two seeds once).

## 7. References

- Richardson, M. T., B. H. Kahn, P. M. Kalmus (2024). *Mesoscale air motion and thermodynamics predict heavy hourly U.S. precipitation.* Comms Earth & Environ 5:472. doi:10.1038/s43247-024-01614-1 (PDF in repo root; Methods pp. 6–7 are the implementation spec).
- `convective_initiation_lit_review.md` (repo root) — mechanism menu and regime-stratification rationale for Phase 6.
