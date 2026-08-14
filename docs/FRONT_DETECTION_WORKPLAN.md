# Work Plan: Front & Dryline Detection from AIRS Data

**Goal.** Train a UNET3+ semantic-segmentation model — the FrontFinder methodology
of Justin, McGovern & Allen (2025, *AIES*, AIES-D-24-0043) — to detect cold, warm,
stationary, and occluded fronts (and, in Phase 2, drylines) **from AIRS sounder
observations**, using the vendored `fronts/` codebase (branch `dev-master`) as a
library. Truth labels are NWS Coded Surface Bulletin analyst fronts
(`data/Fronts/CODSUS_netCDF_MERRA2_2003-2018/CODSUS/MERRA2/`); the authors'
published model output (`data/Fronts/merra2_fronts_merra2-1deg_1wide_1980-2018/`)
is the external benchmark. The object produced is a per-pixel front-type
probability field on the 1° MERRA-2 label grid, from AIRS temperature/humidity
profiles at overpass time.

Strategy in one line: **pretrain on plentiful reanalysis (2003–2015), bridge with
a degraded-reanalysis "AIRS simulator" stage, fine-tune on real AIRS inputs
(2016–2017), test on embargoed 2018** — with retrieval gaps and height-dependent
retrieval accuracy treated as first-class design features, not nuisances.

---

## 0. The findings that shape everything

Three audit facts drive every design choice below:

1. **Winds dominate FrontFinder's skill, and AIRS does not retrieve winds.** The
   paper's permutation studies rank v-wind/u-wind top for cold, warm, and occluded
   fronts. Our inputs carry winds only by grace of the HYSPLIT/WRF processing
   (`fullgrid` files have per-level WRF u/v/w; `nogrid` step-to-step parcel
   displacements give layer winds) — i.e. *model* winds, not observations.
   Decision (2026-08-04, Zach): **allow WRF winds as channels for now**; a
   thermo-only ablation is mandatory because the AIRS-direct path has none, and
   the A-wind − A-thermo pretrain gap quantifies exactly what winds buy.
2. **AIRS accuracy is worst precisely where FrontFinder looks.** The paper's
   levels (2 m/10 m, 1000–850 hPa) sit in the boundary layer, where IR weighting
   functions are broad (~1–2 km effective vertical resolution; T ≈ 1 K/1-km layer,
   q ≈ 15–20 %/2-km layer at best; worse under cloud — AIRS/AMSU validation:
   Susskind et al. 2003, 2014; Divakarla et al. 2006). Mid-troposphere is AIRS's
   strength. The paper's own permutation results soften the blow: warm-front
   signal peaks at 850–900 hPa and dryline signal at 900–850 hPa — *aloft, not at
   the surface*. Hence the upward-shifted level set (§3.2).
3. **Retrieval gaps are large and structured.** On the audit day the five granules
   had 1.2–74 % valid footprints; failures cluster under cloud, which is
   *anticorrelated with clear sky exactly where fronts live*. Gaps therefore enter
   as (a) an input-validity channel the model can exploit, and (b) a bank of
   *real* gap masks applied to reanalysis during pretraining — synthetic speckle
   would miss the cloud–front correlation. Analyst fronts exist under cloud, so
   gap pixels are **not** removed from the loss (§3.4).

A fourth fact reframes the sample-size problem: the "few overlapping years"
constraint (2016–2018) applies to the **HYSPLIT-processed archive**, not the
instrument. AIRS flies since 2002, so the AIRS-L2-direct input path overlaps
CODSUS for ~16 years (2003–2018). Decision (2026-08-04, Zach): support **both**
input paths behind one schema — HYSPLIT files (Zach pulls 2016–2021) and AIRS L2
direct (longer record, cleanliness unknown).

---

## Design constraints (non-negotiable — inherited from the CAPE/kernel work)

1. **Source-agnostic ingest.** The pipeline never cares whether a profile came
   from HYSPLIT nogrid, AIRS L2, or degraded MERRA-2 — every source is reduced to
   one tidy schema (footprint/cell × level: lat, lon, pres, t, q [, u, v],
   valid flag) before anything downstream runs.
2. **One function = one rule, with a citation** in its docstring.
3. **All constants in `src/front_finder/config.py`**, each annotated with its
   source (paper section, AIRS validation paper, or dated decision).
4. **Explicit sequential stages, each usable and inspectable alone**:
   `ingest → derive → regrid → dataset → train → evaluate`. No monolith-with-flags.
5. **Minimal, surgical diffs in `fronts/`** (keep it upstream-diffable): exactly
   one new loss function; everything else consumes `fronts` as a library.
6. **Unit tests with analytic answers** (like `tests/test_gini.py`).
7. **Thin notebooks** — one driver per experiment, logic in the library.
8. **Every skill number carries an explicit km scale.** At 1°, one pixel ≈ 100 km;
   nothing gets reported in "pixels".

---

## 1. The data (confirmed by audit, 2026-08-04)

| Dataset | Grid / geometry | Time | Period | Notes |
|---|---|---|---|---|
| CODSUS labels | 1° MERRA-2, 68×141 (10–77 °N, 171–31 °W) | 3-hourly (missing bulletins omitted) | 2003–2018 | `fronts(time, front=5, lat, lon)` ubyte {0,1,2=fill}; classes cold/warm/stationary/occluded/**none — no dryline**; `_masked` + `mask` variants exist |
| `merra2_fronts` benchmark | identical schema | hourly (complete axis) | 1980–2018 | the authors' model output; inner-join on time vs CODSUS |
| HYSPLIT `nogrid` | native AIRS swath, 45×30 footprints × 57 levels | step 0 = overpass (~18:50/20:35 UTC), steps 1–6 hourly | 2019-06-05 only; 2016–2021 to be pulled | t, q, pres, alt, lat, lon, q_excess; **no u/v/w**; NaN = failed retrieval/below surface |
| HYSPLIT `fullgrid` | 1° (25.5–52.5 °N, 106.5–64.5 °W), 33 30-hPa bins | 7 slots | same | means + std + N, **includes WRF u/v/w**, CAPE family |
| AIRS L2 (to acquire) | native swath granules | 2 overpasses/day | 2002– | product choice (v7 Std/Support vs CLIMCAPS) open |
| MERRA-2 profiles (to acquire) | native 0.5°×0.625° → sample to label grid | 3-hourly | 2003–2015 | pretrain corpus; ERA5 fallback via `fronts/download_era5.py` |

Environment: WSL2, GTX 1070 Max-Q 8 GB (Pascal sm_61), driver 560.81. New conda
env `fronts-tf`: python 3.10 + `tensorflow[and-cuda]==2.15.*` (last Keras-2
default; bundles CUDA 12.2 libs — no system CUDA needed). The vendored
`requirements.txt` is UTF-16 and TF-2.10-era; it is *not* installed.

---

## 2. Package: `src/front_finder/`

| Module | Role |
|---|---|
| `config.py` | flat-YAML config (`configs/front_finder.yaml`) with key ownership à la `convection_skill.config`; level sets, variables, degradation constants (cited), split years, per-stage LRs, class weights, dilation↔km table, paths |
| `ingest_hysplit.py` | wraps `trajectory_kernels.trajectories.load_granule/load_day`; step-0 snapshot + displacement winds; `fullgrid` reader for WRF u/v/w |
| `ingest_airs_l2.py` | AIRS L2 granule reader → the same tidy schema |
| `derive.py` | Td, RH, θe, Tv, r from t/q/pres via `fronts/utils/variables.py`, **at footprint level before regridding** (nonlinear derived vars from cell-mean T,q would be biased) |
| `regrid.py` | footprint → 1° cell binning (or `trajectory_kernels.fuzz.deposit_gaussian`); emits `valid_fraction(level)`, `n_footprints`, `obs_time_offset` per cell |
| `labels.py` | CODSUS loader, 8-connected dilation (`config.LABEL_DILATION`, 0 since 2026-08-10), overpass↔nearest-bulletin pairing (±1.5 h), one-hot + loss-weight channel |
| `degrade.py` | stage-B operators: log-p Gaussian smoothing (FWHM 1.5–2 km) *before* level extraction & derivation; T noise N(0, 1 K); q multiplicative 15–20 % AR(1)-correlated in level; real-gap application |
| `mask_bank.py` | harvest/sample real per-level 1° valid-fraction fields (train years only) |
| `dataset.py` | tf.data pipelines; zero-pad 68×141 → 72×144 (pad weight 0); frozen min-max normalization; weight appended as trailing y_true channel |
| `model.py` | thin `fronts.models.unets.unet_3plus` wrapper from config |
| `train.py` | stage-aware driver (A/B/C); checkpoints SavedModel + `.h5` |
| `evaluate.py` | neighborhood CSI/POD/FAR/FB on 1° with explicit km mapping; swath-mask-paired baseline scoring; isotonic calibration; **day-block bootstrap CIs** (block machinery from `convection_skill`; iid bootstrap understates CIs 2–3× on this kind of data) |
| `permutation.py` | POD-based permutation importance over variable×level, **including the mask channel** (high mask importance ⇒ clouds-mark-fronts shortcut — must be reported, it would inflate apparent skill) |

`fronts/` diffs: **(1)** `models/custom_losses.py` gains
`masked_fractions_skill_score` (weight = trailing y_true channel, applied
post-pooling in the MSE ratio; existing function untouched); **(2)** optionally
the `train_unet.py` `all_years` int fix — we do not otherwise use `train_unet.py`
(the curriculum needs LR/source/loss changes it cannot express).

---

## 3. Core design decisions

### 3.1 Grid: 1° label grid, padded 72×144
Labels, benchmark, `fullgrid`, FCST, and `trajectory_kernels` grids all live at
1°. Native-swath modeling (13.5 km) would discard the entire eval stack and label
pairing for a resolution gain smaller than analyst-placement uncertainty; 0.25°
re-rasterization of CSB XML joins Phase 2. `unet_3plus` with `levels=4` needs
÷8 dims → zero-pad to 72×144, padded pixels weight 0.

### 3.2 Vertical levels: shifted up for AIRS
Default **Set A: 1000, 925, 850, 700, 500 hPa**; paper-faithful
**Set B: 1000, 950, 900, 850, 700** as ablation (formerly E5 — dropped
2026-08-10 along with the U-Net-depth ablation; the architecture was rescaled to
the 1° grid instead). Interpolated from the
57-level support grid (nogrid) or 33-bin fullgrid.

### 3.3 Variables
T, q, r, Td, θe, Tv, RH per level (+ optional u, v block; + per-level
`valid_fraction` as one extra *variable channel* — input lon×lat×5×(V+1), zero
architecture change). Invalid cells imputed to the normalized midpoint **after**
the mask channel is attached, so "imputed" is always distinguishable.

### 3.4 Two masks, never conflated
- **Input-validity** (cloud gaps): input channel only. The loss still scores gap
  pixels — analyst fronts exist under cloud, and we want gap-bridging inference.
  Masking the loss there would teach the model that gaps are consequence-free.
- **Loss mask**: padding, never-observed/off-swath pixels, and (via the CODSUS
  `_masked` variant if adopted) outside-analysis-region pixels.

### 3.5 Architecture (config only, `unets.py` untouched)
`unet_3plus`, `levels=3` (18×36 bottleneck ≈ 445 km/px), `filter_num=[32,64,128]`
(sized for one A100-80GB), `pool/upsample=(2,2,1)`, `kernel_size=3`, GELU, batch
norm, deep supervision, `squeeze_axes=3`, softmax, 5 classes (6 with drylines in
Phase 2). Loss `masked_fractions_skill_score`, `mask_size=(3,3,1)` (1-px ≈ 100 km
neighborhood, stated everywhere); the (3,3) FSS mask is a **uniform boxcar, not
distance-decaying** — kept deliberately. Class weights: **decision (2026-08-10)
— train unweighted**, paper-faithful. Imbalance is milder at 1° than at native
resolution (line-pixel fraction scales with grid spacing); the epoch-41 quicklook
showed no class collapse (max prob 0.92–0.99 for every class); weights would
raise FB. Reopen only if a full sweep shows a class collapsing, then try 3–5× on
that class. `dataset.class_weights()` and the loss's weight hook remain for
diagnostics only (documented in both docstrings).
Implementation note (2026-08-04): the stock loss applies `class_weights`
*pre-sigmoid* (scales the fields before discretization); the masked variant
folds them into the post-pooling weighted mean — deliberate divergence, the
weighted-mean semantics are what a weighted skill score means. Landed with 4
analytic tests (`tests/test_front_losses.py`, incl. w≡1 ≡ stock to 1e-6).

**Label dilation: 0 (decision 2026-08-10).** Training labels are undilated
(`config.LABEL_DILATION = 0`, threaded through dataset/materialize/calibrate;
`make_shard_tf_dataset` refuses shards whose meta dilation disagrees). Rationale:
an undilated 1-px line at 1° is already ~111 km wide — thicker than the paper's
dilated 75 km target — and the FSS pooling already supplies the spatial
tolerance. Dilation-1 training produced FB 2.4–3.5, which dilutes downstream
per-cell Gini discrimination for convective-initiation risk scores. Existing
y shards were rebuilt fast via `materialize --labels-only` (x/times untouched;
meta updated atomically, y tmp renamed before the meta rewrite).

### 3.6 Transfer curriculum
| Stage | Data | Years (train/val) | LR | Notes |
|---|---|---|---|---|
| A | clean MERRA-2 | 2003–2014 / 2015 | 1e-4 | A-wind (ceiling, never fine-tuned) + A-thermo (transfer base); Adam, batch 64 @ lr 1e-4 on one A100-80GB (paper pairing), 10 steps/epoch, FSS, early stopping — paper-faithful |
| B | degraded MERRA-2 | same | 5e-5 | smoothing/noise/gap masks/overpass-hours; severity ramped over the first 41 epochs (= 3 full passes of the ~8,800-sample overpass-hours corpus). Vertical smoothing is per-variable (FWHM: T 1.5 km, q 2.5 km); retrieval noise is horizontally correlated (Gaussian-filtered, e-folding 300 km). Gap fields come from `synth_gaps.py` synthetic swath+cloud heuristics (1650-km swath band, −0.33 dlon/dlat ground track, 400-km cloud correlation, per-level yields 0.55–0.80) until the real bank reaches 30 fields, then the real bank. Validation uses a **frozen** noise/gap realization (identical every epoch). Note (2026-08-04): the corpus stores only the 5 target levels (bandwidth), so vertical smoothing is a 5-level log-p Gaussian **mixing matrix**, not full-profile convolution — still applied to T/q before derived-variable computation |
| C | real AIRS | 2016–2017 / 2017 tail | 1e-5–3e-5 | all layers unfrozen, BN in training mode, patience 15, lon-crop augmentation, no flips; **2018 embargoed test** |

Flip augmentation (paper §2c): 25 % chance per horizontal axis, independently
(`dataset.random_flip`), applied to the stage-A/B **train** streams only — never
validation, never stage C.

Normalization constants computed once (pretrain corpus) and **frozen across all
stages** — recomputing on AIRS would silently change what the imputation value 0
means. 2016–2018 never enter pretraining.

### 3.7 Temporal pairing
Overpass mid-time → nearest 3-hourly bulletin (±1.5 h; fronts ≲30 km/h ⇒ ≤45 km
displacement, inside dilation + neighborhood tolerances). No polyline
interpolation between bulletins.

---

## 4. Experiments (ordered; each gates the next)

| ID | Experiment | Go criterion |
|----|-----------|--------------|
| E0 | Score published `merra2_fronts` vs CODSUS, 2018, 1° — **no training** | ✅ **PASSED 2026-08-04** (criterion revised, see note) |
| E1a/b | Pretrain A-wind / A-thermo | E1a ≈ E0-level skill (loop canary); E1b ≥70 % of E1a cold/warm CSI @100 km (<50 % ⇒ winds become required) |
| E2 | Degraded stage B | ≥60–70 % of E1b on degraded val; degradation vs valid-fraction graceful, no cliff |
| E3 | Fine-tune real AIRS 2016–17, test 2018 | beats E2-zero-shot **and** TFP baseline (`fronts/nfa/methods.py` on AIRS θw@850), block-bootstrap 95 % CI excluding 0 |
| E4 | Ablation: mask channel off | reported regardless |
| E6 | Winds: WRF-fullgrid vs displacement vs thermo-only | wind variant must beat E3 with CI excluding 0 to stay |
| E7 | **Phase 2**: CSB re-rasterized labels (drylines, through 2021) via `fronts/convert_front_xml_to_netcdf.py` | dryline CSI @100 km > TFP-on-moisture baseline |

All baselines scored on identical pixels/times (swath mask applied to everything,
including `merra2_fronts`). A-wind − A-thermo isolates the wind penalty;
A-thermo − E3 isolates the retrieval penalty.

**E0 result (2026-08-04).** Two findings during E0 changed the eval design:

1. The `merra2_fronts` benchmark is **DL-FRONT** (Biard & Kunkel 2019, NCICS —
   the producers of these files), *not* FrontFinder, so the FrontFinder class
   ordering was the wrong go-criterion. Validation rests instead on: per-class
   pixel **rates** agree closely (occluded 27.2 truth vs 27.0 pred px/timestep;
   stationary 103 vs 98; cold 77 vs 91; warm 33 vs 21 — warm genuinely
   underpredicted, FB 0.67), scores rise monotonically with neighborhood, CIs
   bracket points, and any-front POD reaches 0.77 @ ~334 km (CSI 0.51).
2. The upstream FrontFinder contingency convention (TP/FP vs dilated truth, FN
   vs exact truth) assumes thick probability blobs and caps POD near 0.5 for a
   thin binary line offset by 1 px. `evaluate.py` therefore uses **symmetric
   neighborhood matching** (POD on truth pixels vs dilated prediction, FAR on
   predicted pixels vs dilated truth, CSI via the paper's Eq. 4 identity —
   Niebler et al. 2022 convention). Our own model's *probability* outputs can
   be scored both ways; line-vs-line comparisons use symmetric only.
   Per-class CSI @ ~111 km (masked region, 2018): occluded 0.32, stationary
   0.25, cold 0.25, warm 0.20 — the DL-FRONT baseline our AIRS model must
   approach. Tables: `results/front_finder/e0_*.csv`.

---

## 5. Run log

- 2026-08-04: env `fronts-tf` (TF 2.15.1/Keras 2, GPU OK). E0 passed. Masked FSS
  landed (+ tests). `src/front_finder/` package + 41 tests green.
  `filter_num` halved to [16,32,64,128] after OOM at batch 8 on the GTX 1070;
  batch 4 × 160 steps preserves the paper's 640-samples/epoch semantics.
- 2026-08-04: MERRA-2 M2I3NPASM 2003–2015 downloading to `data/MERRA2/daily/`
  (label grid, 5 levels, ~3.8 MB/day). E1 chain armed:
  `results/front_finder/run_e1_chain.sh` waits for the download, verifies
  ≥360 days/yr, computes frozen norm stats (2003–2014), then trains
  `E1b-thermo` and `E1a-wind` (logs: `results/front_finder/e1_chain.log`,
  models: `results/front_finder/models/`).
  Manual equivalents:
  `PYTHONPATH=src <fronts-tf python> -m front_finder.train --name E1b-thermo --no-winds`
  (and `--name E1a-wind --winds`).

- 2026-08-05 (Zach): background jobs killed — downloads are Zach's to run from
  here. Corpus state: `data/MERRA2/daily/{YYYY}/m2_{YYYYMMDD}.nc`; 2009–2015
  ≈complete, 2008 partial (31 d), 2003–2007 empty (resume:
  `python3 -m front_finder.acquire_merra2 2008 2007 ...`). E1 not yet run —
  launch `results/front_finder/run_e1_chain.sh` (or the manual commands
  above) when ready; **recompute norm stats first** (the current
  `data/MERRA2/norm_stats.json` is a 2015-only smoke version).
- 2026-08-05: **stage-C fine-tune pipeline built and validated** on the one
  sample fullgrid file (2019-06-05): `ingest_hysplit.py` (fill/units/vertical
  interp 33→5 levels/half-degree→label-grid embed), `dataset.airs_x/airs_samples/
  make_airs_tf_dataset` (loss weight = label-valid ∩ swath), `mask_bank.py`
  (real-gap harvest for stage B), `train.py --airs-glob ... --retrain <ckpt>`
  (stage-C LR 2e-5, patience 15). Demo `notebooks/10_front_detection_finetune_demo.py`:
  frozen MERRA-2 normalization holds on AIRS values (100% in [-0.25, 1.25]);
  overpass swath = 679 label-grid cells (7.1% of domain) → few loss pixels per
  sample, so the multi-year archive matters; masked-FSS gradient step decreases
  loss. 2019 has no CODSUS labels — demo pairing used 2018-06-05 as an explicit
  placeholder; real E3 waits on 2016–2018 fullgrid files.

- 2026-08-05: **full pipeline code-complete, training-agnostic** (Zach: prep all
  code now, iterate after real training). New: `predict.py` (checkpoint →
  probability netCDF, MERRA-2 years or AIRS files with `observed` swath mask),
  `evaluate.threshold_sweep`/`best_csi`, `calibrate.py` (isotonic + reliability),
  `permutation.py` (POD-based, incl. mask channel), `nfa_baseline.py` (TFP
  1965 baseline; note: vendored locator returns a continuous TFL field —
  sign convention pinned empirically, `np.diff(append=0)` edge artifact
  excluded), stage-B wiring (`train --degraded --retrain <A ckpt>`, severity
  ramp callback, full-severity validation), `dataset.class_weights()`.
  Notebook `11_front_detection_eval_template.py` = the whole post-training
  eval loop, verified end-to-end on the untrained smoke checkpoint (uniform
  probs → the exact degenerate outputs expected). **278 tests green.**
  Stage runbook: A `train --name E1b-thermo --no-winds` → B `train --name E2
  --degraded --retrain <E1 ckpt>` → C `train --name E3 --airs-glob
  'data/HYSPLIT/*/*/fullgrid_*.nc' --retrain <E2 ckpt> --winds`; then
  notebook 11 with real CHECKPOINT/LIMIT=None.
- 2026-08-10: **retool for the 1° grid + A100.** Architecture rescaled: levels
  4→3 (18×36 bottleneck ≈ 445 km/px), kernel 5→3, `filter_num=[32,64,128]`;
  batch 64 × 10 steps @ lr 1e-4 (paper pairing). Labels now dilation 0
  (`materialize --labels-only` rebuilt the y shards). Class weights: documented
  decision to train unweighted. Stage B: severity ramp 41 epochs, per-variable
  vertical FWHM (T 1.5 / q 2.5 km), 300-km-correlated noise, `synth_gaps.py`
  synthetic swath+cloud gaps until the real bank ≥ 30 fields, frozen val
  realization; 25 %-per-axis flips on A/B train streams. `threshold_sweep`
  granularity 0.05→0.01. E5 (level-set/depth ablation) dropped. **Pre-2026-08-10
  checkpoints cannot be `--retrain`'d into the new architecture**; E1b-thermo
  epoch-41 remains the dilation-1 reference/fallback.

## 6. Open items (dated)

- 2026-08-04: night overpasses (~07–09 UTC, IR-only) — start day-only, add as ablation.
- 2026-08-04: AIRS L2 product choice (v7 Std vs Support vs CLIMCAPS) — decide at `ingest_airs_l2.py` time.
- 2026-08-04: if AIRS-direct 2003–2015 is clean, real-AIRS training at scale (16-yr overlap) — revisit after E3.
- 2026-08-04: SMAP L4 `ulay1_av` as low-level wind auxiliary — only if E6 shows winds matter and HYSPLIT coverage is patchy.
