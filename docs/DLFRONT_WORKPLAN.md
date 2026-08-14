# Work Plan: DL-FRONT Replication + Dryline Extension

**Goal.** Replicate Biard & Kunkel (2019, *ASCMO* 5, 147-160 — the DL-FRONT
2-D CNN; `docs/papers/Biard_Kunkel_2019.pdf`) on MERRA-2 surface fields vs
CODSUS analyst fronts, extend it to a 6th **dryline** class using the
NOAA-XML label conversion, and run both inside the same stage A/B/C
reanalysis → degraded → real-AIRS framework as the UNET3+/FrontFinder track
(`docs/FRONT_DETECTION_WORKPLAN.md`), ending with fine-tuning on real AIRS
overpass/forecast data on the JPL laptop.

Package: `src/dl_front/` (sibling of `src/front_detection/`, which it
imports for labels, evaluation, degradation noise, and the gap-mask bank).

## 1. The paper's recipe (all constants cited in `dl_front/config.py`)

| Item | Paper value |
|---|---|
| Inputs | MERRA-2 M2I1NXASM `T2M, QV2M, SLP, U10M, V10M`, 3-hourly, bicubic → 1° over 10-77° N, 171-31° W |
| Labels | CSB fronts rasterized **3 cells wide** (= the published CODSUS `3wide` files), one-hot cold/warm/stationary/occluded/none |
| Region mask | Fig. 2 envelope (>40 crossings/yr) = `codsus_merra2-1deg_mask.nc`, applied to loss AND metrics |
| Network | 3× [zero-pad → 5×5 conv, 80 filters → ReLU → 50% spatial dropout] + [zero-pad → 5×5 conv, n_cls → softmax]; ~331k params |
| Loss | weighted categorical cross-entropy, none = 0.35, others 1.0 |
| Optimizer | Adam, LR 1e-4; batch size unstated (we use 32) |
| Training | 2003-2007 (14,353 pairs), 3-fold CV (random 2/3 each), stop when loss stalls 100 epochs (~1140 epochs); best fold kept |
| Validation | 2008-2015; accuracy ~88% (all-cat), ~90% (front/no-front), AUC 0.90; Tables 1-4 |

Inferred where the paper is silent (documented in config): per-variable
standardization frozen from train years; val_loss as the early-stopping
signal.

## 2. Deviations & extensions

- **Drylines (6-class):** labels =
  `data/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/`
  (NOAA XML analyses, 2006-2022, dryline channel, already one-type-per-cell).
  Train 2006-2012, eval 2013-2015; **2016-2018 stay embargoed** for stage C.
  Dryline weight 1.0 initially (paper found inverse-frequency weighting
  *worse*); revisit if dryline recall is poor.
- **Overlap resolution:** published CODSUS per-type channels can overlap at
  crossings; resolved warm > occluded > stationary > cold (the renderer's
  painter priority, reverse-engineered in `src/codsus_regen.py`).
- **Skill at explicit km:** beyond the paper's cell-fraction/confusion/ROC
  metrics, line-vs-line CSI/POD/FAR via `front_detection.evaluate`
  (symmetric neighborhood, cos-lat weighting) so DL-FRONT and the UNET3+
  track share one skill currency.

## 3. Stage curriculum (mirrors front_detection workplan 3.6)

**Revised 2026-08-12:** stages B and C now run on KRIGED gap-filled caches
(`dl_front/krige_fill.py` ->
`data/front_id/{degraded_reanalysis,kriged_airs_fcst}/`, manifest reorg
2026-08-13) instead of the
on-the-fly `--degraded` noise path (which remains available). Full
JPL-laptop runbook: `docs/JPL_DEPLOYMENT_DLFRONT.md` (chain driver
`scripts/dlfront_jpl_chain.sh`, sbatch files under `slurm/`).

| Stage | Data | Command |
|---|---|---|
| A | clean MERRA-2 sfc | `python -m dl_front.train --name D6A-f0 --classes 6` (and `--classes 5`) |
| B | kriged AIRS-mask degradation: reanalysis T2M/QV2M NaN'd where AIRS unobserved (real fullgrid swath, else seasonal gap-bank draw), refilled by ordinary kriging; SLP/winds clean (`python -m dl_front.krige_fill build-degraded --years 2007-2015`) | `--source kriged-degraded --retrain <A ckpt>` (DEGRADED_LR default) |
| C | kriged real AIRS-FCST surface fields (fullgrid slot 0 + hourly forecasts, hours 18/21/00 UTC; SLP copied clean from reanalysis; `python -m dl_front.krige_fill build-airs --years 2007-2021`) | `--source kriged-airs --retrain <B ckpt>` (FINETUNE_LR / FINETUNE_PATIENCE defaults) |

Held-out scoring: `python -m dl_front.evaluate_test --ckpt <h5> --classes 6
--source {reanalysis,kriged-airs} --years 2016-2021` (both sources default to
the same AIRS-hours filter, so scores are directly comparable).

## 4. Open items (dated 2026-08-09)

- Stage C ingest: map AIRS near-surface retrievals (or HYSPLIT fullgrid
  surface slots) + forecast SLP/U10/V10 into the `sfc_daily` schema; add an
  input-validity channel (`build(n_channels=6)`) and swath-mask loss weights
  (`make_tf_dataset(weights=...)` already accepts per-sample masks).
- Polyline extraction (paper section 3.2 ridgeline/MCP) not replicated —
  crossing-rate climatologies out of scope; gridded metrics don't need it.
- Gap-mask bank currently has 1 real day; grows with Zach's fullgrid pull.

## 5. Run log

- 2026-08-09: package + 24 tests green (analytic arch/loss checks).
  M2I1NXASM download 2003-2015 launched (`results/dl_front/download_sfc.log`).
