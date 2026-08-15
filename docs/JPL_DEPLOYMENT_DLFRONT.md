# Deploying the dl_front AIRS-FCST curriculum at JPL

Runs the DL-FRONT replication curriculum end-to-end on the JPL **gattaca2**
cluster. Manifest reorg 2026-08-13: ALL data lives on the cluster under
`/gpfs/scratch/smap-convection` (`AIRS_SMAP_Front_data/` = the data root,
with the `AIRS_FCST_1deg/` fullgrid archive as its sibling) -- no copying
from weather2 is needed anymore:

| Phase | What | Hardware |
|---|---|---|
| 0 (opt) | download 2016-2021 MERRA-2 `sfc_daily` (test-period reanalysis) | login node, internet |
| pre-2a (opt) | `--with-swath-bank`: composite the 16-day AIRS swath climatology from 2007-2021 fullgrids (the archive spans 2003-2022 since the manifest reorg 2026-08-13; the default years match the training span) → `masks/swath_bank.npz` (the `gap_type` cloud/out-of-swath splitter; 2a and 3a wait on it) | CPU |
| 1 | `D6A-f<k>`: train on clean reanalysis, folds k=0,1,2 | GPU |
| 2a | build kriged **degraded-reanalysis** cache 2007-2015 (AIRS gap masks + ordinary kriging) — runs ON the cluster as a chain phase, writing `front_id/degraded_reanalysis/` | CPU |
| 2b | `D6B-f<k>`: continue from D6A best on kriged-degraded inputs | GPU |
| 3a | build kriged **AIRS-FCST** cache 2007-2021 (real satellite retrievals) — runs ON the cluster as a chain phase, writing `front_id/kriged_airs_fcst/` | CPU |
| 3b | `D6C-f<k>`: continue from D6B best on kriged-AIRS inputs | GPU |
| 4 | held-out 2016-2018 three-way test eval: every stage checkpoint (D6A/D6B/D6C) scored on both reanalysis and kriged-AIRS inputs (6 runs/fold), plus one checkpoint-free run of the published BK19 predictions (`--source bk19`), plus a final `compare` job → `comparison.csv` | CPU (GPU optional) |

Everything cluster-specific is confined to environment variables and
`slurm/dlfront_*.sbatch`; `scripts/dlfront_jpl_chain.sh` submits (or, without
SLURM, runs) the whole thing.

**Analysis domain (user decision 2026-08-13, cache schema v3):** the 6-class
track's product is the box lat 32-53 N / lon 107-64 W intersected with land
(interpolated land fraction >= 0.5 from `masks/land_surface_mask.nc`) — the
old full codsus region mask was far too large. ALL scoring is restricted to
this analysis domain: stage-B/C training loss, every phase-4 leg (bk19
included), and the krige-validation metrics. Kriging (phases 2a/3a) fills the
box plus a minimal halo of `dataset.halo_px()` = 8 px (the derived network
receptive-field radius — beyond box+halo nothing can influence an in-box
prediction), using ALL real observations that fall in the halo (the fullgrid
archive reaches 25.5 N, so the southern halo has real obs); reanalysis is
never substituted into the halo. Stage A trains on box+halo
(`dataset.crop_domain()`) to harvest nearby front examples. **Rebuild
everything downstream of the domain:** schema-v2 kriged caches (the old
region-mask fills) are refused by the loader — rerun phases 2a/3a with
`--force` (or delete `front_id/degraded_reanalysis/` and
`front_id/kriged_airs_fcst/`). The swath bank
(`masks/swath_bank.npz`) is full-grid and unaffected. Config knobs:
`domain:` section of `configs/dl_front.yaml`.

## 1. Prerequisites

```bash
git clone https://github.com/ziglaser/JPL_AIRS.git && cd JPL_AIRS
conda env create -f slurm/fronts-tf-environment.yml   # includes pykrige
```

`tensorflow[and-cuda]==2.15.1` bundles CUDA 12.2 — no CUDA module load, only
an NVIDIA driver >= 525 on the GPU nodes. Verify:

```bash
conda activate fronts-tf
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For phase 0 (and any reanalysis re-download) you need NASA Earthdata
credentials in `~/.netrc` (mode 600):

```
machine urs.earthdata.nasa.gov login <user> password <pass>
```

and the GES DISC application authorized on your Earthdata account.

## 2. Environment variables

Cluster-first (manifest reorg 2026-08-13, user decision): everything is
already on gattaca2 under `/gpfs/scratch/smap-convection` — set the two data
variables to the canonical roots and go:

```bash
export JPL_AIRS_DATA=/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data
export JPL_AIRS_FCST=/gpfs/scratch/smap-convection/AIRS_FCST_1deg
```

(`JPL_AIRS_FCST` is technically redundant — with `JPL_AIRS_DATA` set, the
code and the chain script auto-resolve the sibling `AIRS_FCST_1deg/`
archive — but being explicit costs nothing.) GPFS is assumed visible from
the compute nodes; verify once with an `ls $JPL_AIRS_DATA` from a compute
node (e.g. `srun ls $JPL_AIRS_DATA`) before submitting the chain.

| Variable | Meaning | Default |
|---|---|---|
| `JPL_AIRS_REPO` | repo checkout (sbatch scripts `cd` here) | `$PWD` when running the chain script |
| `JPL_AIRS_DATA` | data root (`front_id/`, `masks/`, ...); cluster: `/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data` | `<repo>/data` |
| `JPL_AIRS_FCST` | AIRS-FCST fullgrid root (`YYYY/wrf27km_YYYYMMDD/fullgrid_*`, 2003-2022); cluster: `/gpfs/scratch/smap-convection/AIRS_FCST_1deg` | the sibling `$JPL_AIRS_DATA/../AIRS_FCST_1deg` when it exists, else `$JPL_AIRS_DATA/HYSPLIT_demo` |
| `JPL_AIRS_RESULTS` | checkpoints + eval output root | `<repo>/results` |
| `JPL_BK19_DIR` | published BK19 prediction rasters (yearly `1deg_{w}wide/{freq}/merra2_merra2-1deg_*_{year}.nc`, 1980-2018); the default now resolves inside the data root on the cluster too, so override only for an out-of-tree archive | `$JPL_AIRS_DATA/front_id/predicted_fronts/bk19` |
| `CONDA_PREFIX_ROOT` | conda install root | `$HOME/miniconda3` |
| `SBATCH_PARTITION` / `SBATCH_ACCOUNT` / `SBATCH_GRES` | injected at submit time (`sbatch -p/-A/--gres`); leave unset to use cluster defaults | unset |
| `SBATCH_GPU_PARTITION` | partition for the GPU (train) jobs only; CPU jobs (krige/quicklook/eval/compare) keep `SBATCH_PARTITION`/cluster default. gattaca2: set this to `gpu`, leave `SBATCH_PARTITION` unset (default `compute`) | unset |
| `CLASSES` | label classes | `6` |
| `FOLDS` | folds to run | `"0 1 2"` |
| `WARM_START` | existing stage-A `.h5` to `--retrain` phase 1 from | unset |
| `KRIGE_WORKERS` | worker processes for `krige_fill` | `8` |
| `FORCE=1` | rerun phases whose done-marker exists | `0` |
| `DRY_RUN=1` | print every command, execute nothing | `0` |

The kriged/degraded caches are **built on the cluster** — that is exactly
what chain phases 2a and 3a are — into
`$JPL_AIRS_DATA/front_id/degraded_reanalysis/` and
`$JPL_AIRS_DATA/front_id/kriged_airs_fcst/`; there is nothing to copy in
or out.

Sanity check before submitting anything:

```bash
ls $JPL_AIRS_DATA/front_id/reanalysis/MERRA2/sfc_daily
```

## 3. Run

### With SLURM

```bash
cd $JPL_AIRS_REPO
conda activate fronts-tf                         # phase 0 runs python here
export JPL_AIRS_DATA=... JPL_AIRS_FCST=... JPL_AIRS_RESULTS=...
export SBATCH_GPU_PARTITION=<gpu partition>      # GPU jobs only (gattaca2: gpu);
                                                 # CPU jobs use SBATCH_PARTITION
                                                 # or the cluster default
DRY_RUN=1 bash scripts/dlfront_jpl_chain.sh      # inspect the plan first
bash scripts/dlfront_jpl_chain.sh --with-acquire --with-swath-bank  # submit for real
```

`--with-swath-bank` prepends a CPU job (`python -m dl_front.swath build-bank
--years 2007-2021`) that composites the 16-day swath-coverage climatology
into `$JPL_AIRS_DATA/masks/swath_bank.npz`; phases 2a/3a depend on it so
their `gap_type` layer can split missing pixels into cloud vs out-of-swath
using the climatological footprint instead of each day's own envelope. Omit
the flag if the bank already exists (the npz is also the skip marker).

(The chain also self-activates `fronts-tf` via `$CONDA_PREFIX_ROOT` before
any foreground python, so a forgotten activate fails over gracefully; jobs
are submitted with `--export=ALL` so the `JPL_AIRS_*` variables reach them
even where the cluster defaults to `--export=NONE`.)

`--with-acquire` runs phase 0 in the foreground on the submitting node
(compute nodes often have no internet) before submitting the chain; omit it
if 2016-2021 `sfc_daily` already exists. The script prints every job id,
writes a manifest to `results/dl_front/chain_<timestamp>.txt`, and ends with
a `squeue -u $USER` hint. Partition/account/gres are injected at submit time
from `SBATCH_PARTITION`/`SBATCH_ACCOUNT`/`SBATCH_GRES` — the `.sbatch` files
contain no cluster names.

### Without SLURM (plain GPU box)

The chain script detects the absence of `sbatch` and runs the identical
steps sequentially in the foreground. That is days of wall-clock, so detach
it from your terminal:

```bash
cd $JPL_AIRS_REPO
setsid nohup bash scripts/dlfront_jpl_chain.sh > logs/chain.out 2>&1 &
tail -f logs/chain.out   # per-step logs land in logs/<phase-label>.log
```

`setsid` + `nohup` keeps the chain alive after logout; each step also gets
its own log under `logs/`.

### Kriging validation (run during phase 1, read before stage B)

The kriged caches are only as good as the interpolator, so validate it on
days where the truth is known — reanalysis fields degraded with real AIRS
gap masks, reconstructed over the box+halo crop, and scored on the held-out
analysis-domain pixels:

```bash
# 1. the gap bank must be re-harvested first: the repo ships a 1-field demo
#    masks/gap_bank.npz, but every sampled 2007-2015 date WITHOUT a fullgrid
#    file draws its availability mask from the bank, and the study aborts
#    below front_finder.mask_bank.MIN_REAL_BANK (30) fields (same
#    prerequisite as phase 2a, see section 8). One-time harvest:
PYTHONPATH=src python -c "from pathlib import Path; from front_finder import mask_bank; mask_bank.harvest(sorted(Path('$JPL_AIRS_FCST').rglob('fullgrid_*')))"
#    (for a local smoke run only, skip the harvest and pass
#    --allow-small-bank to krige_validate instead)
# 2. the swath bank must exist too (it splits gaps into cloud vs
#    out-of-swath strata for the metrics) — either the chain's
#    --with-swath-bank job, or by hand:
PYTHONPATH=src python -m dl_front.swath build-bank --years 2007-2021
# 3. the study itself (CPU, ~an hour; runs fine alongside phase 1 GPU jobs):
PYTHONPATH=src python -m dl_front.krige_validate --years 2007-2015 --n-days 40
```

Outputs land in `results/dl_front/krige_validation/`: `metrics.csv` (tidy
RMSE/MAE/bias/gradient-ratio per date × channel × method, stratified overall
/ by gap_type / by distance-to-observation), `projection_methods.csv` (the
swath-footprint predictor race), `panels/` (truth | kriged | bias maps), and
`summary.md`. **Read `summary.md` before launching stage B (phase 2b)** — it
states the cloud-gap vs out-of-swath error comparison and the winning
variogram explicitly. If a model other than the default wins, swap
`kriging.variogram_model` in `configs/dl_front.yaml` (or point
`JPL_DLFRONT_CONFIG` at an experiment copy) *before* phases 2a/3a build the
caches.

### Spot-check quicklooks (automatic)

The chain renders human-checkable PNGs of every CPU-built product
(`QUICKLOOK=0` disables): after the swath-bank job, one map per period hour
of all 16 cycle-day coverage frequencies with the footprint threshold drawn
(`quicklook/swath_bank/`); after each krige phase, a handful of cache steps
— all five kriged channels plus the gap_type decomposition, framed to the
crop window with the analysis domain outlined
(`quicklook/kriged-degraded/`, `quicklook/kriged-airs/`). Everything lands
under `$JPL_AIRS_RESULTS/dl_front/quicklook/`. Sampling is deterministic
(evenly spaced over the caches' pooled time axis), so a rebuild overwrites
the same filenames. The jobs are leaves of the dependency graph — a
quicklook failure never blocks training. **Eyeball them before the stage-B/C
trainings get far**: wrong-looking swaths, an empty gap_type panel, or a
mis-scaled channel is a cache-build problem caught in minutes instead of a
bad loss curve caught next morning. Manual reruns:

```bash
PYTHONPATH=src python -m dl_front.quicklook swath-bank
PYTHONPATH=src python -m dl_front.quicklook kriged-degraded --years 2007-2015
PYTHONPATH=src python -m dl_front.quicklook kriged-airs --years 2007-2021 --n 8
```

## 4. Expected wall-clock

- **Training** (phases 1/2b/3b, per fold): ~2-3 h on an A100, ~10-16 h on a
  GTX-1070-class card. Folds are independent GPU jobs; with 3 GPUs all folds
  of a phase run concurrently.
- **Kriging** (2a/3a): ordinary kriging of <= `KRIGE_MAX_OBS`=1500 obs points
  per field, 2 gap-filled channels x 3 label hours/day for 2a (4 channels for
  3a) — order seconds per field, so roughly 0.5-2 days serial for 2a's 9
  years; `--workers 8` brings it to hours. Check the timing note in
  `src/dl_front/krige_fill.py`'s module docstring for measured numbers once
  the module has run. 2a runs concurrently with phase 1, 3a with everything
  before 3b, so kriging is usually off the critical path.
- **Eval** (phase 4): minutes per run on GPU, tens of minutes on CPU.

## 5. Checkpoints and outputs

Each training job writes to `$JPL_AIRS_RESULTS/dl_front/models/<name>/`:

- `<name>.h5` — **best-so-far** weights, overwritten on every validation
  improvement (these are the "intermediate bests"; downstream phases retrain
  from this file).
- `<name>_final.h5` — separate end-of-fold weights, written once at job end.
  Also the chain's done-marker for idempotency.
- `history.csv`, `run_config.yaml` (records `source`/`hours`).

Phase 4 is the **three-way comparison** (user decision 2026-08-13): our
checkpoints on reanalysis inputs, our checkpoints on kriged-AIRS inputs, and
the *published* Biard & Kunkel (2019) DL-FRONT predictions (`--source bk19`,
checkpoint-free — the files under `JPL_BK19_DIR` are the model output). The
BK19 archive ends in 2018, so **every leg scores 2016-2018** (yaml
`splits.eval_years_6class`) — identical years for consistency.

Each leg writes `$JPL_AIRS_RESULTS/dl_front/test_eval/<ckpt-stem>_<source>.csv`
(tidy CSI/POD/FAR/FB table; the BK19 leg's stem is just `bk19`) and
`..._paper.json` (accuracy/AUC/confusion — skipped for bk19: its hard binary
predictions make the ROC sweep meaningless, noted in `bk19_run.json`; BK19
also has no dryline class, so dryline rows are all-miss by construction).
All legs are filtered to the AIRS label hours (18, 21, 00 UTC), **and**
reanalysis/bk19 runs additionally intersect their time steps with the
kriged-AIRS cache's time axis (the AIRS archive is sparse, so days without a
fullgrid file must not be scored by only some legs). Note the matching only
*drops* steps — it cannot restore steps a leg is missing on its own (e.g. a
partially downloaded `sfc_daily` year silently shrinks the reanalysis leg) —
so the same-sample property is *verified*, not assumed: each `..._run.json`
provenance file records per-year step counts and a SHA-1 of the scored
timestamps, and the `compare` job cross-checks the SHA-1s of every leg,
warning loudly (naming the odd leg out) when they disagree. Pass
`--no-match` to `evaluate_test` only when you deliberately want the
full-period score.

The final chain job runs `python -m dl_front.evaluate_test compare` (no
further arguments — it errors rather than silently ignoring any), which
pivots the pooled CSI of every leg CSV into
`$JPL_AIRS_RESULTS/dl_front/test_eval/comparison.csv` (rows = front x
dilation km, one column per leg, NaN where a leg is missing) and prints the
table.

The chain's phase-4 idempotency is provenance-aware: an eval is skipped only
when its CSV exists **and** its `_run.json` shows the full 2016-2018 span
with the kriged-AIRS time match (kriged-airs legs record no match — they
*are* the cache steps). A stale `--no-match` or partial-`--years` debugging
CSV under the same stem is detected and rerun, never compared.

## 6. Monitoring

```bash
squeue -u $USER
tail -f logs/dlfront_train_<jobid>.out
tail -f $JPL_AIRS_RESULTS/dl_front/models/D6A-f0/history.csv
# CPU-side CSI probe against a live run (works alongside GPU training):
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/csi_probe.py \
    --name D6A-f0 --classes 6 --n-samples 1500
```

## 7. Resume / rerun

- The chain is **idempotent-safe**: rerunning `dlfront_jpl_chain.sh` skips
  every train phase whose `<name>_final.h5` exists, krige phases whose cache
  files exist for **every** requested year (zero-coverage years still get an
  empty cache file, so a finished build is never mistaken for a partial one),
  and evals whose CSV exists — so after a crash, just rerun the script and
  only the missing pieces are submitted (with correct dependencies among
  them).
- The krige builders are themselves **resumable per year**: each year's
  cache is written as soon as that year finishes, and a rerun skips years
  whose file exists (`--force` rebuilds). A walltime kill mid-build loses at
  most the year in flight.
- A fold that died mid-training has a best-so-far `<name>.h5` but no
  `_final.h5`; the rerun resubmits it from scratch. To warm-start instead,
  submit manually:
  `sbatch slurm/dlfront_train.sbatch --name D6A-f0 --classes 6 --fold 0 --source reanalysis --retrain $JPL_AIRS_RESULTS/dl_front/models/D6A-f0/D6A-f0.h5`
- `FORCE=1 bash scripts/dlfront_jpl_chain.sh` resubmits everything.
- `WARM_START=/path/to/existing_stageA.h5` starts phase 1 from an existing
  checkpoint instead of random init.

## 8. Gotchas

- The 2016-2021 reanalysis test evals **require** phase 0 (`sfc_daily` for
  those years). If `evaluate_test --source reanalysis` fails on missing
  files, run `PYTHONPATH=src python -m dl_front.acquire_merra2_sfc 2016 2017
  2018 2019 2020 2021` on a node with internet + `~/.netrc`.
- `front_id/reanalysis/MERRA2/sfc_norm_stats.json` is frozen (2026-08); do
  not recompute it on the cluster.
- Keep `--workers` <= the sbatch `--cpus-per-task` (8) in
  `slurm/dlfront_krige.sbatch` if you raise `KRIGE_WORKERS`.
- Days without an AIRS-FCST fullgrid file are skipped by `build-airs` with a
  logged line — a sparse `$JPL_AIRS_FCST` tree is expected, not an error.
  Malformed/truncated fullgrid files are likewise skipped with a note, never
  fatal.
- **Re-harvest the gap bank before phase 2a**: the repo ships a 1-field
  `masks/gap_bank.npz` (one demo day). `build-degraded` refuses to run with
  fewer than `front_finder.mask_bank.MIN_REAL_BANK` (30) fields — every
  degraded step would reuse the same gap geometry. On a machine that sees
  the fullgrid archive:
  `PYTHONPATH=src python -c "from pathlib import Path; from front_finder import mask_bank; mask_bank.harvest(sorted(Path('$JPL_AIRS_FCST').rglob('fullgrid_*')))"`
  (`--allow-small-bank` exists for smoke tests only.)
