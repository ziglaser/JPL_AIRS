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

**Analysis domain (user decision 2026-08-13; caches are written at
schema v4 as of 2026-08-18, and v3 is still readable — see the
compatibility table below):** the 6-class
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
(`dataset.crop_domain()`) to harvest nearby front examples.

**Channel sourcing, and why no cache rebuild is ever needed (user
decision 2026-08-18).** A kriged cache is no longer the sole input source.
`dl_front.dataset.kriged_year_arrays` now splits by provenance:

* channels in `airs.kriged_channels` (currently `T2M`, `QV2M`) are read
  **from the cache** — they carry the satellite-shaped gap fills, which
  exist nowhere else;
* every other input channel (`SLP`, `U10M`, `V10M`) is read **from the
  MERRA-2 reanalysis** `sfc_daily` step at the same timestamp, masked to
  the crop domain.

A "clean" channel *is* the reanalysis by definition, so there is no reason
to trust a copy of it baked into a cache file. The consequence is that a
cache's SLP/wind content is simply never read, which makes the v3→v4
distinction a non-event: v3 kriged `U10M`/`V10M`, but those copies are
ignored, so **a v3 cache is usable at any `--channels` width, including
the full five, with no rebuild.** `config.KRIGE_SCHEMA_READABLE = (3, 4)`;
v1 (full-grid fills, no `gap_type`) and v2 (old region-mask domain) are
genuine format breaks and remain unreadable.

It is also more honest: the model now sees the *same* reanalysis SLP and
winds at train and eval time regardless of which cache generation produced
the `T2M`/`QV2M` fills.

One guard remains, and it is the only one that matters: if a channel the
config calls kriged is held **clean** in the cache, the loader refuses
rather than quietly training on reanalysis where the configuration
promises AIRS information.

> **Deployment consequence:** stage B/C loading now needs `sfc_daily` for
> the cache's years, not just the cache. On gattaca2 that is already
> present (it is what stage A trains from), but a machine that has only
> the kriged caches will now fail with a message naming the missing day
> file and the `acquire_merra2_sfc` command that fetches it.

To rebuild when you do want it — rerun phases 2a/3a with `--force` (or
delete `front_id/degraded_reanalysis/` and `front_id/kriged_airs_fcst/` and
let the chain rebuild from scratch):
```bash
PYTHONPATH=src python -m dl_front.krige_fill build-degraded \
    --years 2007-2015 --workers "$KRIGE_WORKERS" --force
PYTHONPATH=src python -m dl_front.krige_fill build-airs \
    --years 2007-2021 --workers "$KRIGE_WORKERS" --force
```
Both are CPU-bound and multi-hour even with `--workers 8` (see section 4's
kriging wall-clock note) — budget the better part of a day, and start them
well before any overnight run that depends on them, not as the first step
of it. This is *not* on the critical path for the label re-score or the
channel ladder; see the compatibility table above. The swath bank
(`masks/swath_bank.npz`) is full-grid and unaffected by either the domain
or the schema-v4 channel change. Config knobs: `domain:` section and
`airs: kriged_channels:` of `configs/dl_front.yaml`.

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
| `FORCE=1` | rerun phases whose done-marker exists **and rebuild both kriged caches and retrain all checkpoints** — see `FORCE_EVAL=1` below if you only want to re-score | `0` |
| `FORCE_EVAL=1` | resubmit ONLY phase 4 (eval legs + `compare`); training and krige done-markers are left alone — this is "re-score existing checkpoints against new labels, do not retrain" (section 6), the opposite of `FORCE=1` | `0` |
| `FORCE_TRAIN=1` | rerun the training phases even when `<name>_final.h5` exists, WITHOUT rebuilding the krige caches or the swath bank — "retrain on regenerated labels without rebuilding krige caches" (section 6). Evals follow automatically: the chain never skips an eval whose train job it just submitted | `0` |
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
tail -f logs/chain.out   # per-step logs land in logs/dlfront/main/<run-ts>/<phase-label>.log
```

`setsid` + `nohup` keeps the chain alive after logout; each step also gets
its own log under the run's `logs/dlfront/main/<run-timestamp>/` directory
(`logs/dlfront/main/latest/` always points at the newest run, and the same
timestamp names the manifest `results/dl_front/chain_<ts>.txt`).

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
#    prerequisite as phase 2a, see section 9). One-time harvest:
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

- **Training** (phases 1/2b/3b, per fold): ~4-6 h on an A100, ~20-32 h on a
  GTX-1070-class card at the current `max_epochs: 1200` (back to the paper's
  figure, user decision 2026-08-18 — roughly 2x the old 600-epoch
  wall-clock; the 600 cap was binding, so runs now end where early stopping
  says). Folds are independent GPU jobs; with 3 GPUs all folds of a phase
  run concurrently.
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

**Input channels are configurable (2026-08-18):** every training/eval job
consumes `config.INPUT_CHANNELS` (default: all five `SFC_VARS` — T2M, QV2M,
SLP, U10M, V10M), settable per job with `--channels T2M,QV2M,...` on
`dl_front.train` and `dl_front.evaluate_test`. `run_config.yaml` in the
checkpoint dir always records the resolved list, both under `run_args:
channels:` and under the tunables block as `INPUT_CHANNELS:` — a checkpoint
directory is self-describing, so you never have to guess what a model's
inputs were. `evaluate_test` reads that file and adopts its channel list
automatically when you don't pass `--channels` yourself (see section 10 for
the concrete stage-A ablation this exists for).

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

## 6. Re-scoring after a label change

The front labels are not static: a bug in `front_formats/xml_to_codsus.
parse_xml` fed raw [-180, 180] longitudes straight to a plain-lat/lon
rasterizer, so any antimeridian-crossing frontal polyline painted a spurious
full-width horizontal bar across the analysis domain — this hit **~36% of
2016-2018 analyses**. Fixed 2026-08-17 (`unwrap_lon` + `split_valid`); all
2007-2022 labels were regenerated and the fix is strictly subtractive
(2,584,855 spurious in-domain cells removed across the archive, zero added).
Re-scoring the existing checkpoints against the clean labels raised the
reported reanalysis CSI substantially (occluded +0.21, cold +0.10) — and it
raised the BK19 baseline too, so the margin over BK19 survives the fix; the
kriged-AIRS leg has **not** been re-scored yet as of this writing, which is
exactly what the overnight run referenced below is for.

**The regenerated labels must be copied to the cluster before you re-score
anything there.** (Corrected 2026-08-18: the procedure below replaces a
previous version of this section whose `rsync` had three bugs at once —
no remote host, so it copied locally on whichever machine you typed it on;
a source path (`data/front_id/...`) that does not exist, since the
repo-local `data/` mount is gone and the real root is
`/mnt/d/JPL_AIRS/data`; and, at the time it was written, it pointed at a
tree that was still pre-fix. An operator following it verbatim would have
re-scored against the old, buggy labels while every provenance check
reported everything self-consistent, because the digest in section 6 only
ever compares the labels *actually on disk* against themselves.) Verified
current state as of today: the regenerated (fixed) labels are promoted and
live at
`/mnt/d/JPL_AIRS/data/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/{1,3}wide`
(~172 MB, 16 years x 2 widths); the pre-fix set is preserved at the sibling
`NOAA_1deg_gridded.pre_2026-08-17_datelinebug/`; the same 32 files also sit
in the flat staging dir `front_id/NOAA_to_CODSUS_staging/` (both are local
to the machine with the `/mnt/d` mount, not the cluster).

1. **Confirm the local tree is the fixed one** before copying anything —
   the antimeridian bug (see above) painted a spurious full-width bar
   across lat 33N on 2017-07-25 in every buggy file; the sharp test is
   counting flagged cells on that one date/latitude: a buggy `3wide` file
   has ~58 front-flagged cells there, a fixed one has 0.
   ```bash
   python -c "
   import xarray as xr
   ds = xr.open_dataset('/mnt/d/JPL_AIRS/data/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/3wide/noaa_fronts_merra2-1deg_3wide_2017.nc')
   # 'fronts' is (time, front, lat, lon), front axis = [cold, warm,
   # stationary, occluded, dryline, none] -- exclude the 'none' (no-front)
   # channel, or a real spurious bar and a real absence both read as
   # 'front axis has something set' and the test washes out to 58 in BOTH
   # trees (verified 2026-08-18: this is why the naive sum(>0) over all six
   # channels is a false negative here -- restricting to the five real
   # front types is what actually separates buggy from fixed).
   real = [i for i, t in enumerate(ds['front_type'].values) if t != 'none']
   sel = ds['fronts'].sel(lat=33, method='nearest').sel(time='2017-07-25T00:00:00').isel(front=real)
   print('real front-flagged cells at 33N, 2017-07-25T00Z:', int((sel > 0).sum()))
   "
   # expect 0 -- if you see ~58 (a contiguous stationary-front bar spanning
   # the whole longitude range at this latitude), this tree is the pre-fix
   # NOAA_1deg_gridded.pre_2026-08-17_datelinebug/ set, not the fixed one.
   ```
2. **Back up whatever the cluster already has** before overwriting it, in
   case the copy is interrupted or the wrong source gets rsynced:
   ```bash
   ssh user@cluster.jpl.example \
       'cp -a "$JPL_AIRS_DATA/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded" \
              "$JPL_AIRS_DATA/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded.bak_$(date +%Y%m%d)"'
   ```
3. **Dry-run the transfer** and read the file list before trusting it —
   `--dry-run` costs nothing and catches a wrong path before it catches you:
   ```bash
   rsync -av --dry-run \
       /mnt/d/JPL_AIRS/data/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/ \
       user@cluster.jpl.example:"\$JPL_AIRS_DATA/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/"
   ```
4. **Copy for real**, resumable and visible (172 MB is small, but
   `--partial --progress` costs nothing and saves a restart from zero if
   the link drops):
   ```bash
   rsync -av --partial --progress \
       /mnt/d/JPL_AIRS/data/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/ \
       user@cluster.jpl.example:"\$JPL_AIRS_DATA/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/"
   ```
5. **Verify with matching checksums on both ends** — a copy that "completed"
   silently truncated is worse than one that visibly failed:
   ```bash
   md5sum /mnt/d/JPL_AIRS/data/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/*/*.nc \
       > /tmp/labels_local.md5
   ssh user@cluster.jpl.example \
       'cd "$JPL_AIRS_DATA/front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded" && md5sum */*.nc' \
       > /tmp/labels_cluster.md5
   diff <(awk '{print $1}' /tmp/labels_local.md5 | sort) \
        <(awk '{print $1}' /tmp/labels_cluster.md5 | sort)
   # no output == every checksum matched
   ```
6. **Re-score everything, without retraining**:
   ```bash
   FORCE_EVAL=1 bash scripts/dlfront_jpl_chain.sh
   ```
   `FORCE_EVAL=1` reruns phase 4 only — training and kriging are left
   alone — so this is exactly step 1 of the label-fix cleanup (re-score,
   no retrain). See the env-var table in section 2 for how this differs
   from plain `FORCE=1`.
7. **Full retrain on the new labels** (step 2, once the re-score has told
   you it is worth the GPU budget):
   ```bash
   FORCE_TRAIN=1 bash scripts/dlfront_jpl_chain.sh
   ```
   trains D6A/D6B/D6C fresh on the labels currently on disk; the phase-4
   evals follow automatically (the chain never skips an eval whose train
   job it submitted this invocation, and the new `_run.json`s carry the new
   checkpoints' `ckpt_sha1`), while the krige caches and swath bank stay
   untouched — they hold input fields only, and labels play no part in
   them. **Ordering (user decision 2026-08-18): all analysis runs on the
   NEW checkpoints.** Nothing is scored against the pre-fix weights: run
   the retrain first, and only then the ablation chain (section 10) — its
   step-2 permutation will target the freshly trained `D6A`/`D6C`, whose
   `run_config.yaml` records the current `airs.kriged_channels`, so the
   `check_kriged_split` gate passes without any config gymnastics.
   (`FORCE_TRAIN=1` overwrites the old model dirs in place; if you want the
   pre-fix weights archived at all, `mv` the model dirs aside first.)
8. **Fully unattended alternative**: `scripts/dlfront_full_sequence.sh`
   runs the whole thing — kickoff (main chain), ablation chain, prediction
   export 2016-2021 (opportunistic GPU, CPU fallback), and the
   FCST_SMAP_MRMS flag injection — in sequence with no human intervention,
   waiting on each phase's done-marker files and aborting loudly if a
   phase's SLURM jobs drain without producing them:
   ```bash
   setsid nohup bash scripts/dlfront_full_sequence.sh > logs/full_sequence.log 2>&1 &
   ```
   See its header for the timeout/poll knobs.
9. **Analysis-only alternative** (no training, no krige builds): when every
   checkpoint and cache is already final and only the numbers/figures need
   refreshing against the labels now on disk, `scripts/dlfront_analysis.sh`
   reruns all evals, the fold-pooled compare, the permutation tables, and
   every figure in one pass — see section 12.

**If the labels are ever lost and must be regenerated from the raw XML**
(this is a from-scratch rebuild, not what step 1 above needs — only run it
if the promoted tree above is gone): the raw archive lives at
`front_id/raw_met_drawn_fronts/NOAA_USA_fronts` (46,694 XML files) and the
conversion is
```bash
PYTHONPATH=src python -m front_formats.xml_to_codsus all
```
which takes ~1.5-2 h end-to-end over `/mnt/d` (network-mount I/O bound, not
CPU bound — splitting into 4 processes by year batches brings it to
~30-40 min). Output lands in the flat staging dir
`front_id/NOAA_to_CODSUS_staging/`, not the canonical tree directly — it
still needs to be promoted (moved/renamed into
`front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/{1,3}wide/`) before
step 1's spot check and the rsync above will find it.

Because a metrics CSV and a model checkpoint are no longer enough to
interpret a number — you also need to know *which* labels produced it — the
chain now digests the labels it is about to score against
(`dl_front.dataset.label_digest`, a SHA-1 content hash of the per-class
scored-cell counts, **not** a file hash) and writes it into every eval leg's
`_run.json` as `labels_sha1` (alongside `labels_dir`, the resolved label
root). A `_run.json` with no `labels_sha1` (every CSV written before
2026-08-18) is treated as stale and rerun automatically, with a note
explaining why. To confirm which labels produced an existing number by hand:

```bash
python -c "import json; print(json.load(open('$JPL_AIRS_RESULTS/dl_front/test_eval/D6C-f0_reanalysis_run.json'))['labels_sha1'])"
python -m dl_front.evaluate_test label-digest --classes 6 --years 2016-2018
```

and compare the two digests — equal means that CSV was scored against the
labels currently on disk. The chain computes the second command's output
**once per invocation** and reuses it across all ~19 legs rather than
shelling out per leg. If the digest command fails to run at all (e.g. from
a submitting shell with no `fronts-tf` env — this already happens for
the krige schema probe), the chain degrades gracefully to the pre-2026-08-18 exists-
only staleness check and prints a loud warning that label staleness could
not be verified; it does not hard-fail the run.

## 7. Monitoring

```bash
squeue -u $USER
# chain-submitted jobs log per run: <label>_<jobid>.out under the run dir
tail -f logs/dlfront/main/latest/phase1-D6A-f0_<jobid>.out
# (only a HAND-submitted `sbatch slurm/dlfront_train.sbatch ...` falls back
#  to the .sbatch file's own logs/dlfront_train_<jobid>.out)
tail -f $JPL_AIRS_RESULTS/dl_front/models/D6A-f0/history.csv
# CPU-side CSI probe against a live run (works alongside GPU training):
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/csi_probe.py \
    --name D6A-f0 --classes 6 --n-samples 1500
```

## 8. Resume / rerun

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

## 9. Gotchas

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

## 10. Ablation chain: permutation importance + stage-A channel ladder

`scripts/dlfront_ablation_chain.sh` (new 2026-08-18) runs two experiments
that are **deliberately kept separate** from the main curriculum in section
3 — they answer "how much does AIRS actually buy us?", not "train the
production model" — so they never touch `scripts/dlfront_jpl_chain.sh` or
its manifest. It shares that script's shape: same `JPL_AIRS_REPO/DATA/FCST/
RESULTS`, `CONDA_PREFIX_ROOT`, `SBATCH_PARTITION`/`SBATCH_ACCOUNT`/
`SBATCH_GRES`/`SBATCH_GPU_PARTITION`, `CLASSES`, `FOLDS`, `FORCE`, `DRY_RUN`
env knobs, submits via `sbatch` when present and otherwise runs the same
steps sequentially in the foreground, writes its own manifest, and
`--help` prints its header comment block.

**Step 2 — permutation importance** (GPU preferred, CPU-ok): for each fold
in `FOLDS`, for each stage checkpoint named in `PERM_CKPTS` (default `D6C`;
accepts a space-separated list, e.g. `PERM_CKPTS="D6A D6B D6C"`), for each
`--source` in `reanalysis` and `kriged-airs`, submits
`slurm/dlfront_permutation.sbatch` — `python -m dl_front.permutation` under
the hood — with `--repeats $PERM_REPEATS` (default 3) over `--years
2016-2018`. This sizes how much skill lives in each of the 5 input channels
(in particular SLP and the two wind components, which are not AIRS-derived
— see the background note below) before spending GPU budget on retraining a
restricted model. Output: `$JPL_AIRS_RESULTS/dl_front/permutation/
<ckpt-stem>_<source>.csv` (per-channel CSI/POD/FAR/FB and their deltas
against the unpermuted baseline, at every dilation) plus a `_run.json`. The
done-marker is the CSV existing *and* its `_run.json`'s `labels_sha1`
matching the current label digest — the same staleness check as section 6,
reimplemented locally in this script rather than sourcing
`dlfront_jpl_chain.sh`.

**Step 3 — stage-A channel ladder** (GPU): the point of this step is to
measure how much front skill a stage-A-only model retains when it is
restricted to the channels AIRS actually retrieves. Of the 5 surface
inputs, only T2M and QV2M carry AIRS information — U10M/V10M come from the
WRF-27km met driving HYSPLIT, and SLP is copied clean from MERRA-2
reanalysis (AIRS retrieves no SLP). **This ladder is not the AIRS-only
skill ceiling** (reworded 2026-08-18: it was previously overclaimed as
one) — the production AIRS model is `D6C`, which reaches its 2-channel
input through the full stage A -> B -> C curriculum on progressively more
AIRS-realistic inputs, whereas this ladder trains stage A alone on clean
reanalysis and then evaluates it on kriged-airs inputs it never saw a
gap-filled version of during training. A 2-channel model that skipped B
and C is a **lower bound** from an out-of-domain model, not a ceiling: the
honest ceiling would need the 2-channel stage A carried through the same
B/C curriculum as the production model (call them `D6B2`/`D6C2`), and this
chain does not queue that — it only measures what stage A alone learns
from each channel subset. The 5-channel rung already exists as `D6A-f<k>`
(the ordinary stage-A checkpoint from section 3) and is **not** retrained
here — the ladder below only adds the two smaller rungs to complete it:

| Name | `--channels` |
|---|---|
| `D6A-f<k>` (existing, not retrained) | `T2M,QV2M,SLP,U10M,V10M` |
| `D6A3-f<k>` | `T2M,QV2M,SLP` |
| `D6A2-f<k>` | `T2M,QV2M` |

(`CHANNEL_SETS` env var controls the two new rungs; default
`"D6A3:T2M,QV2M,SLP" "D6A2:T2M,QV2M"`.) For each fold and each set: train
`<name>-f<k>` (`dl_front.train --source reanalysis --classes $CLASSES
--fold k --channels <list>`), then, `afterok`, evaluate it with `--source
reanalysis` **and** `--source kriged-airs` on 2016-2018 — `evaluate_test`
picks the channel list up automatically from the checkpoint's
`run_config.yaml`, so the eval legs don't need `--channels` repeated by
hand. Done-markers: the usual `<name>-f<k>_final.h5` for training, the
labels-aware CSV check (section 6) for evals.

**The reanalysis leg is the primary ladder result; the kriged-airs leg is
secondary and confounded (noted 2026-08-18).** Only T2M/QV2M are kriged
now (`airs.kriged_channels` in `configs/dl_front.yaml`, schema v4) — SLP
and both wind components are copied clean from reanalysis into the
kriged-airs cache too. So on the kriged-airs eval leg, the 5-channel
model's input distribution is mostly what it trained on (3 of 5 channels
are the identical clean-reanalysis fields at train and test time), while
none of the 2-channel model's input distribution is (both its channels
were kriged at eval time and clean reanalysis at train time). A 5 -> 2
channel drop scored on that leg mixes two different effects — "less
information" and "more train/test distribution shift" — and the second
effect biases the comparison *against* the low-channel rungs, i.e. it can
make dropping channels look worse than it would if every rung were
evaluated on inputs it actually trained on. The reanalysis leg has no such
asymmetry (every rung sees the same input distribution at train and eval
time), so read the reanalysis-leg numbers as the ladder's answer to "how
much skill does each channel subset carry" and treat the kriged-airs-leg
numbers as a secondary, confounded data point about deployment-time
behavior, not a second independent estimate of the same quantity.

Either step can be run alone or together: `STEPS="2"`, `STEPS="3"`, or the
default `STEPS="2 3"`.

Rough cost: step 2 is `(1 + 5 channels) x repeats` forward passes per
(fold, checkpoint, source) — no training. Step 3 is 2 channel sets x
`|FOLDS|` stage-A trainings (plus 4 eval legs per set-fold, 2 sources x 2
channel sets), i.e. the same per-fold wall-clock as one `D6A` training
(section 4) times 2, plus minutes of CPU eval.

```bash
DRY_RUN=1 bash scripts/dlfront_ablation_chain.sh          # inspect the plan
PERM_CKPTS="D6A D6B D6C" PERM_REPEATS=3 STEPS="2" \
    bash scripts/dlfront_ablation_chain.sh                # step 2 only
STEPS="3" bash scripts/dlfront_ablation_chain.sh           # step 3 only
```

## 11. Post-eval phases: export predictions → inject flags

Two manual phases (added 2026-08-18), run **after** phase 4 and *not* wired
into `scripts/dlfront_jpl_chain.sh` yet. They turn checkpoints into a
BK19-schema raster archive and then into flag variables on the
`FCST_SMAP_MRMS` year files that the convection analysis reads.

**11a. Export** — `src/dl_front/export_predictions.py`, submitted via
`slurm/dlfront_export.sbatch` (CPU-only, inference; 4 cpus / 16 G / 6 h):

```bash
sbatch slurm/dlfront_export.sbatch \
    --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f0/D6C-f0.h5" \
    --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f1/D6C-f1.h5" \
    --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f2/D6C-f2.h5" \
    --source kriged-airs --years 2016-2021
```

Writes, per checkpoint **and** for the softmax-averaged ensemble,
`$JPL_AIRS_DATA/front_id/predicted_fronts/dlfront_<stem>_<source>/1deg_3wide/3hr/merra2_merra2-1deg_3wide_3hr_<YYYY>.nc`
(+ `_run.json`) — hard classes, ubyte, `_FillValue=2` outside
`dataset.analysis_domain()`, BK19-identical dims/attrs/dtypes except
`front = 6` (dryline kept). Score any archive through the unmodified BK19
leg: `JPL_BK19_DIR=<root>/<tag> python -m dl_front.evaluate_test --source
bk19 --classes 6`. `--force` rewrites existing years; `--ensemble-only`
suppresses the per-fold archives; `--class-scale warm=1.3` picks a non-default
operating point and lands in the tag. Idempotent per year (done-marker =
`.nc` **and** `_run.json`), and **exits 1 when an archive tag got no year at
all** while years failed (audit 2026-08-18) — so an `afterok` dependant never
runs on an empty archive; a fully-complete requeue still exits 0.

**11b. Inject** — `scripts/add_front_flags.py` (pure xarray/netCDF4, no TF).
`convection_skill.config.DATA_DIR` honours `JPL_AIRS_DATA`, but pass
`--primary-dir` explicitly anyway — the `FCST_SMAP_MRMS` year files live
under the data root, and naming the exact tree being copied keeps a wrong
`JPL_AIRS_DATA` from silently injecting into the wrong archive:

```bash
python scripts/add_front_flags.py --years 2016-2021 --label-source noaa \
    --primary-dir "$JPL_AIRS_DATA/FCST_SMAP_MRMS" \
    --out-dir     "$JPL_AIRS_DATA/FCST_SMAP_MRMS_fronts" \
    --pred-dir    "$JPL_AIRS_DATA/front_id/predicted_fronts/dlfront_D6C-ens3_kriged-airs" \
    --pred-tag    D6C-ens3_kriged-airs
```

Copies each primary year file and *appends* (netCDF4 mode `a`, no re-encode)
`front_{cold,warm,stationary,occluded,dryline,any}_{1,3}w` from the met-drawn
labels plus `pred_front_*_3w` and `pred_front_valid_frac` from the archive —
same 2×2 max-pool + concurrent-bulletin alignment as
`convection_skill.fronts` (verified bit-equal to `year_front_flags`). ~4 GB
of copies for six years; `--in-place` appends into the primaries instead
(opt-in: a crash mid-append damages an irreplaceable file). Refuses a target
that already carries flag variables unless `--force`.

Caveat: the exported prediction domain (`analysis_domain()`, land ≥ 0.5) is
*smaller* than the FCST grid, so `pred_front_*` is NaN outside it (~lat
31.5-52.5) while the met-drawn flags cover every cell — never compare the two
without masking to `pred_front_valid_frac > 0`.

## 12. Analysis-only reruns

`scripts/dlfront_analysis.sh` (new 2026-08-20) reruns **every analysis
against the artifacts already on disk** — it never trains, never builds a
krige cache, and never builds the swath bank. Use it when the checkpoints
and caches are done and you only need the downstream numbers and figures
refreshed: after a label regeneration (this is the analysis-side companion
to section 6's `FORCE_EVAL=1` re-score, but it also refreshes the
permutation tables and every figure in the same pass), after pulling a
finished cluster run to another box, or after an eval-core change
invalidated the CSVs.

What it runs, in dependency order (see `--help` for the full contract):

1. **Pre-flight + discovery.** Aborts loudly unless the kriged-airs cache
   exists with a readable schema for every `EVAL_YEARS` year *and* at least
   one finished checkpoint (`<name>_final.h5`) sits under
   `$JPL_AIRS_RESULTS/dl_front/models` — with nothing trained there is
   nothing to analyze, and this script will not train it. It then
   **discovers** every finished checkpoint on disk and splits them into
   main-curriculum stems (`D6A/D6B/D6C-f<k>`) and everything else (the
   `D6A5/D6A3/D6A2` ladder rungs, one-offs). Discovery is disk-driven, not
   `FOLDS`-driven: whatever finished training gets analyzed.
2. **Eval legs + compare.** Every main checkpoint × `{reanalysis,
   kriged-airs}` through `dl_front.evaluate_test`, plus the checkpoint-free
   BK19 leg, plus the final fold-pooled `compare` into `comparison.csv` —
   all with the main chain's skip discipline (years/match_source/
   `labels_sha1`/`ckpt_sha1`). Non-main checkpoints are evaluated too (no
   `--channels` flag needed: `evaluate_test` adopts each checkpoint's own
   `run_config.yaml` channel list) and their CSVs are moved into
   `ablation_eval/` exactly like the ablation chain's move step, so a
   reduced-channel rung can never contaminate `comparison.csv`; the compare
   job waits on those moves.
3. **Permutation importance.** Every discovered checkpoint (main *and*
   non-main) × both sources through `dl_front.permutation --repeats
   $PERM_REPEATS`, with the ablation chain's skip discipline
   (`labels_sha1` + `ckpt_sha1`). Ladder stems land in the same
   `permutation/` directory under their naturally distinct names.
4. **Six-panel figures** for each fold in `FOLDS` (non-fatal).
5. **`scripts/plot_dlfront_results.py`** (`--all-folds`, non-fatal), so the
   summary figure set is refreshed in the same pass.

Knobs mirror the sibling chains (`JPL_AIRS_*`, `CLASSES`, `FOLDS` —
six-panel only, `EVAL_YEARS` default 2016-2018, `PERM_REPEATS`, `FORCE=1`,
`DRY_RUN=1`, `SBATCH_*`). With SLURM it submits a dependency chain; without
it, the same steps run sequentially in the foreground:

```bash
DRY_RUN=1 bash scripts/dlfront_analysis.sh   # inspect the plan first
bash scripts/dlfront_analysis.sh             # run it
FORCE=1 bash scripts/dlfront_analysis.sh     # rerun every leg regardless
```

Like the sibling chains it duplicates their helpers rather than sourcing
them (independent editability), stamps a manifest at
`results/dl_front/analysis_<timestamp>.txt`, and degrades gracefully (loud
warning, existence-only staleness checks) when the submitting shell cannot
compute the label digest.
