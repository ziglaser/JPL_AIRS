# Deploying front-detection training on a SLURM cluster

Everything cluster-specific is confined to `slurm/` and four environment
variables; the package itself is location-agnostic.

## 1. Data (already on the cluster — nothing to copy)

Manifest reorg 2026-08-13: the full data tree lives on gattaca2 at
`/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data` (with the AIRS-FCST
fullgrid archive `AIRS_FCST_1deg/` as its sibling), and the local `data/`
checkout mirrors the same layout — so deployment is just pointing
`JPL_AIRS_DATA` (and `JPL_AIRS_FCST`) at the GPFS paths. What each stage
reads, relative to `$JPL_AIRS_DATA`:

| What | Path (relative to `$JPL_AIRS_DATA`) | Needed for |
|---|---|---|
| MERRA-2 daily corpus | `front_id/reanalysis/MERRA2/daily/{2003..2015}/` | stages A/B (2016-2021 only for later work) |
| Frozen norm stats | `front_id/reanalysis/MERRA2/norm_stats.json` | everything (do NOT recompute — frozen 2026-08-07) |
| CODSUS (WPC) labels | `front_id/met_drawn_fronts/WPC_CODSUS/WPC_1deg_gridded/` | training labels |
| AIRS gap-mask bank | `masks/gap_bank.npz` | stage B |
| DL-FRONT benchmark (BK19) | `front_id/predicted_fronts/bk19/` | evaluation only |
| AIRS-FCST fullgrid archive | `$JPL_AIRS_FCST` = sibling `../AIRS_FCST_1deg/` (2003-2022) | stage C |
| Existing checkpoints (optional) | `$JPL_AIRS_RESULTS/front_finder/models/` | resuming instead of retraining |

Materialized shards (`front_id/reanalysis/MERRA2/shards/`, ~87 GB) are
regenerated on the cluster by the chain itself — an embarrassingly parallel
~20-minute array job (`slurm/materialize.sbatch`).

GPFS is assumed visible from the compute nodes; verify once with
`srun ls $JPL_AIRS_DATA/front_id/reanalysis/MERRA2` from a compute node
before submitting the chain.

## 2. Environment

```bash
conda env create -f slurm/fronts-tf-environment.yml
```

`tensorflow[and-cuda]==2.15.1` bundles CUDA 12.2 — no CUDA module load needed,
only an NVIDIA driver >= 525 on the GPU nodes. Verify with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 3. Path configuration (env vars, read by `src/front_finder/config.py` and `src/dl_front/config.py`)

| Variable | Meaning | Default |
|---|---|---|
| `JPL_AIRS_REPO` | repo checkout (sbatch scripts `cd` here) | — (required by sbatch scripts) |
| `JPL_AIRS_DATA` | data root; cluster: `/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data` | `<repo>/data` |
| `JPL_AIRS_FCST` | AIRS-FCST fullgrid archive (stage C); cluster: `/gpfs/scratch/smap-convection/AIRS_FCST_1deg` | sibling `$JPL_AIRS_DATA/../AIRS_FCST_1deg` when it exists, else `$JPL_AIRS_DATA/HYSPLIT_demo` |
| `JPL_AIRS_RESULTS` | results/checkpoints root | `<repo>/results` |

## 4. Run

```bash
cd $JPL_AIRS_REPO
bash slurm/submit_chain.sh     # materialize (array 2003-2015) -> E1b -> E2, + E1a
```

Stage C (fine-tune on real AIRS), against the fullgrid archive:

```bash
sbatch slurm/train.sbatch --name E3 --winds \
    --airs-glob "$JPL_AIRS_FCST/*/*/fullgrid_*" \
    --retrain "$JPL_AIRS_RESULTS/front_finder/models/E2-degraded/E2-degraded.h5"
```

(Archive layout is `YYYY/wrf27km_YYYYMMDD/fullgrid_wrf27km_GOOD_1p00deg_*`;
some files have no `.nc` suffix, hence the suffix-less glob.)

## 5. Sizing notes

- The in-repo defaults (`filter_num=[16,32,64,128]`, batch 4 × 160 steps) are
  halved for the local 8 GB GTX 1070. On cluster GPUs pass the paper-faithful
  `--filter-num 32,64,128,256 --batch 64 --steps 10` (both keep the paper's
  640-samples-per-epoch semantics); `submit_chain.sh` already does.
- Training reads materialized shards via memmap — memory use is bounded
  (~3-4 GB host RSS) and I/O is sequential-friendly; 32 G `--mem` is generous.
- **Never enable the tf.data file cache for training** (`cache=True` in
  `dataset.make_tf_dataset`): TF 2.15's cache writer leaks ~0.4 MB per element
  until finalization (post-mortem 2026-08-09, `docs/FRONT_DETECTION_WORKPLAN.md`
  run log). Shards exist precisely so nothing needs that cache.

## 6. Sanity checks after deployment

```bash
PYTHONPATH=src python -m pytest tests/ -k "front" -q   # 106 tests
PYTHONPATH=src python -m front_finder.train --name smoke --no-winds --smoke
```
