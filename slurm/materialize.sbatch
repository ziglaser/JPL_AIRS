#!/bin/bash
#SBATCH --job-name=ff-materialize
#SBATCH --array=2003-2015
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/materialize_%a.out
# One array task per year: derives + normalizes every training sample once and
# writes memmap shards (materialize.py docstring has the full rationale).
# CPU-only; ~6.7 GB/year of shards under
# $JPL_AIRS_DATA/front_id/reanalysis/MERRA2/shards/ (front_finder.config
# SHARD_DIR; manifest reorg 2026-08-13).

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
# JPL_AIRS_DATA / JPL_AIRS_RESULTS: exported in submit_chain.sh (scratch paths)

# Extra args pass through, e.g. `sbatch slurm/materialize.sbatch --labels-only`
# rebuilds only y_*.npy at config.LABEL_DILATION (minutes/year).
python -m front_finder.materialize "$SLURM_ARRAY_TASK_ID" "$@"
