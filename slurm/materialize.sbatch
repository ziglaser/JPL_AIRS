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
# conda bootstrap, robust to install location: explicit CONDA_PREFIX_ROOT wins;
# else ask the conda on PATH (sbatch propagates the submission environment);
# else probe the common install dirs.
if [[ -n "${CONDA_PREFIX_ROOT:-}" ]]; then
    source "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
        if [[ -f "$_c/etc/profile.d/conda.sh" ]]; then source "$_c/etc/profile.d/conda.sh"; break; fi
    done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found: set CONDA_PREFIX_ROOT to the conda install root" >&2; exit 1; }
conda activate fronts-tf
# repo + data roots: explicit exports win; otherwise the repo is where sbatch
# was invoked (SLURM_SUBMIT_DIR) and slurm/env.sh probes the known machine
# layouts -- submitted from the checkout, this job needs NO exports at all.
JPL_AIRS_REPO="${JPL_AIRS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$JPL_AIRS_REPO/slurm/env.sh" ]]; then
    echo "JPL_AIRS_REPO=$JPL_AIRS_REPO is not the repo checkout: submit from" \
         "the repo root or export JPL_AIRS_REPO" >&2
    exit 1
fi
source "$JPL_AIRS_REPO/slurm/env.sh"
cd "$JPL_AIRS_REPO"
export PYTHONPATH=src
# JPL_AIRS_DATA / JPL_AIRS_RESULTS: exported in submit_chain.sh (scratch paths)

# Extra args pass through, e.g. `sbatch slurm/materialize.sbatch --labels-only`
# rebuilds only y_*.npy at config.LABEL_DILATION (minutes/year).
python -m front_finder.materialize "$SLURM_ARRAY_TASK_ID" "$@"
