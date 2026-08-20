#!/bin/bash
#SBATCH --job-name=dlfront-swath-bank
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dlfront_swath_bank_%j.out
# CPU-only 16-day swath-climatology builder: all dl_front.swath arguments
# pass through, e.g.
#   sbatch slurm/dlfront_swath_bank.sbatch build-bank --years 2007-2021
# One pass over the fullgrid archive (read masks only, no kriging), so a
# single CPU suffices.
# Output: $JPL_AIRS_DATA/masks/swath_bank.npz

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
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}"

python -m dl_front.swath "$@"
