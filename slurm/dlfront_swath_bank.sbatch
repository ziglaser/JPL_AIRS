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
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}"

python -m dl_front.swath "$@"
