#!/bin/bash
#SBATCH --job-name=dlfront-quicklook
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dlfront_quicklook_%j.out
# CPU-only spot-check PNG renderer: all dl_front.quicklook arguments pass
# through, e.g.
#   sbatch slurm/dlfront_quicklook.sbatch swath-bank
#   sbatch slurm/dlfront_quicklook.sbatch kriged-degraded --years 2007-2015
#   sbatch slurm/dlfront_quicklook.sbatch kriged-airs     --years 2007-2021
# The chain script submits one of these after each CPU build phase
# (swath bank / 2a / 3a) so every dataset lands with human-checkable maps.
# Output: $JPL_AIRS_RESULTS/dl_front/quicklook/<product>/*.png

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python -m dl_front.quicklook "$@"
