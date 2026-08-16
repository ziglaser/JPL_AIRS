#!/bin/bash
#SBATCH --job-name=dlfront-krige
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dlfront_krige_%j.out
# CPU-only kriged-cache builder: all dl_front.krige_fill arguments pass
# through, e.g.
#   sbatch slurm/dlfront_krige.sbatch build-degraded --years 2007-2015 --workers 8
#   sbatch slurm/dlfront_krige.sbatch build-airs     --years 2007-2021 --workers 8
# Output (manifest reorg 2026-08-13):
#   $JPL_AIRS_DATA/front_id/{degraded_reanalysis,kriged_airs_fcst}/
#   kriged_sfc_<year>.nc
# Keep --workers <= --cpus-per-task above.

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
# one BLAS thread per Pool worker: the 8 forked --workers each spawn
# node-core-count OpenBLAS threads by default, oversubscribing the 8-CPU
# cgroup and slowing the dense OrdinaryKriging solves several-fold
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python -m dl_front.krige_fill "$@"
