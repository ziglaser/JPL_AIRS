#!/bin/bash
#SBATCH --job-name=dlfront-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dlfront_train_%j.out
# Generic dl_front training step: all dl_front.train arguments pass through:
#   sbatch slurm/dlfront_train.sbatch --name D6A-f0 --classes 6 --fold 0 \
#       --source reanalysis
# Partition / account / gres overrides are injected at submit time by
# scripts/dlfront_jpl_chain.sh (SBATCH_PARTITION / SBATCH_ACCOUNT /
# SBATCH_GRES env vars) -- do not hardcode cluster names here.
# Checkpoints land in $JPL_AIRS_RESULTS/dl_front/models/<name>/:
#   <name>.h5        best-so-far (saved on every val improvement)
#   <name>_final.h5  end-of-run weights (also the chain's done-marker)

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
# JPL_AIRS_DATA / JPL_AIRS_FCST / JPL_AIRS_RESULTS: exported by the chain
# script (which submits with --export=ALL so a cluster-side --export=NONE
# default cannot strip them).  Echo the resolved roots so a job that DID
# lose them (manual submit on a stripped-env cluster) is diagnosable from
# the log instead of silently using repo-local defaults.
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python -m dl_front.train "$@"
