#!/bin/bash
#SBATCH --job-name=dlfront-export
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dlfront_export_%j.out
# Export trained checkpoints as BK19-schema hard-class prediction rasters
# (deliverable 1, 2026-08-17): all dl_front.export_predictions arguments pass
# through, e.g.
#   sbatch slurm/dlfront_export.sbatch \
#       --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f0/D6C-f0.h5" \
#       --source kriged-airs --years 2016-2021
#   # three folds -> per-fold archives PLUS the softmax-averaged D6C-ens3:
#   sbatch slurm/dlfront_export.sbatch \
#       --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f0/D6C-f0.h5" \
#       --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f1/D6C-f1.h5" \
#       --ckpt "$JPL_AIRS_RESULTS/dl_front/models/D6C-f2/D6C-f2.h5" \
#       --source kriged-airs
# CPU-ok by design (inference only, no --gres line; partition/account/gres are
# injected at submit time via SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_GRES
# -- do not hardcode cluster names here).  To run on a GPU node, submit it
# yourself with `sbatch --gres=gpu:1 slurm/dlfront_export.sbatch ...`.
# Idempotent: a year whose .nc AND _run.json already exist is skipped unless
# --force is passed, so a requeued job resumes where it stopped.
# Output: $JPL_AIRS_DATA/front_id/predicted_fronts/dlfront_<ckpt>_<source>/
#         1deg_3wide/3hr/merra2_merra2-1deg_3wide_3hr_<year>.nc (+ _run.json)

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python -m dl_front.export_predictions "$@"
