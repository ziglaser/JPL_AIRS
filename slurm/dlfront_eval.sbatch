#!/bin/bash
#SBATCH --job-name=dlfront-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dlfront_eval_%j.out
# Held-out test evaluation (2016-2018, user decision 2026-08-13): all
# dl_front.evaluate_test arguments
# pass through, e.g.
#   sbatch slurm/dlfront_eval.sbatch \
#       --ckpt results/dl_front/models/D6C-f0/D6C-f0.h5 --classes 6 \
#       --source kriged-airs --years 2016-2018
#   sbatch slurm/dlfront_eval.sbatch --source bk19 --classes 6   # BK19 leg
#   sbatch slurm/dlfront_eval.sbatch compare                     # pivot table
# CPU-ok by design (inference only, no --gres line).  The chain script
# submits eval jobs WITHOUT a GPU (SBATCH_GRES only affects the training
# jobs); to run an eval on a GPU node, submit it yourself with
# `sbatch --gres=gpu:1 slurm/dlfront_eval.sbatch ...`.
# Output: $JPL_AIRS_RESULTS/dl_front/test_eval/<ckpt-stem>_<source>.csv
#         + <ckpt-stem>_<source>_paper.json

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"
# Force CPU-friendly TF behavior if no GPU was allocated (TF just warns).

python -m dl_front.evaluate_test "$@"
