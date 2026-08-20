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
     "JPL_AIRS_FCST=${JPL_AIRS_FCST:-<unset: data/HYSPLIT_demo>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"
# Force CPU-friendly TF behavior if no GPU was allocated (TF just warns).

python -m dl_front.evaluate_test "$@"
