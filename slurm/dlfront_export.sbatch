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

python -m dl_front.export_predictions "$@"
