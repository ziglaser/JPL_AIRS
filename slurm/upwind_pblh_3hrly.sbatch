#!/bin/bash
#SBATCH --job-name=upwind-pblh
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/upwind_pblh_%j.out
# One-shot preliminary: pool the raw Guo et al. (2024) 0.25-degree merged-PBLH
# tree ($JPL_AIRS_DATA/PBL_depth/Guo2024_model/) to the 1-degree 3-hourly
# aggregate every downstream upwind job reads (GriddedPBL contact gate + m*,
# and gamma_gap = z_LFC - z_i; UPWIND_INDEX_REVIEW.md sections 1.5, 1.8, 4.1d).
# Extra args pass through to scripts/build_pblh_3hrly_1deg.py (e.g. --years).
# Partition/account/gres are injected at submit time via SBATCH_PARTITION /
# SBATCH_ACCOUNT / SBATCH_GRES -- do not hardcode cluster names here. CPU-only.
# Idempotent: per-year (sum, count) blocks are cached under the output
# directory's _cache/, so a requeued job resumes instead of re-reading source
# files.
#
# Full 2017-2021 production run, in order (this job first, once):
#   export UPWIND_TRAJ_ROOT=/path/to/HYSPLIT/per_day_archive   # once, for the
#                                    # features array (not read by this job)
#   sbatch slurm/upwind_pblh_3hrly.sbatch
#   for Y in 2017 2018 2019 2020 2021; do
#       UPWIND_YEAR=$Y sbatch --array=0-274%40 slurm/upwind_features.sbatch
#   done
#   for Y in 2017 2018 2019 2020 2021; do
#       UPWIND_YEAR=$Y sbatch slurm/upwind_merge.sbatch \
#           --daily-dir "${JPL_AIRS_RESULTS:-results}/upwind_features/daily"
#   done
# The merge's --daily-dir matches the features array's --out-dir default.
# 2016 has no assessed PBLH (the Guo product starts 2017), so its gamma_gap
# and pblh_anom are honestly NaN -- nothing to build or backfill here.

set -euo pipefail
# conda bootstrap, robust to install location: explicit CONDA_PREFIX_ROOT wins;
# else ask the conda on PATH (sbatch propagates the submission environment);
# else probe the common install dirs. Fail with a pointed message, not line-43
# noise from a wrong hardcoded path.
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
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python scripts/build_pblh_3hrly_1deg.py --jobs 10 "$@"
