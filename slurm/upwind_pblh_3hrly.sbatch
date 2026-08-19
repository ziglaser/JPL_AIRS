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
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

python scripts/build_pblh_3hrly_1deg.py --jobs 10 "$@"
