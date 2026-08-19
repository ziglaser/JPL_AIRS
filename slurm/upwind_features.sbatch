#!/bin/bash
#SBATCH --job-name=upwind-feat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/upwind_features_%A_%a.out
# Per-day upwind soil-moisture features (UPWIND_INDEX_REVIEW.md section 4.2):
# build the trajectory kernels for one day's HYSPLIT archive and reduce them
# to the per-day predictor file (psi_anom, phi, m_star, omega, gamma_gap, ...).
# Array index -> date: task N (0-based) is the N-th day counting 1 March
# through 30 November of $UPWIND_YEAR -- the Mar-Nov season SMAP_L4 covers,
# 275 days. Submit one year as e.g.
#   export UPWIND_TRAJ_ROOT=/path/to/HYSPLIT/per_day_archive
#   UPWIND_YEAR=2019 sbatch --array=0-274%40 slurm/upwind_features.sbatch
# UPWIND_TRAJ_ROOT is the HYSPLIT per-day archive root (the directory holding
# the wrf27km_<YYYYMMDD> day dirs); this script forwards it as --traj-root.
# Extra args pass through to scripts/build_upwind_features.py and are placed
# AFTER --traj-root, so a CLI --traj-root still wins (argparse keeps the last
# occurrence of a plain store option). Partition/account/gres are injected at
# submit time via SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_GRES -- do not
# hardcode cluster names here. CPU-only.
# Idempotent: the script skips a date whose per-day output already exists, so
# a requeued array resumes where it stopped.
# 2016 has no assessed PBLH (the Guo product starts 2017), so a 2016 sweep
# gets honestly-NaN gamma_gap and pblh_anom -- the trajectory features still
# build.
#
# Full 2017-2021 production run, in order:
#   export UPWIND_TRAJ_ROOT=/path/to/HYSPLIT/per_day_archive   # once
#   sbatch slurm/upwind_pblh_3hrly.sbatch                      # once
#   for Y in 2017 2018 2019 2020 2021; do
#       UPWIND_YEAR=$Y sbatch --array=0-274%40 slurm/upwind_features.sbatch
#   done
#   for Y in 2017 2018 2019 2020 2021; do
#       UPWIND_YEAR=$Y sbatch slurm/upwind_merge.sbatch \
#           --daily-dir "${JPL_AIRS_RESULTS:-results}/upwind_features/daily"
#   done
# The merge's --daily-dir must be this job's --out-dir; the recipe passes the
# features step's default (<results>/upwind_features/daily) explicitly because
# merge_upwind_features.py defaults --daily-dir to None (trajectory-free-only).

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

: "${UPWIND_YEAR:?set UPWIND_YEAR, e.g. UPWIND_YEAR=2019 sbatch --array=0-274%40 ...}"
: "${UPWIND_TRAJ_ROOT:?set UPWIND_TRAJ_ROOT to the HYSPLIT per-day archive root}"
DATE=$(date -d "$UPWIND_YEAR-03-01 + $SLURM_ARRAY_TASK_ID days" +%F)
if [[ "$DATE" > "$UPWIND_YEAR-11-30" ]]; then
    echo "task $SLURM_ARRAY_TASK_ID -> $DATE is past Nov 30; nothing to do"
    exit 0
fi
echo "task $SLURM_ARRAY_TASK_ID -> $DATE"

# "$@" after --traj-root: argparse keeps the last occurrence of a plain store
# option, so an explicit CLI --traj-root passed at submit time overrides.
python scripts/build_upwind_features.py --date "$DATE" \
    --traj-root "$UPWIND_TRAJ_ROOT" "$@"
