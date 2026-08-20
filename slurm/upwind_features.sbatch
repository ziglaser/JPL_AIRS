#!/bin/bash
#SBATCH --job-name=upwind-feat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/upwind_features_%A.out
#SBATCH --open-mode=append
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
