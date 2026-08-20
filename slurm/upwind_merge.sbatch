#!/bin/bash
#SBATCH --job-name=upwind-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/upwind_merge_%j.out
# Per-year merge (UPWIND_INDEX_REVIEW.md section 4.2, final step): concatenate
# the per-day upwind predictor files for $UPWIND_YEAR onto the FCST_SMAP_MRMS
# (date, time, lat, lon) axes as UPW_* variables. Submit one year as
#   UPWIND_YEAR=2019 sbatch slurm/upwind_merge.sbatch
# --daily-dir defaults to the features array's own --out-dir default
# (<results>/upwind_features/daily), so the handoff cannot silently disagree;
# override both together if you moved it. --no-daily is the explicit
# trajectory-free mode (kernel-borne features NaN).
# Extra args pass through to scripts/merge_upwind_features.py. Partition/
# account/gres are injected at submit time via SBATCH_PARTITION /
# SBATCH_ACCOUNT / SBATCH_GRES -- do not hardcode cluster names here.
# CPU-only. Idempotent: rerunning simply rewrites the year's merged file from
# whatever per-day files exist; days the array never produced stay NaN.
# 2016 has no assessed PBLH (the Guo product starts 2017), so its gamma_gap
# and pblh_anom columns are honestly NaN.
#
# Full 2017-2021 production run, in order:
#   export UPWIND_TRAJ_ROOT=/path/to/HYSPLIT/per_day_archive   # once
#   sbatch slurm/upwind_pblh_3hrly.sbatch                      # once
#   for Y in 2017 2018 2019 2020 2021; do
#       UPWIND_YEAR=$Y sbatch --array=0-274%40 slurm/upwind_features.sbatch
#   done
#   for Y in 2017 2018 2019 2020 2021; do
#       UPWIND_YEAR=$Y sbatch slurm/upwind_merge.sbatch
#   done
#
# Pass 2 (AFTER all years are merged) -- the multi-year Omega climatology and
# the slot-comparable UPW_omega_anom (ablation tier, not auto-included in the
# forest; multi-year ONLY by design -- a within-year base would erase the
# interannual signal). Cheap (minutes), run on the login node or as one job:
#   python scripts/merge_upwind_features.py --build-omega-clim   # >= 2 years
#   for Y in 2017 2018 2019 2020 2021; do
#       python scripts/merge_upwind_features.py --year $Y --add-omega-anom
#   done

set -euo pipefail
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate fronts-tf
cd "${JPL_AIRS_REPO:?set JPL_AIRS_REPO to the repo checkout}"
export PYTHONPATH=src
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

: "${UPWIND_YEAR:?set UPWIND_YEAR, e.g. UPWIND_YEAR=2019 sbatch slurm/upwind_merge.sbatch}"

python scripts/merge_upwind_features.py --year "$UPWIND_YEAR" "$@"
