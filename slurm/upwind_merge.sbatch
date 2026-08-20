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

: "${UPWIND_YEAR:?set UPWIND_YEAR, e.g. UPWIND_YEAR=2019 sbatch slurm/upwind_merge.sbatch}"

python scripts/merge_upwind_features.py --year "$UPWIND_YEAR" "$@"
