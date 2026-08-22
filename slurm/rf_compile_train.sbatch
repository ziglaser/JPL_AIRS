#!/bin/bash
#SBATCH --job-name=rf-compile-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rf_compile_train_%j.out
# One CPU job, two phases (user spec 2026-08-22):
#
#   Phase 1  scripts/compile_rf_dataset.py: build the ONE big RF-ready netCDF
#            (all of FCST_SMAP_MRMS + sm_anom + UPW_psi_anom/UPW_omega +
#            time-INTERPOLATED UPW_pblh/UPW_pblh_anom/UPW_gamma_gap_mu +
#            met_front_*_3w (NOAA analyst, incl. drylines) +
#            pred_front_*_3w (D6C kriged-AIRS ensemble)) at
#            <gpfs scratch root>/RF_dataset_2016-2021.nc, and print/write the
#            categorised variable list.  IDEMPOTENT: if the file already
#            exists the script prints the list and exits 0 immediately, so
#            this job always proceeds straight to phase 2.
#   Phase 2  python -m convection_skill.rf_cluster: the 13 block experiments
#            (full met/pred swap, leave-one-block-out, each block solo;
#            met and pred fronts never co-present; thermo = MU-only) in all
#            three model modes (wide / pooled / perhour), Gini-scored at the
#            P95 and P99.5 set points.  Resumable per experiment.
#
# Prerequisites on the cluster (fail loudly, checked by the scripts):
#   - FCST_SMAP_MRMS year files under $JPL_AIRS_DATA/FCST_SMAP_MRMS/
#   - per-day kernel sweep output UPW_<YYYYMMDD>.nc under
#     $JPL_AIRS_RESULTS/upwind_features/daily/ (psi/omega; user confirmed
#     these already exist -- missing days come through as honest NaN)
#   - assessed PBLH 3-hourly + climatology under $JPL_AIRS_DATA/PBL_depth/derived/
#   - NOAA 3wide front labels (met_drawn_fronts/NOAA_CODSUS/...)
#   - the kriged-AIRS D6C ensemble archive
#     $JPL_AIRS_DATA/front_id/predicted_fronts/dlfront_D6C-ens3_kriged-airs/
#     (the full-sequence chain's step-6 export)
#
# Submit from the repo root on gattaca2 (no exports needed; slurm/env.sh
# probes the machine layout):
#   sbatch slurm/rf_compile_train.sbatch
# Env knobs: RF_YEARS (default 2016-2021), RF_MODES (default
# wide,pooled,perhour), RF_DATASET (override the output/input path),
# RF_FORCE=1 (rebuild the big file AND rerun all experiments).
# Partition/account are injected at submit time via SBATCH_PARTITION /
# SBATCH_ACCOUNT -- do not hardcode cluster names here.  CPU-only by design:
# the forests parallelise over --cpus-per-task via n_jobs=-1.

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
command -v conda >/dev/null 2>&1 || { echo "conda not found: set CONDA_PREFIX_ROOT" >&2; exit 1; }
conda activate fronts-tf

# repo + data roots: explicit exports win; otherwise the repo is where sbatch
# was invoked (SLURM_SUBMIT_DIR) and slurm/env.sh probes the known layouts.
JPL_AIRS_REPO="${JPL_AIRS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$JPL_AIRS_REPO/slurm/env.sh" ]]; then
    echo "JPL_AIRS_REPO=$JPL_AIRS_REPO is not the repo checkout: submit from" \
         "the repo root or export JPL_AIRS_REPO" >&2
    exit 1
fi
source "$JPL_AIRS_REPO/slurm/env.sh"
cd "$JPL_AIRS_REPO"
export PYTHONPATH=src
# keep BLAS single-threaded; sklearn's n_jobs=-1 owns the cgroup's CPUs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "JPL_AIRS_DATA=${JPL_AIRS_DATA:-<unset: repo-local data/>}" \
     "JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-<unset: repo-local results/>}"

RF_YEARS=${RF_YEARS:-2016-2021}
RF_MODES=${RF_MODES:-wide,pooled,perhour}
Y0=${RF_YEARS%-*} Y1=${RF_YEARS#*-}
SCRATCH_ROOT=$(dirname "${JPL_AIRS_DATA:?slurm/env.sh did not resolve JPL_AIRS_DATA}")
RF_DATASET=${RF_DATASET:-$SCRATCH_ROOT/RF_dataset_${Y0}-${Y1}.nc}
FORCE_FLAG=""
[[ "${RF_FORCE:-0}" = 1 ]] && FORCE_FLAG="--force"

echo "== phase 1: compile $RF_DATASET (skips itself if present) =="
python scripts/compile_rf_dataset.py --years "$RF_YEARS" \
    --out "$RF_DATASET" $FORCE_FLAG

echo "== phase 2: RF block experiments (modes: $RF_MODES) =="
python -m convection_skill.rf_cluster --dataset "$RF_DATASET" \
    --modes "$RF_MODES" $FORCE_FLAG

echo "== done: results under \$JPL_AIRS_RESULTS/rf_cluster/<mode>/ =="
