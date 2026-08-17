#!/bin/bash
# One-shot cluster kickoff (gattaca2): gap-bank harvest (if needed) -> submit
# the full dl_front chain.  Run detached so it survives logout:
#   setsid nohup bash scripts/cluster_kickoff.sh > logs/kickoff.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

export JPL_AIRS_DATA=${JPL_AIRS_DATA:-/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data}
export JPL_AIRS_FCST=${JPL_AIRS_FCST:-/gpfs/scratch/smap-convection/AIRS_FCST_1deg}
export JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-$PWD/results}
unset SBATCH_PARTITION
export SBATCH_GPU_PARTITION=${SBATCH_GPU_PARTITION:-gpu}
export PYTHONPATH=src

CONDA_ROOT=${CONDA_PREFIX_ROOT:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
# sbatch jobs re-source conda from CONDA_PREFIX_ROOT (slurm/dlfront_*.sbatch)
# and inherit it via --export=ALL -- must be exported or every job dies at
# "conda.sh: No such file or directory" on clusters where conda isn't at
# ~/miniconda3 (gattaca2 post-mortem 2026-08-15, job 572450).
export CONDA_PREFIX_ROOT="$CONDA_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fronts-tf

log(){ echo "[kickoff $(date '+%F %T')] $*"; }

git submodule update --init fronts   # no-op if already populated

NFULL=$(find "$JPL_AIRS_FCST" -name 'fullgrid_*' -type f | wc -l)
[ "$NFULL" -gt 0 ] || { log "FATAL: no fullgrid_* files under $JPL_AIRS_FCST"; exit 1; }
log "fullgrid archive: $NFULL files"

# (the old front_finder gap-bank harvest step is GONE: stage-B fallback
# masks now come from masks/sfc_gap_bank.npz, harvested by the chain's own
# --with-swath-bank phase with the terrain-following extraction, 2026-08-16)
if [ ! -e "$JPL_AIRS_DATA/masks/surface_elevation.nc" ]; then
    log "building surface elevation map (terrain-following extraction needs it)..."
    python scripts/build_surface_elevation.py
fi

# sfc_daily completeness, per year (a bare dir-existence test is repair-blind:
# acquire creates the year dir before its first download succeeds).  build-airs
# kriges 2007-2021 and hard-requires the clean-SLP reanalysis step for every
# built day -- a hole doesn't just weaken eval legs, it kills phase 3a and
# dependency-cancels D6C + ALL evals + compare.  365 approximates leap years.
SFC="$JPL_AIRS_DATA/front_id/reanalysis/MERRA2/sfc_daily"
PRE_MISSING=() ACQ_MISSING=()
for y in $(seq 2007 2021); do
    n=$(find "$SFC/$y" -name 'm2sfc_*' -type f 2>/dev/null | wc -l)
    if [ "$n" -lt 365 ]; then
        if [ "$y" -ge 2016 ]; then ACQ_MISSING+=("$y($n)")
        else PRE_MISSING+=("$y($n)"); fi
    fi
done
if [ ${#PRE_MISSING[@]} -gt 0 ]; then
    log "FATAL: sfc_daily incomplete for pre-2016 years: ${PRE_MISSING[*]}."
    log "Phase 0 only fetches 2016-2021; the archive itself needs fixing. Not submitting."
    exit 1
fi
EXTRA=()
if [ ${#ACQ_MISSING[@]} -gt 0 ]; then
    if [ -f "$HOME/.netrc" ]; then
        EXTRA+=(--with-acquire)
        log "sfc_daily incomplete for ${ACQ_MISSING[*]} -> adding --with-acquire"
    else
        log "FATAL: sfc_daily incomplete for ${ACQ_MISSING[*]} and no ~/.netrc."
        log "build-airs 2007-2021 would crash mid-run and cancel D6C + all evals. Not submitting."
        exit 1
    fi
fi

log "submitting chain..."
bash scripts/dlfront_jpl_chain.sh --with-swath-bank ${EXTRA[@]+"${EXTRA[@]}"}
log "submitted. queue snapshot:"
squeue -u "$USER" -o "%.10i %.16j %.8T %r"
log "DONE"
