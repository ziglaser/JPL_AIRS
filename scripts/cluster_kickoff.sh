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
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fronts-tf

log(){ echo "[kickoff $(date '+%F %T')] $*"; }

git submodule update --init fronts   # no-op if already populated

NFULL=$(find "$JPL_AIRS_FCST" -name 'fullgrid_*' -type f | wc -l)
[ "$NFULL" -gt 0 ] || { log "FATAL: no fullgrid_* files under $JPL_AIRS_FCST"; exit 1; }
log "fullgrid archive: $NFULL files"

BANK="$JPL_AIRS_DATA/masks/gap_bank.npz"
bank_size(){ python - "$BANK" 2>/dev/null <<'PY' || echo 0
import sys, numpy as np
print(len(np.load(sys.argv[1])["date"]))
PY
}
if [ "$(bank_size)" -lt 30 ]; then
    log "harvesting gap bank from $NFULL fullgrids (this is the slow part)..."
    python -c "import os; from pathlib import Path; from front_finder import mask_bank; mask_bank.harvest(sorted(Path(os.environ['JPL_AIRS_FCST']).rglob('fullgrid_*')))"
fi
SZ=$(bank_size)
[ "$SZ" -ge 30 ] || { log "FATAL: gap bank size $SZ < 30 after harvest"; exit 1; }
log "gap bank OK: $SZ fields"

EXTRA=()
if [ ! -d "$JPL_AIRS_DATA/front_id/reanalysis/MERRA2/sfc_daily/2016" ]; then
    if [ -f "$HOME/.netrc" ]; then
        EXTRA+=(--with-acquire); log "sfc_daily 2016 missing -> adding --with-acquire"
    else
        log "WARNING: sfc_daily 2016-2018 missing and no ~/.netrc; reanalysis eval legs will fail (trainings unaffected; rerun chain after acquiring)"
    fi
fi

log "submitting chain..."
bash scripts/dlfront_jpl_chain.sh --with-swath-bank ${EXTRA[@]+"${EXTRA[@]}"}
log "submitted. queue snapshot:"
squeue -u "$USER" -o "%.10i %.16j %.8T %r"
log "DONE"
