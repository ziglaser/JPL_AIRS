# JPL_AIRS environment bootstrap -- source this, never execute it.
#
#   source /path/to/JPL_AIRS/slurm/env.sh
#
# Fills in every JPL_AIRS_* root the pipelines need, probing the known
# machine layouts. EXPLICIT EXPORTS ALWAYS WIN: any variable already set is
# left untouched, so a deliberate override (an ablation against different
# data, a scratch checkout) still works. A variable whose probe finds nothing
# is left UNSET, so downstream :?-guards and Python fallbacks keep their own
# clear failure messages instead of inheriting an empty string.
#
# Known layouts probed, in order:
#   cluster (gattaca2):  /gpfs/scratch/smap-convection/{AIRS_SMAP_Front_data,
#                        AIRS_FCST_1deg}
#   dev (WSL2, the "My Passport" drive):  /mnt/d/JPL_AIRS/data, with the
#                        HYSPLIT demo day standing in for the FCST archive
#
# Every slurm/*.sbatch sources this after inferring JPL_AIRS_REPO from
# SLURM_SUBMIT_DIR, so batch jobs need NO exports when submitted from the
# checkout. For interactive shells, add the source line to ~/.bashrc.

_jpl_first_dir() {
    local d
    for d in "$@"; do
        if [ -n "$d" ] && [ -d "$d" ]; then printf '%s' "$d"; return 0; fi
    done
    return 1
}

# the repo root: this file's own parent-of-parent (works however it is sourced)
if [ -z "${JPL_AIRS_REPO:-}" ]; then
    JPL_AIRS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export JPL_AIRS_REPO

if [ -z "${JPL_AIRS_DATA:-}" ]; then
    if _d="$(_jpl_first_dir /gpfs/scratch/smap-convection/AIRS_SMAP_Front_data \
                            /mnt/d/JPL_AIRS/data)"; then
        export JPL_AIRS_DATA="$_d"
    fi
fi

if [ -z "${JPL_AIRS_FCST:-}" ]; then
    if _d="$(_jpl_first_dir /gpfs/scratch/smap-convection/AIRS_FCST_1deg \
                            "${JPL_AIRS_DATA:+$JPL_AIRS_DATA/HYSPLIT_demo}")"; then
        export JPL_AIRS_FCST="$_d"
    fi
fi

# the upwind per-day trajectory archive IS the FCST_1deg tree (confirmed
# 2026-08-19: <root>/YYYY/wrf27km_<YYYYMMDD> holds the nogrid parcel files)
if [ -z "${UPWIND_TRAJ_ROOT:-}" ] && [ -n "${JPL_AIRS_FCST:-}" ]; then
    export UPWIND_TRAJ_ROOT="$JPL_AIRS_FCST"
fi

export JPL_AIRS_RESULTS="${JPL_AIRS_RESULTS:-$JPL_AIRS_REPO/results}"

unset -f _jpl_first_dir
unset _d
