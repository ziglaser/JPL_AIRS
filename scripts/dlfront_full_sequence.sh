#!/bin/bash
# scripts/dlfront_full_sequence.sh -- run the whole post-label-fix deployment
# (runbook steps 4-7) unattended, in sequence, with no human intervention
# between phases:
#
#   Step 4  scripts/cluster_kickoff.sh (preflight checks + submit the full
#           main chain: swath bank -> krige builds -> D6A/B/C x 3 folds ->
#           evals -> compare), then WAIT until every fold's three stage
#           checkpoints and the kriged-airs caches exist.
#   Step 5  scripts/dlfront_ablation_chain.sh (permutation importance on the
#           NEW checkpoints + the D6A5/D6A3/D6A2 channel ladder).  Submitted
#           as soon as the main chain's trainings are done; it runs
#           concurrently with steps 6-7 (nothing downstream reads it) and is
#           only WAITED on at the very end.
#   Step 6  prediction export 2016-2021 (slurm/dlfront_export.sbatch, D6C
#           three folds + softmax ensemble, --source kriged-airs).  GPU is
#           OPPORTUNISTIC: the job is CPU-capable by design (inference only),
#           so it is submitted with --gres=gpu:1 only when the GPU partition
#           has an idle node RIGHT NOW -- otherwise it runs on CPU instead of
#           queueing behind training jobs.  WAITs on the ensemble archive.
#   Step 7  scripts/add_front_flags.py: inject the met-drawn (1w+3w) and
#           predicted per-front-type binary flags into copies of the
#           FCST_SMAP_MRMS year files (foreground -- pure xarray, minutes).
#
# Run detached on the gattaca2 login node (survives logout; the whole
# sequence is ~1.5-2 days dominated by training):
#
#   setsid nohup bash scripts/dlfront_full_sequence.sh \
#       > logs/full_sequence.log 2>&1 &
#
# Progress: tail -f logs/full_sequence.log; squeue -u $USER.
#
# Failure model: each WAIT polls for the phase's done-marker FILES while
# watching that phase's SLURM jobs; if every watched job has left the queue
# and the files still do not exist, the sequence aborts loudly (a failed
# train job dependency-cancels its downstream jobs, so the queue draining
# without artifacts IS the failure signal).  Static prerequisites that step
# 7 needs (the FCST_SMAP_MRMS primaries) are checked up front, so a missing
# input fails in seconds, not after two days of training.
#
# Idempotent: every phase it drives is itself idempotent (done-markers with
# labels_sha1/ckpt_sha1 staleness checks), so rerunning this script after a
# partial failure resumes where it stopped instead of redoing finished work.
#
# Env knobs (all optional):
#   POLL_SECS            poll interval while waiting          (default 300)
#   MAIN_TIMEOUT_H       step-4 wait budget, hours            (default 72)
#   EXPORT_TIMEOUT_H     step-6 wait budget, hours            (default 12)
#   ABLATION_TIMEOUT_H   step-5 final wait budget, hours      (default 36)
#   EXPORT_YEARS         export + inject span                 (default 2016-2021)
#   FOLDS                fold list, matches the chains        (default "0 1 2")
#   plus everything cluster_kickoff.sh / the chains read (JPL_AIRS_DATA,
#   SBATCH_GPU_PARTITION, CONDA_PREFIX_ROOT, ...).
set -euo pipefail
cd "$(dirname "$0")/.."

export JPL_AIRS_REPO=${JPL_AIRS_REPO:-$PWD}
export JPL_AIRS_DATA=${JPL_AIRS_DATA:-/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data}
export JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-$PWD/results}
export SBATCH_GPU_PARTITION=${SBATCH_GPU_PARTITION:-gpu}
export PYTHONPATH=src

POLL_SECS=${POLL_SECS:-300}
MAIN_TIMEOUT_H=${MAIN_TIMEOUT_H:-72}
EXPORT_TIMEOUT_H=${EXPORT_TIMEOUT_H:-12}
ABLATION_TIMEOUT_H=${ABLATION_TIMEOUT_H:-36}
EXPORT_YEARS=${EXPORT_YEARS:-2016-2021}
FOLDS=${FOLDS:-0 1 2}
Y0=${EXPORT_YEARS%-*} Y1=${EXPORT_YEARS#*-}

MODELS=$JPL_AIRS_RESULTS/dl_front/models
KRIGED_AIRS=$JPL_AIRS_DATA/front_id/kriged_airs_fcst
ENS_TAG=dlfront_D6C-ens3_kriged-airs   # export_predictions.ensemble_stem for
                                       # the three D6C folds + export_tag
ENS_DIR=$JPL_AIRS_DATA/front_id/predicted_fronts/$ENS_TAG/1deg_3wide/3hr

log(){ echo "[sequence $(date '+%F %T')] $*"; }
die(){ log "FATAL: $*"; exit 1; }
mkdir -p logs

HAVE_SLURM=0
command -v sbatch > /dev/null 2>&1 && HAVE_SLURM=1

# newest_manifest <glob-prefix>: the chains stamp their manifests with a
# timestamp, so the lexically-last match is the run this script just started
newest_manifest(){ ls -1 "$JPL_AIRS_RESULTS"/dl_front/"$1"_*.txt 2>/dev/null | tail -1; }
# manifest_jids <manifest>: the numeric SLURM job ids it recorded (label=jid
# lines; header comments and local-done/DRY_RUN entries are not job ids and
# drop out -- header lines like "force=0" would otherwise match)
manifest_jids(){
    [ -n "$1" ] && grep -v '^#' "$1" | grep -oE '=[0-9]+$' | tr -d = | paste -sd, -
}

# wait_for <label> <timeout-h> <jids-comma-or-empty> <file...>: poll until
# every file exists.  While jobs are known, a drained queue with files still
# missing (after one grace poll for filesystem lag) is a failure; with no
# jobs to watch (no-SLURM foreground runs finished before we got here) the
# files must already exist.
wait_for(){
    local label=$1 timeout_h=$2 jids=$3 grace=0 missing f
    shift 3
    local deadline=$(( $(date +%s) + timeout_h * 3600 ))
    while :; do
        missing=""
        for f in "$@"; do [ -e "$f" ] || { missing=$f; break; }; done
        if [ -z "$missing" ]; then log "$label: complete"; return 0; fi
        [ "$(date +%s)" -lt "$deadline" ] \
            || die "$label: timed out after ${timeout_h}h (first missing: $missing)"
        if [ -z "$jids" ] || [ "$HAVE_SLURM" = 0 ]; then
            die "$label: no jobs to wait on and artifacts missing ($missing)"
        fi
        if [ "$(squeue -h -j "$jids" 2>/dev/null | wc -l)" = 0 ]; then
            grace=$((grace + 1))
            [ "$grace" -ge 2 ] && die "$label: every watched job left the" \
                "queue but artifacts are missing (first: $missing) --" \
                "a job failed; check sacct -j $jids and logs/"
        else
            grace=0
        fi
        sleep "$POLL_SECS"
    done
}

# test hook: `FULLSEQ_SOURCE_ONLY=1 source <this file>` loads just the
# helpers above (for the shell-level tests) without activating conda or
# running the sequence
if [ "${FULLSEQ_SOURCE_ONLY:-0}" = 1 ]; then return 0 2>/dev/null || exit 0; fi

CONDA_ROOT=${CONDA_PREFIX_ROOT:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
export CONDA_PREFIX_ROOT="$CONDA_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fronts-tf

# ---- static preflight for the LAST step, checked FIRST ------------------- #
for ((y = Y0; y <= Y1; y++)); do
    compgen -G "$JPL_AIRS_DATA/FCST_SMAP_MRMS/*${y}*" > /dev/null \
        || die "no FCST_SMAP_MRMS primary file for $y under" \
               "$JPL_AIRS_DATA/FCST_SMAP_MRMS -- step 7 would fail after" \
               "days of compute; fix this first"
done
log "preflight ok: FCST_SMAP_MRMS primaries present for $EXPORT_YEARS"

# ---- step 4: main chain --------------------------------------------------- #
log "step 4: cluster_kickoff (preflight + main-chain submission)"
bash scripts/cluster_kickoff.sh
MAIN_MANIFEST=$(newest_manifest chain)
MAIN_JIDS=$(manifest_jids "$MAIN_MANIFEST" || true)
log "main-chain manifest: ${MAIN_MANIFEST:-<none>} jobs: ${MAIN_JIDS:-<none>}"

MAIN_TARGETS=()
for k in $FOLDS; do
    for s in A B C; do
        MAIN_TARGETS+=("$MODELS/D6$s-f$k/D6$s-f${k}_final.h5")
    done
done
for ((y = Y0; y <= Y1; y++)); do
    MAIN_TARGETS+=("$KRIGED_AIRS/kriged_sfc_$y.nc")
done
wait_for "step4 main chain (9 checkpoints + kriged-airs $EXPORT_YEARS)" \
         "$MAIN_TIMEOUT_H" "${MAIN_JIDS:-}" "${MAIN_TARGETS[@]}"

# ---- step 5: ablation chain (runs concurrently with 6-7) ------------------ #
log "step 5: ablation chain (permutation on the new checkpoints + channel ladder)"
bash scripts/dlfront_ablation_chain.sh
ABL_MANIFEST=$(newest_manifest ablation)
ABL_JIDS=$(manifest_jids "$ABL_MANIFEST" || true)
log "ablation manifest: ${ABL_MANIFEST:-<none>} jobs: ${ABL_JIDS:-<none>}"

# ---- step 6: prediction export (opportunistic GPU, CPU fallback) ---------- #
EXPORT_ARGS=(--ckpt "$MODELS/D6C-f0/D6C-f0.h5"
             --ckpt "$MODELS/D6C-f1/D6C-f1.h5"
             --ckpt "$MODELS/D6C-f2/D6C-f2.h5"
             --source kriged-airs --years "$EXPORT_YEARS")
EXPORT_TARGETS=()
for ((y = Y0; y <= Y1; y++)); do
    EXPORT_TARGETS+=("$ENS_DIR/merra2_merra2-1deg_3wide_3hr_$y.nc"
                     "$ENS_DIR/merra2_merra2-1deg_3wide_3hr_${y}_run.json")
done
if [ "$HAVE_SLURM" = 1 ]; then
    # the export job is CPU-capable by design (inference only); take a GPU
    # only when one is guaranteed free THIS instant (a fully idle node --
    # `mixed` nodes may have both GPUs busy), otherwise a CPU submission
    # starts immediately instead of queueing behind training jobs
    GPU_OPTS=()
    if sinfo -h -p "$SBATCH_GPU_PARTITION" -t idle -o %n 2>/dev/null | grep -q .; then
        GPU_OPTS=(-p "$SBATCH_GPU_PARTITION" --gres=gpu:1)
        log "step 6: idle node in '$SBATCH_GPU_PARTITION' -> exporting on GPU"
    else
        log "step 6: no idle GPU node -> exporting on CPU (job is CPU-capable)"
    fi
    EXPORT_JID=$(sbatch --parsable --export=ALL ${GPU_OPTS[@]+"${GPU_OPTS[@]}"} \
                 slurm/dlfront_export.sbatch "${EXPORT_ARGS[@]}")
    log "step 6: export submitted -> job $EXPORT_JID"
else
    log "step 6: no SLURM -> exporting in the foreground"
    EXPORT_JID=""
    python -m dl_front.export_predictions "${EXPORT_ARGS[@]}" \
        > logs/full_sequence_export.log 2>&1
fi
wait_for "step6 export ($ENS_TAG $EXPORT_YEARS)" \
         "$EXPORT_TIMEOUT_H" "${EXPORT_JID:-}" "${EXPORT_TARGETS[@]}"

# ---- step 7: inject flags into FCST_SMAP_MRMS ------------------------------ #
# foreground: pure xarray/netCDF4, minutes of work.  --force: the out-dir is
# derived data (copies of the primaries), so overwriting a previous run's
# copies keeps this sequence rerunnable end to end.
log "step 7: injecting front flags into FCST_SMAP_MRMS copies"
python scripts/add_front_flags.py --years "$EXPORT_YEARS" --label-source noaa \
    --primary-dir "$JPL_AIRS_DATA/FCST_SMAP_MRMS" \
    --out-dir     "$JPL_AIRS_DATA/FCST_SMAP_MRMS_fronts" \
    --pred-dir    "$JPL_AIRS_DATA/front_id/predicted_fronts/$ENS_TAG" \
    --pred-tag    "${ENS_TAG#dlfront_}" --force \
    > logs/full_sequence_inject.log 2>&1
log "step 7: done -> $JPL_AIRS_DATA/FCST_SMAP_MRMS_fronts"

# ---- final: let the ablation chain drain ----------------------------------- #
if [ -n "${ABL_JIDS:-}" ] && [ "$HAVE_SLURM" = 1 ]; then
    log "waiting for the ablation chain to drain (jobs: $ABL_JIDS)"
    deadline=$(( $(date +%s) + ABLATION_TIMEOUT_H * 3600 ))
    while [ "$(squeue -h -j "$ABL_JIDS" 2>/dev/null | wc -l)" != 0 ]; do
        [ "$(date +%s)" -lt "$deadline" ] \
            || die "ablation chain still running after ${ABLATION_TIMEOUT_H}h"
        sleep "$POLL_SECS"
    done
fi
log "ablation artifacts:" \
    "$(ls "$JPL_AIRS_RESULTS"/dl_front/permutation/*.csv 2>/dev/null | wc -l) permutation CSVs," \
    "$(ls "$JPL_AIRS_RESULTS"/dl_front/ablation_eval/*.csv 2>/dev/null | wc -l) ladder eval CSVs"
log "SEQUENCE COMPLETE: main chain + ablation + export + inject all done"
