#!/bin/bash
# scripts/dlfront_full_sequence.sh -- run the whole post-label-fix deployment
# (runbook steps 4-7) unattended, in sequence, with no human intervention
# between phases:
#
#   Step 4  scripts/cluster_kickoff.sh (preflight checks + submit the full
#           main chain: swath bank -> krige builds -> D6A/B/C x 3 folds ->
#           evals -> compare), then WAIT until every fold's three stage
#           checkpoints and the kriged-airs caches exist.
#   (then)  dl_front.six_panel qualitative figures for SIXPANEL_FOLDS
#           (default fold 0) -- non-fatal, like the chain's quicklooks: it
#           needs checkpoints + kriged-airs + bk19 together, which first
#           exists at this point in the sequence.
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
#   (then)  ensemble eval leg + compare rerun (added 2026-08-21; user
#           decision: user-facing evaluations report the ENSEMBLE, not
#           per-fold splits).  The just-exported softmax-ensemble archive is
#           scored as its own named leg through evaluate_test's BK19-schema
#           reader (--source bk19 --pred-dir --leg-name
#           D6C-ens3_kriged-airs, checkpoint-free), then `evaluate_test
#           compare` reruns so comparison.csv includes it next to the
#           fold-pooled recipe legs.  No afterok on the export job needed:
#           the step-6 wait already blocked until the archive existed.
#           Runs BEFORE step 7 (inject does not depend on it, but the
#           sequence must finish with a complete comparison); on SLURM the
#           two jobs drain with the ablation chain at the very end.
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
# logs/full_sequence.log is the ONE deliberately flat log: it is created by
# the shell redirect above, before this script (and hence its per-run log
# dir) even starts.  Everything the sequence itself writes -- the six-panel,
# export and inject logs, plus the export job's SLURM .out -- nests under
# logs/dlfront/sequence/<run-timestamp>/ (logs/dlfront/sequence/latest/ is a
# symlink to the newest run); the chains it drives nest their own logs the
# same way under logs/dlfront/main|ablation/<their-own-timestamp>/.
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
# (Analysis-only reruns against ALREADY-finished artifacts -- no training,
# no krige builds -- are scripts/dlfront_analysis.sh, runbook section 12.)
#
# Env knobs (all optional):
#   POLL_SECS            poll interval while waiting          (default 300)
#   MAIN_TIMEOUT_H       step-4 wait budget, hours            (default 72)
#   EXPORT_TIMEOUT_H     step-6 wait budget, hours            (default 12)
#   ABLATION_TIMEOUT_H   step-5 final wait budget, hours      (default 36)
#   EXPORT_YEARS         export + inject span                 (default 2016-2021)
#   EVAL_YEARS           ensemble-eval-leg span               (default 2016-2018,
#                        the chains' fixed comparison span: the BK19 archive
#                        ends 2018, so the compare needs identical years)
#   CLASSES              ensemble-eval-leg class count        (default 6)
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
# the ensemble eval leg's span/classes match the chains' fixed comparison
# settings (BK19 ends 2018), NOT the export span -- evaluate_test compare
# refuses legs scored on different samples, so a 2016-2021 leg would poison
# the very comparison this step exists to complete
EVAL_YEARS=${EVAL_YEARS:-2016-2018}
CLASSES=${CLASSES:-6}
FOLDS=${FOLDS:-0 1 2}
Y0=${EXPORT_YEARS%-*} Y1=${EXPORT_YEARS#*-}

MODELS=$JPL_AIRS_RESULTS/dl_front/models
KRIGED_AIRS=$JPL_AIRS_DATA/front_id/kriged_airs_fcst
ENS_TAG=dlfront_D6C-ens3_kriged-airs   # export_predictions.ensemble_stem for
                                       # the three D6C folds + export_tag
ENS_DIR=$JPL_AIRS_DATA/front_id/predicted_fronts/$ENS_TAG/1deg_3wide/3hr

log(){ echo "[sequence $(date '+%F %T')] $*"; }
die(){ log "FATAL: $*"; exit 1; }
# ONE timestamp per invocation (logs reorg 2026-08-21); this script has no
# manifest of its own, so RUN_TS names only the per-run log nest.  The
# mkdir/symlink live AFTER the FULLSEQ_SOURCE_ONLY hook below so sourcing
# the helpers for tests creates nothing on disk.
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/dlfront/sequence/$RUN_TS

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
                "a job failed; check sacct -j $jids and logs/dlfront/"
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

# Per-run log nest, created before any submission: SLURM refuses to START a
# job whose --output directory does not exist, and the step-6 export
# submission below points its job output into $LOG_DIR.
mkdir -p "$LOG_DIR"
# convenience symlink: logs/dlfront/sequence/latest/ is always the newest run
ln -sfn "$RUN_TS" logs/dlfront/sequence/latest

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

# ---- six-panel qualitative figures (non-fatal, like the chain quicklooks) - #
# dl_front.six_panel is not part of any chain: it needs the finished D6A/D6C
# checkpoints AND the kriged-airs cache AND the bk19 archive at once, which
# only exists here, after the step-4 wait.  A figure failure must never block
# the ablation/export/inject pipeline -- render, note, move on.  Foreground:
# a handful of instants through a small CNN is minutes on CPU.
SIXPANEL_FOLDS=${SIXPANEL_FOLDS:-0}
for k in $SIXPANEL_FOLDS; do
    log "six-panel figures, fold $k (non-fatal)"
    python -m dl_front.six_panel --fold "$k" \
        > "$LOG_DIR/six_panel_f$k.log" 2>&1 \
        || log "six_panel fold $k FAILED (non-fatal, see $LOG_DIR/six_panel_f$k.log)"
done

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
    # --output nests the job log in this run's $LOG_DIR (mkdir'd above --
    # SLURM refuses to start a job whose --output dir is missing); this
    # submit-time flag OVERRIDES dlfront_export.sbatch's own
    # `#SBATCH --output=logs/...` line, which stays as the fallback for
    # hand submissions.
    EXPORT_JID=$(sbatch --parsable --export=ALL \
                 --output="$LOG_DIR/export_%j.out" \
                 ${GPU_OPTS[@]+"${GPU_OPTS[@]}"} \
                 slurm/dlfront_export.sbatch "${EXPORT_ARGS[@]}")
    log "step 6: export submitted -> job $EXPORT_JID"
else
    log "step 6: no SLURM -> exporting in the foreground"
    EXPORT_JID=""
    python -m dl_front.export_predictions "${EXPORT_ARGS[@]}" \
        > "$LOG_DIR/export.log" 2>&1
fi
wait_for "step6 export ($ENS_TAG $EXPORT_YEARS)" \
         "$EXPORT_TIMEOUT_H" "${EXPORT_JID:-}" "${EXPORT_TARGETS[@]}"

# ---- step 6.5: ensemble eval leg + compare rerun --------------------------- #
# User decision 2026-08-21: user-facing evaluations report the ENSEMBLE (the
# deployed product), not per-fold splits -- the fold-pooled D6C legs the main
# chain scored are recipe estimates; this leg is the shipped archive's own
# score.  Checkpoint-free: the archive scores through evaluate_test's
# BK19-schema reader (--source bk19 --pred-dir), so there is no ckpt behind
# it and no afterok needed (the step-6 wait above already blocked until the
# archive existed).  The compare rerun pivots it into comparison.csv next to
# the recipe legs.  Kept BEFORE step 7: inject does not read it, but the
# sequence must finish with a complete comparison.
ENS_LEG=${ENS_TAG#dlfront_}
ENS_EVAL_ARGS=(--source bk19
               --pred-dir "$JPL_AIRS_DATA/front_id/predicted_fronts/$ENS_TAG"
               --leg-name "$ENS_LEG" --classes "$CLASSES"
               --years "$EVAL_YEARS")
ENS_JIDS=""
if [ "$HAVE_SLURM" = 1 ]; then
    # CPU jobs (inference-free scoring): the eval .sbatch's own resources
    # apply; compare afterok-chains on the eval so the pivot always sees the
    # fresh leg CSV.  Both drain with the ablation chain at the very end.
    JENSEVAL=$(sbatch --parsable --export=ALL \
               --output="$LOG_DIR/eval_ens_%j.out" \
               slurm/dlfront_eval.sbatch "${ENS_EVAL_ARGS[@]}")
    JENSCMP=$(sbatch --parsable --export=ALL \
              --output="$LOG_DIR/compare_ens_%j.out" \
              --dependency="afterok:$JENSEVAL" \
              slurm/dlfront_eval.sbatch compare)
    ENS_JIDS=$JENSEVAL,$JENSCMP
    log "step 6.5: ensemble eval leg -> job $JENSEVAL," \
        "compare rerun -> job $JENSCMP (afterok)"
else
    log "step 6.5: no SLURM -> ensemble eval + compare in the foreground"
    python -m dl_front.evaluate_test "${ENS_EVAL_ARGS[@]}" \
        > "$LOG_DIR/eval_ens.log" 2>&1
    python -m dl_front.evaluate_test compare \
        > "$LOG_DIR/compare_ens.log" 2>&1
fi

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
    > "$LOG_DIR/inject.log" 2>&1
log "step 7: done -> $JPL_AIRS_DATA/FCST_SMAP_MRMS_fronts"

# ---- final: let the ablation chain + step-6.5 eval/compare drain ----------- #
# One combined drain: the step-6.5 ensemble eval + compare jobs were
# deliberately NOT waited on inline (step 7 does not read them), so they are
# collected here with the ablation jobs before the sequence declares itself
# complete.
DRAIN_JIDS=${ABL_JIDS:-}
[ -n "$ENS_JIDS" ] && DRAIN_JIDS="${DRAIN_JIDS:+$DRAIN_JIDS,}$ENS_JIDS"
if [ -n "$DRAIN_JIDS" ] && [ "$HAVE_SLURM" = 1 ]; then
    log "waiting for the ablation chain + ensemble eval/compare to drain" \
        "(jobs: $DRAIN_JIDS)"
    deadline=$(( $(date +%s) + ABLATION_TIMEOUT_H * 3600 ))
    while [ "$(squeue -h -j "$DRAIN_JIDS" 2>/dev/null | wc -l)" != 0 ]; do
        [ "$(date +%s)" -lt "$deadline" ] \
            || die "ablation/ensemble-eval jobs still running after ${ABLATION_TIMEOUT_H}h"
        sleep "$POLL_SECS"
    done
fi
# A drained queue is not success: if the ensemble leg's CSV never appeared,
# the eval job failed (and dependency-cancelled the compare rerun) and the
# comparison this sequence promised is incomplete -- fail loudly, exactly
# like every other wait in this script.  (The no-SLURM branch already ran
# both foreground under set -e, so this check is a no-op there.)
[ -e "$JPL_AIRS_RESULTS/dl_front/test_eval/$ENS_LEG.csv" ] \
    || die "ensemble eval leg CSV missing" \
           "($JPL_AIRS_RESULTS/dl_front/test_eval/$ENS_LEG.csv) --" \
           "the step-6.5 eval failed; check $LOG_DIR and sacct"
log "ablation artifacts:" \
    "$(ls "$JPL_AIRS_RESULTS"/dl_front/permutation/*.csv 2>/dev/null | wc -l) permutation CSVs," \
    "$(ls "$JPL_AIRS_RESULTS"/dl_front/ablation_eval/*.csv 2>/dev/null | wc -l) ladder eval CSVs"
log "SEQUENCE COMPLETE: main chain + ablation + export + ensemble eval" \
    "+ inject all done"
