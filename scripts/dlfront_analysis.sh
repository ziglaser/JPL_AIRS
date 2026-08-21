#!/bin/bash
# scripts/dlfront_analysis.sh -- ANALYSIS-ONLY reruns: run every dl_front
# analysis against the artifacts ALREADY on disk.  This script never trains,
# never builds a krige cache, and never builds the swath bank (user request
# 2026-08-20) -- it is the "the checkpoints and caches are done, refresh
# every downstream number and figure" lever, e.g. after a label
# regeneration, after pulling a finished cluster run to another box, or
# after an eval-core change that invalidates the CSVs.
#
# What it runs, in dependency order:
#
#   Step 1  Pre-flight + checkpoint discovery [foreground]: the kriged-airs
#           cache must exist with a readable schema for every year of
#           EVAL_YEARS (every leg below either reads it or intersects its
#           time steps with its time axis), and at least one finished
#           checkpoint (<name>_final.h5) must exist under
#           $JPL_AIRS_RESULTS/dl_front/models -- otherwise the whole
#           invocation aborts loudly, because with nothing trained there is
#           nothing to analyze and this script will not train it (that is
#           the main chain's job: scripts/dlfront_jpl_chain.sh).
#           Discovery: every models/<name>/<name>_final.h5 is picked up and
#           split into MAIN-curriculum stems (D6A/D6B/D6C-f<k>) and
#           everything else (the D6A5/D6A3/D6A2 ladder rungs, one-off
#           experiments...).  Discovery is DISK-driven, not FOLDS-driven:
#           whatever finished training gets analyzed, and a fold that never
#           trained is simply absent rather than an error.
#   Step 2  Eval legs [CPU]: every MAIN checkpoint x {--source reanalysis,
#           --source kriged-airs} through dl_front.evaluate_test on
#           EVAL_YEARS, plus the checkpoint-free BK19 published-prediction
#           leg, plus a final `evaluate_test compare` pivoting the leg CSVs
#           in test_eval/ into comparison.csv (fold-pooled: legs named
#           <stage>-f<k>_<source> are averaged across folds into
#           <stage>_<source> columns -- see evaluate_test.compare).
#           NON-main checkpoints are evaluated too (no --channels flag:
#           evaluate_test adopts the checkpoint's own run_config.yaml
#           channel list), but their CSVs are then MOVED out of test_eval/
#           into ablation_eval/ exactly like the ablation chain's move
#           step, so a reduced-channel rung can never be pivoted into
#           comparison.csv as if it were a peer of D6A/B/C.  The compare
#           job therefore also waits on those move jobs.
#   Step 3  Permutation importance [GPU preferred]: EVERY discovered
#           checkpoint (main and non-main) x both sources through
#           dl_front.permutation --repeats $PERM_REPEATS.  Ladder rungs
#           land in the same permutation/ directory under their naturally
#           distinct stems (D6A5-f<k>_<source> etc.).
#   Step 4  Six-panel qualitative figures for each fold in FOLDS
#           (dl_front.six_panel) -- non-fatal: a figure failure never
#           blocks or fails the rest of the analysis.
#   Step 5  scripts/plot_dlfront_results.py (training curves + the
#           fold-POOLED CSI/permutation/ablation figures by default, plus
#           --per-fold --all-folds so the per-fold debug figures on disk
#           are refreshed too, not left stale) -- non-fatal, same policy.
#
# Idempotency (FORCE=1 defeats all of it):
#   eval legs   skip when their CSV exists AND its _run.json proves a
#               matched full-EVAL_YEARS run (years + kriged-airs
#               match_source for reanalysis/bk19 legs) AND its labels_sha1
#               equals the CURRENT front-label content digest (computed
#               ONCE per invocation via `dl_front.evaluate_test
#               label-digest`) AND its ckpt_sha1 equals the SHA-1 of the
#               checkpoint .h5 currently on disk (bk19 has no checkpoint
#               and skips that clause).  Main legs are checked in
#               test_eval/, non-main legs in ablation_eval/ (their true
#               post-move home -- a leftover in test_eval/ means the move
#               failed, not that the leg is done).
#   permutation skip when the CSV exists AND its _run.json matches both the
#               current labels_sha1 and the checkpoint's ckpt_sha1.
#   compare     skip when comparison.csv exists and no eval ran this
#               invocation.
#   figures     always re-rendered (deterministic filenames, overwrite is
#               the refresh; they are cheap next to one eval leg).
# If the label digest cannot be computed (submitting shell without the
# fronts-tf env), the labels_sha1 comparison degrades to existence-only and
# a loud WARNING is printed instead of aborting -- same policy as both
# sibling chains.
#
# Env knobs (same names/defaults as scripts/dlfront_jpl_chain.sh and
# scripts/dlfront_ablation_chain.sh where they overlap -- this script
# deliberately DUPLICATES their helpers rather than sourcing either, so the
# three chains stay independently editable and a bugfix or interface change
# in one can never silently break another):
#   JPL_AIRS_REPO      repo checkout            (default: $PWD)
#   JPL_AIRS_DATA      data root                (default: $JPL_AIRS_REPO/data)
#   JPL_AIRS_FCST      AIRS-FCST fullgrid root  (default: sibling
#                      AIRS_FCST_1deg, else $JPL_AIRS_DATA/HYSPLIT_demo --
#                      same resolution as dl_front.config)
#   JPL_AIRS_RESULTS   results root             (default: $JPL_AIRS_REPO/results)
#   JPL_BK19_DIR       BK19 published-prediction root (default:
#                      $JPL_AIRS_DATA/front_id/predicted_fronts/bk19)
#   CONDA_PREFIX_ROOT  conda install root       (default: $HOME/miniconda3)
#   SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_GRES  injected at submit time
#   SBATCH_GPU_PARTITION  partition for GPU-preferred jobs (permutation);
#                      CPU-submitted jobs (eval, figures) keep SBATCH_PARTITION
#   CLASSES            default 6
#   FOLDS              default "0 1 2" -- used ONLY by the step-4 six-panel
#                      figures; the eval/permutation legs come from disk
#                      discovery, never from this list (see step 1)
#   EVAL_YEARS         default 2016-2018 (the BK19 archive ends 2018, so a
#                      wider span breaks the three-way comparison -- change
#                      it only for checkpoint-only side analyses)
#   PERM_REPEATS       step-3 --repeats passed to dl_front.permutation
#                      (default 3)
#   FORCE=1            rerun every leg whose done-marker already exists
#   DRY_RUN=1          print every sbatch/python command without executing
#                      (also degrades the step-1 pre-flight aborts to
#                      warnings, so the plan can be previewed on a box
#                      where neither the data root nor the models exist)
#
# Logs (reorg 2026-08-21): every run nests its logs under
# logs/dlfront/analysis/<run-timestamp>/ -- SLURM job .out files
# (<label>_<jobid>.out, submit-time --output overriding the .sbatch
# fallbacks; the --wrap move/figure jobs get one too, instead of dumping
# slurm-<jid>.out into the repo root) and the no-SLURM branch's per-step
# <label>.log files.  logs/dlfront/analysis/latest/ is a symlink to the
# newest run, and the same timestamp names the manifest
# (results/dl_front/analysis_<ts>.txt) so runs pair up by eye.
#
# No SLURM?  The script detects the absence of `sbatch` and runs the same
# steps sequentially in the foreground, exactly like the sibling chains.
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        # help = the header comment block only (everything up to the first
        # non-comment line), so code changes can never leak into --help --
        # the same trick as both sibling chains.
        -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' \
                   | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg (this script takes no positional" \
                "arguments -- configure it with env vars, see --help)" >&2
           exit 2 ;;
    esac
done

export JPL_AIRS_REPO=${JPL_AIRS_REPO:-$PWD}
cd "$JPL_AIRS_REPO"
export JPL_AIRS_DATA=${JPL_AIRS_DATA:-$JPL_AIRS_REPO/data}
# same AIRS-FCST resolution as the sibling chains (manifest reorg
# 2026-08-13) -- duplicated here rather than shared, see the header note on
# deliberate duplication.
if [ -z "${JPL_AIRS_FCST:-}" ]; then
    if [ -d "$(dirname "$JPL_AIRS_DATA")/AIRS_FCST_1deg" ]; then
        JPL_AIRS_FCST=$(dirname "$JPL_AIRS_DATA")/AIRS_FCST_1deg
    else
        JPL_AIRS_FCST=$JPL_AIRS_DATA/HYSPLIT_demo
    fi
fi
export JPL_AIRS_FCST
export JPL_AIRS_RESULTS=${JPL_AIRS_RESULTS:-$JPL_AIRS_REPO/results}
CLASSES=${CLASSES:-6}
FOLDS=${FOLDS:-0 1 2}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
PERM_REPEATS=${PERM_REPEATS:-3}
# default matches the sibling chains' fixed phase-4 span (user decision
# 2026-08-13: the BK19 archive ends 2018, identical years for every leg of
# a comparison) -- overridable here because an analysis-only rerun is also
# the natural place for a checkpoint-only side span.
EVAL_YEARS=${EVAL_YEARS:-2016-2018}

MODELS=$JPL_AIRS_RESULTS/dl_front/models
PERM_DIR=$JPL_AIRS_RESULTS/dl_front/permutation
# EVAL_DIR is where dl_front.evaluate_test ALWAYS writes (it has no out-dir
# flag) and where compare() globs; ABLATION_EVAL_DIR is where non-main leg
# CSVs are moved so they never reach that glob -- same layout contract as
# the ablation chain.
EVAL_DIR=$JPL_AIRS_RESULTS/dl_front/test_eval
ABLATION_EVAL_DIR=$JPL_AIRS_RESULTS/dl_front/ablation_eval
KRIGED_AIRS=$JPL_AIRS_DATA/front_id/kriged_airs_fcst
# ONE timestamp per invocation (logs reorg 2026-08-21): shared by the
# manifest filename and the per-run log dir, so a manifest can always be
# paired with its logs by eye instead of matching two nearby date stamps.
RUN_TS=$(date +%Y%m%d_%H%M%S)
# Per-chain, per-run log nest -- the old flat logs/ had every chain's every
# run interleaved and was unbrowsable.  Created RIGHT HERE, before any
# submission: SLURM refuses to START a job whose --output directory does
# not exist, and submit()/submit_move()/submit_pyjob() below point job
# output into $LOG_DIR.
LOG_DIR=logs/dlfront/analysis/$RUN_TS
mkdir -p "$LOG_DIR" "$JPL_AIRS_RESULTS/dl_front" "$PERM_DIR" "$ABLATION_EVAL_DIR"
# convenience symlink: logs/dlfront/analysis/latest/ is always the newest run
ln -sfn "$RUN_TS" logs/dlfront/analysis/latest
MANIFEST=$JPL_AIRS_RESULTS/dl_front/analysis_$RUN_TS.txt
{
    echo "# dl_front analysis-only manifest  $(date -Is)"
    echo "# repo=$JPL_AIRS_REPO data=$JPL_AIRS_DATA fcst=$JPL_AIRS_FCST"
    echo "# results=$JPL_AIRS_RESULTS classes=$CLASSES folds='$FOLDS'"
    echo "# eval_years=$EVAL_YEARS perm_repeats=$PERM_REPEATS"
    echo "# force=$FORCE dry_run=$DRY_RUN"
} > "$MANIFEST"

HAVE_SLURM=0
command -v sbatch > /dev/null 2>&1 && HAVE_SLURM=1

note() { echo "[analysis] $*" >&2; }
record() { echo "$1=$2" >> "$MANIFEST"; }

# ---- step 1: kriged-airs cache pre-flight --------------------------------- #
# KRIGE_SCHEMA_READABLE: the schema versions the LOADER will read, which is
# what this probe must test -- not the one the builder stamps.  Duplicated
# from the sibling chains deliberately (see the header note); keep all
# three in sync by hand if src/dl_front/krige_fill.py's schema_version ever
# moves again.  v3 and v4 share an identical on-disk layout; v4 only marks
# U10M/V10M as clean reanalysis instead of kriged fills, a per-channel
# provenance difference dl_front.dataset gates against INPUT_CHANNELS, so a
# v3 cache stays readable for wind-free models.  v1/v2 are genuine format
# breaks and stay excluded.
KRIGE_SCHEMA_READABLE="3 4"
# cache_is_current <cache.nc>: 0 iff the cache carries a schema_version in
# $KRIGE_SCHEMA_READABLE.  Identical probe to both sibling chains --
# duplicated here, not sourced, per this script's stated policy of
# independent editability.
cache_is_current() {
    python3 - "$1" $KRIGE_SCHEMA_READABLE <<'PY' 2>/dev/null
import sys
path = sys.argv[1]
readable = {int(v) for v in sys.argv[2:]}
def version():
    try:
        from netCDF4 import Dataset
        with Dataset(path) as ds:
            return getattr(ds, "schema_version", None)
    except ImportError:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return ds.attrs.get("schema_version")
try:
    v = version()
except ImportError:
    sys.exit(0)                 # no reader available: existence-only
except Exception:
    sys.exit(1)                 # unreadable cache: not a valid current file
sys.exit(0 if v in readable else 1)
PY
}
# preflight_fail <msg...>: abort the WHOLE invocation loudly -- except under
# DRY_RUN, whose whole purpose is previewing the plan a submitting shell
# WOULD produce, including on a dev box where neither the data root nor the
# models are mounted; there it degrades to a warning, same policy as the
# ablation chain's cache pre-flight.  A real invocation hard-aborts,
# because every step below would otherwise fail leg-by-leg inside SLURM,
# burning an allocation per leg to rediscover the same one root cause.
preflight_fail() {
    if [ "$DRY_RUN" = 1 ]; then
        note "WARNING (DRY_RUN, not aborting): $*"
    else
        note "ERROR: $*"
        exit 3
    fi
}
check_kriged_airs_cache() {
    local first=${EVAL_YEARS%-*} last=${EVAL_YEARS#*-} y f bad=0
    for ((y = first; y <= last; y++)); do
        f=$KRIGED_AIRS/kriged_sfc_$y.nc
        if [ ! -e "$f" ] || ! cache_is_current "$f"; then
            bad=1
        fi
    done
    [ "$bad" = 0 ] && return 0
    preflight_fail "$KRIGED_AIRS is missing or has a stale (schema outside" \
        "$KRIGE_SCHEMA_READABLE) kriged-airs cache for some year in" \
        "$EVAL_YEARS.  Every eval and permutation leg either scores" \
        "--source kriged-airs or intersects its time steps with that" \
        "cache's time axis, so the whole analysis is dead without it." \
        "This script NEVER builds caches (user request: analysis only) --" \
        "run the main chain's phase 3a first:" \
        "PYTHONPATH=src python -m dl_front.krige_fill build-airs" \
        "--years $EVAL_YEARS --force"
}
check_kriged_airs_cache

# ---- step 1: checkpoint discovery ------------------------------------------ #
# Every models/<name>/<name>_final.h5 counts as a finished checkpoint
# (train.py deletes _final.h5 at training START and rewrites it only at the
# end, so a concurrently-retraining checkpoint drops out of discovery
# instead of being analyzed half-trained -- the same readiness gate as the
# ablation chain's step 2).  MAIN = the curriculum grid D6A/D6B/D6C-f<k>
# (the stems evaluate_test.compare fold-pools into comparison.csv);
# everything else (ladder rungs D6A5/D6A3/D6A2, one-off experiments) is
# analyzed too but kept OUT of test_eval/ (see step 2).
MAIN_CKPTS=()
EXTRA_CKPTS=()
for d in "$MODELS"/*/; do
    [ -d "$d" ] || continue                 # unmatched glob stays literal
    name=$(basename "$d")
    [ -e "$MODELS/$name/${name}_final.h5" ] || continue
    if [[ $name =~ ^D6[ABC]-f[0-9]+$ ]]; then
        MAIN_CKPTS+=("$name")
    else
        EXTRA_CKPTS+=("$name")
    fi
done
if [ $(( ${#MAIN_CKPTS[@]} + ${#EXTRA_CKPTS[@]} )) -eq 0 ]; then
    preflight_fail "no finished checkpoint (<name>_final.h5) under $MODELS" \
        "-- there is nothing to analyze, and this script never trains" \
        "(user request: analysis only).  Train first via" \
        "scripts/dlfront_jpl_chain.sh (curriculum) and/or" \
        "scripts/dlfront_ablation_chain.sh (channel ladder)."
fi
note "discovered ${#MAIN_CKPTS[@]} main checkpoint(s):" \
     "${MAIN_CKPTS[*]:-<none>}"
note "discovered ${#EXTRA_CKPTS[@]} non-main checkpoint(s):" \
     "${EXTRA_CKPTS[*]:-<none>}"
record discovered-main "${MAIN_CKPTS[*]:-none}"
record discovered-extra "${EXTRA_CKPTS[*]:-none}"

# ---- label-content digest, computed ONCE per invocation ------------------- #
# Same discipline as both sibling chains: a metrics CSV is only as
# trustworthy as the label grid it was scored against.  Computed once (not
# once per leg -- there can be dozens of legs) and cached in LABELS_SHA1.
# python3 (not the conda env) deliberately: the submitting shell may not
# have fronts-tf active -- degrade to existence-only staleness checking
# rather than hard-failing the whole run.
LABELS_SHA1=""
LABELS_SHA1=$(PYTHONPATH=src python3 -m dl_front.evaluate_test label-digest \
    --classes "$CLASSES" --years "$EVAL_YEARS" 2>/dev/null) || true
if [ -z "$LABELS_SHA1" ]; then
    note "WARNING: could not compute the current front-label digest" \
         "('python -m dl_front.evaluate_test label-digest' failed --" \
         "likely no fronts-tf env in this shell).  Falling back to" \
         "existence-only staleness checks for every leg this invocation;" \
         "a genuinely stale CSV from before a label regeneration may be" \
         "reused.  Rerun from a fronts-tf shell to get the real check."
else
    note "current labels_sha1=$LABELS_SHA1 (classes=$CLASSES years=$EVAL_YEARS)"
fi

# ---- skip predicates (FORCE=1 defeats all) -------------------------------- #
# An eval CSV's filename encodes only ckpt+source, so its existence alone
# cannot prove it is the run the comparison needs: a manual debugging run
# (--no-match, partial --years) writes the SAME stem.  Trust only a
# _run.json recording the full $EVAL_YEARS span, (for reanalysis/bk19 legs)
# the kriged-airs time-axis match, a ckpt_sha1 equal to the digest of the
# checkpoint .h5 currently on disk (checkpoint legs only -- bk19 has none;
# a _run.json with NO ckpt_sha1 while the checkpoint exists is stale), and
# -- when the current digest was computable -- a labels_sha1 equal to
# $LABELS_SHA1.  Duplicated from the main chain deliberately (see header).
eval_run_matches() {  # eval_run_matches <run.json> <source> <ckpt-path-or-empty>
    python3 - "$1" "$2" "$EVAL_YEARS" "$LABELS_SHA1" "$3" <<'PY' 2>/dev/null
import hashlib, json, os, sys
path, source, span, labels_sha1, ckpt = sys.argv[1:6]
try:
    run = json.load(open(path))
except (OSError, ValueError):
    sys.exit(1)                     # missing/broken provenance: rerun
first, last = span.split("-")
ok = (run.get("years") == list(range(int(first), int(last) + 1))
      and (source == "kriged-airs"
           or run.get("match_source") == "kriged-airs"))
if ok and ckpt:
    if not os.path.exists(ckpt):
        ok = False
    else:
        h = hashlib.sha1()
        with open(ckpt, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ok = run.get("ckpt_sha1") == h.hexdigest()
# Empty labels_sha1 = the digest could not be computed this invocation --
# degrade gracefully rather than mark every CSV stale just because we
# cannot check.  A run.json with no labels_sha1 at all still fails whenever
# the digest WAS computable, which is the point.
if ok and labels_sha1:
    ok = run.get("labels_sha1") == labels_sha1
sys.exit(0 if ok else 1)
PY
}
# skip_eval <dir> <csv-stem> <source> <ckpt-path-or-empty>
# One predicate, two homes: main legs (and bk19) live in EVAL_DIR, non-main
# legs in ABLATION_EVAL_DIR -- their true post-move location; a leftover in
# EVAL_DIR means the move step failed, not that the leg is done.
skip_eval() {
    [ "$FORCE" = 1 ] && return 1
    [ -e "$1/$2.csv" ] || return 1
    eval_run_matches "$1/${2}_run.json" "$3" "$4" && return 0
    note "eval $2: existing CSV is not a matched $EVAL_YEARS run on current" \
         "labels, or was scored by a checkpoint other than the one now on" \
         "disk (stale/debug ${2}_run.json) -- rerunning"
    return 1
}
# labels_match / ckpt_matches: the permutation done-marker's two halves,
# duplicated from the ablation chain (see header) -- permutation _run.jsons
# carry labels_sha1 + ckpt_sha1 but no years/match_source fields to check.
labels_match() {  # labels_match <run.json>
    [ -e "$1" ] || return 1
    [ -z "$LABELS_SHA1" ] && return 0
    python3 - "$1" "$LABELS_SHA1" <<'PY' 2>/dev/null
import json, sys
path, want = sys.argv[1:3]
try:
    run = json.load(open(path))
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if run.get("labels_sha1") == want else 1)
PY
}
ckpt_matches() {  # ckpt_matches <run.json> <ckpt.h5>
    python3 - "$1" "$2" <<'PY' 2>/dev/null
import hashlib, json, sys
path, ckpt = sys.argv[1:3]
try:
    run = json.load(open(path))
except (OSError, ValueError):
    sys.exit(1)
h = hashlib.sha1()
try:
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
except OSError:
    sys.exit(1)                     # no checkpoint on disk: cannot be done
sys.exit(0 if run.get("ckpt_sha1") == h.hexdigest() else 1)
PY
}
skip_perm() {   # skip_perm <csv-stem> <ckpt-path>   (stem = <ckpt>_<source>)
    [ "$FORCE" = 1 ] && return 1
    [ -e "$PERM_DIR/$1.csv" ] || return 1
    if labels_match "$PERM_DIR/${1}_run.json" \
       && ckpt_matches "$PERM_DIR/${1}_run.json" "$2"; then
        return 0
    fi
    note "permutation $1: existing CSV was computed on labels or checkpoint" \
         "weights that have since changed (stale ${1}_run.json) -- rerunning"
    return 1
}
# move_to_ablation_dir <stem>: relocate one non-main eval leg's artifacts
# from EVAL_DIR (evaluate_test's fixed, un-overridable output dir) to
# ABLATION_EVAL_DIR so evaluate_test.compare()'s glob of EVAL_DIR never
# pivots a reduced-channel rung into comparison.csv as a peer of D6A/B/C.
# `mv -f` (not cp): a duplicate left behind in EVAL_DIR would itself be
# exactly the contamination this exists to prevent.  Per-file existence
# check (not a blanket `|| true`): an absent file is tolerated, a REAL move
# failure (ENOSPC, permissions) must surface.  Duplicated from the ablation
# chain (see header).
move_to_ablation_dir() {  # move_to_ablation_dir <stem>
    local f
    for f in "$EVAL_DIR/$1.csv" "$EVAL_DIR/${1}_paper.json" \
             "$EVAL_DIR/${1}_run.json"; do
        if [ -e "$f" ]; then
            mv -f "$f" "$ABLATION_EVAL_DIR/"
        fi
    done
}

# join non-empty job ids with ':' (skipped legs contribute nothing) --
# identical helper to both sibling chains, copied deliberately (see header).
join_deps() {
    local out="" d
    for d in "$@"; do
        [ -n "$d" ] && out="${out:+$out:}$d"
    done
    printf '%s' "$out"
}

# ---- SLURM submitter (sets SUBMIT_JID; not $(...) so counters persist) ---- #
DRYC=0
SUBMIT_JID=""
submit() {  # submit <label> <gpu 0|1> <deps colon-joined> <sbatch script> <args...>
    local label=$1 gpu=$2 deps=$3 script=$4
    shift 4
    # --export=ALL: the JPL_AIRS_* roots MUST reach the job even on a
    # cluster whose slurm defaults force --export=NONE.
    local opts=(--export=ALL)
    # job output nests in this run's $LOG_DIR (mkdir'd at script start --
    # SLURM refuses to start a job whose --output dir is missing).  This
    # submit-time -o OVERRIDES the `#SBATCH --output=logs/...` line inside
    # the slurm/*.sbatch file, which stays as the fallback for hand
    # submissions.
    opts+=(--output="$LOG_DIR/${label}_%j.out")
    local part=${SBATCH_PARTITION:-}
    [ "$gpu" = 1 ] && [ -n "${SBATCH_GPU_PARTITION:-}" ] && part=$SBATCH_GPU_PARTITION
    [ -n "$part" ] && opts+=(-p "$part")
    [ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(-A "$SBATCH_ACCOUNT")
    [ "$gpu" = 1 ] && [ -n "${SBATCH_GRES:-}" ] && opts+=(--gres "$SBATCH_GRES")
    [ -n "$deps" ] && opts+=("--dependency=afterok:$deps")
    if [ "$DRY_RUN" = 1 ]; then
        DRYC=$((DRYC + 1)); SUBMIT_JID=DRY$DRYC
        note "DRY_RUN [$SUBMIT_JID] sbatch --parsable ${opts[*]-} $script $*"
    else
        SUBMIT_JID=$(sbatch --parsable ${opts[@]+"${opts[@]}"} "$script" "$@")
        note "submitted $label -> job $SUBMIT_JID${deps:+ (afterok:$deps)}"
    fi
    record "$label" "$SUBMIT_JID"
}

# submit_move <label> <deps> <stem>: SLURM-only counterpart to
# move_to_ablation_dir -- the eval job hasn't executed yet at submit time,
# so the move must be its own tiny job, afterok-gated on the eval.
# `sbatch --wrap` rather than a new .sbatch file: this script may only edit
# itself, not add files under slurm/.  Duplicated from the ablation chain
# (see header).
submit_move() {  # submit_move <label> <deps> <stem>
    local label=$1 deps=$2 stem=$3
    local opts=(--export=ALL)
    # --wrap jobs have no .sbatch file and hence no #SBATCH --output at all:
    # without this submit-time --output they dump slurm-<jid>.out into the
    # repo root.  Nest them in $LOG_DIR like every other job of this run.
    opts+=(--output="$LOG_DIR/${label}_%j.out")
    [ -n "${SBATCH_PARTITION:-}" ] && opts+=(-p "$SBATCH_PARTITION")
    [ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(-A "$SBATCH_ACCOUNT")
    [ -n "$deps" ] && opts+=("--dependency=afterok:$deps")
    # same per-file loop as move_to_ablation_dir (absent files tolerated,
    # real mv failures surface and fail the job), built into a --wrap
    # string: \$f stays unexpanded here so the JOB's shell expands it.
    # `set -e` because --wrap's shell runs without it.
    local wrap="set -e; for f in '$EVAL_DIR/$stem.csv' '$EVAL_DIR/${stem}_paper.json'"
    wrap="$wrap '$EVAL_DIR/${stem}_run.json'; do"
    wrap="$wrap if [ -e \"\$f\" ]; then mv -f \"\$f\" '$ABLATION_EVAL_DIR/'; fi; done"
    if [ "$DRY_RUN" = 1 ]; then
        DRYC=$((DRYC + 1)); SUBMIT_JID=DRY$DRYC
        note "DRY_RUN [$SUBMIT_JID] sbatch ${opts[*]-} --wrap=\"$wrap\" (afterok:$deps)"
    else
        SUBMIT_JID=$(sbatch --parsable ${opts[@]+"${opts[@]}"} --wrap="$wrap")
        note "submitted $label -> job $SUBMIT_JID (afterok:$deps)"
    fi
    record "$label" "$SUBMIT_JID"
}

# submit_pyjob <label> <deps> <command...>: a figure job with no .sbatch
# script of its own (this script may not add files under slurm/), built as
# `sbatch --wrap` with the same conda bootstrap the slurm/*.sbatch files
# use, condensed: the wrap job's shell starts without fronts-tf.  CPU-only
# (figures), so SBATCH_PARTITION applies, never the GPU partition.
submit_pyjob() {
    local label=$1 deps=$2
    shift 2
    local croot=${CONDA_PREFIX_ROOT:-}
    if [ -z "$croot" ] && command -v conda >/dev/null 2>&1; then
        croot=$(conda info --base 2>/dev/null) || croot=""
    fi
    croot=${croot:-$HOME/miniconda3}
    local opts=(--export=ALL)
    # --wrap jobs have no #SBATCH --output fallback: before the logs reorg
    # (2026-08-21) these figure jobs dumped slurm-<jid>.out into the repo
    # root.  Nest their output in $LOG_DIR like every other job of this run
    # (the dir exists -- mkdir'd at script start, which SLURM requires).
    opts+=(--output="$LOG_DIR/${label}_%j.out")
    [ -n "${SBATCH_PARTITION:-}" ] && opts+=(-p "$SBATCH_PARTITION")
    [ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(-A "$SBATCH_ACCOUNT")
    [ -n "$deps" ] && opts+=("--dependency=afterok:$deps")
    local wrap="set -e; source '$croot/etc/profile.d/conda.sh';"
    wrap="$wrap conda activate fronts-tf; cd '$JPL_AIRS_REPO';"
    wrap="$wrap PYTHONPATH=src $*"
    if [ "$DRY_RUN" = 1 ]; then
        DRYC=$((DRYC + 1)); SUBMIT_JID=DRY$DRYC
        note "DRY_RUN [$SUBMIT_JID] sbatch ${opts[*]-} --wrap=\"$wrap\"${deps:+ (afterok:$deps)}"
    else
        SUBMIT_JID=$(sbatch --parsable ${opts[@]+"${opts[@]}"} --wrap="$wrap")
        note "submitted $label -> job $SUBMIT_JID${deps:+ (afterok:$deps)}"
    fi
    record "$label" "$SUBMIT_JID"
}

# ---- local (no-SLURM) runner: same commands, sequential foreground -------- #
run_local() {  # run_local <label> <module> <args...>
    local label=$1 status=0
    shift
    note "RUN python -m $*   (log: $LOG_DIR/$label.log)"
    if [ "$DRY_RUN" = 1 ]; then
        # never executed -- record a value distinguishable from local-done
        # so a DRY_RUN manifest can't be mistaken for a completed run
        record "$label" DRY_RUN
        return 0
    fi
    PYTHONPATH=src python -m "$@" > "$LOG_DIR/$label.log" 2>&1 || status=$?
    if [ "$status" = 0 ]; then
        record "$label" local-done
    else
        record "$label" "local-FAILED-exit$status"
    fi
    return "$status"
}
# run_local_script <label> <script.py> <args...>: same contract as
# run_local for a plain script (scripts/plot_dlfront_results.py is not an
# importable module of src/).
run_local_script() {
    local label=$1 status=0
    shift
    note "RUN python $*   (log: $LOG_DIR/$label.log)"
    if [ "$DRY_RUN" = 1 ]; then
        record "$label" DRY_RUN
        return 0
    fi
    PYTHONPATH=src python "$@" > "$LOG_DIR/$label.log" 2>&1 || status=$?
    if [ "$status" = 0 ]; then
        record "$label" local-done
    else
        record "$label" "local-FAILED-exit$status"
    fi
    return "$status"
}

# the foreground branch runs python itself and needs the fronts-tf env --
# identical to the sibling chains' activate_env, copied deliberately.
activate_env() {
    [ "${CONDA_DEFAULT_ENV:-}" = fronts-tf ] && return 0
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
    command -v conda >/dev/null 2>&1 || { echo "conda not found: set CONDA_PREFIX_ROOT to the conda install root" >&2; exit 1; }
    conda activate fronts-tf
}

if [ "$HAVE_SLURM" = 1 ]; then
    # ================= SLURM dependency chain ================= #
    note "SLURM detected: submitting dependency chain"

    # ---- step 2: eval legs ---- #
    # EVAL_JIDS collects MAIN eval jobs; MOVE_JIDS the non-main move jobs.
    # compare waits on BOTH: the eval CSVs it pivots, and the moves that
    # pull non-main CSVs OUT of test_eval/ before the glob runs (without
    # that dependency a ladder CSV could sit in test_eval/ during the
    # compare job's window and contaminate comparison.csv).
    EVAL_JIDS=()
    MOVE_JIDS=()
    for ckpt in ${MAIN_CKPTS[@]+"${MAIN_CKPTS[@]}"}; do
        for src in reanalysis kriged-airs; do
            stem=${ckpt}_${src}
            if skip_eval "$EVAL_DIR" "$stem" "$src" "$MODELS/$ckpt/$ckpt.h5"; then
                note "skip eval $stem (matched CSV exists; FORCE=1 to rerun)"
                continue
            fi
            submit "eval-$stem" 0 "" slurm/dlfront_eval.sbatch \
                   --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                   --source "$src" --years "$EVAL_YEARS"
            EVAL_JIDS+=("$SUBMIT_JID")
        done
    done

    # checkpoint-free BK19 published-prediction leg (no ckpt_sha1 clause)
    if skip_eval "$EVAL_DIR" bk19 bk19 ""; then
        note "skip eval bk19 (matched CSV exists; FORCE=1 to rerun)"
    else
        submit eval-bk19 0 "" slurm/dlfront_eval.sbatch \
               --source bk19 --classes "$CLASSES" --years "$EVAL_YEARS"
        EVAL_JIDS+=("$SUBMIT_JID")
    fi

    # non-main checkpoints: eval, then move OUT of test_eval/ (see header).
    # No --channels: evaluate_test adopts the checkpoint's own
    # run_config.yaml channel list, the only correct list for a discovered
    # checkpoint whose rung this script cannot know.
    for ckpt in ${EXTRA_CKPTS[@]+"${EXTRA_CKPTS[@]}"}; do
        for src in reanalysis kriged-airs; do
            stem=${ckpt}_${src}
            if skip_eval "$ABLATION_EVAL_DIR" "$stem" "$src" \
                         "$MODELS/$ckpt/$ckpt.h5"; then
                note "skip eval $stem (matched CSV exists in ablation_eval/;" \
                     "FORCE=1 to rerun)"
                continue
            fi
            submit "eval-$stem" 0 "" slurm/dlfront_eval.sbatch \
                   --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                   --source "$src" --years "$EVAL_YEARS"
            submit_move "move-$stem" "$SUBMIT_JID" "$stem"
            MOVE_JIDS+=("$SUBMIT_JID")
        done
    done

    # fold-pooled comparison table over every main leg CSV in test_eval/
    if [ ${#EVAL_JIDS[@]} -eq 0 ] && [ ${#MOVE_JIDS[@]} -eq 0 ] \
       && [ "$FORCE" != 1 ] && [ -e "$EVAL_DIR/comparison.csv" ]; then
        note "skip compare (comparison.csv exists, no new evals; FORCE=1 to rerun)"
        JCOMPARE=""
    else
        submit compare 0 \
               "$(join_deps ${EVAL_JIDS[@]+"${EVAL_JIDS[@]}"} \
                            ${MOVE_JIDS[@]+"${MOVE_JIDS[@]}"})" \
               slurm/dlfront_eval.sbatch compare
        JCOMPARE=$SUBMIT_JID
    fi

    # ---- step 3: permutation importance, every checkpoint x both sources - #
    PERM_JIDS=()
    for ckpt in ${MAIN_CKPTS[@]+"${MAIN_CKPTS[@]}"} \
                ${EXTRA_CKPTS[@]+"${EXTRA_CKPTS[@]}"}; do
        for src in reanalysis kriged-airs; do
            stem=${ckpt}_${src}
            if skip_perm "$stem" "$MODELS/$ckpt/$ckpt.h5"; then
                note "skip permutation $stem (matched CSV exists; FORCE=1" \
                     "to rerun)"
                continue
            fi
            submit "perm-$stem" 1 "" slurm/dlfront_permutation.sbatch \
                   --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                   --source "$src" --years "$EVAL_YEARS" \
                   --repeats "$PERM_REPEATS"
            PERM_JIDS+=("$SUBMIT_JID")
        done
    done

    # ---- step 4: six-panel figures (non-fatal: nothing depends on them, so
    # a failed figure job blocks nothing downstream) ---- #
    for k in $FOLDS; do
        submit_pyjob "six-panel-f$k" "" \
                     python -m dl_front.six_panel --fold "$k"
    done

    # ---- step 5: summary plots, refreshed once the numbers they read are
    # final (comparison.csv + every eval CSV) ---- #
    submit_pyjob plot-results \
        "$(join_deps "${JCOMPARE:-}" ${EVAL_JIDS[@]+"${EVAL_JIDS[@]}"})" \
        python scripts/plot_dlfront_results.py \
        --results-dir "$JPL_AIRS_RESULTS/dl_front" --per-fold --all-folds

    note "manifest: $MANIFEST"
    note "monitor with: squeue -u \$USER"
else
    # ================= no SLURM: sequential foreground ================= #
    note "sbatch not found: running the same steps sequentially in this shell"
    if [ "$DRY_RUN" != 1 ]; then
        activate_env
    fi

    # ---- step 2: eval legs ---- #
    NEW_EVALS=0
    for ckpt in ${MAIN_CKPTS[@]+"${MAIN_CKPTS[@]}"}; do
        for src in reanalysis kriged-airs; do
            stem=${ckpt}_${src}
            if skip_eval "$EVAL_DIR" "$stem" "$src" "$MODELS/$ckpt/$ckpt.h5"; then
                note "skip eval $stem (matched CSV exists)"
                continue
            fi
            run_local "eval-$stem" dl_front.evaluate_test \
                --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                --source "$src" --years "$EVAL_YEARS"
            NEW_EVALS=$((NEW_EVALS + 1))
        done
    done

    # checkpoint-free BK19 published-prediction leg
    if skip_eval "$EVAL_DIR" bk19 bk19 ""; then
        note "skip eval bk19 (matched CSV exists)"
    else
        run_local eval-bk19 dl_front.evaluate_test \
            --source bk19 --classes "$CLASSES" --years "$EVAL_YEARS"
        NEW_EVALS=$((NEW_EVALS + 1))
    fi

    # non-main checkpoints: eval then move immediately (foreground: no
    # dependency job needed), BEFORE compare runs below so a ladder CSV
    # never sits in test_eval/ when the compare glob happens.  A DRY_RUN
    # skips the eval itself, so there is nothing to move.
    for ckpt in ${EXTRA_CKPTS[@]+"${EXTRA_CKPTS[@]}"}; do
        for src in reanalysis kriged-airs; do
            stem=${ckpt}_${src}
            if skip_eval "$ABLATION_EVAL_DIR" "$stem" "$src" \
                         "$MODELS/$ckpt/$ckpt.h5"; then
                note "skip eval $stem (matched CSV exists in ablation_eval/)"
                continue
            fi
            run_local "eval-$stem" dl_front.evaluate_test \
                --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                --source "$src" --years "$EVAL_YEARS"
            [ "$DRY_RUN" != 1 ] && move_to_ablation_dir "$stem"
        done
    done

    # fold-pooled comparison table over every main leg CSV in test_eval/
    if [ "$NEW_EVALS" -eq 0 ] && [ "$FORCE" != 1 ] \
       && [ -e "$EVAL_DIR/comparison.csv" ]; then
        note "skip compare (comparison.csv exists, no new evals)"
    else
        run_local compare dl_front.evaluate_test compare
    fi

    # ---- step 3: permutation importance, every checkpoint x both sources - #
    for ckpt in ${MAIN_CKPTS[@]+"${MAIN_CKPTS[@]}"} \
                ${EXTRA_CKPTS[@]+"${EXTRA_CKPTS[@]}"}; do
        for src in reanalysis kriged-airs; do
            stem=${ckpt}_${src}
            if skip_perm "$stem" "$MODELS/$ckpt/$ckpt.h5"; then
                note "skip permutation $stem (matched CSV exists)"
                continue
            fi
            run_local "perm-$stem" dl_front.permutation \
                --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                --source "$src" --years "$EVAL_YEARS" \
                --repeats "$PERM_REPEATS"
        done
    done

    # ---- step 4: six-panel figures (non-fatal by design, || note: a
    # rendering failure must not abort the analysis under set -e) ---- #
    for k in $FOLDS; do
        run_local "six-panel-f$k" dl_front.six_panel --fold "$k" \
            || note "six-panel fold $k FAILED (non-fatal, see log)"
    done

    # ---- step 5: summary plots (non-fatal, same policy) ---- #
    run_local_script plot-results scripts/plot_dlfront_results.py \
        --results-dir "$JPL_AIRS_RESULTS/dl_front" --per-fold --all-folds \
        || note "plot_dlfront_results FAILED (non-fatal, see log)"

    note "manifest: $MANIFEST"
fi
