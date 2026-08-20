#!/bin/bash
# scripts/dlfront_ablation_chain.sh -- submit (SLURM) or run (plain bash) the
# two AIRS-input-channel experiments the user asked to be kept SEPARATE from
# the main JPL curriculum (scripts/dlfront_jpl_chain.sh), so an overnight
# ablation run can never accidentally retrain/resubmit the main D6A/B/C
# curriculum or vice versa:
#
#   Step 2  Permutation importance [GPU preferred]: for each fold and each
#           stage checkpoint in PERM_CKPTS (default "D6A D6C" -- see the
#           PERM_CKPTS knob below for why both), for EACH of
#           --source reanalysis and --source kriged-airs, shuffle one of
#           the 5 input channels at a time and record the CSI/POD/FAR/FB
#           cost, over 2016-2018.  This sizes how much skill lives in
#           SLP/U10M/V10M (which AIRS never sees -- SLP is copied clean from
#           MERRA-2 and the winds are the WRF-27km met driving HYSPLIT).
#           REQUIRES the curriculum checkpoint to have FINISHED training,
#           not merely improved once -- see the readiness gate on
#           <ckpt>_final.h5 below, and REQUIRES the kriged-airs cache
#           (see the pre-flight check below).
#           ORDERING (user decision 2026-08-18): all analysis runs on the
#           NEW checkpoints -- run this chain AFTER the main chain's
#           FORCE_TRAIN=1 retrain has finished.  The gates make wrong
#           orderings safe rather than silent: train.py deletes
#           <ckpt>_final.h5 at training START, so a step-2 invocation
#           during a retrain skips the checkpoint instead of permuting a
#           half-trained snapshot, and skip_perm compares the run.json's
#           ckpt_sha1 against the checkpoint on disk, so a permutation CSV
#           written from superseded weights is rerun, not reused (a
#           labels_sha1 match alone cannot notice a retrain on unchanged
#           labels).
#   Step 3  Stage-A channel ablation ladder [GPU]: retrain the stage-A
#           (--source reanalysis) architecture with the input channels cut
#           down to the sets in CHANNEL_SETS (default: a 5-channel, a
#           3-channel, and a 2-channel rung -- see CHANNEL_SETS below),
#           then evaluate each on 2016-2018 with BOTH --source reanalysis
#           and --source kriged-airs.  THE POINT of step 3 is to measure
#           how much front skill survives when the model is restricted to
#           the channels AIRS actually provides: T2M and QV2M come from
#           AIRS retrievals, but U10M/V10M are WRF-27km (not AIRS) and SLP
#           is MERRA-2 reanalysis (AIRS retrieves no SLP).  The 2-channel
#           (T2M,QV2M) rung is therefore the honest AIRS-only skill
#           ceiling; the 3-channel rung shows how much of the gap SLP alone
#           buys back; the 5-channel rung (D6A5, user decision 2026-08-18)
#           is the matched all-channels control.
#           D6A5 -- NOT the main chain's D6A -- is this ladder's top rung:
#           training all three rungs (D6A5/D6A3/D6A2) fresh, under
#           identical conditions and on the same labels, keeps the ladder
#           a pure channel comparison -- reusing a checkpoint trained in a
#           different run (different label vintage, epochs, or warm-start
#           lineage) would conflate the channel effect with a training-
#           provenance effect, biased exactly in the direction this
#           ablation hopes to demonstrate.  This makes step 3 three
#           stage-A trainings per fold instead of two; the main chain's
#           D6A checkpoint is untouched and remains its stage-B retrain
#           source.
#           ablation eval CSVs are written to a SEPARATE results directory
#           (ABLATION_EVAL_DIR below) so they cannot leak into the main
#           chain's three-way comparison.csv -- see the note there.
#           REQUIRES the kriged-airs cache (see the pre-flight check
#           below).
#
# STEPS=2 / STEPS=3 / STEPS="2 3" (default) selects which of the above run;
# they are fully independent of each other.
#
# Env knobs (same names/defaults as scripts/dlfront_jpl_chain.sh where they
# overlap -- this script deliberately DUPLICATES that script's helpers
# rather than sourcing it, so the two chains stay independently editable
# and a bugfix or interface change in one can never silently break the
# other):
#   JPL_AIRS_REPO      repo checkout            (default: $PWD)
#   JPL_AIRS_DATA      data root                (default: $JPL_AIRS_REPO/data)
#   JPL_AIRS_FCST      AIRS-FCST fullgrid root  (default: sibling
#                      AIRS_FCST_1deg, else $JPL_AIRS_DATA/HYSPLIT_demo --
#                      same resolution as dl_front.config)
#   JPL_AIRS_RESULTS   results root             (default: $JPL_AIRS_REPO/results)
#   CONDA_PREFIX_ROOT  conda install root       (default: $HOME/miniconda3)
#   SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_GRES  injected at submit time
#   SBATCH_GPU_PARTITION  partition for GPU jobs only (train, permutation);
#                      CPU-submitted jobs (eval) keep SBATCH_PARTITION
#   CLASSES            default 6
#   FOLDS              default "0 1 2"
#   FORCE=1            resubmit every phase whose done-marker already exists
#   DRY_RUN=1          print every sbatch/python command without executing
#   STEPS              which steps to queue: "2", "3", or "2 3" (default)
#   PERM_CKPTS         step-2 stage checkpoints to permute, space-separated
#                      stage letters matching the main chain's naming
#                      (default: "D6A D6C" -- D6C alone
#                      is the wrong default for what step 2 claims to
#                      answer: D6C has been fine-tuned on kriged-AIRS
#                      inputs and so has a specific incentive to lean on
#                      the clean SLP/winds that AIRS never sees, which
#                      does NOT predict what stage A (the channel ladder
#                      in step 3) will do.  D6A answers "what will the
#                      ladder do" (it IS the ladder's un-fine-tuned
#                      starting point); D6C answers "what does the
#                      production model actually use".  Both are cheap
#                      (inference only) so both run by default.)
#   PERM_REPEATS       step-2 --repeats passed to dl_front.permutation
#                      (default 3 -- one fixed shuffle per (channel, repeat))
#   CHANNEL_SETS       step-3 ladder rungs as "name:comma,separated,channels"
#                      pairs (default:
#                        "D6A5:T2M,QV2M,SLP,U10M,V10M" "D6A3:T2M,QV2M,SLP"
#                        "D6A2:T2M,QV2M"
#                      ) -- each is trained as <name>-f<k> with
#                      --source reanalysis --channels <list>.  D6A5 is the
#                      matched 5-channel control -- see the step-3
#                      description above for why it is retrained here
#                      instead of reusing the main chain's D6A.
#
# Both steps use the fixed 2016-2018 span (EVAL_YEARS below) for permutation
# and step-3 eval, matching the main chain's phase-4 test years (user
# decision 2026-08-13).
#
# Pre-flight: both steps score against --source kriged-airs (and even the
# --source reanalysis legs intersect their time steps with the kriged-airs
# cache's time axis for comparability, exactly like the main chain's phase
# 4) -- so BOTH steps are dead without a current-schema kriged-airs cache
# covering $EVAL_YEARS, not merely the kriged-airs half of each step.  This
# script never builds that cache itself (it belongs to the main chain's
# phase 3a, `krige_fill build-airs`) -- see check_kriged_airs_cache below,
# which aborts the whole invocation with a clear message if the cache is
# missing or stale, rather than letting every leg silently fail one at a
# time inside SLURM.
#
# Idempotency:
#   step 2 done-marker: the permutation CSV
#     $JPL_AIRS_RESULTS/dl_front/permutation/<ckpt-stem>_<source>.csv
#     exists AND its _run.json's labels_sha1 equals the CURRENT front-label
#     content digest (computed once per invocation via
#     `dl_front.evaluate_test label-digest`, same discipline as the main
#     chain's phase-4 staleness check -- a label
#     regeneration invalidates a permutation CSV exactly the way it
#     invalidates an eval CSV, since both score against the label grid)
#     AND its _run.json's ckpt_sha1 equals the SHA-1 of the checkpoint
#     currently on disk (ckpt_matches below -- a retrain on unchanged
#     labels keeps the same labels_sha1, so only the weights fingerprint
#     catches it).
#     If the digest command fails (submitting shell without the fronts-tf
#     env), the labels_sha1 comparison degrades to existence-only and a
#     warning is printed -- it does NOT abort the chain.  The checkpoint
#     itself must show <ckpt>_final.h5, not merely <ckpt>.h5:
#     train.py's ModelCheckpoint rewrites <ckpt>.h5 on EVERY validation
#     improvement, so gating on it would let a permutation job load a
#     still-training 2-epoch snapshot and write a complete,
#     provenance-stamped CSV whose done-marker then makes every later run
#     skip it -- this matters because the main chain may be running
#     concurrently, which the runbook explicitly invites.  train.py
#     deletes _final.h5 at training start, so _final.h5 existence always
#     means the CURRENT <ckpt>.h5 weights finished training (see the
#     ORDERING note in the step-2 description above).
#   step 3 train done-marker: the existing <name>-f<k>_final.h5 convention
#     (identical to the main chain's skip_train).
#   step 3 eval done-marker: the same labels-aware CSV check as step 2,
#     applied to the eval CSV, checked in ABLATION_EVAL_DIR (see below --
#     NOT the main chain's test_eval/, so these reduced-channel legs can
#     never be pivoted into comparison.csv as if they were peers of
#     D6A/B/C).
# FORCE=1 overrides both.
#
# Results layout (coordinate with the eval-core owner if this changes):
# dl_front.evaluate_test has no flag to redirect its output
# directory -- it always writes to $JPL_AIRS_RESULTS/dl_front/test_eval/,
# the same directory `evaluate_test.compare()` globs for the main chain's
# three-way comparison.csv.  Since this script may not add a flag to that
# file (owned by a different agent), each step-3 eval CSV/paper/run.json is
# moved (not copied -- there is no reason to keep a duplicate sitting in
# test_eval/) into ABLATION_EVAL_DIR
# ($JPL_AIRS_RESULTS/dl_front/ablation_eval/) as a post-step once the eval
# job completes, so it never contaminates the main comparison table.  Step
# 2's permutation CSVs already live under permutation/, a directory
# comparison.csv never globs, so they need no such move.
#
# No SLURM?  The script detects the absence of `sbatch` and runs the same
# steps sequentially in the foreground, exactly like the main chain.
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        # help = the header comment block only (everything up to the first
        # non-comment line), so code changes can never leak into --help --
        # copied verbatim from dlfront_jpl_chain.sh's --help trick.
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
# same AIRS-FCST resolution as the main chain (manifest reorg 2026-08-13) --
# duplicated here rather than shared, see the header note on deliberate
# duplication.
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
STEPS=${STEPS:-2 3}
# D6A + D6C: D6A answers "what will the channel ladder
# do" (it is the ladder's own un-fine-tuned starting point); D6C answers
# "what does the production model use" (it has been fine-tuned on
# kriged-AIRS and so has a specific incentive to lean on the clean
# SLP/winds AIRS never sees, which does not predict D6A's ranking) -- see
# the header note.
PERM_CKPTS=${PERM_CKPTS:-"D6A D6C"}
PERM_REPEATS=${PERM_REPEATS:-3}
# D6A5/D6A3/D6A2 (user decision 2026-08-18): all three rungs of the
# stage-A channel ladder are now trained fresh under identical conditions
# on the CURRENT (post 2026-08-17 label fix) labels -- see the header note
# on why D6A5, not the main chain's pre-fix D6A, is the top rung.
CHANNEL_SETS=${CHANNEL_SETS:-"D6A5:T2M,QV2M,SLP,U10M,V10M D6A3:T2M,QV2M,SLP D6A2:T2M,QV2M"}
# fixed test span, matches the main chain's phase-4 EVAL_YEARS (user
# decision 2026-08-13: identical years across every leg of a comparison)
EVAL_YEARS=2016-2018

run_step() { [[ " $STEPS " == *" $1 "* ]]; }   # run_step 2 / run_step 3

MODELS=$JPL_AIRS_RESULTS/dl_front/models
PERM_DIR=$JPL_AIRS_RESULTS/dl_front/permutation
# EVAL_DIR is where dl_front.evaluate_test ALWAYS writes (no out-dir flag
# to override -- see the header's "Results layout" note); ABLATION_EVAL_DIR
# is where this script's step-3 CSVs are moved to afterward so they never
# reach evaluate_test.compare()'s glob of EVAL_DIR.
EVAL_DIR=$JPL_AIRS_RESULTS/dl_front/test_eval
ABLATION_EVAL_DIR=$JPL_AIRS_RESULTS/dl_front/ablation_eval
KRIGED_AIRS=$JPL_AIRS_DATA/front_id/kriged_airs_fcst
mkdir -p logs "$JPL_AIRS_RESULTS/dl_front" "$PERM_DIR" "$ABLATION_EVAL_DIR"
MANIFEST=$JPL_AIRS_RESULTS/dl_front/ablation_$(date +%Y%m%d_%H%M%S).txt
{
    echo "# dl_front ablation chain manifest  $(date -Is)"
    echo "# repo=$JPL_AIRS_REPO data=$JPL_AIRS_DATA fcst=$JPL_AIRS_FCST"
    echo "# results=$JPL_AIRS_RESULTS classes=$CLASSES folds='$FOLDS'"
    echo "# steps='$STEPS' perm_ckpts='$PERM_CKPTS' perm_repeats=$PERM_REPEATS"
    echo "# channel_sets='$CHANNEL_SETS' force=$FORCE dry_run=$DRY_RUN"
} > "$MANIFEST"

HAVE_SLURM=0
command -v sbatch > /dev/null 2>&1 && HAVE_SLURM=1

note() { echo "[ablation] $*" >&2; }
record() { echo "$1=$2" >> "$MANIFEST"; }

# ---- kriged-airs cache pre-flight ----------------------------------------- #
# KRIGE_SCHEMA_READABLE: the schema versions the LOADER will read, which is
# what this probe must test -- not the one the builder stamps.  Duplicated
# from the main chain deliberately (see the header note on why this script
# never sources that one); keep both in sync by hand if
# src/dl_front/krige_fill.py's schema_version ever moves again.  v3 and v4
# share an identical on-disk layout; v4 only marks U10M/V10M as clean
# reanalysis instead of kriged fills, a per-CHANNEL provenance difference
# that dl_front.dataset gates per channel against INPUT_CHANNELS.  So a v3
# cache is fully valid for a T2M/QV2M(/SLP) model and is refused only for a
# run that actually consumes the winds (user correction 2026-08-18: "only
# reducing the number of kriged variables is backwards compatible").
# Testing == 4 here would force a multi-hour rebuild for runs that never
# read a wind channel.  v1/v2 are genuine format breaks and stay excluded.
KRIGE_SCHEMA_READABLE="3 4"
# cache_is_current <cache.nc>: 0 iff the cache carries a schema_version in
# $KRIGE_SCHEMA_READABLE.  Identical probe to the main chain's
# cache_is_current -- duplicated here, not sourced, per this script's
# stated policy of independent editability.
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
# check_kriged_airs_cache: abort the WHOLE invocation up front if the
# kriged-airs cache is missing or stale for any year in $EVAL_YEARS.
# permutation.main and every step-3 eval leg set match_source="kriged-airs"
# (they intersect their time steps with the cache's time axis for
# comparability, exactly like the main chain's phase 4) -- so a missing or
# stale cache fails EVERY leg of both steps, not just the kriged-airs half,
# and letting that happen leg-by-leg inside SLURM burns an allocation per
# leg to discover the same one root cause.  This script never builds the
# cache itself; that is the main chain's phase 3a.
# DRY_RUN=1 degrades this to a warning instead of an abort:
# DRY_RUN's whole purpose is previewing the plan a submitting
# shell WOULD produce, including on a dev box where the data root (and
# hence the cache) is not mounted at all -- the same reasoning that already
# lets the label-digest probe above degrade gracefully rather than abort.
# A real (non-DRY_RUN) invocation still hard-aborts.
check_kriged_airs_cache() {
    local first=${EVAL_YEARS%-*} last=${EVAL_YEARS#*-} y f bad=0
    for ((y = first; y <= last; y++)); do
        f=$KRIGED_AIRS/kriged_sfc_$y.nc
        if [ ! -e "$f" ] || ! cache_is_current "$f"; then
            bad=1
        fi
    done
    [ "$bad" = 0 ] && return 0
    local msg=("$KRIGED_AIRS is missing or has a stale (schema outside $KRIGE_SCHEMA_READABLE)"
               "kriged-airs cache for some year in $EVAL_YEARS.  Both step 2"
               "(permutation) and step 3 (the channel ladder) score against"
               "--source kriged-airs and/or intersect their time steps with"
               "its time axis, so ALL legs of both steps are dead without"
               "it -- run the main chain's phase 3a first:"
               "PYTHONPATH=src python -m dl_front.krige_fill build-airs"
               "--years $EVAL_YEARS --force   (or just"
               "bash scripts/dlfront_jpl_chain.sh, which builds the full"
               "2007-2021 span).")
    if [ "$DRY_RUN" = 1 ]; then
        note "WARNING (DRY_RUN, not aborting): ${msg[*]}"
    else
        note "ERROR: ${msg[*]}"
        exit 3
    fi
}
if run_step 2 || run_step 3; then
    check_kriged_airs_cache
fi

# ---- label-content digest, computed ONCE per invocation ------------------ #
# Same discipline as the main chain's phase-4 staleness check:
# a metrics/permutation CSV is only as trustworthy as the
# label grid it was scored against, and the 2026-08-17 antimeridian-bug fix
# regenerated every year's labels.  Computed once (not once per leg -- there
# are dozens of legs across step 2 and step 3) and cached in LABELS_SHA1.
# python3 (not the conda env) is used deliberately: the SLURM submitting
# shell may not have fronts-tf active, exactly the situation the
# cache_is_current probe above already tolerates -- degrade to
# existence-only staleness checking rather than hard-failing the whole
# chain.
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
skip_train() {  # skip_train <name>
    [ "$FORCE" = 1 ] && return 1
    [ -e "$MODELS/$1/${1}_final.h5" ]
}
# labels_sha1-aware "does this provenance file match the CURRENT labels"
# check, shared by step 2 (permutation) and step 3 (eval) done-markers.
# When LABELS_SHA1 is empty (digest unavailable this invocation) the
# comparison is skipped entirely -- existence of the file is enough,
# matching the WARNING printed above.
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
# An eval CSV's stem alone cannot prove it covers the full $EVAL_YEARS span
# (a manual debugging run can write the same stem) -- reuse the main
# chain's years/match_source discipline (copied deliberately, see header),
# extended with the labels_sha1 check above.
eval_run_matches() {  # eval_run_matches <run.json> <source>
    labels_match "$1" || return 1
    python3 - "$1" "$2" "$EVAL_YEARS" <<'PY' 2>/dev/null
import json, sys
path, source, span = sys.argv[1:4]
try:
    run = json.load(open(path))
except (OSError, ValueError):
    sys.exit(1)
first, last = span.split("-")
ok = (run.get("years") == list(range(int(first), int(last) + 1))
      and (source == "kriged-airs"
           or run.get("match_source") == "kriged-airs"))
sys.exit(0 if ok else 1)
PY
}
skip_perm() {   # skip_perm <csv-stem> <ckpt-path>   (stem = <ckpt>_<source>)
    [ "$FORCE" = 1 ] && return 1
    [ -e "$PERM_DIR/$1.csv" ] || return 1
    if labels_match "$PERM_DIR/${1}_run.json" \
       && ckpt_matches "$PERM_DIR/${1}_run.json" "$2"; then
        return 0
    fi
    note "step2 permutation $1: existing CSV was computed on labels or" \
         "checkpoint weights that have since changed (stale ${1}_run.json)" \
         "-- rerunning"
    return 1
}
# ckpt_matches <run.json> <ckpt.h5>: the recorded ckpt_sha1 must equal the
# SHA-1 of the checkpoint currently on disk -- a labels_sha1 match alone
# cannot notice a retrain on unchanged labels (same class of staleness the
# main chain's eval_run_matches guards since 2026-08-18).  A run.json with
# no ckpt_sha1 while the checkpoint exists counts as stale.
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
# skip_eval checks ABLATION_EVAL_DIR, not EVAL_DIR: a
# completed leg's CSV/paper/run.json are moved OUT of EVAL_DIR (where
# evaluate_test necessarily writes them -- see the header's "Results
# layout" note) into ABLATION_EVAL_DIR by move_to_ablation_dir below, so
# ABLATION_EVAL_DIR is this chain's true done-marker location; a leftover
# file in EVAL_DIR means the move step itself failed/hasn't run yet, not
# that the leg is done.
skip_eval() {   # skip_eval <csv-stem> <source>
    [ "$FORCE" = 1 ] && return 1
    [ -e "$ABLATION_EVAL_DIR/$1.csv" ] || return 1
    eval_run_matches "$ABLATION_EVAL_DIR/${1}_run.json" "$2" && return 0
    note "step3 eval $1: existing CSV is not a matched $EVAL_YEARS run on" \
         "current labels (stale/debug ${1}_run.json) -- rerunning"
    return 1
}
# move_to_ablation_dir <stem>: relocate one eval leg's artifacts from
# EVAL_DIR (evaluate_test's fixed, un-overridable output dir) to
# ABLATION_EVAL_DIR (this chain's own dir) so evaluate_test.compare()'s
# glob of EVAL_DIR never sees a reduced-channel ablation leg and pivots it
# into the main chain's three-way comparison.csv as if it were a peer of
# D6A/B/C.  `mv -f` (not cp): a duplicate left behind in EVAL_DIR serves no
# purpose and would itself be exactly the contamination this exists to
# prevent.  Per-file existence check (not a blanket `|| true`): a genuinely
# absent file (e.g. no _paper.json for a --no-match run) is tolerated, but
# a REAL move failure (ENOSPC, permissions) must surface -- the
# csv+run.json pair is what skip_eval's done-marker check needs.
move_to_ablation_dir() {  # move_to_ablation_dir <stem>
    local f
    for f in "$EVAL_DIR/$1.csv" "$EVAL_DIR/${1}_paper.json" \
             "$EVAL_DIR/${1}_run.json"; do
        if [ -e "$f" ]; then
            mv -f "$f" "$ABLATION_EVAL_DIR/"
        fi
    done
}

# join non-empty job ids with ':' (skipped phases contribute nothing) --
# identical helper to the main chain, copied deliberately (see header).
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
# move_to_ablation_dir -- an eval job's `mv` cannot run in THIS (submitting)
# shell because the eval job itself hasn't executed yet, so the move has to
# be its own tiny job, gated with --dependency=afterok on the eval job so
# it only runs (and only counts as done) once the CSV actually exists.
# `sbatch --wrap` rather than a new .sbatch file: this script may only edit
# itself, not add files under slurm/ (see task ownership).  Reuses the same
# partition/account opts as a CPU `submit` call (gpu=0) since this is a
# few-millisecond mv, never a compute job.
submit_move() {  # submit_move <label> <deps> <stem>
    local label=$1 deps=$2 stem=$3
    local opts=(--export=ALL)
    [ -n "${SBATCH_PARTITION:-}" ] && opts+=(-p "$SBATCH_PARTITION")
    [ -n "${SBATCH_ACCOUNT:-}" ] && opts+=(-A "$SBATCH_ACCOUNT")
    [ -n "$deps" ] && opts+=("--dependency=afterok:$deps")
    # same per-file loop as move_to_ablation_dir (absent files tolerated,
    # real mv failures surface and fail the job), built into a --wrap
    # string: \$f stays unexpanded here so the JOB's shell expands it.
    # `set -e` because --wrap's shell runs without it -- otherwise a later
    # missing-file iteration would mask an earlier mv failure's status.
    local wrap="set -e; for f in '$EVAL_DIR/$stem.csv' '$EVAL_DIR/${stem}_paper.json'"
    wrap="$wrap '$EVAL_DIR/${stem}_run.json'; do"
    wrap="$wrap if [ -e \"\$f\" ]; then mv -f \"\$f\" '$ABLATION_EVAL_DIR/'; fi; done"
    if [ "$DRY_RUN" = 1 ]; then
        DRYC=$((DRYC + 1)); SUBMIT_JID=DRY$DRYC
        note "DRY_RUN [$SUBMIT_JID] sbatch --wrap=\"$wrap\" (afterok:$deps)"
    else
        SUBMIT_JID=$(sbatch --parsable ${opts[@]+"${opts[@]}"} --wrap="$wrap")
        note "submitted $label -> job $SUBMIT_JID (afterok:$deps)"
    fi
    record "$label" "$SUBMIT_JID"
}

# ---- local (no-SLURM) runner: same commands, sequential foreground -------- #
run_local() {  # run_local <label> <module> <args...>
    local label=$1 status=0
    shift
    note "RUN python -m $*   (log: logs/$label.log)"
    if [ "$DRY_RUN" = 1 ]; then
        # never executed -- record a value distinguishable from local-done
        # so a DRY_RUN manifest can't be mistaken for a completed run
        record "$label" DRY_RUN
        return 0
    fi
    PYTHONPATH=src python -m "$@" > "logs/$label.log" 2>&1 || status=$?
    if [ "$status" = 0 ]; then
        record "$label" local-done
    else
        record "$label" "local-FAILED-exit$status"
    fi
    return "$status"
}

# the submitting/foreground shell needs the fronts-tf env whenever it runs
# python itself (the whole no-SLURM branch) -- identical to the main
# chain's activate_env, copied deliberately (see header).
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

    if run_step 2; then
        note "step2: permutation importance -- ckpts='$PERM_CKPTS'" \
             "repeats=$PERM_REPEATS years=$EVAL_YEARS"
        for k in $FOLDS; do
            for stage in $PERM_CKPTS; do
                ckpt=$stage-f$k
                # gate on ${ckpt}_final.h5, NOT the plain ${ckpt}.h5 that
                # train.py's ModelCheckpoint rewrites on every validation
                # improvement: the main chain may be training this exact
                # checkpoint concurrently (the runbook invites it), and
                # permuting a still-training 2-epoch snapshot would write
                # a complete, provenance-stamped CSV whose done-marker
                # then makes every later, correct run silently skip it.
                # train.py deletes _final.h5 at training start, so this
                # gate stays closed for the whole duration of any
                # (re)train, including a main-chain FORCE_TRAIN=1 retrain
                # (see the ORDERING note in the header).  --ckpt below
                # still points at the plain .h5 -- that IS the file
                # permutation should load (the best-val weights); only
                # the READINESS gate is _final.h5.
                if [ ! -e "$MODELS/$ckpt/${ckpt}_final.h5" ]; then
                    note "step2 skip $ckpt: no ${ckpt}_final.h5 at" \
                         "$MODELS/$ckpt/ (curriculum training not finished" \
                         "-- train it first, e.g. via the main chain; this" \
                         "script never trains step-2 checkpoints)"
                    continue
                fi
                for src in reanalysis kriged-airs; do
                    stem=${ckpt}_${src}
                    if skip_perm "$stem" "$MODELS/$ckpt/$ckpt.h5"; then
                        note "skip step2 $stem (matched CSV exists; FORCE=1" \
                             "to rerun)"
                        continue
                    fi
                    submit "step2-perm-$stem" 1 "" \
                           slurm/dlfront_permutation.sbatch \
                           --ckpt "$MODELS/$ckpt/$ckpt.h5" \
                           --classes "$CLASSES" --source "$src" \
                           --years "$EVAL_YEARS" --repeats "$PERM_REPEATS"
                done
            done
        done
    fi

    if run_step 3; then
        note "step3: stage-A channel ladder -- sets='$CHANNEL_SETS'." \
             "All rungs (5/3/2-channel) are trained fresh on the CURRENT" \
             "labels -- the main chain's D6A is NOT reused (it predates" \
             "the 2026-08-17 label fix; see the header note)."
        for k in $FOLDS; do
            for pair in $CHANNEL_SETS; do
                name=${pair%%:*}
                chans=${pair#*:}
                ckpt=$name-f$k

                JTRAIN=""
                if skip_train "$ckpt"; then
                    note "skip step3 train $ckpt (${ckpt}_final.h5 exists;" \
                         "FORCE=1 to rerun)"
                else
                    submit "step3-train-$ckpt" 1 "" \
                           slurm/dlfront_train.sbatch \
                           --name "$ckpt" --classes "$CLASSES" --fold "$k" \
                           --source reanalysis --channels "$chans"
                    JTRAIN=$SUBMIT_JID
                fi

                for src in reanalysis kriged-airs; do
                    stem=${ckpt}_${src}
                    if skip_eval "$stem" "$src"; then
                        note "skip step3 eval $stem (matched CSV exists;" \
                             "FORCE=1 to rerun)"
                        continue
                    fi
                    # --channels is redundant here -- C4 has evaluate_test
                    # auto-adopt run_args.channels from the checkpoint's
                    # run_config.yaml when --channels is not passed -- but
                    # it is passed anyway as belt-and-braces: if a future
                    # change to train.py or a hand-built checkpoint ever
                    # omits/mis-writes run_config.yaml, this still forces
                    # the correct channel count instead of silently
                    # evaluating a mismatched model with plausible-looking
                    # (but meaningless) numbers.
                    submit "step3-eval-$stem" 0 "$JTRAIN" \
                           slurm/dlfront_eval.sbatch \
                           --ckpt "$MODELS/$ckpt/$ckpt.h5" \
                           --classes "$CLASSES" --source "$src" \
                           --years "$EVAL_YEARS" --channels "$chans"
                    # relocate this leg's CSV out of EVAL_DIR once the eval
                    # job finishes (see the header's "Results layout" note
                    # and move_to_ablation_dir/submit_move above) -- MUST
                    # be its own afterok-gated job, not a command run in
                    # this submitting shell, since the eval job itself has
                    # not executed yet.
                    submit_move "step3-move-$stem" "$SUBMIT_JID" "$stem"
                done
            done
        done
    fi

    note "manifest: $MANIFEST"
    note "monitor with: squeue -u \$USER"
else
    # ================= no SLURM: sequential foreground ================= #
    note "sbatch not found: running the same steps sequentially in this shell"
    if [ "$DRY_RUN" != 1 ]; then
        activate_env
    fi

    if run_step 2; then
        note "step2: permutation importance -- ckpts='$PERM_CKPTS'" \
             "repeats=$PERM_REPEATS years=$EVAL_YEARS"
        for k in $FOLDS; do
            for stage in $PERM_CKPTS; do
                ckpt=$stage-f$k
                # same ${ckpt}_final.h5 readiness gate as the SLURM branch
                # above -- see the comment there for why plain ${ckpt}.h5
                # (rewritten on every improvement) is the wrong gate.
                if [ ! -e "$MODELS/$ckpt/${ckpt}_final.h5" ]; then
                    note "step2 skip $ckpt: no ${ckpt}_final.h5 at" \
                         "$MODELS/$ckpt/ (curriculum training not finished)"
                    continue
                fi
                for src in reanalysis kriged-airs; do
                    stem=${ckpt}_${src}
                    if skip_perm "$stem" "$MODELS/$ckpt/$ckpt.h5"; then
                        note "skip step2 $stem (matched CSV exists)"
                        continue
                    fi
                    run_local "step2-perm-$stem" dl_front.permutation \
                        --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                        --source "$src" --years "$EVAL_YEARS" \
                        --repeats "$PERM_REPEATS"
                done
            done
        done
    fi

    if run_step 3; then
        note "step3: stage-A channel ladder -- sets='$CHANNEL_SETS'." \
             "All rungs (5/3/2-channel) are trained fresh on the CURRENT" \
             "labels -- the main chain's D6A is NOT reused."
        for k in $FOLDS; do
            for pair in $CHANNEL_SETS; do
                name=${pair%%:*}
                chans=${pair#*:}
                ckpt=$name-f$k
                if skip_train "$ckpt"; then
                    note "skip step3 train $ckpt (final exists)"
                else
                    run_local "step3-train-$ckpt" dl_front.train \
                        --name "$ckpt" --classes "$CLASSES" --fold "$k" \
                        --source reanalysis --channels "$chans"
                fi
                for src in reanalysis kriged-airs; do
                    stem=${ckpt}_${src}
                    if skip_eval "$stem" "$src"; then
                        note "skip step3 eval $stem (matched CSV exists)"
                        continue
                    fi
                    run_local "step3-eval-$stem" dl_front.evaluate_test \
                        --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                        --source "$src" --years "$EVAL_YEARS" \
                        --channels "$chans"
                    # relocate this leg's CSV out of EVAL_DIR immediately
                    # (foreground: no dependency job needed) -- see the
                    # header's "Results layout" note.  A DRY_RUN skips the
                    # eval itself, so there is nothing to move; guard on it
                    # the same way run_local guards the eval command.
                    [ "$DRY_RUN" != 1 ] && move_to_ablation_dir "$stem"
                done
            done
        done
    fi

    note "manifest: $MANIFEST"
fi
