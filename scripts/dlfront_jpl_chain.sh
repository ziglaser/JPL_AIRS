#!/bin/bash
# scripts/dlfront_jpl_chain.sh -- submit (SLURM) or run (plain bash) the full
# dl_front JPL curriculum:
#
#   Phase 0  (optional, --with-acquire, foreground -- needs internet+~/.netrc):
#            dl_front.acquire_merra2_sfc 2016..2021 (test-period reanalysis)
#   Pre-2a   (optional, --with-swath-bank): swath build-bank --years 2007-2021
#            -> $JPL_AIRS_DATA/masks/swath_bank.npz (16-day coverage
#            climatology for gap_type); 2a and 3a depend on it     [CPU]
#   Phase 1  per fold k: D6A-f<k>  train --source reanalysis        [GPU]
#   Phase 2a           : krige_fill build-degraded --years 2007-2015 [CPU]
#   Phase 2b per fold k: D6B-f<k>  train --source kriged-degraded
#                        --retrain D6A best          (afterok: 1[k] + 2a)
#   Phase 3a           : krige_fill build-airs --years 2007-2021     [CPU]
#   Phase 3b per fold k: D6C-f<k>  train --source kriged-airs
#                        --retrain D6B best          (afterok: 2b[k] + 3a)
#   Phase 4  per fold k: evaluate_test on 2016-2018 for EACH stage checkpoint
#                        D6A-f<k>, D6B-f<k>, D6C-f<k>, each with --source
#                        reanalysis AND --source kriged-airs (6 evals/fold);
#            once      : the checkpoint-free BK19 published-prediction leg
#                        (--source bk19), and a final `evaluate_test compare`
#                        job pivoting every leg CSV into comparison.csv.
#                        Years 2016-2018 everywhere (user decision 2026-08-13:
#                        the BK19 archive ends 2018, identical years for the
#                        three-way comparison).
#
# Env knobs (all optional unless noted; manifest reorg 2026-08-13 -- on the
# cluster (gattaca2) everything lives under /gpfs/scratch/smap-convection):
#   JPL_AIRS_REPO      repo checkout            (default: $PWD)
#   JPL_AIRS_DATA      data root                (default: $JPL_AIRS_REPO/data)
#                      cluster: /gpfs/scratch/smap-convection/AIRS_SMAP_Front_data
#   JPL_AIRS_FCST      AIRS-FCST fullgrid root  (default: the sibling
#                      $JPL_AIRS_DATA/../AIRS_FCST_1deg when it exists,
#                      else $JPL_AIRS_DATA/HYSPLIT_demo -- mirrors
#                      dl_front.config._resolve_airs_fcst_root)
#                      cluster: /gpfs/scratch/smap-convection/AIRS_FCST_1deg
#   JPL_AIRS_RESULTS   results root             (default: $JPL_AIRS_REPO/results)
#   JPL_BK19_DIR       BK19 published-prediction root (default:
#                      $JPL_AIRS_DATA/front_id/predicted_fronts/bk19 -- now
#                      inside the data root on the cluster too, so the
#                      override is rarely needed)
#   CONDA_PREFIX_ROOT  conda install root       (default: $HOME/miniconda3)
#   SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_GRES  injected at submit time
#   CLASSES            default 6
#   FOLDS              default "0 1 2"
#   WARM_START         optional existing stage-A .h5 to --retrain phase 1 from
#   KRIGE_WORKERS      default 8 (matches dlfront_krige.sbatch cpus)
#   FORCE=1            resubmit phases whose done-marker already exists
#   DRY_RUN=1          print every sbatch/python command without executing
#
# Idempotency: a train phase is skipped when
#   $JPL_AIRS_RESULTS/dl_front/models/<name>/<name>_final.h5 exists;
# krige phases when EVERY year's cache file exists (the builder itself also
# resumes per year, writing an empty cache for years with no coverage); eval
# runs (bk19 included) when their CSV exists AND its _run.json proves it is
# the matched full-span run (years == the 2016-2018 span; reanalysis/bk19
# legs additionally matched to the kriged-airs time axis -- a stale
# --no-match or partial-years debugging CSV is rerun, never compared); the
# compare job when comparison.csv exists AND no eval ran this submission.
# FORCE=1 overrides
# all of these (and passes --force to the krige builders so existing year
# caches are rebuilt).
#
# No SLURM?  The script detects the absence of `sbatch` and runs the exact
# same steps sequentially in the foreground (see the runbook for
# `setsid nohup` guidance).
set -euo pipefail

WITH_ACQUIRE=0
WITH_SWATH_BANK=0
for arg in "$@"; do
    case "$arg" in
        --with-acquire) WITH_ACQUIRE=1 ;;
        --with-swath-bank) WITH_SWATH_BANK=1 ;;
        # help = the header comment block only (everything up to the first
        # non-comment line), so code changes can never leak into --help
        -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' \
                   | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg" \
                "(accepted: --with-acquire, --with-swath-bank)" >&2
           exit 2 ;;
    esac
done

export JPL_AIRS_REPO=${JPL_AIRS_REPO:-$PWD}
cd "$JPL_AIRS_REPO"
export JPL_AIRS_DATA=${JPL_AIRS_DATA:-$JPL_AIRS_REPO/data}
# manifest reorg 2026-08-13: the AIRS-FCST archive is a SIBLING of the data
# root (cluster: /gpfs/scratch/smap-convection/AIRS_FCST_1deg); fall back to
# the local demo day -- same resolution as dl_front.config.
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
WARM_START=${WARM_START:-}
KRIGE_WORKERS=${KRIGE_WORKERS:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
# phase-4 test years (user decision 2026-08-13): the BK19 published
# predictions end 2018, so the three-way test uses identical 2016-2018 years
# for every leg (matches configs/dl_front.yaml eval_years_6class)
EVAL_YEARS=2016-2018

MODELS=$JPL_AIRS_RESULTS/dl_front/models
# kriged cache dirs (manifest reorg 2026-08-13; = config.KRIGED_SOURCE_DIRS)
KRIGED_DEGRADED=$JPL_AIRS_DATA/front_id/degraded_reanalysis
KRIGED_AIRS=$JPL_AIRS_DATA/front_id/kriged_airs_fcst
SWATH_BANK=$JPL_AIRS_DATA/masks/swath_bank.npz
mkdir -p logs "$JPL_AIRS_RESULTS/dl_front"
MANIFEST=$JPL_AIRS_RESULTS/dl_front/chain_$(date +%Y%m%d_%H%M%S).txt
{
    echo "# dl_front JPL chain manifest  $(date -Is)"
    echo "# repo=$JPL_AIRS_REPO data=$JPL_AIRS_DATA fcst=$JPL_AIRS_FCST"
    echo "# results=$JPL_AIRS_RESULTS classes=$CLASSES folds='$FOLDS'"
    echo "# warm_start='${WARM_START}' force=$FORCE dry_run=$DRY_RUN"
} > "$MANIFEST"

HAVE_SLURM=0
command -v sbatch > /dev/null 2>&1 && HAVE_SLURM=1

note() { echo "[chain] $*" >&2; }
record() { echo "$1=$2" >> "$MANIFEST"; }

# ---- skip predicates (FORCE=1 defeats all) -------------------------------- #
skip_train() {  # skip_train <name>
    [ "$FORCE" = 1 ] && return 1
    [ -e "$MODELS/$1/$1_final.h5" ]
}
# cache_is_v3 <cache.nc>: 0 iff the cache carries schema_version=3.
# Submit-time schema probe (review 2026-08-13): a cluster whose kriged cache
# dirs predate the 2026-08-13 domain decision holds v1/v2 caches that pass
# the existence check --
# 2a/3a would be skipped and the 2b/3b GPU jobs would die at data-load
# time in the loader's schema guard, wasting the allocation.  When the
# submitting shell has no netCDF reader (SLURM branch may run outside
# fronts-tf), fall back to existence-only: the loader still guards at run
# time.
cache_is_v3() {
    python3 - "$1" <<'PY' 2>/dev/null
import sys
path = sys.argv[1]
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
    sys.exit(1)                 # unreadable cache: not a valid v3 file
sys.exit(0 if v == 3 else 1)
PY
}
skip_krige() {  # skip_krige <cache-dir> <first-year> <last-year>
    # every year must exist: the builder writes even zero-step years, so a
    # missing file means unfinished work, not an empty year
    [ "$FORCE" = 1 ] && return 1
    local y f
    for ((y = $2; y <= $3; y++)); do
        f=$1/kriged_sfc_$y.nc
        [ -e "$f" ] || return 1
        if ! cache_is_v3 "$f"; then
            # resubmitting without FORCE cannot fix this: the builder
            # skips existing year files, so the stale cache would survive
            note "ERROR: $f is not a schema-v3 cache (pre-domain v1/v2 or" \
                 "unreadable).  Rerun with FORCE=1 (passes --force to the" \
                 "krige builders) or delete $1 first."
            exit 3
        fi
    done
    return 0
}
KRIGE_FORCE=()
[ "$FORCE" = 1 ] && KRIGE_FORCE=(--force)
skip_bank() {   # the bank npz itself is the done-marker
    [ "$FORCE" = 1 ] && return 1
    [ -e "$SWATH_BANK" ]
}
# An eval CSV's filename encodes only ckpt+source, so its existence alone
# cannot prove it is the run the three-way comparison needs: a manual
# debugging run (--no-match, partial --years) writes the SAME stem.  Trust
# only a _run.json provenance recording the full $EVAL_YEARS span and (for
# reanalysis/bk19 legs, which would otherwise score every step) the
# kriged-airs time-axis match; kriged-airs legs ARE the cache steps, so no
# match is recorded for them (evaluate_test sets match_source=null there).
eval_run_matches() {  # eval_run_matches <run.json> <source>
    python3 - "$1" "$2" "$EVAL_YEARS" <<'PY'
import json, sys
path, source, span = sys.argv[1:4]
try:
    run = json.load(open(path))
except (OSError, ValueError):
    sys.exit(1)                     # missing/broken provenance: rerun
first, last = span.split("-")
ok = (run.get("years") == list(range(int(first), int(last) + 1))
      and (source == "kriged-airs"
           or run.get("match_source") == "kriged-airs"))
sys.exit(0 if ok else 1)
PY
}
skip_eval() {   # skip_eval <csv-stem> <source>   (stem <ckpt>_<source>, or 'bk19')
    [ "$FORCE" = 1 ] && return 1
    local dir=$JPL_AIRS_RESULTS/dl_front/test_eval
    [ -e "$dir/$1.csv" ] || return 1
    eval_run_matches "$dir/${1}_run.json" "$2" && return 0
    note "phase4 eval $1: existing CSV is not a matched $EVAL_YEARS run" \
         "(stale/debug ${1}_run.json) -- rerunning"
    return 1
}

# join non-empty job ids with ':' (skipped phases contribute nothing)
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
    # --export=ALL: the JPL_AIRS_* roots MUST reach the job even on clusters
    # whose slurm defaults force --export=NONE (silent repo-local fallbacks
    # otherwise)
    local opts=(--export=ALL)
    [ -n "${SBATCH_PARTITION:-}" ] && opts+=(-p "$SBATCH_PARTITION")
    [ -n "${SBATCH_ACCOUNT:-}" ]   && opts+=(-A "$SBATCH_ACCOUNT")
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

# ---- local (no-SLURM) runner: same commands, sequential foreground -------- #
run_local() {  # run_local <label> <module> <args...>
    local label=$1
    shift
    note "RUN python -m $*   (log: logs/$label.log)"
    if [ "$DRY_RUN" != 1 ]; then
        PYTHONPATH=src python -m "$@" > "logs/$label.log" 2>&1
    fi
    record "$label" local-done
}

# the submitting shell needs the fronts-tf env whenever it runs python
# itself (phase 0 and the whole no-SLURM branch) -- a fresh login shell's
# system python lacks xarray/requests and would abort the chain under set -e
activate_env() {
    [ "${CONDA_DEFAULT_ENV:-}" = fronts-tf ] && return 0
    source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    conda activate fronts-tf
}

# ---- Phase 0: acquire 2016-2021 sfc_daily (foreground: needs internet) ---- #
if [ "$WITH_ACQUIRE" = 1 ]; then
    note "phase0: acquiring 2016-2021 sfc_daily reanalysis (Earthdata ~/.netrc" \
         "required; runs in the foreground on this node)"
    if [ "$DRY_RUN" = 1 ]; then
        note "DRY_RUN python -m dl_front.acquire_merra2_sfc 2016 2017 2018 2019 2020 2021"
    else
        activate_env
        PYTHONPATH=src python -m dl_front.acquire_merra2_sfc \
            2016 2017 2018 2019 2020 2021 2>&1 | tee logs/dlfront_acquire_sfc.log
    fi
    record phase0-acquire foreground
fi

if [ "$HAVE_SLURM" = 1 ]; then
    # ================= SLURM dependency chain ================= #
    note "SLURM detected: submitting dependency chain"

    # pre-phase: swath-coverage climatology (gap_type input for 2a/3a caches)
    JSB=""
    if [ "$WITH_SWATH_BANK" = 1 ]; then
        if skip_bank; then
            note "skip pre-swath-bank ($SWATH_BANK exists; FORCE=1 to rebuild)"
        else
            submit pre-swath-bank 0 "" slurm/dlfront_swath_bank.sbatch \
                   build-bank --years 2007-2021
            JSB=$SUBMIT_JID
        fi
    fi

    J2A=""
    if skip_krige "$KRIGED_DEGRADED" 2007 2015; then
        note "skip phase2a (kriged degraded caches 2007-2015 exist; FORCE=1 to rebuild)"
    else
        submit phase2a-krige-degraded 0 "$JSB" slurm/dlfront_krige.sbatch \
               build-degraded --years 2007-2015 --workers "$KRIGE_WORKERS" \
               ${KRIGE_FORCE[@]+"${KRIGE_FORCE[@]}"}
        J2A=$SUBMIT_JID
    fi

    J3A=""
    if skip_krige "$KRIGED_AIRS" 2007 2021; then
        note "skip phase3a (kriged AIRS caches 2007-2021 exist; FORCE=1 to rebuild)"
    else
        submit phase3a-krige-airs 0 "$JSB" slurm/dlfront_krige.sbatch \
               build-airs --years 2007-2021 --workers "$KRIGE_WORKERS" \
               ${KRIGE_FORCE[@]+"${KRIGE_FORCE[@]}"}
        J3A=$SUBMIT_JID
    fi

    EVAL_JIDS=()   # every phase-4 eval submitted this run (compare-job deps)
    for k in $FOLDS; do
        A=D6A-f$k B=D6B-f$k C=D6C-f$k

        J1=""
        if skip_train "$A"; then
            note "skip phase1 $A (${A}_final.h5 exists; FORCE=1 to rerun)"
        else
            args=(--name "$A" --classes "$CLASSES" --fold "$k" --source reanalysis)
            [ -n "$WARM_START" ] && args+=(--retrain "$WARM_START")
            submit "phase1-$A" 1 "" slurm/dlfront_train.sbatch "${args[@]}"
            J1=$SUBMIT_JID
        fi

        J2B=""
        if skip_train "$B"; then
            note "skip phase2b $B (${B}_final.h5 exists; FORCE=1 to rerun)"
        else
            submit "phase2b-$B" 1 "$(join_deps "$J1" "$J2A")" \
                   slurm/dlfront_train.sbatch --name "$B" --classes "$CLASSES" \
                   --fold "$k" --source kriged-degraded --retrain "$MODELS/$A/$A.h5"
            J2B=$SUBMIT_JID
        fi

        J3B=""
        if skip_train "$C"; then
            note "skip phase3b $C (${C}_final.h5 exists; FORCE=1 to rerun)"
        else
            submit "phase3b-$C" 1 "$(join_deps "$J2B" "$J3A")" \
                   slurm/dlfront_train.sbatch --name "$C" --classes "$CLASSES" \
                   --fold "$k" --source kriged-airs --retrain "$MODELS/$B/$B.h5"
            J3B=$SUBMIT_JID
        fi

        for ckpt in "$A" "$B" "$C"; do
            for src in reanalysis kriged-airs; do
                if skip_eval "${ckpt}_${src}" "$src"; then
                    note "skip phase4 eval $ckpt/$src (matched CSV exists; FORCE=1 to rerun)"
                    continue
                fi
                # each eval waits on its OWN stage's train job (which
                # transitively covers the earlier phases) -- plus 3a ALWAYS:
                # kriged-airs runs read the cache fields, and reanalysis runs
                # intersect their time steps with the cache's time axis
                # (evaluate_test comparability guarantee).  Skipped phases
                # drop out.
                case "$ckpt" in
                    "$A") deps=$J1 ;;
                    "$B") deps=$J2B ;;
                    *)    deps=$J3B ;;
                esac
                deps=$(join_deps "$deps" "$J3A")
                submit "phase4-eval-$ckpt-$src" 0 "$deps" \
                       slurm/dlfront_eval.sbatch \
                       --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                       --source "$src" --years "$EVAL_YEARS"
                EVAL_JIDS+=("$SUBMIT_JID")
            done
        done
    done

    # fold-independent BK19 published-prediction leg (checkpoint-free); needs
    # 3a only, for the time-axis intersection with the kriged-airs cache
    if skip_eval bk19 bk19; then
        note "skip phase4 eval bk19 (matched CSV exists; FORCE=1 to rerun)"
    else
        submit phase4-eval-bk19 0 "$J3A" slurm/dlfront_eval.sbatch \
               --source bk19 --classes "$CLASSES" --years "$EVAL_YEARS"
        EVAL_JIDS+=("$SUBMIT_JID")
    fi

    # final three-way comparison table over every leg CSV in test_eval/
    if [ ${#EVAL_JIDS[@]} -eq 0 ] && [ "$FORCE" != 1 ] \
       && [ -e "$JPL_AIRS_RESULTS/dl_front/test_eval/comparison.csv" ]; then
        note "skip phase4 compare (comparison.csv exists, no new evals; FORCE=1 to rerun)"
    else
        submit phase4-compare 0 \
               "$(join_deps ${EVAL_JIDS[@]+"${EVAL_JIDS[@]}"})" \
               slurm/dlfront_eval.sbatch compare
    fi

    note "manifest: $MANIFEST"
    note "monitor with: squeue -u \$USER"
else
    # ================= no SLURM: sequential foreground ================= #
    note "sbatch not found: running the same steps sequentially in this shell"
    note "(long! consider: setsid nohup bash scripts/dlfront_jpl_chain.sh" \
         "> logs/chain.out 2>&1 & -- see docs/JPL_DEPLOYMENT_DLFRONT.md)"
    if [ "$DRY_RUN" != 1 ]; then
        activate_env
    fi

    if [ "$WITH_SWATH_BANK" = 1 ]; then
        if skip_bank; then
            note "skip pre-swath-bank ($SWATH_BANK exists)"
        else
            run_local pre-swath-bank dl_front.swath build-bank --years 2007-2021
        fi
    fi
    if skip_krige "$KRIGED_DEGRADED" 2007 2015; then
        note "skip phase2a (caches exist)"
    else
        run_local phase2a-krige-degraded dl_front.krige_fill \
            build-degraded --years 2007-2015 --workers "$KRIGE_WORKERS" \
            ${KRIGE_FORCE[@]+"${KRIGE_FORCE[@]}"}
    fi
    if skip_krige "$KRIGED_AIRS" 2007 2021; then
        note "skip phase3a (caches exist)"
    else
        run_local phase3a-krige-airs dl_front.krige_fill \
            build-airs --years 2007-2021 --workers "$KRIGE_WORKERS" \
            ${KRIGE_FORCE[@]+"${KRIGE_FORCE[@]}"}
    fi

    NEW_EVALS=0   # evals run this pass (compare-job trigger)
    for k in $FOLDS; do
        A=D6A-f$k B=D6B-f$k C=D6C-f$k
        if skip_train "$A"; then
            note "skip phase1 $A (final exists)"
        else
            args=(--name "$A" --classes "$CLASSES" --fold "$k" --source reanalysis)
            [ -n "$WARM_START" ] && args+=(--retrain "$WARM_START")
            run_local "phase1-$A" dl_front.train "${args[@]}"
        fi
        if skip_train "$B"; then
            note "skip phase2b $B (final exists)"
        else
            run_local "phase2b-$B" dl_front.train --name "$B" \
                --classes "$CLASSES" --fold "$k" --source kriged-degraded \
                --retrain "$MODELS/$A/$A.h5"
        fi
        if skip_train "$C"; then
            note "skip phase3b $C (final exists)"
        else
            run_local "phase3b-$C" dl_front.train --name "$C" \
                --classes "$CLASSES" --fold "$k" --source kriged-airs \
                --retrain "$MODELS/$B/$B.h5"
        fi
        for ckpt in "$A" "$B" "$C"; do
            for src in reanalysis kriged-airs; do
                if skip_eval "${ckpt}_${src}" "$src"; then
                    note "skip phase4 eval $ckpt/$src (matched CSV exists)"
                else
                    run_local "phase4-eval-$ckpt-$src" dl_front.evaluate_test \
                        --ckpt "$MODELS/$ckpt/$ckpt.h5" --classes "$CLASSES" \
                        --source "$src" --years "$EVAL_YEARS"
                    NEW_EVALS=$((NEW_EVALS + 1))
                fi
            done
        done
    done

    # fold-independent BK19 published-prediction leg (checkpoint-free)
    if skip_eval bk19 bk19; then
        note "skip phase4 eval bk19 (matched CSV exists)"
    else
        run_local phase4-eval-bk19 dl_front.evaluate_test \
            --source bk19 --classes "$CLASSES" --years "$EVAL_YEARS"
        NEW_EVALS=$((NEW_EVALS + 1))
    fi

    # final three-way comparison table over every leg CSV in test_eval/
    if [ "$NEW_EVALS" -eq 0 ] && [ "$FORCE" != 1 ] \
       && [ -e "$JPL_AIRS_RESULTS/dl_front/test_eval/comparison.csv" ]; then
        note "skip phase4 compare (comparison.csv exists, no new evals)"
    else
        run_local phase4-compare dl_front.evaluate_test compare
    fi

    note "manifest: $MANIFEST"
fi
