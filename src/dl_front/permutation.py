"""Single-pass permutation importance over the DL-FRONT input channels.

WHY this exists (user decision 2026-08-18): the model consumes five surface
channels, but only T2M and QV2M carry AIRS information -- U10M/V10M come
from the WRF-27km met driving HYSPLIT and SLP is copied clean from MERRA-2
("AIRS retrieves no SLP", ``krige_fill``).  Before spending GPU hours on the
stage-A channel ablation ladder (``scripts/dlfront_ablation_chain.sh``) we
want a cheap upper bound on how much skill actually lives in SLP and the
winds.  Permutation importance answers exactly that with ONE existing
checkpoint and no retraining.

Method (``front_finder.permutation``'s single-pass convention, which this
module deliberately mirrors but cannot reuse -- that one is 3-D
level x channel and bound to front_finder's dataset/predict): permuting a
channel means shuffling it along the SAMPLE axis with one fixed permutation
per (channel, repeat), every other channel and the whole spatial field of
each donor step held fixed.  That keeps each permuted channel a physically
real field -- just one from the wrong day -- so the score drop measures the
information the model draws from the channel's temporal co-variation with
the labels, not the model's response to noise.

Currency (user decision 2026-08-18): the scores are ``dl_front.evaluate``
neighborhood CSI/POD/FAR/FB at the same dilations and over the same scoring
mask as ``dl_front.evaluate_test``, so a permutation delta is directly
comparable to a cross-leg CSI delta from the test-eval table.  No bespoke
metric is invented here.

TWO different spreads are reported, and conflating them is the easy mistake:

* ``csi_lo``/``csi_hi`` -- SAMPLING uncertainty: the day-block circular
  bootstrap ``evaluate.block_bootstrap`` that ``evaluate_test`` attaches to
  every CSI it writes, for the reason recorded there (audit + user decision
  2026-08-15): "cross-leg CSI deltas -- dryline especially, ~150-190
  event-bearing steps/year and strongly autocorrelated -- are not
  interpretable without uncertainty; deltas inside overlapping CIs are
  sampling noise".  A permutation delta is a cross-leg CSI delta, so the
  same rule applies to it.  Present on every "raw" row, baseline and
  permuted alike.
* ``stat == "std"`` rows -- SHUFFLE uncertainty: the spread of the delta
  over ``--repeats`` draws of the permutation, on the SAME ~2000 steps.
  It answers "would another shuffle of this channel have given a different
  answer?", NOT "would another three years of days have?".  A tiny std
  therefore does not make a delta real.

How to read a delta: take the channel's ``csi`` on a row and the baseline
row's ``csi`` at the same (front, dilation).  If the two ``[csi_lo,
csi_hi]`` intervals overlap, the delta is within sampling noise and must
not be spent GPU hours on; only a delta whose permuted CI sits clear of the
baseline CI is evidence that the channel carries information.  Because
``block_bootstrap`` uses a fixed ``BOOT_SEED``, both legs are resampled on
the same day blocks, so the comparison is paired-in-days and the
overlapping-interval test is the conservative side of a proper paired-delta
test -- an overlap is a genuine "not shown", not a proof of no effect.

Cost note: this is (1 + n_channels * repeats) full inference passes AND the
same number of neighborhood-CSI passes (the dilation loop is usually the
more expensive half).  With five channels and ``--repeats 3`` that is 16
passes; budget the slurm wall clock accordingly
(``slurm/dlfront_permutation.sbatch`` asks for more time than the eval
script for this reason).

CLI::

    python -m dl_front.permutation \
        --ckpt $JPL_AIRS_RESULTS/dl_front/models/D6C-f0/D6C-f0.h5 \
        --classes 6 --source kriged-airs --years 2016-2018 --repeats 3

(``dl_front.train`` writes checkpoints to ``RESULTS_DIR/models/<name>/``,
which is also where both chain scripts look for them -- the ``models_v2``
tree on some dev boxes is a superseded hand-made copy, not the convention.
Integration note 2026-08-18.)

Outputs under ``$JPL_AIRS_RESULTS/dl_front/permutation/``:

* ``<ckpt-stem>_<source>.csv``       tidy per-(channel, repeat, front,
                                     dilation) table with the deltas and
                                     the day-block ``csi_lo``/``csi_hi``
* ``<ckpt-stem>_<source>_run.json``  the same provenance an eval leg
                                     writes, including ``labels_sha1``, the
                                     resolved ``channels`` and the
                                     kriged/clean input split
                                     ``kriged_channels``
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, dataset, evaluate, evaluate_test, predict

#: Channel name used for the un-permuted reference rows.  Angle brackets so
#: it can never collide with a real ``config.SFC_VARS`` entry.
BASELINE = "<baseline>"

#: Column order of the tidy permutation CSV (interface 2026-08-18).
#: ``stat`` distinguishes the per-repeat rows ("raw") from the per-channel
#: aggregate rows ("mean"/"std", emitted only when ``repeats > 1``); keeping
#: the aggregate in the SAME frame as long-form rows means a reader never
#: has to join two files, and ``df[df.stat == "raw"]`` recovers the raw
#: table exactly.
CSV_COLUMNS = ["channel", "repeat", "stat", "front", "dilation", "km",
               "csi", "csi_lo", "csi_hi", "pod", "far", "fb",
               "csi_delta", "pod_delta"]

#: ``repeat`` value on rows that are not one specific shuffle (the baseline
#: row and the aggregate rows).  A sentinel int rather than NaN/None so the
#: column survives a CSV round trip as an integer dtype (audit 2026-08-18:
#: pandas turns a column with one NaN into float64 and the chain scripts
#: then grep for "repeat,0.0").
NO_REPEAT = -1


# --------------------------------------------------------------------------- #
# Provenance (identical fields to an evaluate_test leg, C4/C5 2026-08-18)
# --------------------------------------------------------------------------- #
# The channel resolution / checkpoint-alignment / label-digest logic is
# IMPORTED from evaluate_test and dataset rather than reimplemented: a
# permutation table that adopted a checkpoint's channels by slightly
# different rules than the eval leg it is compared against would be a very
# quiet way to produce two incomparable numbers.  Only the thin assembly
# below is local.

def label_provenance(years, n_classes: int) -> dict:
    """``labels_sha1`` / ``labels_dir`` for the run json (C5 2026-08-18).

    Same two keys, same meanings, as an eval leg's ``_run.json``, so the
    chain scripts' staleness check works on a permutation artifact without
    knowing which kind of artifact it is looking at.
    """
    return {"labels_sha1": dataset.label_digest(years, n_classes),
            "labels_dir": str(evaluate_test.labels_dir(n_classes))}


# --------------------------------------------------------------------------- #
# The only place this module touches a model
# --------------------------------------------------------------------------- #

def predict_classes(model, x: np.ndarray, batch_size: int = 128
                    ) -> np.ndarray:
    """(n, 68, 141) argmax class grid from ``model.predict``.

    The sole model interaction in this module, so a numpy stub with a
    ``.predict`` method exercises :func:`single_pass` end to end without
    TensorFlow (``front_finder.permutation`` isolates the model behind
    ``predict.predict_batch`` for the same reason).

    Batched, and the argmax is taken per batch: only the class indices are
    kept (n x 68 x 141 uint8, ~20 MB for three years) instead of the full
    softmax (n x 68 x 141 x 6 float32, ~500 MB).  Nothing downstream of the
    argmax is needed here -- the paper metrics live in evaluate_test, this
    module only scores neighborhood CSI.
    """
    out = np.empty((len(x), *x.shape[1:3]), dtype=np.uint8)
    for start in range(0, len(x), batch_size):
        batch = np.asarray(x[start:start + batch_size], dtype=np.float32)
        probs = np.asarray(model.predict(batch, batch_size=batch_size,
                                         verbose=0))
        out[start:start + len(batch)] = probs.argmax(-1).astype(np.uint8)
    return out


def _score(cls: np.ndarray, y: np.ndarray, times, n_classes: int,
           mask: np.ndarray) -> pd.DataFrame:
    """Neighborhood CSI/POD/FAR/FB + day-block CSI CI per (front, dilation).

    ``evaluate.csi_counts`` returns PER-DAY contingency counts, which is
    exactly what ``evaluate.block_bootstrap`` consumes, so the SAMPLING
    uncertainty attached here is computed the same way and with the same
    knobs (``front_finder.config.BLOCK_DAYS`` / ``N_BOOT_REPS`` /
    ``CONFIDENCE_LEVEL`` / ``BOOT_SEED``) as the CIs on every
    ``evaluate_test`` CSI row -- see ``evaluate_test.evaluate_ckpt``.  Doing
    it here rather than only on the baseline is what
    makes a permutation delta readable: a delta far smaller than the CI
    width is sampling noise no matter how tight its across-repeat std is.

    ``evaluate.csi_scores`` returns a frame INDEXED by (front, dilation);
    the bootstrap bounds share that index, so they are assigned before
    reset_index() and callers get flat columns to merge on.
    """
    counts = evaluate.csi_counts(cls, y, times, n_classes, mask=mask)
    scores = evaluate.csi_scores(counts)
    boot = evaluate.block_bootstrap(counts)
    scores["csi_lo"] = boot.lo["csi"]
    scores["csi_hi"] = boot.hi["csi"]
    return scores.reset_index()


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #

def single_pass(model, x: np.ndarray, y: np.ndarray, times, n_classes: int,
                channels, rng, mask: np.ndarray | None = None,
                repeats: int = 1, batch_size: int = 128) -> pd.DataFrame:
    """Single-pass permutation importance over the input channels.

    ``x`` is (n, 68, 141, len(channels)) as produced by
    ``evaluate_test.load_year``; ``channels`` names its trailing axis IN
    ORDER (``config.INPUT_CHANNELS``).  ``y`` is the (n, 68, 141) truth
    class grid, ``times`` its DatetimeIndex.  ``mask`` defaults to the
    scoring region ``evaluate_test`` uses for the same class count
    (``dataset.analysis_domain()`` for 6, the Fig. 2 region mask for 5), so
    the deltas are in the same currency as the test-eval CSVs.

    One fixed permutation per (channel, repeat), applied along the SAMPLE
    axis only (front_finder's convention): each step's spatial field is
    handed over intact from a different day, which is what makes this
    permutation importance rather than noise injection.

    Returns a tidy frame with :data:`CSV_COLUMNS`.  ``csi_delta`` and
    ``pod_delta`` are ``baseline - permuted``, so POSITIVE means the model
    got worse without that channel's information, i.e. the channel matters.
    The baseline itself is present as its own row with
    ``channel == "<baseline>"`` and zero deltas, so the CSV is
    self-contained -- a reader never has to fetch the eval leg's CSV to
    interpret a delta.

    Every "raw" row also carries ``csi_lo``/``csi_hi``, the day-block
    bootstrap CI on that row's own CSI: a delta is only
    interpretable against them -- if a permuted leg's interval overlaps the
    baseline row's interval at the same (front, dilation), the delta is
    inside sampling noise.  This is deliberately NOT collapsed into a
    "significant" flag: the module reports the interval and the reader
    applies the repo's standing rule.  See the module docstring for how the
    CI differs from the across-repeat std.

    Cost: the baseline is scored ONCE and reused for every channel; only the
    permuted passes are repeated.  The bootstrap itself is negligible next
    to the dilation loop -- it resamples a (day, front, dilation, 4) count
    cube, never the grids.  Memory: one permuted copy of ``x`` exists
    at a time (never ``len(channels)`` of them), and the shuffled column is
    gathered directly into it rather than by fancy-indexing all of ``x``.
    """
    channels = list(channels)
    if x.shape[-1] != len(channels):
        raise ValueError(
            f"x has {x.shape[-1]} channel(s) but {len(channels)} channel "
            f"name(s) were given ({channels}); the array's trailing axis "
            f"must be config.INPUT_CHANNELS in order -- pass "
            f"'--channels {','.join(channels)}' consistently to the loader "
            f"and to this call")
    if mask is None:
        mask = (dataset.analysis_domain() if n_classes == 6
                else dataset.region_mask().astype(bool))

    base = _score(predict_classes(model, x, batch_size), y, times,
                  n_classes, mask)
    keys = ["front", "dilation"]
    base_ref = base[keys + ["csi", "pod"]].rename(
        columns={"csi": "_csi0", "pod": "_pod0"})

    rows = [base.assign(channel=BASELINE, repeat=NO_REPEAT, stat="raw",
                        csi_delta=0.0, pod_delta=0.0)]
    n = len(x)
    for ci, name in enumerate(channels):
        for rep in range(repeats):
            perm = rng.permutation(n)
            xp = x.copy()
            # Gather only the permuted column: x[perm, ..., ci] copies one
            # channel, whereas the more obvious x[perm][..., ci] would
            # materialise a second full-size array first.
            xp[..., ci] = x[perm, ..., ci]
            scored = _score(predict_classes(model, xp, batch_size), y, times,
                            n_classes, mask)
            del xp                       # free before the next channel's copy
            scored = scored.merge(base_ref, on=keys, how="left")
            scored["csi_delta"] = scored["_csi0"] - scored["csi"]
            scored["pod_delta"] = scored["_pod0"] - scored["pod"]
            rows.append(scored.drop(columns=["_csi0", "_pod0"]).assign(
                channel=name, repeat=rep, stat="raw"))

    df = pd.concat(rows, ignore_index=True)[CSV_COLUMNS]
    if repeats > 1:
        df = pd.concat([df, _aggregate(df)], ignore_index=True)
    return df


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-(channel, front, dilation) mean/std across the repeats.

    Spread across repeats is SHUFFLE variance -- how much of a delta is the
    luck of one draw of the permutation -- and is a different quantity from
    the ``csi_lo``/``csi_hi`` day-block CI, which is SAMPLING variance over
    the scored days (review 2026-08-18; see the module docstring for how to
    read the two together).  ``std`` is the sample std (ddof=1), undefined
    and therefore NaN for a single repeat -- which is why this is only
    called when ``repeats > 1``.  The baseline rows are excluded: they are
    not a distribution over shuffles.

    ``csi_lo``/``csi_hi`` are averaged on the "mean" rows (the typical CI a
    shuffle of this channel earns) but left NaN on the "std" rows: the
    scatter of a confidence bound across shuffles is not a quantity anyone
    reads, and printing it invites confusing it with the CI width.
    """
    perm = raw[raw["channel"] != BASELINE]
    value_cols = ["km", "csi", "csi_lo", "csi_hi", "pod", "far", "fb",
                  "csi_delta", "pod_delta"]
    grouped = perm.groupby(["channel", "front", "dilation"], as_index=False)
    out = []
    for stat in ("mean", "std"):
        agg = grouped[value_cols].agg(stat)
        agg["km"] = grouped[["km"]].first()["km"]   # km is a label, not data
        if stat == "std":
            agg[["csi_lo", "csi_hi"]] = np.nan
        out.append(agg.assign(repeat=NO_REPEAT, stat=stat))
    return pd.concat(out, ignore_index=True)[CSV_COLUMNS]


# --------------------------------------------------------------------------- #
# Data assembly (same comparability discipline as an eval leg)
# --------------------------------------------------------------------------- #

def load_arrays(years, n_classes: int, source: str, hours,
                match_source: str | None, info: dict | None = None):
    """Concatenate ``evaluate_test.load_year`` over ``years`` -> (x, y, times).

    Applies the SAME two-step comparability discipline as
    ``evaluate_test.evaluate_ckpt`` (module docstring there): the hours
    filter equalizes the label hours, and -- for a reanalysis run -- the
    timestamp intersection with the kriged-AIRS cache equalizes the days, so
    a permutation table computed on reanalysis inputs is comparable to one
    computed on kriged-AIRS inputs.  Deliberately NOT streamed year by year
    (unlike evaluate_ckpt): a permutation is a shuffle along the sample
    axis, which is only meaningful if every scored step is in memory at
    once.  Three years at the two AIRS hours is ~2000 x 68 x 141 x 5
    float16, ~190 MB.
    """
    stats = dataset.load_norm_stats()
    xs, ys, ts, n_steps = [], [], [], {}
    for year in years:
        x, y, times = dataset.filter_hours(
            *evaluate_test.load_year(year, n_classes, stats, source), hours)
        if match_source is not None and match_source != source:
            ref = evaluate_test.kriged_cache_times(year, match_source)
            keep = times.isin(ref)
            if (~keep).any():
                print(f"{year}: {int((~keep).sum())} steps absent from the "
                      f"{match_source} cache dropped for comparability",
                      flush=True)
            x, y, times = x[keep], y[keep], times[keep]
        n_steps[year] = len(x)
        print(f"{year}: {len(x)} steps loaded", flush=True)
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        ts.append(times)
    if not xs:
        raise RuntimeError(f"no data found for years {list(years)} at hours "
                           f"{tuple(hours)} (source={source})")
    times = ts[0].append(ts[1:])
    if info is not None:
        info.update(
            n_steps_per_year={int(y): int(n) for y, n in n_steps.items()},
            times_sha1=hashlib.sha1("\n".join(
                str(t) for t in sorted(times)).encode()).hexdigest(),
            match_source=match_source)
    return np.concatenate(xs), np.concatenate(ys), times


def write_outputs(df: pd.DataFrame, ckpt, source: str, years, hours,
                  channels, repeats: int, seed: int,
                  out_dir: Path | None = None, out_path: Path | None = None,
                  info: dict | None = None) -> dict:
    """Write the CSV + provenance json; return their paths.

    The provenance fields mirror an eval leg's ``_run.json`` (same keys,
    same meanings) so the two artifact families can be checked by the same
    chain-script helpers -- notably ``labels_sha1``, which is what lets a
    chain notice that a stored table was computed on the pre-2026-08-17
    front labels.
    """
    if out_path is not None:
        out_path = Path(out_path)
        out_dir = out_path.parent
        stem = out_path.stem
    else:
        out_dir = Path(out_dir) if out_dir is not None \
            else config.RESULTS_DIR / "permutation"
        stem = f"{Path(ckpt).stem}_{source}"
        out_path = out_dir / f"{stem}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    run_path = out_dir / f"{stem}_run.json"
    run_path.write_text(json.dumps(
        {"ckpt": str(Path(ckpt).resolve()),
         # weights fingerprint, same key as an eval leg: lets the ablation
         # chain's skip_perm notice that the checkpoint was retrained since
         # this CSV was written (a labels_sha1 match alone cannot -- a
         # retrain on unchanged labels keeps the same digest)
         **({"ckpt_sha1": evaluate_test.ckpt_sha1(ckpt)}
            if Path(ckpt).exists() else {}),
         "source": source,
         "years": [int(y) for y in years], "hours": [int(h) for h in hours],
         "channels": list(channels), "repeats": int(repeats),
         "seed": int(seed),
         **(info or {}),
         "git_rev": evaluate_test._git_rev(),
         "created": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        indent=1))
    return {"csv": out_path, "run": run_path}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(
        description="Single-pass permutation importance over the DL-FRONT "
                    "input channels: how much neighborhood CSI/POD does the "
                    "model lose when one channel is shuffled along the "
                    "sample axis?  Sizes how much skill lives in the "
                    "non-AIRS channels (SLP is MERRA-2, U10M/V10M are WRF) "
                    "before spending GPU on the channel ablation ladder.")
    ap.add_argument("--ckpt", required=True,
                    help="trained .h5 checkpoint to probe")
    ap.add_argument("--classes", type=int, default=6, choices=(5, 6))
    ap.add_argument("--source", default="reanalysis",
                    choices=("reanalysis", "kriged-airs"),
                    help="input fields to probe (bk19 has no model to "
                         "permute)")
    ap.add_argument("--years", default=None,
                    help="'A-B' range or comma list (default: the yaml eval "
                         f"split of --classes, e.g. "
                         f"{list(config.EVAL_YEARS_6)} for 6)")
    ap.add_argument("--hours", default=None,
                    help="comma-separated UTC hours to score "
                         f"(default: AIRS hours {config.AIRS_HOURS})")
    ap.add_argument("--channels", default=None,
                    help="comma-separated subset of "
                         f"{list(config.SFC_VARS)} the checkpoint consumes "
                         "(default: adopt the checkpoint's "
                         "run_config.yaml, else the configured default)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="permutations per channel; >1 adds mean/std "
                         "aggregate rows (cost is linear in this)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the permutations (recorded in the "
                         "run json, so a table is reproducible)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None,
                    help="explicit output CSV path (default: "
                         "$JPL_AIRS_RESULTS/dl_front/permutation/"
                         "<ckpt-stem>_<source>.csv)")
    ap.add_argument("--no-match", action="store_true",
                    help="do NOT intersect a reanalysis run's time steps "
                         "with the kriged-airs cache (the result is then "
                         "NOT comparable to a kriged-airs run)")
    a = ap.parse_args(argv)

    years = evaluate_test.resolve_years(a.years, a.classes)
    hours = (tuple(int(h) for h in a.hours.split(","))
             if a.hours else config.AIRS_HOURS)
    match_source = ("kriged-airs"
                    if a.source == "reanalysis" and not a.no_match else None)
    # Channels first: they decide how many channels the loader stacks, so
    # this MUST happen before any data loading or model construction.
    channels = evaluate_test.resolve_channels(a.ckpt, a.channels)

    model = predict.load_model(a.ckpt)
    evaluate_test.check_model_channels(model, a.ckpt)
    # INPUT provenance, same guard and same position as evaluate_test.main
    # (before any data is loaded): which input channels come from the kriged
    # cache is a yaml tunable, so a checkpoint trained under a different
    # airs.kriged_channels would otherwise be silently fed e.g. clean
    # reanalysis winds where it learned kriged ones, corrupting every
    # importance below.  Raises with the remedy on a true mismatch; the
    # returned split is recorded in the run json like an eval leg's.
    info: dict = {"kriged_channels":
                  evaluate_test.check_kriged_split(a.ckpt, a.source)}
    x, y, times = load_arrays(years, a.classes, a.source, hours,
                              match_source, info=info)
    info.update(label_provenance(years, a.classes))
    df = single_pass(model, x, y, times, a.classes, channels,
                     np.random.default_rng(a.seed), repeats=a.repeats,
                     batch_size=a.batch_size)
    paths = write_outputs(df, a.ckpt, a.source, years, hours, channels,
                          a.repeats, a.seed,
                          out_path=Path(a.out) if a.out else None, info=info)

    # Headline: the mean delta at the coarsest dilation, biggest first --
    # the number the ablation decision is actually made on.
    summary = df[(df["channel"] != BASELINE)
                 & (df["stat"] == ("mean" if a.repeats > 1 else "raw"))]
    coarse = summary[summary["dilation"] == summary["dilation"].max()]
    print("\nCSI drop when each channel is permuted "
          f"(dilation {int(coarse['dilation'].max())} = "
          f"{int(coarse['km'].max())} km):")
    print(coarse.pivot_table(index="channel", columns="front",
                             values="csi_delta")
          .to_string(float_format=lambda v: f"{v:+.4f}"))
    # ... and immediately next to it the yardstick those deltas have to
    # clear: the baseline's day-block bootstrap CI at
    # the same dilation.  A delta printed above that is smaller than the
    # half-width printed below is sampling noise, and the operator should
    # see both numbers in the same screenful rather than have to open the
    # CSV to find out.
    ref = df[(df["channel"] == BASELINE)
             & (df["dilation"] == df["dilation"].max())]
    print("\nbaseline CSI with its day-block bootstrap CI at the same "
          "dilation (a delta inside this width is sampling noise):")
    print(ref.assign(ci=lambda d: d["csi_hi"] - d["csi_lo"])
          [["front", "csi", "csi_lo", "csi_hi", "ci"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nwrote " + "\n      ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    sys.exit(main())
