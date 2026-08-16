"""Held-out test evaluation: three-way checkpoint/benchmark comparison.

Scores one evaluation "leg" on the held-out test years (yaml
``splits.eval_years_*``; 2016-2018 for 6-class, user decision 2026-08-13),
from one of three input sources:

* ``reanalysis``  -- clean MERRA-2 surface fields (``dataset.year_arrays``),
* ``kriged-airs`` -- kriged gap-filled AIRS-FCST fields
  (``dataset.kriged_year_arrays``; caches built by ``dl_front.krige_fill``),
* ``bk19``        -- the PUBLISHED Biard & Kunkel (2019) DL-FRONT prediction
  rasters (``config.BK19_DIR``; no checkpoint involved -- the files ARE the
  model output, hard binary classes, no dryline).

All legs are restricted to the SAME time steps, two ways:

* an hours filter (default the AIRS-covered label hours ``config.AIRS_HOURS``)
  equalizes the label hours, and
* a TIMESTAMP intersection with the kriged-AIRS cache (default for
  ``--source reanalysis`` and ``--source bk19``; disable with ``--no-match``)
  equalizes the days: the AIRS archive is sparse, so without it those legs
  would score every test-period step while the kriged run scores only
  AIRS-covered days, confounding input quality with sample composition.

The legs are then directly comparable (the only difference is the input
fields / predictor, never the sample set) -- but note the matching is
ONE-DIRECTIONAL: it only DROPS steps absent from the cache, it cannot
resurrect steps a leg is missing on its own (e.g. a partially downloaded
``sfc_daily`` year makes ``dataset.year_arrays`` silently skip days, so the
reanalysis leg could score fewer steps than the others).  The per-year step
counts and a SHA-1 of the scored timestamps therefore land in the
``_run.json`` provenance file, and :func:`compare` cross-checks the SHA-1s
across legs, warning loudly when they did not score identical steps.

Outputs under ``results/dl_front/test_eval/`` (created on demand):

* ``<ckpt-stem>_<source>.csv``        tidy neighborhood-CSI table
                                      (front, dilation, km, csi, pod, far, fb)
                                      (the bk19 leg's stem is just ``bk19``)
* ``<ckpt-stem>_<source>_paper.json`` paper metrics: accuracy dict, ROC AUC,
                                      per-class confusion (% of masked cells).
                                      SKIPPED for bk19: hard binary
                                      predictions make the ROC none-scaling
                                      sweep meaningless.
* ``<ckpt-stem>_<source>_run.json``   provenance: ckpt path, source, years,
                                      hours, git revision, timestamp
* ``comparison.csv``                  (``compare`` subcommand) pooled CSI
                                      pivoted (front, dilation_km) x leg

CLI::

    python -m dl_front.evaluate_test --ckpt runs/xyz/model.h5 --classes 6 \
        --source kriged-airs [--years 2016-2018] [--hours 18,21,0]
    python -m dl_front.evaluate_test --source bk19 --classes 6
    python -m dl_front.evaluate_test compare

Evaluation is streamed one year at a time (``PaperMetrics.update`` +
per-year CSI counts), so the test span never sits in memory at once.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, dataset, evaluate, predict

#: Column order of the tidy CSI CSV (frozen interface, 2026-08-12).
CSI_CSV_COLUMNS = ["front", "dilation", "km", "csi", "csi_lo", "csi_hi",
                   "pod", "far", "fb"]


# --------------------------------------------------------------------------- #
# Per-year data loading (with actionable missing-data errors)
# --------------------------------------------------------------------------- #

def _kriged_cache_path(year: int, source: str) -> Path:
    """Path of one kriged cache file; FileNotFoundError names the fix."""
    # manifest reorg 2026-08-13: KRIGED_SOURCE_DIRS holds the full cache dirs
    path = config.KRIGED_SOURCE_DIRS[source] / f"kriged_sfc_{year}.nc"
    if not path.exists():
        build = {"kriged-airs": "build-airs",
                 "kriged-degraded": "build-degraded"}[source]
        raise FileNotFoundError(
            f"no kriged cache for {year}: {path} does not exist; build "
            f"it with 'python -m dl_front.krige_fill {build} "
            f"--years {year}'")
    return path


def kriged_cache_times(year: int, source: str = "kriged-airs"
                       ) -> pd.DatetimeIndex:
    """Time axis of one kriged cache file (cheap: coords only)."""
    import xarray as xr

    with xr.open_dataset(_kriged_cache_path(year, source)) as ds:
        return pd.DatetimeIndex(ds["time"].values)


def bk19_path(year: int) -> Path:
    """Path of one BK19 published-prediction file; actionable error if absent.

    The published archive covers 1980-2018 (which is why the 6-class eval
    years are 2016-2018, user decision 2026-08-13).
    """
    w = config.LABEL_WIDTH
    path = (config.BK19_DIR / f"1deg_{w}wide" / "3hr"
            / f"merra2_merra2-1deg_{w}wide_3hr_{year}.nc")
    if not path.exists():
        raise FileNotFoundError(
            f"no BK19 published predictions for {year}: {path} does not "
            f"exist (the archive covers 1980-2018 only); the default "
            f"resolves inside the data root "
            f"(front_id/predicted_fronts/bk19, manifest reorg 2026-08-13) "
            f"-- set JPL_BK19_DIR only for an out-of-tree archive")
    return path


def bk19_class_grid(bk, n_classes: int) -> np.ndarray:
    """(time, lat, lon) uint8 class index from a BK19 prediction dataset.

    Same painter as ``dataset.class_grid`` (lowest ``config.TYPE_PRIORITY``
    first, so higher priority overwrites; 'none' = last index), with two
    BK19-specific accommodations: ``front_type`` names are matched
    case-insensitively, and classes absent from the file (dryline -- BK19
    never predicts it) are simply never painted.
    """
    names = dataset.class_names(n_classes)
    front_type = [str(s).lower() for s in bk["front_type"].values]
    n_time, _, n_lat, n_lon = bk["fronts"].shape
    cls = np.full((n_time, n_lat, n_lon), len(names) - 1, dtype=np.uint8)
    for name in reversed(config.TYPE_PRIORITY):
        if name not in names or name.lower() not in front_type:
            continue                       # dryline: not in the BK19 files
        hit = bk["fronts"].isel(front=front_type.index(name.lower())).values == 1
        cls[hit] = names.index(name)
    return cls


def bk19_year_arrays(year: int, n_classes: int
                     ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One year of the BK19 leg: (pred-as-x, y, times).

    "x" is the published prediction class grid carried as a trailing
    channel, (n, 68, 141, 1) -- :class:`BK19Predictions` one-hot encodes it
    at "inference" time, so the leg runs through the exact
    :func:`evaluate_ckpt` machinery (hours filter, kriged-cache time
    matching, provenance) as the checkpoint legs.  Labels/times come from
    the SAME ``valid_label_steps`` + exact-timestamp inner join as
    ``dataset.year_arrays``.
    """
    import xarray as xr

    with xr.open_dataset(bk19_path(year)) as bk:
        bk = bk.load()
    pred = bk19_class_grid(bk, n_classes)
    bk_times = pd.DatetimeIndex(bk["time"].values)

    with dataset.load_label_ds(year, n_classes) as lab:
        keep = dataset.valid_label_steps(lab, n_classes)
        cls = dataset.class_grid(lab, n_classes)[keep]
        label_times = pd.DatetimeIndex(lab["time"].values)[keep]

    common = bk_times.intersection(label_times)
    x = pred[bk_times.get_indexer(common)][..., None].astype(np.float16)
    y = cls[label_times.get_indexer(common)]
    return x, y, common


class BK19Predictions:
    """Duck-typed 'model' for the bk19 leg: one-hot encode the class grid.

    ``x`` (from :func:`bk19_year_arrays`) already holds the published class
    indices; "prediction" is a certainty one-hot, so downstream argmax
    reproduces the published classes exactly and PaperMetrics still
    accumulates (its ROC sweep is meaningless on hard classes -- the paper
    json is skipped for this leg).
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def predict(self, x, batch_size=64, verbose=0):
        cls = np.rint(np.asarray(x)[..., 0]).astype(np.int64)
        probs = np.zeros((*cls.shape, self.n_classes), dtype=np.float32)
        np.put_along_axis(probs, cls[..., None], 1.0, axis=-1)
        return probs


def load_year(year: int, n_classes: int, stats: dict, source: str
              ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One year of (x, y, times) from the requested source.

    Raises FileNotFoundError naming the missing data AND the command that
    produces it, so a failed cluster run tells the operator exactly what to
    do next.
    """
    if source == "bk19":
        return bk19_year_arrays(year, n_classes)   # stats unused: no z-score

    if source == "reanalysis":
        year_dir = config.SFC_DIR / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(
                f"no reanalysis surface data for {year}: {year_dir} does not "
                f"exist; fetch it with "
                f"'python -m dl_front.acquire_merra2_sfc {year}'")
        return dataset.year_arrays(year, n_classes, stats)

    if source in config.KRIGED_SOURCE_DIRS:
        _kriged_cache_path(year, source)          # actionable error if absent
        return dataset.kriged_year_arrays(year, n_classes, stats, source)

    raise ValueError(f"unknown source {source!r}; expected 'reanalysis', "
                     f"'bk19' or one of {sorted(config.KRIGED_SOURCE_DIRS)}")


# --------------------------------------------------------------------------- #
# Core evaluation (importable; model is anything with .predict)
# --------------------------------------------------------------------------- #

def evaluate_ckpt(model, years, n_classes: int, source: str,
                  hours=None, stats: dict | None = None,
                  loader=None, batch_size: int = 64,
                  match_source: str | None = None,
                  info: dict | None = None
                  ) -> tuple[evaluate.PaperMetrics, pd.DataFrame]:
    """Streamed test evaluation -> (PaperMetrics, pooled CSI scores).

    ``model``: any object with ``.predict(x, batch_size=, verbose=)``
    returning (n, 68, 141, n_classes) softmax outputs.
    ``hours``: UTC label hours to keep; default ``config.AIRS_HOURS`` for
    EVERY source.
    ``loader``: optional ``loader(year) -> (x, y, times)`` override of
    :func:`load_year` (tests inject synthetic data here).
    ``match_source``: name of a kriged cache whose per-year TIME AXIS the
    scored steps are intersected with -- the identical-sample-set guarantee
    for cross-source comparisons (a sparse AIRS archive covers only some
    days; see module docstring).  The CLI sets ``'kriged-airs'`` for
    reanalysis runs.
    ``info``: optional dict, filled with per-year step counts and a SHA-1 of
    every scored timestamp (comparability provenance).

    Scoring mask (user decision 2026-08-13): the 6-class dryline/AIRS
    track scores EVERY source -- reanalysis, kriged-airs AND bk19 -- over
    ``dataset.analysis_domain()`` (box ∩ land), so the three legs are
    compared on identical pixels; the 5-class paper replication keeps the
    Fig. 2 region mask.
    """
    hours = config.AIRS_HOURS if hours is None else tuple(hours)
    if loader is None:
        if stats is None and source != "bk19":   # bk19 has no z-scoring
            stats = dataset.load_norm_stats()
        loader = lambda year: load_year(year, n_classes, stats, source)

    mask = (dataset.analysis_domain() if n_classes == 6
            else dataset.region_mask().astype(bool))
    pm = evaluate.PaperMetrics(n_classes, mask=mask)
    counts, n_steps, scored_times = [], {}, []
    for year in years:
        x, y, times = dataset.filter_hours(*loader(year), hours)
        if match_source is not None and match_source != source:
            ref = kriged_cache_times(year, match_source)
            keep = times.isin(ref)
            if (~keep).any():
                print(f"{year}: {int((~keep).sum())} steps absent from the "
                      f"{match_source} cache dropped for comparability",
                      flush=True)
            x, y, times = x[keep], y[keep], times[keep]
        if len(x) == 0:
            print(f"{year}: no steps at hours {hours}, skipped", flush=True)
            n_steps[year] = 0
            continue
        probs = np.asarray(model.predict(x.astype(np.float32),
                                         batch_size=batch_size, verbose=0))
        pm.update(probs, y)
        counts.append(evaluate.csi_counts(probs.argmax(-1), y, times,
                                          n_classes, mask=mask))
        n_steps[year] = len(x)
        scored_times.extend(times)
        print(f"{year}: {len(x)} steps scored", flush=True)
    if not counts:
        raise RuntimeError(f"no data found for years {list(years)} at hours "
                           f"{hours} (source={source})")
    if info is not None:
        digest = hashlib.sha1("\n".join(
            str(t) for t in sorted(scored_times)).encode()).hexdigest()
        info.update(n_steps_per_year={int(y): int(n)
                                      for y, n in n_steps.items()},
                    times_sha1=digest, match_source=match_source)
    all_counts = pd.concat(counts, ignore_index=True)
    scores = evaluate.csi_scores(all_counts)
    # Day-block bootstrap CIs (audit + user decision 2026-08-15): cross-leg
    # CSI deltas -- dryline especially, ~150-190 event-bearing steps/year
    # and strongly autocorrelated -- are not interpretable without
    # uncertainty; deltas inside overlapping CIs are sampling noise.
    boot = evaluate.block_bootstrap(all_counts)
    scores["csi_lo"] = boot.lo["csi"]
    scores["csi_hi"] = boot.hi["csi"]
    return pm, scores


# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #

def _git_rev() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, text=True,
            capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_outputs(pm: evaluate.PaperMetrics, scores: pd.DataFrame,
                  ckpt: Path | None, source: str, years, hours,
                  out_dir: Path | None = None,
                  info: dict | None = None,
                  write_paper: bool = True) -> dict:
    """Write the CSV + paper json + provenance json; return their paths.

    ``ckpt=None`` is the checkpoint-free bk19 leg: the stem is just the
    source name.  ``write_paper=False`` (bk19) skips the paper json -- hard
    binary predictions make the ROC none-scaling sweep meaningless -- and
    records that in the run json instead.
    """
    out_dir = Path(out_dir) if out_dir is not None \
        else config.RESULTS_DIR / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source if ckpt is None else f"{Path(ckpt).stem}_{source}"

    tidy = scores.reset_index()[CSI_CSV_COLUMNS]
    csv_path = out_dir / f"{stem}.csv"
    tidy.to_csv(csv_path, index=False)
    paths = {"csv": csv_path}

    if write_paper:
        conf = pm.confusion_table(percent=True)
        paper = {"accuracy": pm.accuracy(),
                 "auc": pm.auc(),
                 "confusion_percent": {actual: dict(row)
                                       for actual, row in conf.iterrows()}}
        paper_path = out_dir / f"{stem}_paper.json"
        paper_path.write_text(json.dumps(paper, indent=1))
        paths["paper"] = paper_path

    run_path = out_dir / f"{stem}_run.json"
    run_path.write_text(json.dumps(
        {"ckpt": None if ckpt is None else str(Path(ckpt).resolve()),
         "source": source,
         "years": [int(y) for y in years], "hours": [int(h) for h in hours],
         **({} if write_paper
            else {"paper_metrics": "skipped (binary baseline)"}),
         **(info or {}),
         "git_rev": _git_rev(),
         "created": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        indent=1))
    paths["run"] = run_path
    return paths


# --------------------------------------------------------------------------- #
# Leg comparison (the three-way table, user decision 2026-08-13)
# --------------------------------------------------------------------------- #

def _check_same_sample(provenance: dict) -> bool:
    """Warn loudly when the legs did not score identical time steps.

    ``provenance`` maps leg stem -> parsed ``_run.json`` dict (or None when
    the file is missing/unreadable).  The scored-timestamp SHA-1s of all
    legs must agree for the comparison table to be a like-for-like one
    (evaluate_ckpt's time matching only DROPS steps, so a leg with missing
    source days -- or a stale --no-match / partial --years run -- scores a
    different sample without any error).  Returns True when the check
    passes.
    """
    shas = {stem: (run or {}).get("times_sha1") for stem, run in
            provenance.items()}
    if len({s for s in shas.values() if s is not None}) <= 1 \
            and all(s is not None for s in shas.values()):
        return True
    print("\nWARNING: the legs did NOT score identical time steps -- the "
          "columns below are NOT a like-for-like comparison.  Per-leg "
          "scored-timestamp SHA-1 and step counts:", flush=True)
    for stem, run in sorted(provenance.items()):
        if run is None:
            print(f"  {stem}: no readable {stem}_run.json (unknown sample)",
                  flush=True)
        else:
            print(f"  {stem}: sha1={run.get('times_sha1')} years="
                  f"{run.get('years')} match_source="
                  f"{run.get('match_source')} "
                  f"n_steps={run.get('n_steps_per_year')}", flush=True)
    print("re-run the odd leg(s) out with the default --years and time "
          "matching (no --no-match), then compare again", flush=True)
    return False


def compare(out_dir: Path | None = None) -> pd.DataFrame:
    """Pivot pooled CSI across every leg CSV -> comparison.csv + printed table.

    Dumb and robust by design: every ``*.csv`` in the eval dir carrying the
    frozen :data:`CSI_CSV_COLUMNS` is a leg named by its stem; anything else
    (including ``comparison.csv`` itself) is skipped with a note, and a leg
    missing a (front, dilation_km) row simply shows NaN there.

    One check IS performed: the same-sample guarantee.  ``evaluate_ckpt``'s
    time matching only drops steps, so a leg can silently score fewer steps
    than the others (missing sfc_daily day files, a stale ``--no-match`` or
    partial ``--years`` run); each leg's ``_run.json`` records a SHA-1 of
    its scored timestamps, and any disagreement across legs is reported
    loudly here AND encoded in the output name (the table is written as
    ``comparison_MISMATCHED_SAMPLE.csv``; the warning tells you which leg
    to rerun).
    """
    out_dir = Path(out_dir) if out_dir is not None \
        else config.RESULTS_DIR / "test_eval"
    legs, provenance = {}, {}
    for path in sorted(out_dir.glob("*.csv")):
        if path.name == "comparison.csv":
            continue
        df = pd.read_csv(path)
        if not set(CSI_CSV_COLUMNS) <= set(df.columns):
            print(f"skipping {path.name}: not a CSI leg CSV", flush=True)
            continue
        idx = df.set_index(["front", "km"])
        legs[path.stem] = idx["csi"]
        legs[f"{path.stem}_lo"] = idx["csi_lo"]
        legs[f"{path.stem}_hi"] = idx["csi_hi"]
        run_path = path.with_name(f"{path.stem}_run.json")
        try:
            provenance[path.stem] = json.loads(run_path.read_text())
        except (OSError, ValueError):
            provenance[path.stem] = None      # reported below
    if not legs:
        raise FileNotFoundError(f"no leg CSVs found in {out_dir}; run some "
                                f"evaluations first")
    same_sample = _check_same_sample(provenance)
    table = pd.DataFrame(legs)                # aligns on the index union
    table.index.names = ["front", "dilation_km"]
    # a non-like-for-like table must not masquerade as the headline result:
    # the stdout warning never reaches CSV consumers, so the mismatch is
    # encoded in the artifact's NAME (audit 2026-08-15)
    out_path = out_dir / ("comparison.csv" if same_sample
                          else "comparison_MISMATCHED_SAMPLE.csv")
    table.to_csv(out_path)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_path}")
    return table


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_years(spec: str) -> list[int]:
    """'2016-2018' (inclusive range) or '2016,2018,2020' -> list of ints."""
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def resolve_years(spec: str | None, n_classes: int) -> list[int]:
    """--years spec, or (None) the yaml eval split of the class count."""
    if spec:
        return parse_years(spec)
    return list(config.EVAL_YEARS_6 if n_classes == 6
                else config.EVAL_YEARS_5)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare"]:               # leg-comparison subcommand
        if argv[1:]:                          # never silently drop arguments
            sys.exit(f"evaluate_test compare takes no arguments (got "
                     f"{argv[1:]}); it always pivots every leg CSV in "
                     f"{config.RESULTS_DIR / 'test_eval'}")
        compare()
        return

    ap = argparse.ArgumentParser(
        description="Score one evaluation leg on the held-out test years "
                    "(paper metrics + neighborhood CSI, AIRS label hours): "
                    "a checkpoint on reanalysis or kriged-AIRS inputs, or "
                    "the published BK19 predictions.  The 'compare' "
                    "subcommand pivots every leg CSV into comparison.csv.")
    ap.add_argument("--ckpt", default=None,
                    help="trained .h5 checkpoint (required unless "
                         "--source bk19, which has no checkpoint)")
    ap.add_argument("--classes", type=int, default=6, choices=(5, 6))
    ap.add_argument("--source", default="reanalysis",
                    choices=("reanalysis", "kriged-airs", "bk19"),
                    help="input fields / predictor to score (every leg is "
                         "scored on the same AIRS-hour time steps)")
    ap.add_argument("--years", default=None,
                    help="'A-B' range or comma list (default: the yaml eval "
                         "split of --classes, e.g. "
                         f"{list(config.EVAL_YEARS_6)} for 6)")
    ap.add_argument("--hours", default=None,
                    help="comma-separated UTC hours to score "
                         f"(default: AIRS hours {config.AIRS_HOURS})")
    ap.add_argument("--no-match", action="store_true",
                    help="do NOT intersect a reanalysis/bk19 run's time "
                         "steps with the kriged-airs cache (scores every "
                         "step, so the result is NOT comparable to a "
                         "kriged-airs run on a sparse AIRS archive)")
    a = ap.parse_args(argv)
    if a.source == "bk19" and a.ckpt:
        ap.error("--ckpt does not apply to --source bk19 (the published "
                 "prediction files ARE the model output)")
    if a.source != "bk19" and not a.ckpt:
        ap.error(f"--ckpt is required for --source {a.source}")

    years = resolve_years(a.years, a.classes)
    hours = (tuple(int(h) for h in a.hours.split(","))
             if a.hours else config.AIRS_HOURS)
    match_source = ("kriged-airs"
                    if a.source in ("reanalysis", "bk19") and not a.no_match
                    else None)

    is_bk19 = a.source == "bk19"
    model = BK19Predictions(a.classes) if is_bk19 \
        else predict.load_model(a.ckpt)
    info: dict = {}
    if is_bk19:
        info["dryline"] = ("not predicted: the BK19 files carry no dryline "
                           "class, so dryline CSV rows are all-miss by "
                           "construction")
    pm, scores = evaluate_ckpt(model, years, a.classes, a.source, hours,
                               match_source=match_source, info=info)
    paths = write_outputs(pm, scores,
                          None if is_bk19 else Path(a.ckpt),
                          a.source, years, hours, info=info,
                          write_paper=not is_bk19)

    acc = pm.accuracy()
    line = (f"\naccuracy: all={acc['all_categories']:.4f} "
            f"front/no-front={acc['front_no_front']:.4f}")
    if not is_bk19:                # ROC AUC is meaningless on hard classes
        line += f"  auc={pm.auc():.4f}"
    print(line)
    print(scores.reset_index()[CSI_CSV_COLUMNS].to_string(index=False))
    print("\nwrote " + "\n      ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    sys.exit(main())
