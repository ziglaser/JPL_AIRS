"""Mark's RF on the compiled cluster dataset, scored by Gini at fixed set
points -- the block-ablation experiment battery for gattaca2.

Consumes the ONE big netCDF written by ``scripts/compile_rf_dataset.py``
(FCST_SMAP_MRMS + sm_anom + UPW_psi_anom/UPW_omega + interpolated PBL family
+ met/pred front flags) and runs Mark's verbatim RF (via
:mod:`convection_skill.rf_experiments`'s three mode fit paths) over a small,
pre-specified set of BLOCK experiments.

Headline metric (user decision 2026-08-22): the **Gini coefficient** of the
RF prediction as a risk score for qpe-exceedance events at two set points,
**P95** and **P99.5** of the base sample -- the same ``convection_skill.gini``
coefficient the hypothesis battery reports, so forests and battery tests read
on one scale.  Thresholds are ABSOLUTE values taken from ALL in-domain land
hour-rows BEFORE screening (the paper's "thresholds are based on all data"
convention), so every experiment and mode scores against identical event
definitions.  R^2 and tail capture are retained as secondary columns.

Feature blocks (user decisions 2026-08-22)
------------------------------------------
- ``THERMO``  (hourly): mu_cape, mu_cin, mu_el, mu_lcl, fcst_q, fcst_t --
  **MU parcel ONLY**; every MML variable is deliberately absent.
- ``SM``      (daily):  sm_anom -- the timing-guarded surface soil-moisture
  index.
- ``UPWIND``  (hourly): UPW_psi_anom, UPW_omega -- the kernel-borne pair kept
  together as their own upwind-coupling block.
- ``PBL``     (hourly): UPW_pblh, UPW_pblh_anom, UPW_gamma_gap_mu -- the
  time-INTERPOLATED boundary-layer family (MU gamma gap only).
- ``FRONTS_MET`` / ``FRONTS_PRED`` (hourly): the six 3-wide flag families;
  analyst-drawn and model-predicted fronts are NEVER co-present -- where a
  fronts block appears, the two sources are swapped for comparison.

Experiment set (13, not a full factorial -- "a few logical choices"):
- ``full-met`` / ``full-pred``: every block, with the two front sources
  swapped (the head-to-head the fronts axis exists for).
- ``drop-<block>``: full-met minus one block at a time (leave-one-block-out;
  ``drop-fronts`` removes fronts entirely) -- each block's value READ AS the
  Gini it costs to remove it with everything else present.
- ``solo-<block>``: each block alone (fronts twice, met and pred) -- each
  block's standalone signal.

All experiments in all three modes (wide / pooled / perhour, see
rf_experiments) share ONE common finite sample and ONE cell-day train/test
partition, so every number is paired.

Training window (Zach, 2026-08-22): **March-November, 2017-2021** (--years /
--months defaults).  The compiled RF_dataset netCDF may stay a 2016-2021
full-year SUPERSET -- the window is applied here at experiment time, BEFORE
the set-point thresholds are computed, so thresholds describe the base
sample the forests actually train on.  ``sample_info.json`` records the
window and :func:`rf_experiments.check_window_consistency` refuses to mix
results produced under a different one.

CLI::

    python -m convection_skill.rf_cluster --dataset <big.nc> \
        [--years 2017-2021] [--months 3-11] \
        [--modes wide,pooled,perhour] [--out-dir DIR] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, data_loading as dl, mark_screens as mk, models
from . import rf_experiments as rfe
from .config import AnalysisConfig

TARGET = "qpe"
#: Gini set points: results.csv column gini_p95 / gini_p99_5.  The ONE
#: definition lives in rf_experiments (labels double as tail_cdf suffixes).
GINI_SET_POINTS: dict[str, float] = rfe.SET_POINTS

#: compiled-file name -> cell-day stem, for the slot-level (hourly) features.
SLOT_RENAME: dict[str, str] = {
    "FCST_MU_CAPE": "mu_cape", "FCST_MU_CIN": "mu_cin",
    "FCST_MU_EL": "mu_el", "FCST_MU_LCL": "mu_lcl",
    "FCST_q": "fcst_q", "FCST_t": "fcst_t",
}
FRONT_TYPES = ("cold", "warm", "stationary", "occluded", "dryline", "any")

BLOCKS: dict[str, tuple[str, ...]] = {
    "THERMO": ("mu_cape", "mu_cin", "mu_el", "mu_lcl", "fcst_q", "fcst_t"),
    "SM": ("sm_anom",),
    "UPWIND": ("UPW_psi_anom", "UPW_omega"),
    "PBL": ("UPW_pblh", "UPW_pblh_anom", "UPW_gamma_gap_mu"),
    "FRONTS_MET": tuple(f"met_front_{t}_3w" for t in FRONT_TYPES),
    "FRONTS_PRED": tuple(f"pred_front_{t}_3w" for t in FRONT_TYPES),
}
#: the non-fronts blocks, in drop/solo order.
CORE_BLOCKS = ("THERMO", "SM", "UPWIND", "PBL")


@dataclass(frozen=True)
class BlockExperiment:
    """Duck-typed for rf_experiments' fit paths: needs .id and .features."""

    id: str
    blocks: tuple[str, ...]

    @property
    def features(self) -> list[str]:
        out: list[str] = []
        for b in self.blocks:
            out.extend(BLOCKS[b])
        return out


def experiment_set() -> list[BlockExperiment]:
    """The 13 pre-specified block experiments (module docstring)."""
    full_met = CORE_BLOCKS + ("FRONTS_MET",)
    exps = [
        BlockExperiment("full-met", full_met),
        BlockExperiment("full-pred", CORE_BLOCKS + ("FRONTS_PRED",)),
    ]
    for b in CORE_BLOCKS:
        exps.append(BlockExperiment(
            f"drop-{b.lower()}",
            tuple(x for x in full_met if x != b)))
    exps.append(BlockExperiment("drop-fronts", CORE_BLOCKS))
    for b in CORE_BLOCKS:
        exps.append(BlockExperiment(f"solo-{b.lower()}", (b,)))
    exps.append(BlockExperiment("solo-fronts-met", ("FRONTS_MET",)))
    exps.append(BlockExperiment("solo-fronts-pred", ("FRONTS_PRED",)))
    assert not any({"FRONTS_MET", "FRONTS_PRED"} <= set(e.blocks)
                   for e in exps), "met and pred fronts must never co-occur"
    return exps


# --------------------------------------------------------------------------- #
# Compiled file -> screened cell-day table (mirrors the suite's semantics)
# --------------------------------------------------------------------------- #
def load_cell_days(dataset_path: Path, years: tuple[int, ...],
                   months: tuple[int, int],
                   ) -> tuple[pd.DataFrame, dict[str, float]]:
    """(screened cell-day wide table, absolute Gini thresholds), restricted
    to the (years, months-inclusive) training window.

    The compiled file may be a 2016-2021 full-year superset; the window is
    cut FIRST, before the thresholds, because the window defines the base
    sample (same convention as ``AnalysisConfig.years/months`` upstream) --
    quality screens, by contrast, must never move the event bar.

    Reconstructs the suite's preparation directly from the compiled file:
    grid-mean ``qpe`` (= _av * _cnt / 81, the audited reconstruction),
    Mark's master-mask components verbatim (mark_screens), land >= the
    suite's default fraction, then the slot 1..6 pivot to ``<stem>_h<slot>``
    columns.  Thresholds are computed from ALL in-window land hour-rows
    BEFORE Mark's screens (paper convention).
    """
    ds = xr.open_dataset(dataset_path)
    # the training window: years exact-set, months inclusive lo-hi
    dates = pd.DatetimeIndex(ds["date"].values)
    in_window = (np.isin(dates.year, list(years))
                 & (dates.month >= months[0]) & (dates.month <= months[1]))
    if not in_window.any():
        raise ValueError(f"{dataset_path} has no dates in years {years} "
                         f"months {months}; wrong file or window?")
    ds = ds.isel(date=np.flatnonzero(in_window))
    cfg = AnalysisConfig()  # read the suite's default screen constants
    # explicit slot labels: the raw files carry no time coord, and the pivot
    # below must name columns _h1.._h6, never positional 0..5 after the isel
    ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))

    # the audited grid-mean reconstruction; MRMS variables live on the
    # ``nhours`` axis, hour-aligned with ``time`` (see mark_screens.add_gridav)
    qpe = (ds[config.QPE_VAR] * ds[config.QPE_CNT_VAR]
           / config.WET_CELL_MAX_CNT).astype("float64")
    axis = next(d for d in qpe.dims if d in ("time", "nhours"))
    qpe = xr.DataArray(
        qpe.transpose("date", axis, "lat", "lon").values,
        dims=("date", "time", "lat", "lon"),
        coords={k: ds[k] for k in ("date", "time", "lat", "lon")})
    land = dl.load_land_fraction_grid(ds["lat"].values, ds["lon"].values)
    land_ok = land >= cfg.land_fraction_min

    # Gini set-point thresholds: all land hour-rows (slots 1..6), pre-screen.
    # (date, time, lat, lon)[..., 2-D land mask] -> (date, time, n_land_cells)
    qpe_land = qpe.isel(time=slice(1, None)).values[..., land_ok]
    qpe_land = qpe_land[np.isfinite(qpe_land)]
    thresholds = {label: float(np.percentile(qpe_land, pct))
                  for label, pct in GINI_SET_POINTS.items()}

    comp = mk.master_mask_components(ds)  # alt_max / dry_start_qpe / valid7
    keep = (xr.DataArray(land_ok, dims=("lat", "lon"))
            & (comp["alt_max"] < cfg.alt_max_m)
            & (comp["dry_start_qpe"] <= cfg.dry_start_mm)
            & comp["valid7"])

    slot_vars = {**SLOT_RENAME,
                 **{f"{p}_front_{t}_3w": f"{p}_front_{t}_3w"
                    for p in ("met", "pred") for t in FRONT_TYPES},
                 "UPW_psi_anom": "UPW_psi_anom", "UPW_omega": "UPW_omega",
                 "UPW_pblh": "UPW_pblh", "UPW_pblh_anom": "UPW_pblh_anom",
                 "UPW_gamma_gap_mu": "UPW_gamma_gap_mu"}
    missing = [v for v in slot_vars if v not in ds]
    if missing:
        raise KeyError(f"{dataset_path} is missing {missing}; rebuild it with "
                       f"scripts/compile_rf_dataset.py")

    sub = ds[list(slot_vars)].rename(SLOT_RENAME)  # rest keep their names
    sub[TARGET] = qpe.astype("float32")
    sub = sub.isel(time=range(1, ds.sizes["time"]))  # forecast slots only
    sub = sub.where(keep)

    df = sub.to_dataframe(dim_order=["date", "time", "lat", "lon"]
                          ).reset_index()
    df["day"] = df["date"].values.astype("datetime64[D]")
    stems = [c for c in df.columns
             if c not in ("date", "time", "lat", "lon", "day", "sm_anom")]
    wide = df.pivot(index=["day", "lat", "lon"], columns="time", values=stems)
    wide.columns = [f"{v}_h{int(s)}" for v, s in wide.columns]
    daily = (ds["sm_anom"].to_dataframe().reset_index()
             .rename(columns={"date": "day"}))
    daily["day"] = daily["day"].values.astype("datetime64[D]")
    out = (wide.reset_index()
           .merge(daily, on=["day", "lat", "lon"], how="left"))
    ds.close()
    return out, thresholds


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
RESULT_COLUMNS = ("id", "blocks", "n_samples", "gini_p95", "gini_p99_5",
                  "r2_test", "r2_h1", "r2_h2", "r2_h3", "r2_h4", "r2_h5",
                  "r2_h6", "tail_top10_capture")


def write_report(out_dir: Path, mode: str, results: pd.DataFrame,
                 facts: dict) -> None:
    lines = [f"# RF block experiments on the compiled dataset -- mode {mode}",
             "",
             f"Training window: years {facts.get('years')}, months "
             f"{facts.get('months')} (inclusive; enforced by the window-"
             f"consistency guard).",
             "",
             f"Headline metric: Gini of the RF prediction vs qpe exceedance "
             f"at P95 ({facts['thresholds']['p95']:.3f} mm/h) and P99.5 "
             f"({facts['thresholds']['p99_5']:.3f} mm/h), thresholds from "
             f"all in-window pre-screen land hour-rows. R^2 is a SECONDARY "
             f"column (iid-split optimism; the set-point Ginis are what "
             f"ranks). n_samples = {facts['n_samples']} common cell-days.",
             "", rfe.MODE_BLURBS[mode], "",
             "| id | blocks | gini_p95 | gini_p99_5 | r2_test |",
             "|---|---|---|---|---|"]
    for _, r in results.sort_values("gini_p99_5", ascending=False).iterrows():
        lines.append(f"| {r['id']} | {r['blocks']} | {r['gini_p95']:.4f} | "
                     f"{r['gini_p99_5']:.4f} | {r['r2_test']:.4f} |")
    (Path(out_dir) / "REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Mark's RF block experiments on the compiled dataset, "
                    "Gini-scored at P95/P99.5")
    ap.add_argument("--dataset", required=True, type=Path)
    # window defaults = the project's training window (Mar-Nov 2017-2021);
    # the compiled file may hold 2016-2021 full years -- filtered here.
    ap.add_argument("--years", default="2017-2021",
                    help="year range/list of the training window")
    ap.add_argument("--months", default="3-11",
                    help="inclusive month window lo-hi (default Mar-Nov)")
    ap.add_argument("--modes", default="wide,pooled,perhour")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    years = rfe.parse_years(args.years)
    months = rfe.parse_months(args.months)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in rfe.MODES]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; choose from "
                         f"{sorted(rfe.MODES)}")
    root = Path(args.out_dir) if args.out_dir else (
        config.RESULTS_DIR / "rf_cluster")

    cell_days, thresholds = load_cell_days(args.dataset, years, months)
    exps = experiment_set()
    all_stems = sorted({s for e in exps for s in e.features})
    missing = rfe.missing_stems(cell_days, tuple(all_stems))
    if missing:
        raise AssertionError(f"cell-day table is missing stems {missing}")

    samples = models.samples_from_cell_days(cell_days)
    samples = models.finite_samples(samples, all_stems + [TARGET])
    facts = {"years": list(years), "months": list(months),
             "n_samples": int(samples.sizes["sample"]),
             "thresholds": thresholds,
             "dataset": str(args.dataset)}
    print(f"common sample: {facts['n_samples']} cell-days "
          f"(window years {years}, months {months}); thresholds "
          f"{thresholds}")

    for mode in modes:
        fit_fn, _ = rfe.MODES[mode]
        out_dir = root / mode
        (out_dir / "importances").mkdir(parents=True, exist_ok=True)
        (out_dir / "tail_cdf").mkdir(parents=True, exist_ok=True)
        # refuse/discard rows from a different training window BEFORE the
        # sample_info.json below overwrites the recorded provenance
        results = rfe.check_window_consistency(out_dir, years, months,
                                               force=args.force,
                                               columns=RESULT_COLUMNS)
        done = set() if args.force else set(results["id"])
        (out_dir / "sample_info.json").write_text(json.dumps(facts, indent=2))
        for exp in exps:
            if exp.id in done:
                continue
            print(f"=== [{mode}] {exp.id} ({len(exp.features)} stems) ===")
            m = fit_fn(samples, exp, thresholds=thresholds)
            m["importances"].to_csv(out_dir / "importances" / f"{exp.id}.csv",
                                    index_label="feature")
            rfe.write_tail_curves(out_dir, exp.id, m["tail_cdfs"])
            row = {"id": exp.id, "blocks": "+".join(exp.blocks),
                   "n_samples": facts["n_samples"],
                   "gini_p95": m["gini_p95"], "gini_p99_5": m["gini_p99_5"],
                   "r2_test": m["r2_test"], **m["r2_per_hour"],
                   "tail_top10_capture": m["tail_top10_capture"]}
            results = pd.concat(
                [results[results["id"] != exp.id], pd.DataFrame([row])],
                ignore_index=True)
            results.to_csv(res_path, index=False)   # resumable per experiment
        write_report(out_dir, mode, results, facts)
        print(f"[{mode}] {len(results)}/{len(exps)} experiments -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
