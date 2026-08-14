"""Render the battery results: markdown report + standard figures.

The report leads with verdicts (expected sign vs observed, FDR-adjusted), keeps
the full (hypothesis x scope) table for reference, shows the naive-vs-block CI
inflation the autocorrelation machinery exists for, and ends with the two
sections the user asked to be explicit: variable-assumption corrections for the
source hypothesis table, and the trajectory-kernel extension map.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .hypotheses import REGISTRY  # noqa: E402

_SIGN = {True: "+", False: "-"}


def _verdict(row: pd.Series) -> str:
    if row["testability"] == "untestable":
        return "not testable locally"
    if not np.isfinite(row["gini"]):
        return "no estimate"
    sig = bool(row["fdr_significant"])
    sign = _SIGN[row["gini"] > 0]
    if row["expected"] in ("+", "-"):
        if not sig:
            return "null (n.s. after FDR)"
        return ("SUPPORTED" if sign == row["expected"]
                else "CONTRADICTED (sign reversed)")
    if row["expected"] == "regime":
        return f"{sign}{'*' if sig else ' (n.s.)'} (regime test: see strata)"
    if row["expected"] == "shape":
        return "see event-rate curve"
    return "—"


def _fmt_ci(r) -> str:
    if not np.isfinite(r["gini"]):
        return "—"
    return f"{r['gini']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"


def _header_lines(results: pd.DataFrame, run_label: str, cfg=None) -> list[str]:
    """Header stating the run's actual settings (falls back to the module
    defaults when no config is passed, e.g. for pre-refactor callers)."""
    block_days = cfg.block_days if cfg else config.BLOCK_DAYS
    n_reps = cfg.n_boot_reps if cfg else config.N_BOOT_REPS
    fdr_alpha = cfg.fdr_alpha if cfg else config.FDR_ALPHA
    heavy_pct = cfg.heavy_percentile if cfg else config.HEAVY_PERCENTILE
    method = results["inference"].iloc[0] if "inference" in results else "block"
    inference = (
        f"{block_days}-day circular block bootstrap over days"
        if method == "block"
        else "iid row bootstrap (the paper's construction; NOTE the audit "
             "measured 2-3x SE overconfidence vs day blocks on this data)")
    return [
        f"# Hypothesis suite results — {run_label}",
        "",
        f"Inference: {inference} ({n_reps} reps), "
        f"BH-FDR α={fdr_alpha} across all "
        f"{len(results)} tests (Wilks 2016). Predictors are per-cell standardized "
        f"harmonic anomalies; SM predictors use pre-window L4 slots only "
        f"(16:30/19:30 UTC — the Tuttle & Salvucci timing guard). "
        f"Targets from base-sample thresholds (heavy = cell-mean QPE > "
        f"P{heavy_pct}).",
    ]


def _verdicts_lines(results: pd.DataFrame) -> list[str]:
    lines = ["", "## Overall verdicts", "",
             "| ID | Hypothesis | Gini [95% CI] | p | FDR sig | verdict |",
             "|----|----|----|----|----|----|"]
    overall = results[results["scope"] == "overall"].set_index("id")
    run_ids = set(results["id"])  # only what this run actually tested
    for spec in REGISTRY:
        if spec.id not in run_ids:
            continue
        if spec.id not in overall.index:
            lines.append(f"| {spec.id} | {spec.title} | — | — | — | not testable locally |")
            continue
        r = overall.loc[spec.id]
        p = f"{r['p']:.3g}" if np.isfinite(r["p"]) else "—"
        sig = "yes" if r["fdr_significant"] else "no"
        lines.append(f"| {spec.id} | {spec.title} | {_fmt_ci(r)} | {p} | {sig} | {_verdict(r)} |")
    return lines


def _increments_lines(results: pd.DataFrame) -> list[str]:
    lines = ["", "## Skill increments (conditional Gini within control bins)", "",
             "| ID | control | conditional Gini [95% CI] | p | FDR sig |",
             "|----|----|----|----|----|"]
    controls = results[results["scope"].str.startswith("ctrl:")]
    for _, r in controls.iterrows():
        sig = "yes" if r["fdr_significant"] else "no"
        lines.append(f"| {r['id']} | {r['scope'][5:]} | {_fmt_ci(r)} | {r['p']:.3g} | {sig} |")
    return lines


def _strata_lines(results: pd.DataFrame) -> list[str]:
    lines = ["", "## Regime strata (sign tests)", "",
             "| ID | stratum | n_events | Gini [95% CI] | FDR sig |",
             "|----|----|----|----|----|"]
    is_stratum = (~results["scope"].isin(["overall", "—"])
                  & ~results["scope"].str.startswith("ctrl:"))
    for _, r in results[is_stratum].iterrows():
        sig = "yes" if r["fdr_significant"] else "no"
        lines.append(f"| {r['id']} | {r['scope']} | {int(r['n_events'])} | {_fmt_ci(r)} | {sig} |")
    return lines


def _honesty_lines(results: pd.DataFrame) -> list[str]:
    inflation = results["inflation"].dropna()
    if not len(inflation):
        return []
    return ["", "## Autocorrelation honesty check", "",
            f"Block-vs-naive SE inflation across tests: median {inflation.median():.2f}, "
            f"p90 {inflation.quantile(0.9):.2f}, max {inflation.max():.2f}. "
            f"Values > 1 quantify how much the paper-style iid bootstrap would "
            f"overstate confidence here."]


#: Audit corrections for the source hypothesis table (see docs/CODEBASE_GUIDE.md).
_CORRECTIONS = [
    "- `FCST_N` is a parcel COUNT (identically 0 in this cut), **not** "
    "Brunt–Väisälä frequency; unusable.",
    "- All non-L4 `SMAP_*` fields are all-NaN in every year; every SMAP "
    "operationalization must use `SMAP_L4_*` (done here).",
    "- `FCST_q_excess` = mixing ratio above saturation (condensate proxy, nonzero "
    "in only ~5% of samples), not a general moisture surplus — weak support for "
    "A4 as written.",
    "- `MRMS_*_av` is the precipitating-area conditional mean; the cell mean is "
    "`_av*_cnt/81` (used here). `_cnt` is area-weighted (non-integer). `_max` "
    "confirmed sub-pixel max; `_sk` sub-pixel skewness (0 for 76.5% of rows).",
    "- `SMAP_L4_ulay1_av` is a wind-speed magnitude (≥0), not a u-component — "
    "S2's gate is directly available, the direction is not.",
    "- L4 slots 25.5/28.5 = 01:30/04:30 UTC **next day** (after the QPE window): "
    "same-day daily means leak post-rain SM into predictors; pre-window slots "
    "only (implemented).",
]


def _closing_lines(results: pd.DataFrame) -> list[str]:
    lines = ["", "## Variable-assumption corrections (for the source hypothesis table)",
             "", *_CORRECTIONS, "", "## Trajectory-kernel extension map", ""]
    run_ids = set(results["id"])
    for spec in REGISTRY:
        if spec.kernel_extension and spec.id in run_ids:
            lines.append(f"- **{spec.id}** — {spec.kernel_extension}")
    return lines


def write_report(results: pd.DataFrame, out_path: Path, run_label: str,
                 cfg=None) -> Path:
    """The battery-level markdown report, one section builder per block.

    Pass the run's :class:`AnalysisConfig` so the header states the settings
    actually used (reps, FDR alpha, heavy-QPE percentile).
    """
    lines = (_header_lines(results, run_label, cfg)
             + _verdicts_lines(results)
             + _increments_lines(results)
             + _strata_lines(results)
             + _honesty_lines(results)
             + _closing_lines(results))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def plot_forest(results: pd.DataFrame, save_path: Path,
                overall_only: bool = True) -> None:
    """Forest plot of the topline (overall) Gini, one row per hypothesis.

    Stratum-level detail lives in each hypothesis folder's ``strata.png``;
    ``overall_only=False`` restores the old every-scope wall (every control
    and stratum row) if a single all-in figure is ever wanted.
    """
    sub = results[results["scope"] != "—"].copy()
    if overall_only:
        sub = sub[sub["scope"] == "overall"]
    sub = sub[np.isfinite(sub["gini"])]
    sub["label"] = (sub["id"] if overall_only
                    else sub["id"] + " · " + sub["scope"])
    sub = sub.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(sub) + 1.5))
    y = np.arange(len(sub))
    colors = np.where(sub["fdr_significant"], "#b13f3f", "#9a9a9a")
    ax.hlines(y, sub["ci_lo"], sub["ci_hi"], color=colors, lw=2, alpha=0.8)
    ax.scatter(sub["gini"], y, c=colors, s=18, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8 if overall_only else 7)
    ax.set_xlabel("Gini (bootstrap 95% CI); red = FDR-significant")
    ax.set_title("Hypothesis battery — signed discrimination of heavy precip"
                 + ("" if overall_only else " (all scopes)"))
    fig.savefig(save_path, dpi=130, bbox_inches="tight")



def plot_curves(curves: dict, save_path: Path) -> None:
    """Event-rate curves for the non-monotonic specs (A2, A4)."""
    if not curves:
        return
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (hid, c) in zip(axes, curves.items()):
        ax.fill_between(c["centers"], c["lo"], c["hi"], alpha=0.25, color="steelblue")
        ax.plot(c["centers"], c["rate"], "o-", color="steelblue", ms=3)
        ax.set_title(hid)
        ax.set_xlabel("predictor (bin median)")
        ax.set_ylabel("P(event)")
    fig.suptitle("Event-rate curves (block-bootstrap 95% bands)")
    fig.savefig(save_path, dpi=130, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Per-hypothesis result folders
# --------------------------------------------------------------------------- #
_RUN_PALETTE = ("#3b6ea5", "#c98137", "#5c9e6e", "#a05a9e", "#7a7a7a")


def _run_colors(runs) -> dict:
    """Stable color per run label, in first-appearance order."""
    return {run: _RUN_PALETTE[i % len(_RUN_PALETTE)]
            for i, run in enumerate(runs)}


def _ci_str(r) -> str:
    return (f"{r['gini']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
            if np.isfinite(r["gini"]) else "—")


def _write_topline(spec, rows: pd.DataFrame, out: Path) -> None:
    """topline.md: the hypothesis header + overall/control Ginis per run."""
    lines = [
        f"# {spec.id} — {spec.title}", "",
        f"- **Expected:** {spec.expected}   ·   **Testability:** {spec.testability}",
        f"- **Predictor:** `{spec.predictor}`   ·   **Target:** `{spec.target}`"
        f"   ·   **Sample:** {spec.sample}",
        f"- **Trajectory-kernel extension:** {spec.kernel_extension or '—'}",
    ]
    if spec.notes:
        lines.append(f"- **Notes:** {spec.notes}")
    lines += ["", "## Topline (block-bootstrap 95% CI)", "",
              "| run | scope | n_events | Gini [95% CI] | p | FDR sig |",
              "|----|----|----|----|----|----|"]
    top = rows[rows["scope"].isin(["overall"])
               | rows["scope"].str.startswith("ctrl:")]
    for _, r in top.iterrows():
        p = f"{r['p']:.3g}" if np.isfinite(r["p"]) else "—"
        lines.append(f"| {r['run']} | {r['scope']} | {int(r['n_events'])} | "
                     f"{_ci_str(r)} | {p} | "
                     f"{'yes' if r['fdr_significant'] else 'no'} |")
    (out / "topline.md").write_text("\n".join(lines))


def _plot_strata(spec, rows: pd.DataFrame, out: Path) -> None:
    """strata.png: one panel per stratifier (season / region / time / regime),
    Gini with CI whiskers, runs side by side."""
    strat_rows = rows[~rows["scope"].isin(["overall", "—"])
                      & ~rows["scope"].str.startswith("ctrl:")].copy()
    strat_rows = strat_rows[np.isfinite(strat_rows["gini"])]
    if strat_rows.empty:
        return
    strat_rows["stratifier"] = strat_rows["scope"].str.split("=").str[0]
    strat_rows["level"] = strat_rows["scope"].str.split("=").str[1]
    stratifiers = list(dict.fromkeys(strat_rows["stratifier"]))

    fig, axes = plt.subplots(1, len(stratifiers),
                             figsize=(0.8 + 3.2 * len(stratifiers), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    runs = list(dict.fromkeys(strat_rows["run"]))
    colors = _run_colors(runs)
    for ax, strat in zip(axes, stratifiers):
        sub = strat_rows[strat_rows["stratifier"] == strat]
        levels = list(dict.fromkeys(sub["level"]))
        x = np.arange(len(levels), dtype=float)
        for k, run in enumerate(runs):
            rr = sub[sub["run"] == run].set_index("level").reindex(levels)
            xo = x + (k - (len(runs) - 1) / 2) * 0.18
            ax.errorbar(xo, rr["gini"],
                        yerr=[rr["gini"] - rr["ci_lo"], rr["ci_hi"] - rr["gini"]],
                        fmt="o", ms=4, capsize=3,
                        color=colors[run], label=run)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=8)
        ax.set_title(strat, fontsize=10)
    axes[0].set_ylabel("Gini (95% CI)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"{spec.id}: stratified results", fontsize=11)
    fig.savefig(out / "strata.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_hyp_curve(spec, curves_by_run: dict, out: Path) -> None:
    per_run = {run: c[spec.id] for run, c in curves_by_run.items()
               if spec.id in c}
    if not per_run:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = _run_colors(per_run)
    for run, c in per_run.items():
        color = colors[run]
        ax.fill_between(c["centers"], c["lo"], c["hi"], alpha=0.2, color=color)
        ax.plot(c["centers"], c["rate"], "o-", ms=3, color=color, label=run)
    ax.set_xlabel(f"{spec.predictor} (bin median)")
    ax.set_ylabel("P(event)")
    ax.set_title(f"{spec.id}: event rate vs predictor (95% bands)")
    ax.legend(fontsize=8)
    fig.savefig(out / "event_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_hypothesis_folders(
    results_by_run: dict[str, pd.DataFrame],
    curves_by_run: dict[str, dict],
    base: Path,
) -> None:
    """One folder per hypothesis under ``base``: results.csv (all scopes, all
    runs), topline.md (headline Ginis + uncertainties + spec metadata),
    strata.png (results by season / region / time-of-window / regime), and
    event_rate.png where the spec has a curve. This is the regeneration entry
    point -- rerunning the driver rebuilds every folder from scratch.
    """
    combined = pd.concat(
        [df.assign(run=run) for run, df in results_by_run.items()],
        ignore_index=True)
    for spec in REGISTRY:
        rows = combined[combined["id"] == spec.id]
        if rows.empty:
            continue
        out = base / spec.id
        out.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out / "results.csv", index=False)
        _write_topline(spec, rows, out)
        if spec.testability != "untestable":
            _plot_strata(spec, rows, out)
            _plot_hyp_curve(spec, curves_by_run, out)
