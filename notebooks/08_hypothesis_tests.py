# %% [markdown]
# # Hypothesis suite driver (unified architecture)
#
# One `AnalysisConfig` = one comparable run: the cached base table is screened
# per the config, event flags come from base-sample thresholds, and the
# config's hypotheses run through the same `test_hypothesis` function under
# the run's ONE inference convention. Artifacts land under `results/suite/`:
#
# - `RESULTS_<label>.md` — suite report (verdicts, strata, honesty check)
# - `results_<label>.csv`, `curves_<label>.npz` — tidy results + curves
# - `figures/ht_{forest,curves}_<label>.png` — suite-level figures
# - `<HYP_ID>/` — per-hypothesis folder (results.csv, topline.md, strata.png)
#
# THE CANONICAL DRIVER IS `src/run_suite.py` (same artifacts incl. the
# per-hypothesis folders):
#
#     PYTHONPATH=src python src/run_suite.py [data_table.yaml] [hypothesis.yaml]
#
# Files in either order (recognized by keys), either omissible (defaults to
# `configs/data_table.yaml` + `configs/hypothesis_tests.yaml`). Everything —
# years, hypotheses, strata, screens, curves — comes from the config; to
# compare runs (e.g. pooled vs paper years), give the hypothesis file a
# `defaults:` + `runs:` spec, or pass different config files per invocation.
#
# This notebook exists for interactive exploration (run_one / load_curves
# cells); running it headless with the same config args additionally builds
# the convective-mode diagnostics figures.

# %%
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convection_skill import dataset as D, report as R, suite  # noqa: E402
from convection_skill.config import AnalysisConfig  # noqa: E402
from run_suite import load_configs  # noqa: E402  (shared two-file CLI loader)

OUT = Path("results/suite")
FIG = OUT / "figures"

# %% [markdown]
# ## One suite run (prepare -> run_suite -> report + figures)

# %%
def run_one(cfg: AnalysisConfig):
    label = cfg.label()
    print(f"[{label}] preparing dataset for {cfg.years} ...", flush=True)
    prepared = D.prepare(cfg)
    print(f"[{label}] {len(prepared.table):,} rows, "
          f"{prepared.table['day'].nunique()} days, "
          f"{len(prepared.onset):,} precipitating cell-days", flush=True)

    results, curves = suite.run_suite(prepared)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / f"results_{label}.csv", index=False)
    flat_curves = {f"{hid}__{k}": v for hid, c in curves.items() for k, v in c.items()}
    np.savez(OUT / f"curves_{label}.npz", **flat_curves)

    R.write_report(results, OUT / f"RESULTS_{label}.md",
                   run_label=f"{label} ({cfg.years})", cfg=cfg)
    R.plot_forest(results, FIG / f"ht_forest_{label}.png")
    R.plot_curves(curves, FIG / f"ht_curves_{label}.png")
    n_significant = int(results["fdr_significant"].sum())
    print(f"[{label}] {len(results)} tests, {n_significant} FDR-significant",
          flush=True)
    return label, results, curves

# %% [markdown]
# ## Rehydrating persisted curves (re-style figures without a rerun)

# %%
def load_curves(label):
    path = OUT / f"curves_{label}.npz"
    if not path.exists():
        return {}
    flat = np.load(path)
    curves = {}
    for key in flat.files:
        hid, k = key.rsplit("__", 1)
        curves.setdefault(hid, {})[k] = flat[key]
    return curves


if __name__ == "__main__":
    configs = load_configs(sys.argv[1:])

    from convective_id import plotting as cid_plotting

    # convective-mode diagnostics on the FIRST config's sample (its years and
    # screens) — previously hard-coded to the pooled years regardless of args
    diagnostics_df = D.build_dataset(configs[0])
    cid_plotting.plot_convective_by_percentile(
        diagnostics_df, FIG / "convective_frac_vs_qpe.png")
    cid_plotting.animate_cape_qpe_convective_threshold(
        diagnostics_df, FIG / "cape_qpe_convective_threshold.gif")
    del diagnostics_df

    results_by_run = {}
    curves_by_run = {}
    for cfg in configs:
        label, results, curves = run_one(cfg)
        results_by_run[label] = results
        curves_by_run[label] = curves

    R.write_hypothesis_folders(results_by_run, curves_by_run, OUT)
    print(f"per-hypothesis folders written -> {OUT}/<HYP_ID>/", flush=True)
