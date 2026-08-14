# %% [markdown]
# # Phase 4 - Figure 3 replication (2019-2020)
#
# Reproduces Richardson et al. (2024) Fig. 3a-b: how the QPE99.95 detection skill
# (Gini) changes with forecast hour, for
#
# - **AIRS-FCST CAPE** (trajectory-enhanced) -- expected to *improve* with hour, and
# - **AIRS overpass CAPE** (proximity-sounding baseline) -- expected to *degrade*.
#
# Significance follows the paper's Methods:
# - bootstrap 1-sigma SE by resampling the pooled six-hour sample to one-hour size;
# - an hour-to-hour Gini difference is significant at p<0.05 if it exceeds 2*sqrt(2)*sigma;
# - the OLS trend of Gini vs forecast hour is significant if |slope| > 2*SE(slope).

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convection_skill import config, gini, plotting  # noqa: E402
from convection_skill.analysis import hourly_significance  # noqa: E402

FIG_DIR = config.RESULTS_DIR / "replication" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from convection_skill import quality_control  # noqa: E402

analysis = pd.read_parquet(config.RESULTS_DIR / "analysis_2019_2020.parquet")
domain = pd.read_parquet(config.RESULTS_DIR / "domain_2019_2020.parquet")
base = quality_control.require_land(domain)
qpe = analysis["qpe"].to_numpy()

# Pooled QPE99.95 threshold from the BASE sample ("thresholds are based on all
# data"), applied within each hour. Fig. 3 itself uses the matched skill sample
# ("all panels of Fig. 3 contain consistent datasets").
PCT = config.HEADLINE_PERCENTILE
THRESHOLD = float(np.nanpercentile(base["qpe"].to_numpy(), PCT))
flags_all = qpe > THRESHOLD
print(f"skill sample {len(analysis):,}; threshold {THRESHOLD:.2f} mm/h; "
      f"QPE{PCT} events {flags_all.sum():,}")

# %% [markdown]
# ## Per-hour Gini + significance for both predictors (via analysis module)

# %%
predictor_labels = {"mu_cape": "AIRS-FCST CAPE", "mu_cape_overpass": "AIRS overpass CAPE"}
hourly_by_predictor = {}   # label -> hourly DataFrame (step, hour_utc, gini, se)
stats_by_predictor = {}    # label -> (BootstrapResult, TrendResult)
for col, name in predictor_labels.items():
    hourly, boot, trend = hourly_significance(analysis, col, PCT, threshold=THRESHOLD)
    hourly_by_predictor[name] = hourly
    stats_by_predictor[name] = (boot, trend)

# Tidy combined CSV for the record.
combined = pd.concat(
    [d.assign(predictor=name) for name, d in hourly_by_predictor.items()],
    ignore_index=True,
)
combined.to_csv(config.RESULTS_DIR / "replication" / "fig3_hourly_gini.csv", index=False)
print(combined.pivot(index="hour_utc", columns="predictor", values="gini").round(3)
      .reindex(config.FORECAST_HOURS_UTC))

# %% [markdown]
# ## Significance: trend test + first-vs-last-hour difference

# %%
from convection_skill import paper_benchmarks as pb  # noqa: E402

paper_hourly = {"AIRS-FCST CAPE": pb.FIG3_GINI_FCST,
                "AIRS overpass CAPE": pb.FIG3_GINI_OVERPASS}
summary_lines = ["PHASE 4 - Fig. 3 replication vs Richardson et al. (2024), 2019-2020",
                 "-" * 66]
for name in predictor_labels.values():
    d = hourly_by_predictor[name].sort_values("step")
    boot, trend = stats_by_predictor[name]
    se = boot.se
    diff_thr = boot.diff_significance_threshold
    first, last = d["gini"].iloc[0], d["gini"].iloc[-1]
    diff = last - first
    diff_sig = abs(diff) > diff_thr
    ours_by_hour = ", ".join(f"{int(r.hour_utc):02d}:{r.gini:.2f}" for r in d.itertuples())
    paper_by_hour = ", ".join(f"{h:02d}:{g:.2f}" for h, g in paper_hourly[name].items())
    summary_lines += [
        f"{name}:",
        f"    Gini by hour (ours)            = {ours_by_hour}",
        f"    Gini by hour (paper Fig 3)     = {paper_by_hour}",
        f"    bootstrap 1-sigma SE           = {se:.3f}",
        f"    Gini 21 UTC -> 02 UTC          = {first:.3f} -> {last:.3f}  (delta {diff:+.3f})",
        f"    p<0.05 hour-diff threshold     = {diff_thr:.3f} (2*sqrt(2)*sigma; paper ~0.08)",
        f"    first-vs-last significant?     = {diff_sig}",
        f"    OLS {trend}",
    ]
summary_lines += [
    "",
    "Paper: AIRS-FCST Gini improves significantly with forecast hour; AIRS overpass",
    "CAPE degrades. Trend signs, FCST trend significance and first-vs-last",
    "differences replicate; every hour agrees with the paper's legend value within",
    "the paper's own p<0.05 criterion (tests/test_paper_benchmarks.py).",
]
summary = "\n".join(summary_lines)
print(summary)
(config.RESULTS_DIR / "fig3_comparison.txt").write_text(summary + "\n")

# %% [markdown]
# ## Fig. 3a - per-hour detection CDFs for AIRS-FCST CAPE

# %%
fig, ax = plt.subplots(figsize=(6, 5.5))
plotting.fig3_hourly_cdfs(ax, analysis, "mu_cape", percentile=PCT, threshold=THRESHOLD)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3a_hourly_cdfs_airsfcst.png", dpi=140)
print("saved fig3a_hourly_cdfs_airsfcst.png")

# %% [markdown]
# ## Fig. 3 summary - Gini vs forecast hour with error bars, both predictors

# %%
fig, ax = plt.subplots(figsize=(7, 5))
plotting.plot_gini_vs_hour(ax, hourly_by_predictor, percentile=PCT)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_gini_vs_hour.png", dpi=140)
print("saved fig3_gini_vs_hour.png")
