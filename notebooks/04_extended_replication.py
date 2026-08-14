# %% [markdown]
# # Phase 5 - Extended replication (2016-2021)
#
# The paper used 2019-2020 (the only two years with complete, constant-version
# output from every product it compared). We have six clean years locally, so we
# re-run the AIRS-FCST Gini analysis on the full record to see how skill estimates
# and their uncertainty behave with ~3x the sample. This is the new-science
# baseline the SMAP work builds on.
#
# We report three samples:
# - **2019-2020** (paper years; reloaded from the Phase-1 cache),
# - **2016-2021** (all six years),
# - **2016-2021 ex. post-Aug-2021** (drops Sep-Nov 2021, which the paper flags as
#   contaminated by an AIRS deep-space-manoeuvre retrieval issue).

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convection_skill import config, plotting  # noqa: E402
from convection_skill.analysis import (  # noqa: E402
    gini_by_percentile, hourly_gini, hourly_significance,
)

FIG_DIR = config.RESULTS_DIR / "replication" / "figures"
PCT = config.HEADLINE_PERCENTILE

# %% [markdown]
# ## Build (and cache) the full six-year QC table

# %%
cache_all = config.RESULTS_DIR / "analysis_2016_2021.parquet"
if cache_all.exists():
    full = pd.read_parquet(cache_all)
    print(f"loaded cached six-year QC table: {len(full):,} rows")
else:
    from convection_skill import dataset
    from convection_skill.config import AnalysisConfig

    # unified builder; valid_datasets=("airs_fcst",) == the paper QC exactly
    # (domain + land + the four MU indices finite; NaN-QPE rows kept, as the
    # old drop_missing_predictor + apply_paper_qc path did)
    full = dataset.build_dataset(
        AnalysisConfig(years=config.ALL_YEARS, valid_datasets=("airs_fcst",)))
    full.to_parquet(cache_all)
    print(f"\ncached -> {cache_all}  ({len(full):,} rows)")

# %% [markdown]
# ## Per-year skill breakdown - is the extended sample homogeneous?
#
# Before pooling, check each year on its own (threshold computed within the year).
# This reveals whether any year is an outlier that would distort the pooled result.

# %%
per_year_rows = []
for yr in config.ALL_YEARS:
    s = full[full["year"] == yr]
    g = gini_by_percentile(s, ["mu_cape"], percentiles=[99.0, 99.9, 99.95])["mu_cape"]
    h = hourly_gini(s, "mu_cape", PCT).sort_values("step")
    per_year_rows.append(dict(
        year=yr, rows_per_hour=len(s) // 6,
        gini_99=g.iloc[0], gini_999=g.iloc[1], gini_9995=g.iloc[2],
        hr21=h["gini"].iloc[0], hr02=h["gini"].iloc[-1],
        hourly_delta=h["gini"].iloc[-1] - h["gini"].iloc[0],
    ))
per_year = pd.DataFrame(per_year_rows).set_index("year")
per_year.to_csv(config.RESULTS_DIR / "replication" / "extended_per_year_gini.csv")
print(per_year.round(3).to_string())
print(
    "\nInterpretation: 2018-2020 replicate the paper cleanly (Gini99.95 ~ 0.92-0.94,\n"
    "small hourly delta). 2016 is a clear outlier (Gini99.95 ~ 0.49) that drags down\n"
    "the pooled six-year skill; 2017 and 2021 show degraded EARLY-hour (21 UTC) skill\n"
    "that recovers by 02 UTC -- inflating their hourly delta and, if anything,\n"
    "amplifying the paper's 'skill improves with forecast hour' result. Recommend the\n"
    "SMAP work prefer 2018-2021 and treat 2016 (and 2021 early hours) with caution."
)

# %% [markdown]
# ## Define the three samples

# %%
paper = pd.read_parquet(config.RESULTS_DIR / "analysis_2019_2020.parquet")
# Post-Aug-2021 contamination flag: 2021, months 9-11.
contaminated = (full["year"] == 2021) & (full["month"] >= 9)
clean = full[~contaminated]
print(f"post-Aug-2021 rows dropped in 'clean' variant: {contaminated.sum():,}")

samples = {
    "2019-2020 (paper)": paper,
    "2016-2021 (all)": full,
    "2016-2021 ex. Sep-Nov'21": clean,
}
for name, s in samples.items():
    per_hour = len(s) // len(config.FORECAST_HOURS_UTC)
    print(f"{name:28s}: {len(s):>9,} rows  ({per_hour:,}/hour)")

# %% [markdown]
# ## Gini vs event rarity, per sample (AIRS-FCST CAPE)

# %%
gini_curves = {}
for name, s in samples.items():
    res = gini_by_percentile(s, ["mu_cape"])
    gini_curves[name] = res["mu_cape"]
gini_table = pd.DataFrame(gini_curves)
gini_table.index.name = "percentile"
gini_table.to_csv(config.RESULTS_DIR / "replication" / "extended_gini_by_percentile.csv")
print(gini_table.round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
for i, (name, col) in enumerate(gini_curves.items()):
    ax.plot(col.index, col.values, "-o", color=f"C{i}", label=name, lw=1.8)
ax.set_xlabel("QPE threshold percentile")
ax.set_ylabel("Gini coefficient (AIRS-FCST CAPE)")
ax.set_title("Skill vs event rarity across sample sizes")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "extended_gini_vs_percentile.png", dpi=140)
print("saved extended_gini_vs_percentile.png")

# %% [markdown]
# ## Hourly trend + bootstrap SE per sample: does uncertainty tighten with data?

# %%
lines = ["PHASE 5 - Extended replication (AIRS-FCST CAPE, QPE99.95)",
         "-" * 62,
         f"{'sample':28s} {'per-hour':>9s} {'SE':>7s} {'21->02 delta':>13s} {'trend/hr':>10s}"]
hourly_by_sample = {}
for name, s in samples.items():
    hourly, boot, trend = hourly_significance(s, "mu_cape", PCT)
    hourly_by_sample[name] = hourly
    delta = hourly.sort_values("step")["gini"].iloc[-1] - hourly.sort_values("step")["gini"].iloc[0]
    sig = "*" if trend.significant else " "
    lines.append(f"{name:28s} {len(s)//6:>9,} {boot.se:>7.3f} {delta:>+13.3f} "
                 f"{trend.slope:>+9.4f}{sig}")
lines += ["", "'*' = OLS hourly trend significant at p<0.05.",
          "Note: the pooled six-year bootstrap SE is LARGER than the 2019-2020 SE",
          "despite 3x the data -- a signature of year-to-year heterogeneity (see the",
          "per-year table: 2016 is a low-skill outlier, 2017/2021 have weak early hours).",
          "The positive AIRS-FCST hourly trend persists and strengthens, because the",
          "noisier years have especially low 21-UTC skill that recovers by 02 UTC."]
summary = "\n".join(lines)
print(summary)
(config.RESULTS_DIR / "extended_comparison.txt").write_text(summary + "\n")

# Overlay Gini-vs-hour for the three samples.
fig, ax = plt.subplots(figsize=(7, 5))
for i, (name, hourly) in enumerate(hourly_by_sample.items()):
    d = hourly.sort_values("step")
    ax.errorbar(d["step"], d["gini"], yerr=d["se"], marker="o", capsize=3,
                color=f"C{i}", label=name)
steps = list(range(len(config.FORECAST_HOURS_UTC)))
ax.set_xticks(steps)
ax.set_xticklabels([f"{h:02d}" for h in config.FORECAST_HOURS_UTC])
ax.set_xlabel("forecast hour (UTC)")
ax.set_ylabel("Gini coefficient (QPE$_{99.95}$)")
ax.set_title("AIRS-FCST detection skill vs hour, across samples")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "extended_gini_vs_hour.png", dpi=140)
print("saved extended_gini_vs_hour.png")
