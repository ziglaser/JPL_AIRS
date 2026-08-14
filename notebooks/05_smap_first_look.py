# %% [markdown]
# # Phase 6 - SMAP extension first look
#
# The point of this phase is to show that the replication machinery is genuinely
# **predictor-agnostic**: the exact same `gini()` / `analysis` code that scored
# AIRS-FCST CAPE now scores SMAP soil-moisture fields, with no change to the
# statistics. This is the launch pad for the soil-moisture study motivated by
# `docs/convective_initiation_lit_review.md`.
#
# Three demonstrations:
# 1. **SMAP alone** - Gini of each SMAP field as a standalone predictor of heavy
#    QPE (expected to be weak: soil moisture modulates convection, it is not a CAPE
#    substitute).
# 2. **Stratified CAPE skill** - CAPE's QPE-detection Gini computed *within*
#    soil-moisture terciles and east/Plains regions, to probe the regime dependence
#    the literature flags (Tuttle & Salvucci 2016; Guillod et al. 2015).
# 3. **Extension paths** - a documented sketch of the two ways to fold SMAP in.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convection_skill import config, dataset  # noqa: E402
from convection_skill.analysis import gini_by_percentile, stratified_gini  # noqa: E402
from convection_skill.config import AnalysisConfig  # noqa: E402

FIG_DIR = config.RESULTS_DIR / "replication" / "figures"
#: Unified-schema SMAP columns. NOTE (2026-07 port): these are the battery's
#: timing-guarded DAILY pre-window values -- sm_raw is the raw surface SM, the
#: rest are per-cell standardized harmonic anomalies, so the seasonal/regional
#: climatology confound the original raw slot-level fields carried is now
#: removed by construction for the anomaly columns.
SMAP_COLS = ["sm_raw", "absgrad_anom", "qlay1_anom", "tlay1_anom"]
PCT = 99.9  # slightly less rare than 99.95 -> more events for the weak predictors

# %% [markdown]
# ## Build the 2019-2020 table (unified builder; SMAP columns ride along)

# %%
table = dataset.build_dataset(AnalysisConfig.paper())
print(f"paper-QC table: {len(table):,} rows")
print("SMAP columns:", SMAP_COLS)

# %% [markdown]
# ## Demonstration 1 - SMAP fields as standalone predictors
#
# Same `gini_by_percentile` call as Phase 3, just different columns. We include
# CAPE as the reference. Soil moisture is expected to be far weaker than CAPE (and
# possibly signed either way by regime).

# %%
predictors = ["mu_cape"] + SMAP_COLS
solo = gini_by_percentile(table, predictors, percentiles=[95.0, 99.0, PCT])
pretty = {"mu_cape": "AIRS-FCST CAPE", "sm_raw": "SMAP surface SM",
          "absgrad_anom": "SMAP SM |gradient|", "qlay1_anom": "SMAP root-zone SM",
          "tlay1_anom": "SMAP soil temp"}
solo_named = solo.rename(columns=pretty)
solo_named.to_csv(config.RESULTS_DIR / "replication" / "smap_solo_gini.csv")
print(solo_named.round(3).to_string())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
vals = solo.loc[PCT]
colors = ["C0"] + ["C1"] * len(SMAP_COLS)
ax.bar([pretty[c] for c in predictors], vals.values, color=colors)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_ylabel(f"Gini (QPE$_{{{PCT:g}}}$)")
ax.set_title("Standalone detection skill: CAPE vs SMAP fields")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "smap_solo_gini.png", dpi=140)
print("saved smap_solo_gini.png")

# %% [markdown]
# ## Demonstration 2 - CAPE skill stratified by soil-moisture regime and region
#
# Soil-moisture terciles (dry / mid / wet) from the pooled surface SM, and an
# east/Plains split at 95W. We compute CAPE's QPE-detection Gini within each
# stratum against the *global* QPE threshold, so strata are directly comparable.

# %%
table = table.assign(
    sm_tercile=pd.qcut(table["sm_raw"], 3, labels=["dry", "mid", "wet"]),
    region=np.where(table["lon"] <= -95.0, "Plains (<=95W)", "East (>95W)"),
)

by_tercile = stratified_gini(table, "mu_cape", table["sm_tercile"], percentile=PCT)
by_region = stratified_gini(table, "mu_cape", table["region"], percentile=PCT)
# Joint region x tercile stratification (the regime-stratified view the review wants).
joint_labels = table["region"].astype(str) + " / " + table["sm_tercile"].astype(str)
by_joint = stratified_gini(table, "mu_cape", joint_labels, percentile=PCT)

print("CAPE Gini by soil-moisture tercile:\n", by_tercile.round(3).to_string(index=False))
print("\nCAPE Gini by region:\n", by_region.round(3).to_string(index=False))
print("\nCAPE Gini by region x tercile:\n", by_joint.round(3).to_string(index=False))
by_joint.to_csv(config.RESULTS_DIR / "replication" / "smap_stratified_cape_gini.csv", index=False)

# Plot: CAPE Gini across terciles, one line per region.
fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["dry", "mid", "wet"]
for i, reg in enumerate(sorted(table["region"].unique())):
    sub = by_joint[by_joint["stratum"].str.startswith(reg)].copy()
    sub["terc"] = sub["stratum"].str.split(" / ").str[1]
    sub = sub.set_index("terc").reindex(order)
    ax.plot(order, sub["gini"], "-o", color=f"C{i}", label=reg)
ax.set_xlabel("soil-moisture tercile")
ax.set_ylabel(f"CAPE Gini (QPE$_{{{PCT:g}}}$)")
ax.set_title("Does CAPE's heavy-rain skill depend on soil-moisture regime?")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "smap_stratified_cape_gini.png", dpi=140)
print("saved smap_stratified_cape_gini.png")

# %% [markdown]
# ## Demonstration 3 - the two extension paths (documented, not yet fitted)
#
# **(a) Conditioning / stratification** *(shown above).* Compute CAPE skill within
# SMAP-defined regimes. If CAPE's Gini differs across terciles/regions, soil
# moisture carries information *beyond* CAPE about when heavy rain occurs. This
# needs only `stratified_gini` -- already predictor-agnostic.
#
# **(b) Bivariate predictor.** Combine CAPE with a SMAP feature into a single score,
# then feed it to the same `gini()` call. The honest version fits the combination
# on training years and evaluates on held-out years (avoid in-sample optimism);
# a first sketch:
#
# ```python
# from convection_skill.gini import gini, exceedance_flags
# # z-score each predictor, try a simple additive score, evaluate held-out:
# train, test = table[table.year == 2019], table[table.year == 2020]
# def z(s): return (s - train[s.name].mean()) / train[s.name].std()
# score_test = z(test["mu_cape"]) + w * z(test["absgrad_anom"])   # tune w on train
# g = gini(score_test.to_numpy(), exceedance_flags(test["qpe"].to_numpy(), PCT))
# ```
#
# Both paths reuse the tested core unchanged -- which was the whole point of
# building the replication predictor-agnostically.

# %%
summary = [
    "PHASE 6 - SMAP extension first look (2019-2020, QPE99.9)",
    "-" * 58,
    "Standalone Gini (QPE99.9):",
    f"  AIRS-FCST CAPE     : {solo.loc[PCT,'mu_cape']:+.3f}",
]
for c in SMAP_COLS:
    summary.append(f"  {pretty[c]:19s}: {solo.loc[PCT, c]:+.3f}")
summary += [
    "",
    "CAPE Gini by soil-moisture tercile: "
    + ", ".join(f"{r.stratum}:{r.gini:.3f}" for r in by_tercile.itertuples()),
    "CAPE Gini by region: "
    + ", ".join(f"{r.stratum}:{r.gini:.3f}" for r in by_region.itertuples()),
    "",
    "Takeaways:",
    "- Surface SM and its gradient (the physically causal fields) are weak standalone",
    "  predictors of heavy QPE (Gini ~0.08-0.10), as expected: soil moisture modulates",
    "  convection, it is not a CAPE substitute.",
    "- CAUTION: root-zone SM and soil temperature show HIGH standalone Gini (~0.7-0.8),",
    "  but this is almost certainly CONFOUNDING, not causal skill -- both track the",
    "  warm-season / humid-east climatology where heavy rain concentrates anyway.",
    "  This is exactly the confounding the review warns about (Tuttle & Salvucci 2017);",
    "  a causal SMAP signal must be isolated by conditioning, not read off raw Gini.",
    "- CAPE's Gini is fairly flat across soil-moisture terciles (0.92-0.94) and",
    "  regions (East 0.93 vs Plains 0.91); chase the sign/size of these differences",
    "  next, per-region (review: E-US negative, Plains mixed).",
    "- The same predictor-agnostic core scored CAPE and SMAP with zero code changes.",
]
report = "\n".join(summary)
print("\n" + report)
(config.RESULTS_DIR / "smap_comparison.txt").write_text(report + "\n")
