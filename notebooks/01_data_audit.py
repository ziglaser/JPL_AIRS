# %% [markdown]
# # Phase 1 - Data audit
#
# - **Q1** slot -> forecast-hour mapping (verify across all six years)
# - **Q2** what does `FCST_N` (parcel count) actually contain?
# - **Q3** does our pooled QPE99.95 land near the paper's 5.1 mm/h?
# - **Q4** sample size per forecast hour vs the paper's ">160k"
# - **Q5** leap-day / padding-day handling
#
# This file is a `# %%`-cell script: open it as a notebook in VS Code / Jupyter,
# or run it headless with `python notebooks/01_data_audit.py`.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; comment out when running interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convection_skill import config, data_loading, quality_control, paper_benchmarks as pb

FIG_DIR = config.RESULTS_DIR / "replication" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Q1 - Slot -> forecast-hour mapping, verified across all years
#
# `FCST_parceltime` is "nanoseconds since <file-specific epoch>". We decode the
# actual UTC hour at each of the 7 slots, for a few valid days per year, and
# confirm slot 0 is the AIRS overpass and slots 1-6 are 21,22,23,0,1,2 UTC.

# %%
def decode_slot_hours(year, n_days=5):
    """Return the set of UTC hours seen at each of the 7 slots for `year`."""
    ds = data_loading.open_year(year, drop_parceltime=False)
    # xarray decodes FCST_parceltime to datetime64 (with NaT where missing).
    arr = ds["FCST_parceltime"].values  # (date, time, lat, lon) datetime64[ns]
    valid = ~np.isnat(arr)
    per_slot = {}
    valid_days = np.where(valid.any(axis=(1, 2, 3)))[0]
    sample_days = valid_days[:: max(1, len(valid_days) // n_days)][:n_days]
    for slot in range(arr.shape[1]):
        hours = set()
        for d in sample_days:
            vals = arr[d, slot][valid[d, slot]]
            if vals.size:
                hours.update(pd.DatetimeIndex(vals).hour.tolist())
        per_slot[slot] = sorted(hours)
    ds.close()
    return per_slot

def Q1():
    for yr in config.ALL_YEARS:
        per_slot = decode_slot_hours(yr)
        print(f"{yr}: " + " | ".join(f"slot{s}={per_slot[s]}" for s in range(7)))

    print("\nConfig assumes: OVERPASS_SLOT=0, FORECAST_SLOTS=1..6 -> hours 21,22,23,0,1,2")
    
    return None

# %% [markdown]
# ## Q2 - What is in `FCST_N`?
#
# In the 2016 recon it was 0 wherever CAPE was valid. Check every year: is it ever
# a usable parcel count? If it is always 0 (or NaN) where CAPE is valid, the
# ">20 parcels" screen was applied upstream and we correctly skip it in QC.

# %%
def Q2():
    for yr in config.ALL_YEARS:
        ds = data_loading.open_year(yr)
        cape = ds["FCST_MU_CAPE"].values
        n = ds["FCST_N"].values
        print(ds["FCST_N"].to_series().describe())
        valid = np.isfinite(cape)
        n_where_valid = n[valid]
        finite = np.isfinite(n_where_valid)
        print(f"{yr}: FCST_N where CAPE valid -> finite frac {finite.mean():.3f}, "
            f"min {np.nanmin(n_where_valid) if finite.any() else 'n/a'}, "
            f"max {np.nanmax(n_where_valid) if finite.any() else 'n/a'}, "
            f"frac>20 {np.mean(n_where_valid[finite] > 20) if finite.any() else 0:.3f}")
        ds.close()

    return None

# %% [markdown]
# ## Build the 2019-2020 tables (cached for Phases 3-4 and the benchmark tests)
#
# Three nested samples (scheme established by testing against the paper's own
# numbers; see `tests/test_paper_benchmarks.py`):
#
# 1. **domain**: all in-domain rows, incl. ocean cells and rows without valid
#    AIRS data (cached; the regional Supp-Table-1 comparisons need it);
# 2. **base** = land rows of (1): the paper's *threshold* sample ("thresholds
#    are based on all data, including all locations ... wet and dry hours");
# 3. **analysis** = valid-MU-indices rows of (2): the AIRS-FCST skill sample.
#    The stated complete-cell-days rule is NOT applied -- on our files it
#    deletes the wettest 23% of valid rows and inflates every skill score
#    (see quality_control.require_all_timesteps_valid).

def build_tables(years=config.PAPER_YEARS):
    from convection_skill import dataset
    from convection_skill.config import AnalysisConfig

    # unified base superset: domain-sliced at load, no land/validity screening
    domain = dataset.build_base_table(AnalysisConfig(years=tuple(years)))
    domain.to_parquet(config.RESULTS_DIR / "domain_2019_2020.parquet")
    print(f"domain rows (incl. ocean & no-AIRS rows): {len(domain):,}  [cached]")

    base = quality_control.require_land(domain)
    analysis, report = quality_control.apply_paper_qc(domain, return_report=True)
    print(report)

    cache_path = config.RESULTS_DIR / "analysis_2019_2020.parquet"
    analysis.to_parquet(cache_path)
    print(f"\ncached QC'd analysis table -> {cache_path}  ({len(analysis):,} rows)")

    return base, analysis

def build_qpe_thresholds(df):
    qpe = df["qpe"].to_numpy()
    thresholds = {p: float(np.nanpercentile(qpe, p)) for p in config.QPE_PERCENTILES}

    return qpe, thresholds


# %% [markdown]
# ## Q3 - Pooled QPE percentile thresholds vs the paper's QPE99.95 ~ 5.1 mm/h

# %%
# Canonical thresholds come from the BASE sample (paper: "all data").
def Q3(thresholds):    
    thr_df = pd.DataFrame(
        {"percentile": list(thresholds), "qpe_threshold_mm_per_h": list(thresholds.values())}
    )
    thr_df.to_csv(config.RESULTS_DIR / "replication" / "thresholds.csv", index=False)
    print(thr_df.to_string(index=False))
    print(f"\nQPE99.95 = {thresholds[99.95]:.2f} mm/h  (paper: ~{config.PAPER_QPE9995_MM_PER_H} mm/h)")

# %% [markdown]
# ### Q3 resolution (revised 2026-07) - `_av` is a *conditional* mean and must be rescaled
#
# The original audit found our raw `MRMS_GaugeCorrQPE01H_av` tail (~8.4 mm/h at the
# 99.95th) stable across sample definitions and concluded "data-provenance
# difference, percentile-based Gini unaffected". **That conclusion was wrong.**
# `_av` is the mean over *precipitating* sub-cells only (0 when none are wet) --
# not the paper's all-pixel 1x1 cell mean -- so it re-ranks which cell-hours are
# "events" (small intense cells outrank broad moderate storms) and inflated every
# Gini by +0.1-0.2 (the paper's own Supp. Note 8 reports higher Gini for exactly
# this within-precipitating-area QPE). Diagnostics that established this:
#
# - ratio ours/paper across the Fig. 2c threshold ladder falls 9.0x -> 1.65x from
#   the 95th to 99.95th percentile: a wet-area-fraction effect, not a unit offset;
# - `_cnt` > 0 exactly where `_av` > 0 and tops out at 81 at every latitude, so
#   `_cnt`/81 is the cell's precipitating-area fraction;
# - `_av * _cnt/81` reproduces the paper's tail thresholds (5.55 vs 5.1 mm/h) and
#   the Fig. 1 Kansas case-study cell values.
#
# `data_loading` now reconstructs `qpe = _av * _cnt / 81` (grid-cell mean, the
# paper's target) and keeps the raw conditional mean as `qpe_wet`. The full
# paper-vs-us verification lives in `tests/test_paper_benchmarks.py`.

# %%

def Q3_update(df, thresholds):
    qpe_wet = df["qpe_wet"].to_numpy()
    print("wet-area-mean ladder (skill sample) vs paper Supp Fig 13c")
    print(f"{'pctl':>7} {'ours _av':>9} {'paper':>7}")
    for p, ref in pb.WET_QPE_THRESHOLDS_MM_PER_H.items():
        print(f"{p:>7} {np.nanpercentile(qpe_wet, p):>9.2f} {ref:>7.1f}")
    
    print("\ncell-mean ladder (base sample) vs paper Fig 2c:")
    print(f"{'pctl':>7} {'recon qpe':>10} {'paper':>7}")
    for p, ref in pb.QPE_THRESHOLDS_MM_PER_H.items():
        print(f"{p:>7} {thresholds[p]:>10.2f} {ref:>7.1f}")


# %% [markdown]
# ## Q4 - Sample size per forecast hour vs the paper (">160k per forecast hour")
#
# Without the complete-cell-days rule the hours no longer have identical counts
# (the paper's data delivered that property structurally; ours cannot).

# %%
def Q4(df):
    per_hour = df.groupby("hour_utc").size().reindex(config.FORECAST_HOURS_UTC)
    print(per_hour)
    print(f"\nmin per-hour: {per_hour.min():,}  (paper: >160k for 2019-2020)")
    n_days = df.groupby(['year', 'date']).ngroups
    print(f"valid days: {n_days}  (paper Supp Notes 3: 455)")

    return None

# %% [markdown]
# ## Q5 - Leap-day / padding handling
#
# 2020 is a leap year (366 days); the others are 365. We confirm the season filter
# (March-November) and the valid-CAPE drop leave no all-NaN padding rows, and that
# 29 Feb (if present) is naturally excluded by the March-November filter.

# %%
def Q5(df):
    for yr in config.ALL_YEARS:
        ds = data_loading.open_year(yr)
        ndays = ds.sizes["date"]
        ds.close()
        print(f"{yr}: {ndays} date slots")
    feb29 = df[(df["date"].dt.month == 2) & (df["date"].dt.day == 29)]
    print(f"\nrows on 29 Feb in analysis sample: {len(feb29)} (expected 0 - outside Mar-Nov)")
    print(f"months present in analysis: {sorted(df['month'].unique())}")

    return None

def coverage_a(qpe, thresholds, savename="audit_qpe_distribution.png"):
    # (a) QPE distribution with percentile thresholds marked (log-y).
    fig, ax = plt.subplots(figsize=(7, 4))
    wet = qpe[qpe > 0]
    ax.hist(wet, bins=np.logspace(-2, np.log10(np.nanmax(wet)), 60), color="C0", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for p, t in thresholds.items():
        ax.axvline(t, color="C3", lw=1, ls="--")
        ax.text(t, ax.get_ylim()[1] * 0.5, f"  {p:g}", rotation=90, va="top", fontsize=8, color="C3")
    ax.set_xlabel("QPE (mm/h), wet cells only")
    ax.set_ylabel("count")
    ax.set_title("2019-2020 QC sample: QPE distribution + percentile thresholds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / savename, dpi=130)

def coverage_b(df, savename="audit_spatial_coverage.png"):
    # (b) Spatial coverage: number of valid QC samples per cell, and mean MU_CAPE.
    grid = df.groupby(["lat", "lon"]).agg(n=("mu_cape", "size"),
                                                mean_cape=("mu_cape", "mean")).reset_index()
    piv_n = grid.pivot(index="lat", columns="lon", values="n")
    piv_cape = grid.pivot(index="lat", columns="lon", values="mean_cape")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, piv, title, cmap in [
        (axes[0], piv_n, "QC sample count per cell", "viridis"),
        (axes[1], piv_cape, "mean MU_CAPE (J/kg)", "magma"),
    ]:
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=cmap, shading="nearest")
        ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("2019-2020 analysis domain (32-53N, land>=50%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / savename, dpi=130)

def coverage_c(df, savename="audit_land_mask.png"):
    # (c) Land mask over the domain after the >=50% cut.
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    land = data_loading.load_land_fraction_grid(lats, lons)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.pcolormesh(lons, lats, land, cmap="BrBG", vmin=0, vmax=1, shading="nearest")
    ax.set_title("Land fraction (analysis cells, all >= 0.5 by construction)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / savename, dpi=130)


def main():
    return None

if __name__ == "__main__":
    base, analysis = build_tables()
    base_qpe, base_thresholds = build_qpe_thresholds(base)
    analysis_qpe, analysis_thresholds = build_qpe_thresholds(analysis)

    # Q1()
    # Q2()
    Q3(thresholds=analysis_thresholds)
    Q3_update(df=analysis, thresholds=analysis_thresholds)

    Q3(thresholds=base_thresholds)
    Q3_update(df=base, thresholds=base_thresholds)
    # Q4(df=analysis)
    # Q5(df=analysis)

    # coverage_a(qpe=analysis_qpe, thresholds=analysis_thresholds)
    # coverage_b(df=analysis)
    # coverage_c(df=analysis)
