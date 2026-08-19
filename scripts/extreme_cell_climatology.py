#!/usr/bin/env python
"""Find the 1-degree cell with the most extreme-QPE events and characterize it.

What it does, in order:

1. Loads the standard base row table (every (date, slot, cell) row in the
   analysis domain/months, 2016-2021 by default) through the usual
   ``convection_skill.dataset`` builder -- so the QPE reconstruction, daily SM
   predictors and screen columns are exactly the battery's. If the raw netCDF
   tree is unreachable (dev machine without D: mounted) it falls back to the
   newest cached base parquet of any schema version, with a warning.
2. Defines "extreme precip event" the same way the suite defines every event:
   an absolute threshold = the requested percentile (default P99.5) of
   cell-mean QPE over ALL in-domain LAND rows (the paper's "thresholds are
   based on all data" rule -- no validity screening moves the definition).
3. Counts exceeding hours and exceeding days per cell, ranks cells, and picks
   the top one (or ``--rank N`` / an explicit ``--cell LAT LON``).
4. Characterizes that cell's climatology -- mean, SD, median, quantiles -- for
   every thermodynamic / soil-moisture / precip variable in the table, over
   three nested samples:
       all         every row in the cell (the cell's general climatology)
       event_day   every row on days with >=1 exceeding hour (whole-day state)
       event_hour  the exceeding rows themselves
5. Optionally enriches the chosen cell with variables NOT in the cached table,
   each skipped gracefully when its source is unreachable:
       - LFC height + raw (non-anomaly) SM gradients / heterogeneity, read
         from the raw ``FCST_SMAP_MRMS_<year>.nc`` files for the one cell;
       - assessed PBL height / PBLH anomaly / Gamma_gap = z_LFC - z_i, read
         from the ``results/upwind_features/UPWIND_FEATURES_<year>.nc``
         companions (built on the cluster; PBLH source = Guo et al. 2024).
6. Writes CSVs + a Markdown report + figures under ``results/extreme_cell/``.

Parcel note: the AIRS-FCST files carry MOST-UNSTABLE (MU) and MEAN-MIXED-LAYER
(MML) parcels only -- there is no surface-based (SB) CAPE/CIN in the dataset.
MML is the closest available analog to SBCAPE/SBCIN; both families are reported.

Anomaly-column note: ``*_anom`` predictors are deseasonalized (2-harmonic
day-of-year fit) and z-scored PER CELL, so their all-days mean/SD within one
cell are ~0/~1 by construction -- the event-day shift is the informative number.

Usage::

    python scripts/extreme_cell_climatology.py                 # P99.5, top cell
    python scripts/extreme_cell_climatology.py --percentile 99.9 --rank 2
    python scripts/extreme_cell_climatology.py --cell 30.5 -92.5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from convection_skill import config, dataset  # noqa: E402
from convection_skill import data_loading as dl  # noqa: E402
from convection_skill.config import AnalysisConfig  # noqa: E402

OUT_DIR_DEFAULT = config.RESULTS_DIR / "extreme_cell"
UPWIND_DIR = config.RESULTS_DIR / "upwind_features"

# ---------------------------------------------------------------------------
# Variable catalogue: (table column, pretty label, unit)
# Order = report/plot order. Enriched columns appear only when their source
# was reachable. Front-flag columns are deliberately excluded: the v9 cache
# was built while the front loader silently returned all-NaN.
# ---------------------------------------------------------------------------
VARIABLES: list[tuple[str, str, str]] = [
    # precipitation (MRMS)
    ("qpe",          "MRMS cell-mean QPE",            "mm/h"),
    ("qpe_wet",      "MRMS wet-area-mean QPE",        "mm/h"),
    ("qpe_max",      "MRMS sub-pixel max QPE",        "mm/h"),
    ("convective_share", "MRMS convective share of wet px", "-"),
    # instability / inhibition (AIRS-FCST; MU + MML parcels, no SB in files)
    ("mu_cape",      "MU CAPE",                       "J/kg"),
    ("mml_cape",     "MML CAPE",                      "J/kg"),
    ("mu_minus_mml", "MU - MML CAPE (elevatedness)",  "J/kg"),
    ("mu_cin",       "MU CIN",                        "J/kg"),
    ("mml_cin",      "MML CIN",                       "J/kg"),
    # parcel geometry (AIRS-FCST)
    ("mu_lcl",       "MU LCL height",                 "m"),
    ("mml_lcl",      "MML LCL height",                "m"),
    ("mu_lfc",       "MU LFC height",                 "m"),      # enrichment
    ("mml_lfc",      "MML LFC height",                "m"),      # enrichment
    ("mu_el",        "MU equilibrium level",          "m"),
    # boundary layer (Guo et al. 2024 assessed PBLH via upwind companions)
    ("pblh",         "PBL height (assessed)",         "m"),      # enrichment
    ("pblh_anom",    "PBLH anomaly vs mon-diurnal clim", "m"),   # enrichment
    ("gamma_gap_mu", "Gamma_gap = MU LFC - PBLH",     "m"),      # enrichment
    ("gamma_gap_mml","Gamma_gap = MML LFC - PBLH",    "m"),      # enrichment
    # near-surface parcel environment (AIRS-FCST)
    ("fcst_q",       "Parcel near-surface q",         "g/kg"),
    ("fcst_t",       "Parcel near-surface T",         "K"),
    # soil moisture (SMAP L4, pre-window daily means; timing-guarded)
    ("sm_raw",       "Soil moisture (pre-window mean)", "m3/m3"),
    ("sm_anom",      "SM anomaly (deseasonalized, cell-z)", "z"),
    ("smsd_raw",     "SM sub-cell SD (heterogeneity)", "m3/m3"), # enrichment
    ("smsd_anom",    "SM sub-cell SD anomaly",        "z"),
    ("absgrad_raw",  "|grad SM| (raw)",               "m3/m3/deg"),  # enrichment
    ("absgrad_anom", "|grad SM| anomaly",             "z"),
    ("wegrad_raw",   "W-E SM gradient (raw)",         "m3/m3/deg"),  # enrichment
    ("wegrad_anom",  "W-E SM gradient anomaly",       "z"),
    ("sngrad_raw",   "S-N SM gradient (raw)",         "m3/m3/deg"),  # enrichment
    ("sngrad_anom",  "S-N SM gradient anomaly",       "z"),
    # other environment (SMAP L4 / derived)
    ("qlay1_anom",   "Layer-1 humidity anomaly",      "z"),
    ("tlay1_anom",   "Layer-1 temperature anomaly",   "z"),
    ("wind",         "Near-surface wind speed",       "m/s"),
    ("pflux_prewindow", "Same-day pre-window precip flux", "kg/m2/s"),
    ("pflux_ante",   "Antecedent precip flux (1-5 d)", "kg/m2/s"),
]

#: which variables get a distribution panel (raw physical quantities the user
#: asked about first; anomalies second) -- panels only for columns present.
PLOT_VARS = [
    "qpe", "qpe_max", "mu_cape", "mml_cape", "mu_cin", "mml_cin",
    "mu_lcl", "mml_lcl", "mu_lfc", "mml_lfc", "pblh", "gamma_gap_mu",
    "sm_raw", "sm_anom", "absgrad_anom", "wegrad_anom", "sngrad_anom",
    "smsd_anom", "fcst_q", "wind",
]
#: variables whose monthly cycle is drawn in the seasonal figure
SEASONAL_VARS = ["mu_cape", "mu_cin", "mu_lcl", "pblh", "sm_raw", "fcst_q"]

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

GROUP_LABELS = {"all": "all days", "event_day": "event days",
                "event_hour": "event hours"}
GROUP_COLORS = {"all": "0.45", "event_day": "tab:blue", "event_hour": "tab:red"}


# --------------------------------------------------------------------------- #
# Base-table loading (cache-version fallback for machines without raw data)
# --------------------------------------------------------------------------- #
def load_base_table(cfg: AnalysisConfig) -> pd.DataFrame:
    """The standard base table; falls back to the newest cached version."""
    try:
        return dataset.build_base_table(cfg, use_cache=True)
    except Exception as err:  # raw tree unreachable -> hunt older caches
        current = dataset._base_cache_path(cfg)
        pattern = current.name.replace(f"_v{dataset.DATASET_VERSION}_", "_v*_")
        candidates = sorted(
            current.parent.glob(pattern),
            key=lambda p: int(p.name.split("_v")[1].split("_")[0]),
        )
        if not candidates:
            raise RuntimeError(
                f"raw data unreachable ({err}) and no cached base table "
                f"matches {pattern} in {current.parent}") from err
        stale = candidates[-1]
        warnings.warn(
            f"raw data unreachable ({err}); using stale cache {stale.name} "
            f"(current schema v{dataset.DATASET_VERSION}). Front columns in "
            f"v<10 caches are all-NaN; they are excluded here anyway.")
        return pd.read_parquet(stale)


# --------------------------------------------------------------------------- #
# Event counting and cell selection
# --------------------------------------------------------------------------- #
def count_events(base: pd.DataFrame, percentile: float,
                 land_min: float) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Threshold (base-sample convention), event rows, per-cell counts."""
    land = base[base["land_frac"] >= land_min]
    land = land[np.isfinite(land["qpe"])]
    threshold = float(np.nanpercentile(land["qpe"].to_numpy(), percentile))
    events = land[land["qpe"] >= threshold]
    counts = (events.groupby(["lat", "lon"])
              .agg(n_hours=("qpe", "size"),
                   n_days=("day", "nunique"),
                   max_qpe=("qpe", "max"))
              .reset_index()
              .sort_values(["n_hours", "n_days"], ascending=False)
              .reset_index(drop=True))
    return threshold, events, counts


# --------------------------------------------------------------------------- #
# Enrichment: columns not in the cached table, best-effort per source
# --------------------------------------------------------------------------- #
def enrich_from_upwind(cell_rows: pd.DataFrame, years) -> pd.DataFrame:
    """Merge assessed PBLH / PBLH anomaly / Gamma_gap from the companions."""
    rename = {"UPW_pblh": "pblh", "UPW_pblh_anom": "pblh_anom",
              "UPW_gamma_gap_mu": "gamma_gap_mu",
              "UPW_gamma_gap_mml": "gamma_gap_mml"}
    lat, lon = cell_rows["lat"].iloc[0], cell_rows["lon"].iloc[0]
    import xarray as xr
    parts = []
    for year in years:
        path = UPWIND_DIR / f"UPWIND_FEATURES_{year}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            have = [v for v in rename if v in ds]
            if not have:
                continue
            sub = ds[have].sel(lat=lat, lon=lon, method="nearest",
                               tolerance=0.01)
            df = sub.to_dataframe().reset_index().rename(
                columns={**rename, "time": "slot"})
            parts.append(df[["date", "slot"] + [rename[v] for v in have]])
    if not parts:
        print("  [enrich] no UPWIND_FEATURES_<year>.nc found -> "
              "PBLH / Gamma_gap columns skipped")
        return cell_rows
    merged = cell_rows.merge(pd.concat(parts, ignore_index=True),
                             on=["date", "slot"], how="left")
    found = sorted({y for y in years
                    if (UPWIND_DIR / f"UPWIND_FEATURES_{y}.nc").exists()})
    print(f"  [enrich] PBLH/Gamma_gap from upwind companions, years {found}")
    return merged


def enrich_from_raw(cell_rows: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Merge LFC heights + raw SM gradients from the raw netCDFs (one cell)."""
    slot_vars = {"FCST_MU_LFC": "mu_lfc", "FCST_MML_LFC": "mml_lfc"}
    daily_vars = {  # pre-window daily means, same convention as sm_raw
        config.SM_ABSGRAD_VAR: "absgrad_raw",
        config.SM_WEGRAD_VAR: "wegrad_raw",
        config.SM_SNGRAD_VAR: "sngrad_raw",
        config.SM_SD_VAR: "smsd_raw",
    }
    lat, lon = float(cell_rows["lat"].iloc[0]), float(cell_rows["lon"].iloc[0])
    parts = []
    try:
        for year in cfg.years:
            with dl.open_year(year) as probe:
                avail = [v for v in list(slot_vars) + list(daily_vars)
                         if v in probe.data_vars]
            if not avail:
                continue
            raw = dl.load_raw([year], lat_range=(lat, lat), lon_range=(lon, lon),
                              months=cfg.months, variables=avail)
            frames = []
            have_slot = [v for v in slot_vars if v in raw]
            if have_slot:
                uni = dl.make_uniform(raw[have_slot + []], cfg.slots,
                                      smap_time_policy=None)
                df = (uni[have_slot].rename({v: slot_vars[v] for v in have_slot})
                      .to_dataframe().reset_index())
                frames.append(df[["date", "slot"]
                                 + [slot_vars[v] for v in have_slot]])
            have_daily = [v for v in daily_vars if v in raw]
            if have_daily:
                daily = pd.DataFrame({"date": raw["date"].values})
                for v in have_daily:
                    da = dataset._prewindow_daily(raw, v)
                    daily[daily_vars[v]] = da.squeeze(
                        ["lat", "lon"]).to_numpy()
                frames.append(daily)
            year_df = frames[0]
            for f in frames[1:]:
                year_df = year_df.merge(f, on="date", how="outer")
            parts.append(year_df)
    except Exception as err:
        print(f"  [enrich] raw netCDFs unreachable ({err}) -> "
              "LFC / raw-gradient columns skipped")
        return cell_rows
    if not parts:
        print("  [enrich] no LFC / raw-gradient variables found in raw files")
        return cell_rows
    extra = pd.concat(parts, ignore_index=True)
    keys = ["date", "slot"] if "slot" in extra else ["date"]
    print(f"  [enrich] LFC / raw gradients from raw files: "
          f"{[c for c in extra.columns if c not in keys]}")
    return cell_rows.merge(extra, on=keys, how="left")


# --------------------------------------------------------------------------- #
# Climatology statistics
# --------------------------------------------------------------------------- #
def climatology_stats(cell_rows: pd.DataFrame,
                      event_days: set, event_mask: np.ndarray) -> pd.DataFrame:
    """Mean/SD/median/quantiles per variable x sample group."""
    groups = {
        "all": cell_rows,
        "event_day": cell_rows[cell_rows["day"].isin(event_days)],
        "event_hour": cell_rows[event_mask],
    }
    rows = []
    for col, label, unit in VARIABLES:
        if col not in cell_rows.columns:
            continue
        for gname, gdf in groups.items():
            x = pd.to_numeric(gdf[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            rec = dict(variable=col, label=label, unit=unit, group=gname,
                       n=x.size,
                       mean=np.mean(x) if x.size else np.nan,
                       std=np.std(x, ddof=1) if x.size > 1 else np.nan)
            for q in QUANTILES:
                rec[f"q{int(q * 100):02d}"] = (np.quantile(x, q)
                                               if x.size else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


def write_report(out: Path, stats: pd.DataFrame, cell, counts: pd.DataFrame,
                 threshold: float, percentile: float, cfg: AnalysisConfig,
                 n_land_rows: int) -> None:
    lat, lon = cell
    top10 = counts.head(10)
    lines = [
        f"# Extreme-precip cell climatology (P{percentile:g})",
        "",
        f"- Sample: years {cfg.years[0]}-{cfg.years[-1]}, months "
        f"{cfg.months[0]}-{cfg.months[1]}, land rows (land_frac >= "
        f"{config.LAND_FRACTION_MIN}) with finite cell-mean QPE "
        f"(n = {n_land_rows:,}).",
        f"- Event definition: cell-mean QPE >= P{percentile:g} of the base "
        f"land sample = **{threshold:.3f} mm/h** (absolute threshold, "
        "paper convention: all locations/seasons/hours).",
        f"- **Selected cell: {lat:.1f}N, {abs(lon):.1f}W** -- "
        f"{int(top10.iloc[0].n_hours)} exceeding hours on "
        f"{int(top10.iloc[0].n_days)} days "
        f"(max QPE {top10.iloc[0].max_qpe:.2f} mm/h).",
        "",
        "Groups: `all` = every row in the cell; `event_day` = all hours of "
        "days with >=1 exceeding hour; `event_hour` = exceeding rows only.",
        "`*_anom` variables are deseasonalized and z-scored per cell, so "
        "their all-days mean/SD are ~0/~1 by construction.",
        "",
        "## Top 10 cells by exceeding hours",
        "",
        "| rank | lat | lon | hours | days | max QPE (mm/h) |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in top10.iterrows():
        lines.append(f"| {i + 1} | {r.lat:.1f} | {r.lon:.1f} | "
                     f"{int(r.n_hours)} | {int(r.n_days)} | {r.max_qpe:.2f} |")
    lines += ["", "## Climatology of the selected cell", "",
              "| variable | unit | group | n | mean | std | q05 | q25 | "
              "median | q75 | q95 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats.iterrows():
        fmt = (lambda v: "-" if not np.isfinite(v)
               else f"{v:.3g}" if abs(v) < 1e4 else f"{v:.0f}")
        lines.append(
            f"| {r['label']} | {r.unit} | {GROUP_LABELS[r.group]} | {r.n:,} | "
            f"{fmt(r['mean'])} | {fmt(r['std'])} | {fmt(r.q05)} | {fmt(r.q25)} "
            f"| {fmt(r.q50)} | {fmt(r.q75)} | {fmt(r.q95)} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_event_map(out: Path, counts: pd.DataFrame, cell, threshold: float,
                   percentile: float) -> None:
    """Per-cell exceeding-hour counts on a CONUS map; star = selected cell."""
    # grid from the actual 1-degree cell centers (half-degree points; the
    # config DOMAIN_* tuples are outer bounds, not centers)
    lat0, lon0 = counts["lat"].min(), counts["lon"].min()
    lats = np.arange(lat0, counts["lat"].max() + 0.5)
    lons = np.arange(lon0, counts["lon"].max() + 0.5)
    lat_edges = np.append(lats - 0.5, lats[-1] + 0.5)  # N+1 cell edges
    lon_edges = np.append(lons - 0.5, lons[-1] + 0.5)
    grid = np.full((lats.size, lons.size), np.nan)
    ii = ((counts["lat"] - lat0).round().astype(int),
          (counts["lon"] - lon0).round().astype(int))
    grid[ii] = counts["n_hours"]

    try:  # pretty projected map when cartopy is available
        from convection_skill import plotting as cp
        import cartopy.crs as ccrs
        ax = cp.make_conus_axes(figsize=(11, 7))
        mesh = ax.pcolormesh(lon_edges, lat_edges, grid, cmap="magma_r",
                             transform=ccrs.PlateCarree(), zorder=3)
        ax.plot(cell[1], cell[0], marker="*", ms=22, mec="k", mfc="cyan",
                transform=ccrs.PlateCarree(), zorder=5)
        fig = ax.figure
        fig.colorbar(mesh, ax=ax, shrink=0.75,
                     label=f"hours with QPE >= P{percentile:g} "
                           f"({threshold:.2f} mm/h)")
    except Exception as err:  # plain lat/lon fallback
        warnings.warn(f"cartopy map unavailable ({err}); plain axes fallback")
        fig, ax = plt.subplots(figsize=(11, 7))
        mesh = ax.pcolormesh(lon_edges, lat_edges, grid, cmap="magma_r")
        ax.plot(cell[1], cell[0], marker="*", ms=22, mec="k", mfc="cyan")
        ax.set_xlabel("lon"), ax.set_ylabel("lat")
        fig.colorbar(mesh, ax=ax, label=f"hours >= P{percentile:g}")
    ax.set_title(f"Extreme-precip event count per 1° cell "
                 f"(cell-mean QPE >= {threshold:.2f} mm/h, P{percentile:g});"
                 f" star = {cell[0]:.1f}N {abs(cell[1]):.1f}W")
    fig.savefig(out / "event_count_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(out: Path, cell_rows: pd.DataFrame, event_days: set,
                       event_mask: np.ndarray, cell, threshold: float,
                       percentile: float) -> None:
    """Panel per variable: all-days vs event-day vs event-hour distributions."""
    groups = {"all": cell_rows,
              "event_day": cell_rows[cell_rows["day"].isin(event_days)],
              "event_hour": cell_rows[event_mask]}
    meta = {c: (lab, unit) for c, lab, unit in VARIABLES}
    cols = [c for c in PLOT_VARS
            if c in cell_rows.columns
            and np.isfinite(pd.to_numeric(cell_rows[c], errors="coerce")).sum()]
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows))
    for ax, col in zip(np.ravel(axes), cols):
        lab, unit = meta[col]
        log = col.startswith("qpe")  # precip is log-tailed
        for gname, gdf in groups.items():
            x = pd.to_numeric(gdf[col], errors="coerce").to_numpy(float)
            x = x[np.isfinite(x)]
            if log:
                x = x[x > 0]
            if x.size < 5:
                continue
            lo, hi = np.quantile(x, [0.001, 0.999])
            bins = (np.geomspace(max(lo, 1e-3), hi, 40) if log
                    else np.linspace(lo, hi, 40))
            ax.hist(x, bins=bins, density=True, histtype="step", lw=1.8,
                    color=GROUP_COLORS[gname],
                    label=f"{GROUP_LABELS[gname]} (n={x.size:,})")
            ax.axvline(np.mean(x), color=GROUP_COLORS[gname], ls="--", lw=1)
        if log:
            ax.set_xscale("log")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel(unit, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5, frameon=False)
    for ax in np.ravel(axes)[len(cols):]:
        ax.set_axis_off()
    fig.suptitle(f"Cell {cell[0]:.1f}N {abs(cell[1]):.1f}W climatology: "
                 f"all days vs P{percentile:g} event days "
                 f"(QPE >= {threshold:.2f} mm/h); dashed = mean", y=1.001)
    fig.tight_layout()
    fig.savefig(out / "climatology_distributions.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def plot_cycles(out: Path, cell_rows: pd.DataFrame, event_days: set,
                event_mask: np.ndarray, cell, percentile: float) -> None:
    """Seasonal + diurnal structure of the cell and its events."""
    meta = {c: (lab, unit) for c, lab, unit in VARIABLES}
    cols = [c for c in SEASONAL_VARS if c in cell_rows.columns
            and np.isfinite(pd.to_numeric(cell_rows[c], errors="coerce")).sum()]
    fig, axes = plt.subplots(2, max(3, (len(cols) + 2 + 1) // 2),
                             figsize=(4.0 * max(3, (len(cols) + 3) // 2), 6.6))
    axes = np.ravel(axes)

    # panel 0: monthly event-hour counts; panel 1: per-slot event counts
    ev = cell_rows[event_mask]
    axes[0].bar(*np.unique(ev["month"], return_counts=True), color="tab:red")
    axes[0].set_title(f"P{percentile:g} event hours by month", fontsize=10)
    axes[0].set_xlabel("month", fontsize=8)
    slot_hours = dict(zip(config.FORECAST_SLOTS, config.FORECAST_HOURS_UTC))
    su, sc = np.unique(ev["slot"], return_counts=True)
    axes[1].bar([f"{slot_hours.get(s, s):02d}Z" if s != 0 else "ovp"
                 for s in su], sc, color="tab:red")
    axes[1].set_title("event hours by forecast slot", fontsize=10)

    for ax, col in zip(axes[2:], cols):
        lab, unit = meta[col]
        for gname, gdf in {"all": cell_rows,
                           "event_day": cell_rows[cell_rows["day"]
                                                  .isin(event_days)]}.items():
            g = gdf.groupby("month")[col].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=3,
                        capsize=2, lw=1.4, color=GROUP_COLORS[gname],
                        label=GROUP_LABELS[gname])
        ax.set_title(f"{lab} ({unit})", fontsize=10)
        ax.set_xlabel("month", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[2 + len(cols):]:
        ax.set_axis_off()
    fig.suptitle(f"Cell {cell[0]:.1f}N {abs(cell[1]):.1f}W: seasonal / "
                 "diurnal structure (bars = events; mean ± SD by month)",
                 y=1.001)
    fig.tight_layout()
    fig.savefig(out / "seasonal_diurnal.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="QPE percentile defining an extreme event (99.5)")
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(config.ALL_YEARS))
    ap.add_argument("--rank", type=int, default=1,
                    help="use the Nth-ranked cell instead of the top one")
    ap.add_argument("--cell", type=float, nargs=2, metavar=("LAT", "LON"),
                    help="characterize this cell instead of the top-ranked one")
    ap.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-enrich", action="store_true",
                    help="skip the raw-file / upwind-companion enrichment")
    args = ap.parse_args(argv)

    cfg = AnalysisConfig(years=tuple(args.years),
                         heavy_percentile=args.percentile,
                         name=f"extreme_cell_p{args.percentile:g}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading base table, years {cfg.years} ...")
    base = load_base_table(cfg)

    print(f"[2/5] counting events above P{args.percentile:g} ...")
    threshold, events, counts = count_events(
        base, args.percentile, config.LAND_FRACTION_MIN)
    n_land = int((base["land_frac"] >= config.LAND_FRACTION_MIN).sum())
    counts.to_csv(out / "event_counts_by_cell.csv", index=False)
    if args.cell:
        cell = (args.cell[0], args.cell[1])
    else:
        r = counts.iloc[args.rank - 1]
        cell = (float(r.lat), float(r.lon))
    print(f"      threshold = {threshold:.3f} mm/h; selected cell "
          f"{cell[0]:.1f}N {abs(cell[1]):.1f}W")

    print("[3/5] assembling the cell sample ...")
    cell_rows = base[(base["lat"] == cell[0]) & (base["lon"] == cell[1])].copy()
    if not args.no_enrich:
        cell_rows = enrich_from_upwind(cell_rows, cfg.years)
        cell_rows = enrich_from_raw(cell_rows, cfg)
    event_mask = (np.isfinite(cell_rows["qpe"])
                  & (cell_rows["qpe"] >= threshold)).to_numpy()
    event_days = set(cell_rows.loc[event_mask, "day"])

    print("[4/5] climatology statistics ...")
    stats = climatology_stats(cell_rows, event_days, event_mask)
    stats.insert(0, "percentile", args.percentile)
    stats.insert(1, "threshold_mm", threshold)
    stats.insert(2, "cell_lat", cell[0])
    stats.insert(3, "cell_lon", cell[1])
    stats.to_csv(out / "cell_climatology_stats.csv", index=False)
    write_report(out, stats, cell, counts, threshold, args.percentile, cfg,
                 n_land)

    print("[5/5] figures ...")
    plot_event_map(out, counts, cell, threshold, args.percentile)
    plot_distributions(out, cell_rows, event_days, event_mask, cell,
                       threshold, args.percentile)
    plot_cycles(out, cell_rows, event_days, event_mask, cell, args.percentile)
    print(f"done -> {out}/ (REPORT.md, 2 CSVs, 3 figures)")


if __name__ == "__main__":
    main()
