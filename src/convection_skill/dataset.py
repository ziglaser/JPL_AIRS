"""THE unified dataset builder: one Richardson-style preparation for every
analysis (paper replication and hypothesis battery alike).

Three layers, all driven by one :class:`~convection_skill.config.AnalysisConfig`:

1. :func:`build_base_table` -- the cached SUPERSET: every (date, slot, cell)
   row in the configured domain/months, including ocean cells and rows without
   valid AIRS/SMAP/MRMS data. This is the paper's threshold base ("thresholds
   are based on ALL data") and the only expensive artifact; every validity
   mode and screen is a cheap row operation on top, so configs share the cache.
2. :func:`event_thresholds` -- absolute event thresholds from the base's
   in-domain LAND rows, so event definitions never move when the sample
   screening changes.
3. :func:`build_dataset` / :func:`prepare` -- the config's screens applied
   (land, Mark's altitude / dry-start / complete-series screens, product
   validity, post-rain screens) and the event flags built against the base
   thresholds. With ``sample_unit="cell_day"``, :func:`to_cell_days` also
   emits the Mark-style one-row-per-cell-day wide table.

Mark's screens (email 2026-07-21) are the gold-standard sample definition;
his code lives VERBATIM in :mod:`convection_skill.mark_screens` and the
cached columns here (``alt_max``, ``dry_start_qpe``, ``valid7``) are parity-
tested against it. Computing them once at cube level (his approach) replaces
the old per-config pandas groupby completeness pass.

Timing discipline (Tuttle & Salvucci endogeneity fix): every same-day SM
predictor uses ONLY the pre-window L4 slots (16:30/19:30 UTC) -- the 01:30/04:30
UTC "same-day" L4 slots fall AFTER the 21-02 UTC target window and would leak
post-rain soil moisture. That is also why ``make_uniform`` runs with
``smap_time_policy=None`` (no L4-to-slot interpolation).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from . import data_loading as dl
from . import fronts as fr
from . import mark_screens as mk
from . import predictors as P
from .config import AnalysisConfig
from .gini import exceedance_flags  # noqa: F401  (re-export for interactive use)

#: Bump when the table schema/derivation changes: stale caches are then ignored
#: (the version is part of the cache filename). v6 = unified suite schema
#: (hypothesis_tests table v5 + mu_el, mu_cape_overpass, qpe_wet); v7 = hail
#: counted as CONVECTIVE in the precip-mode fractions (Zach 2026-07-23);
#: v8 = Mark's screen columns (alt_max, dry_start_qpe, valid7);
#: v9 = CODSUS surface-front flags (front_{type}_{1,3}w; fronts.py; NaN 2019+).
DATASET_VERSION: int = 9
CACHE_DIR = config.RESULTS_DIR / "suite" / "cache"

#: Slot-level variables (file name -> table name). The union of the paper
#: replication's fields (incl. mu_el for the AIRS-FCST validity rule and the
#: replicated overpass baseline) and the battery's.
SLOT_FIELDS: dict[str, str] = {
    "FCST_MU_CAPE": "mu_cape",
    "FCST_MU_CIN": "mu_cin",
    "FCST_MU_EL": "mu_el",
    "FCST_MU_LCL": "mu_lcl",
    "FCST_MML_CAPE": "mml_cape",
    "FCST_MML_LCL": "mml_lcl",
    "FCST_MU_CAPE_overpass": "mu_cape_overpass",
    "FCST_q": "fcst_q",
    "FCST_t": "fcst_t",
    "MRMS_GaugeCorrQPE01H_max": "qpe_max",
    "MRMS_GaugeCorrQPE01H_sk": "qpe_sk",
}
OPTIONAL_SLOT_FIELDS: dict[str, str] = {"FCST_MML_CIN": "mml_cin"}

#: Targets that are precipitation events (the convective filter and screens act
#: on these). Mediator targets (CAPE/LCL/CIN) are never mode-filtered.
PRECIP_TARGETS: tuple[str, ...] = ("heavy", "heavy_sens", "any", "max_extreme", "sk_high")


# --------------------------------------------------------------------------- #
# Daily (pre-window) predictors
# --------------------------------------------------------------------------- #
def _prewindow_daily(raw: xr.Dataset, var: str) -> xr.DataArray:
    """Mean of an L4 field over the pre-window slots -> (date, lat, lon)."""
    return raw[var].isel(L4_nhours=list(config.L4_PREWINDOW_SLOTS)).mean(
        "L4_nhours", skipna=True)


def _anom(da: xr.DataArray, name: str) -> xr.DataArray:
    """Standardized harmonic anomaly (the battery's default predictor form)."""
    a = P.deseasonalize(da, n_harmonics=config.N_HARMONICS)
    if config.ZSCORE_PER_CELL:
        a = P.zscore_by_cell(a)
    return a.rename(name)


def build_daily(raw: xr.Dataset) -> xr.Dataset:
    """All (date, lat, lon) daily predictors, anomaly-transformed and lagged."""
    out = {}

    sm = _prewindow_daily(raw, config.SM_VAR).rename("sm_raw")
    out["sm_raw"] = sm
    out["sm_anom"] = _anom(sm, "sm_anom")
    local, nonlocal_ = P.local_nonlocal_decomposition(
        out["sm_anom"], halfwidth=config.NEIGHBORHOOD_HALFWIDTH)
    out["sm_local"] = local.rename("sm_local")
    out["sm_nonlocal"] = nonlocal_.rename("sm_nonlocal")

    out["smsd_anom"] = _anom(_prewindow_daily(raw, config.SM_SD_VAR), "smsd_anom")
    out["absgrad_anom"] = _anom(_prewindow_daily(raw, config.SM_ABSGRAD_VAR), "absgrad_anom")
    out["wegrad_anom"] = _anom(_prewindow_daily(raw, config.SM_WEGRAD_VAR), "wegrad_anom")
    out["sngrad_anom"] = _anom(_prewindow_daily(raw, config.SM_SNGRAD_VAR), "sngrad_anom")
    out["qlay1_anom"] = _anom(_prewindow_daily(raw, config.QLAY1_VAR), "qlay1_anom")
    out["tlay1_anom"] = _anom(_prewindow_daily(raw, config.TLAY1_VAR), "tlay1_anom")
    out["wind"] = _prewindow_daily(raw, config.WIND_VAR).rename("wind")

    # antecedent SM ladder (lags of the anomaly; T4) and the antecedent-precip
    # control (full-day pflux on PRIOR days; Tuttle & Salvucci guard)
    for k in config.ANTECEDENT_LAGS:
        out[f"sm_anom_lag{k}"] = out["sm_anom"].shift(date=k).rename(f"sm_anom_lag{k}")
    pflux_daily = raw[config.PFLUX_VAR].mean("L4_nhours", skipna=True)
    lo, hi = config.ANTE_PRECIP_WINDOW
    out["pflux_ante"] = P.antecedent_mean(pflux_daily, lo, hi).rename("pflux_ante")
    # same-day PRE-WINDOW precip: the control that catches same-synoptic-system
    # morning rain masquerading as an SM signal (battery finding 2026-07-20:
    # conditioning on this collapses the same-day sm_anom Gini +0.31 -> ~-0.04).
    out["pflux_prewindow"] = _prewindow_daily(raw, config.PFLUX_VAR).rename("pflux_prewindow")

    #: cell-climatology aridity (a per-cell constant broadcast onto dates)
    clim = sm.mean("date", skipna=True)
    out["sm_cell_clim"] = xr.broadcast(clim, sm)[0].rename("sm_cell_clim")

    return xr.Dataset(out)


def _add_precip_mode_fractions(uniform: xr.Dataset, ds: xr.Dataset) -> None:
    """Convective/stratiform sub-pixel fractions from the MRMS precip flags.

    Audited accounting: classified pixels = TotalCountsAll - CountsNaN -
    NoCoverage (MissingFileCounts sits OUTSIDE the total); cell-hours with no
    classified pixels get NaN. ``*_frac`` = coverage of the cell; ``*_share`` =
    share of ALL precipitating sub-pixels (conv + strat + snow). HAIL COUNTS AS
    CONVECTIVE (Zach 2026-07-23: hail is a convective precip mode, not "other");
    the non-convective frozen residual is snow only. NOTE the MRMS "Convection"
    flag marks strict convective CORES, so shares run low even for heavy events
    (P99.9 mean ~0.2 before the hail change).
    """
    def cnt(category):
        return uniform[f"{dl.PRECIP_FLAG_VAR}_{category}"]

    if f"{dl.PRECIP_FLAG_VAR}_TotalCountsAll" not in uniform:
        return
    classified = cnt("TotalCountsAll") - cnt("CountsNaN") - cnt("NoCoverage")
    classified = classified.where(classified > 0)
    convective = cnt("Convection") + cnt("TropicalConvectiveRain") + cnt("Hail")
    stratiform = (cnt("WarmStratiformRain") + cnt("CoolStratiformRain")
                  + cnt("TropicalStratiformRain"))
    snow = cnt("Snow")
    ds["no_precip_frac"] = cnt("NoPrecipitation") / classified
    ds["convective_frac"] = convective / classified
    ds["stratiform_frac"] = stratiform / classified
    precipitating = (convective + stratiform + snow).where(
        convective + stratiform + snow > 0)
    ds["convective_share"] = convective / precipitating
    ds["stratiform_share"] = stratiform / precipitating


# --------------------------------------------------------------------------- #
# The cached base superset
# --------------------------------------------------------------------------- #
def _base_cache_path(cfg: AnalysisConfig) -> "Path":
    """Cache file keyed by the fields that affect the BASE table only."""
    key = (f"y{'_'.join(str(y) for y in cfg.years)}"
           f"_s{''.join(str(s) for s in cfg.slots)}"
           f"_m{cfg.months[0]}-{cfg.months[1]}")
    domain = f"{cfg.lat_range}{cfg.lon_range}"
    digest = hashlib.md5(domain.encode()).hexdigest()[:8]
    return CACHE_DIR / f"base_v{DATASET_VERSION}_{key}_{digest}.parquet"


def build_base_table(cfg: AnalysisConfig, use_cache: bool = True) -> pd.DataFrame:
    """The superset row table: EVERY (date, slot, cell) row in domain/months.

    No land, validity, or rain screening -- ocean cells and rows without valid
    AIRS/SMAP/MRMS all included, so this table serves as the paper's threshold
    base and every config's starting point. Cached to parquet keyed by
    (years, slots, months, domain, DATASET_VERSION); minutes to build per year,
    seconds to reload.
    """
    cache_path = _base_cache_path(cfg)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    variables = (
        list(SLOT_FIELDS) + list(OPTIONAL_SLOT_FIELDS)
        + ["FCST_alt"]  # Mark's altitude screen (all-hours parcel altitude)
        + [config.QPE_VAR, config.QPE_CNT_VAR, dl.PRECIP_FLAG_VAR]
        + [config.SM_VAR, config.SM_SD_VAR, config.SM_WEGRAD_VAR,
           config.SM_SNGRAD_VAR, config.SM_ABSGRAD_VAR, config.QLAY1_VAR,
           config.TLAY1_VAR, config.WIND_VAR, config.PFLUX_VAR]
    )
    parts = []
    for year in cfg.years:  # one year at a time keeps peak memory bounded
        with dl.open_year(year) as probe:
            available = [v for v in variables if v in probe.data_vars]
        raw = dl.load_raw(
            [year],
            lat_range=cfg.lat_range,
            lon_range=cfg.lon_range,
            months=cfg.months,
            variables=available,
        )
        uniform = dl.make_uniform(raw, cfg.slots, smap_time_policy=None)
        daily = build_daily(raw)

        # slot-level frame (the Richardson preparation)
        keep = {src: dst for src, dst in {**SLOT_FIELDS, **OPTIONAL_SLOT_FIELDS}.items()
                if src in uniform}
        ds = uniform[list(keep) + ["land_frac"]].rename(keep)
        # the audited QPE reconstruction: _av is the precipitating-area mean,
        # the paper's grid-cell mean is _av * _cnt / 81 (see config QPE note)
        ds["qpe"] = uniform[config.QPE_VAR] * (
            uniform[config.QPE_CNT_VAR] / config.WET_CELL_MAX_CNT)
        ds["qpe_wet"] = uniform[config.QPE_VAR]
        _add_precip_mode_fractions(uniform, ds)
        # overpass-hour cell-mean QPE (slot 0): the post-rain screen's
        # "did it already rain before the window" measure. Covers the overpass
        # HOUR only; pflux_prewindow covers the gap hours to 21 UTC.
        axis = next(d for d in dl.FORECAST_DIMS
                    if d in raw[config.QPE_VAR].dims)
        qpe0 = (raw[config.QPE_VAR] * raw[config.QPE_CNT_VAR]
                / config.WET_CELL_MAX_CNT).isel(
                    {axis: config.OVERPASS_SLOT}
                ).transpose("date", "lat", "lon")
        ds["qpe_overpass"] = (("date", "lat", "lon"), qpe0.values)

        # CODSUS surface-front flags (concurrent with the forecast window;
        # all-NaN for years without front files, e.g. 2019+). See fronts.py.
        flags = fr.year_front_flags(
            year, dates=ds["date"].values, slots=cfg.slots,
            lats=ds["lat"].values, lons=ds["lon"].values)
        for name in fr.front_columns():
            ds[name] = (("date", "slot", "lat", "lon"), flags[name].values)

        df = ds.to_dataframe(dim_order=["date", "slot", "lat", "lon"]).reset_index()

        # Mark's screen columns (cell-day level): alt_max / dry_start_qpe are
        # the sufficient statistics of his altitude and dry-start constraints,
        # valid7 his completeness constraint verbatim (mark_screens.py).
        daily = daily.merge(mk.master_mask_components(raw))

        # daily-level frame, merged on (date, lat, lon)
        ddf = daily.to_dataframe().reset_index()
        df = df.merge(ddf, on=["date", "lat", "lon"], how="left")
        parts.append(df)

    table = pd.concat(parts, ignore_index=True)

    # id / stratum columns
    table["year"] = table["date"].dt.year.astype("int16")
    table["day"] = table["date"].values.astype("datetime64[D]")
    table["month"] = table["date"].dt.month.astype("int8")
    season_of = {m: s for s, ms in config.SEASONS.items() for m in ms}
    table["season"] = table["month"].map(season_of)
    table["mu_minus_mml"] = table["mu_cape"] - table["mml_cape"]
    table["local_hour"] = (table["hour_utc"] + table["lon"] / 15.0) % 24.0
    table["is_east"] = table["lon"] >= config.EAST_WEST_SPLIT_LON
    table["is_late_slot"] = table["slot"].isin(config.LATE_SLOTS)

    # memory: floats to float32 except the target
    for c in table.columns:
        if table[c].dtype == np.float64 and c != "qpe":
            table[c] = table[c].astype("float32")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache_path)
    return table


# --------------------------------------------------------------------------- #
# Base-sample event thresholds (absolute, so screens never move them)
# --------------------------------------------------------------------------- #
def event_thresholds(base: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, float]:
    """Absolute thresholds for every target, from the in-domain LAND rows.

    The paper's rule: "thresholds are based on all data, including all
    locations, all seasons, and both wet and dry hours" -- data screening
    applies to the skill evaluation, not the event definition. Computed once
    per run; :func:`build_flags` then cuts on these absolute values so every
    validity mode / screen sees identical event definitions. The percentiles
    themselves come from the run's hypothesis config (``cfg.heavy_percentile``
    et al.).
    """
    land = base[base["land_frac"] >= cfg.land_fraction_min]
    land = land[np.isfinite(land["qpe"])]
    qpe = land["qpe"].to_numpy()
    wet = qpe > cfg.any_precip_mm
    return {
        "heavy": float(np.nanpercentile(qpe, cfg.heavy_percentile)),
        "heavy_sens": float(np.nanpercentile(qpe, cfg.heavy_sens_percentile)),
        "any": cfg.any_precip_mm,
        "max_extreme": float(np.nanpercentile(land["qpe_max"].to_numpy(),
                                              cfg.max_percentile)),
        "sk_high": float(np.nanpercentile(land["qpe_sk"].to_numpy()[wet],
                                          cfg.sk_percentile)),
        "high_mml_cape": float(np.nanpercentile(land["mml_cape"].to_numpy(), 90.0)),
        "low_mml_lcl": float(np.nanpercentile(land["mml_lcl"].to_numpy(), 10.0)),
        "inhibited": float(np.nanpercentile(land["mu_cin"].to_numpy(), 10.0)),
    }


def qpe_percentile_thresholds(
    base: pd.DataFrame, cfg: AnalysisConfig,
    percentiles: tuple[float, ...] = config.QPE_PERCENTILES,
) -> dict[float, float]:
    """The paper's Fig. 2b threshold ladder {percentile -> mm/h} from the base."""
    land = base[base["land_frac"] >= cfg.land_fraction_min]
    qpe = land["qpe"].to_numpy()
    return {p: float(np.nanpercentile(qpe, p)) for p in percentiles}


# --------------------------------------------------------------------------- #
# Config-driven row screens
# --------------------------------------------------------------------------- #
def rain_screen_mask(
    table: pd.DataFrame,
    overpass: bool = True,
    forecast: bool = True,
    rain_mm: float = config.RAIN_SCREEN_MM,
) -> np.ndarray:
    """Keep-mask (True = keep) removing rows AFTER it has rained in the cell.

    The row-deletion complement of the ``pflux_prewindow`` conditioning
    control: once rain has fallen, later rows in that cell are contaminated
    (cold pools, rain-wetted soil), so any SM "signal" there is a consequence
    of rain rather than a predictor of it. Two independent toggles:

    - ``overpass``: drop the ENTIRE cell-day when the overpass-hour QPE
      (``qpe_overpass``, slot 0) exceeds ``rain_mm``.
    - ``forecast``: within each cell-day, drop the slots STRICTLY AFTER the
      first slot whose cell-mean QPE exceeds ``rain_mm``. The first-rain slot
      itself is kept -- that is the event being predicted.

    Rows where the deciding value is NaN are kept (no evidence of prior rain).
    """
    keep = np.ones(len(table), dtype=bool)
    if overpass:
        keep &= ~(table["qpe_overpass"].to_numpy() > rain_mm)
    if forecast:
        wet_slot = table["slot"].where(table["qpe"] > rain_mm)
        onset = wet_slot.groupby(
            [table["day"], table["lat"], table["lon"]]).transform("min")
        keep &= ~(table["slot"] > onset).to_numpy()  # NaN onset -> all kept
    return keep


def apply_screens(base: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Land + Mark's screens + product-validity + post-rain screens, per config.

    The altitude / dry-start / complete-series screens are Mark's master-mask
    constraints (mark_screens.py, verbatim semantics: a NaN in the deciding
    value FAILS the screen), applied here as cheap cuts on the cached columns.
    """
    table = base[base["land_frac"] >= cfg.land_fraction_min]

    # Mark: "Altitude Constraint: lowest parcel must be below z_thresh for ALL
    # 7 hours" -- alt_max is max(FCST_alt) over the hours, skipna=False.
    if cfg.screen_altitude:
        table = table[table["alt_max"].to_numpy() < cfg.alt_max_m]

    # Mark: "Dry Start Constraint: QPE ~ 0 for first 2 hours" (hours 0 and 1).
    if cfg.screen_dry_start:
        table = table[table["dry_start_qpe"].to_numpy() <= cfg.dry_start_mm]

    # Mark: "Validity Constraint: No NaNs for specific vars across all 7
    # hours" (MU_CAPE, MU_CIN, grid-mean QPE) -- the cached valid7 column.
    if cfg.require_complete_days:
        table = table[table["valid7"].to_numpy().astype(bool)]

    cols = cfg.validity_columns()
    if cols:
        valid = np.ones(len(table), dtype=bool)
        for col in cols:
            valid &= np.isfinite(table[col].to_numpy())
        table = table[valid]

    if cfg.screen_overpass_rain or cfg.screen_forecast_rain:
        keep = rain_screen_mask(table, overpass=cfg.screen_overpass_rain,
                                forecast=cfg.screen_forecast_rain,
                                rain_mm=cfg.rain_screen_mm)
        table = table[keep]
    return table.reset_index(drop=True)


def build_dataset(cfg: AnalysisConfig, use_cache: bool = True) -> pd.DataFrame:
    """The analysis table for one config: cached base + that config's screens."""
    return apply_screens(build_base_table(cfg, use_cache=use_cache), cfg)


# --------------------------------------------------------------------------- #
# Event flags (absolute base thresholds -> screens never move definitions)
# --------------------------------------------------------------------------- #
def build_flags(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    convective_min: Optional[float] = None,
    convective_col: str = "convective_share",
) -> dict[str, np.ndarray]:
    """Event flags for every target the registry uses, from ABSOLUTE thresholds.

    ``convective_min`` is the convective-event filter: precip events
    additionally require ``convective_col`` (share of raining sub-pixels;
    alternative ``"convective_frac"`` cell coverage) to meet the threshold.
    NaN (no raining classified pixels) counts as not-convective.
    """
    qpe = df["qpe"].to_numpy()
    mml_lcl = df["mml_lcl"].to_numpy()
    flags = {
        "heavy": qpe > thresholds["heavy"],
        "heavy_sens": qpe > thresholds["heavy_sens"],
        "any": qpe > thresholds["any"],
        "max_extreme": df["qpe_max"].to_numpy() > thresholds["max_extreme"],
        "sk_high": (qpe > thresholds["any"])
                   & (df["qpe_sk"].to_numpy() > thresholds["sk_high"]),
        "high_mml_cape": df["mml_cape"].to_numpy() > thresholds["high_mml_cape"],
        "low_mml_lcl": (mml_lcl < thresholds["low_mml_lcl"]) & np.isfinite(mml_lcl),
        "inhibited": df["mu_cin"].to_numpy() < thresholds["inhibited"],
    }
    if convective_min is not None:
        conv = df[convective_col].to_numpy()
        is_convective = np.isfinite(conv) & (conv >= convective_min)
        for target in PRECIP_TARGETS:
            flags[target] = flags[target] & is_convective
    return flags


def build_onset_table(df: pd.DataFrame, thresholds: dict[str, float],
                      convective_min: Optional[float] = None,
                      convective_col: str = "convective_share") -> pd.DataFrame:
    """One row per precipitating cell-day (S5): onset slot and early flag.

    Onset = first forecast slot whose cell-mean QPE exceeds the any-precip
    threshold; ``early_onset`` = onset within the first half of the window.
    Daily predictors ride along from the first row of the cell-day. With
    ``convective_min`` set, only slots whose ``convective_col`` meets it count
    as precipitating (the same filter as :func:`build_flags`).
    """
    wet = df[df["qpe"] > thresholds["any"]]
    if convective_min is not None:
        conv = wet[convective_col]
        wet = wet[np.isfinite(conv) & (conv >= convective_min)]
    if wet.empty:
        return wet.copy()
    grp = wet.groupby(["day", "lat", "lon"], sort=False, observed=True)
    onset = grp["slot"].min().rename("onset_slot").reset_index()
    first_rows = grp.head(1).drop(columns=["slot"])
    out = first_rows.merge(onset, on=["day", "lat", "lon"], how="left")
    out["early_onset"] = out["onset_slot"] <= max(config.EARLY_SLOTS)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cell-day samples (Mark's extract_samples, row-table form)
# --------------------------------------------------------------------------- #
#: Columns that vary within a cell-day; everything else rides along as a
#: per-day scalar. ``mu_cape_overpass`` is replicated across slots, so it is
#: daily by construction and listed with the daily columns.
_SLOT_LEVEL_COLS: tuple[str, ...] = tuple(
    c for c in {**SLOT_FIELDS, **OPTIONAL_SLOT_FIELDS}.values()
    if c != "mu_cape_overpass"
) + ("qpe", "qpe_wet", "no_precip_frac", "convective_frac", "stratiform_frac",
     "convective_share", "stratiform_share", "mu_minus_mml") + fr.front_columns()


def to_cell_days(table: pd.DataFrame) -> pd.DataFrame:
    """One row per surviving cell-day, slot variables widened to ``<var>_h<slot>``.

    The row-table form of Mark's ``extract_samples``: his stacked
    ``(sample, time)`` arrays become one row per sample with the retained
    forecast hours as columns (his ``target_hours=slice(1, None)`` is our
    ``cfg.slots``, already hours 1+). Daily predictors (SM anomalies, screens,
    strata) ride along once per cell-day.
    """
    keys = ["day", "lat", "lon"]
    slot_cols = [c for c in _SLOT_LEVEL_COLS if c in table.columns]
    wide = table.pivot(index=keys, columns="slot", values=slot_cols)
    wide.columns = [f"{var}_h{int(slot)}" for var, slot in wide.columns]

    daily_cols = [c for c in table.columns
                  if c not in slot_cols
                  and c not in keys + ["slot", "hour_utc", "local_hour",
                                       "is_late_slot"]]
    daily = table.drop_duplicates(subset=keys).set_index(keys)[daily_cols]
    return daily.join(wide).reset_index()


# --------------------------------------------------------------------------- #
# Everything a suite run needs, prepared once
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """One config's fully prepared sample: the shared input of every test.

    ``table``/``flags`` always hold the (date, slot, cell) rows the suite
    consumes; with ``cfg.sample_unit == "cell_day"``, ``cell_days`` also holds
    the Mark-style one-row-per-cell-day table (ML-ready wide format).
    """

    cfg: AnalysisConfig
    table: pd.DataFrame                      # screened analysis rows
    thresholds: dict[str, float]             # absolute, from the base sample
    flags: dict[str, np.ndarray] = field(repr=False, default=None)
    onset: pd.DataFrame = field(repr=False, default=None)
    cell_days: pd.DataFrame = field(repr=False, default=None)


def prepare(cfg: AnalysisConfig, use_cache: bool = True) -> Prepared:
    """Base table -> thresholds -> screens -> flags/onset/cell-days, in one call."""
    base = build_base_table(cfg, use_cache=use_cache)
    thresholds = event_thresholds(base, cfg)
    table = apply_screens(base, cfg)
    flags = build_flags(table, thresholds,
                        convective_min=cfg.convective_min,
                        convective_col=cfg.convective_col)
    onset = build_onset_table(table, thresholds,
                              convective_min=cfg.convective_min,
                              convective_col=cfg.convective_col)
    cell_days = to_cell_days(table) if cfg.sample_unit == "cell_day" else None
    return Prepared(cfg=cfg, table=table, thresholds=thresholds,
                    flags=flags, onset=onset, cell_days=cell_days)
