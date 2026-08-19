from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import math
import json
from pathlib import Path
import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
#: Data root, overridable exactly like ``front_finder.config.DATA_ROOT`` and
#: ``dl_front.config`` (2026-08-18): the same ``JPL_AIRS_DATA`` export now
#: configures all three packages, so one variable moves the whole suite
#: between the cluster tree (/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data)
#: and a development copy off the repo (this machine keeps it on D:,
#: /mnt/d/JPL_AIRS/data).  Was hard-wired to REPO_ROOT/"data".
DATA_DIR: Path = Path(os.environ.get("JPL_AIRS_DATA", REPO_ROOT / "data"))
RESULTS_DIR: Path = Path(os.environ.get("JPL_AIRS_RESULTS",
                                        REPO_ROOT / "results"))
#: Land-sea mask read by data_loading.load_land_fraction_grid: the ``lsm``
#: fraction on 1-deg half-degree centers, which coincide EXACTLY with the
#: FCST_SMAP_MRMS grid.  FIXED 2026-08-18: this pointed at a bare
#: DATA_DIR/"lsm.nc" that has never existed in the post-2026-08-13 manifest
#: tree, so load_land_fraction_grid raised FileNotFoundError and NO base
#: table could be built at all -- not a cosmetic path move.
LSM_PATH: Path = DATA_DIR / "masks" / "land_surface_mask.nc"
YEAR_FILE_TEMPLATE: str = "FCST_SMAP_MRMS/FCST_SMAP_MRMS_{year}.nc"

# --------------------------------------------------------------------------- #
# Time domain
# --------------------------------------------------------------------------- #
ALL_YEARS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020, 2021)
PAPER_YEARS: tuple[int, ...] = (2019, 2020)
ANALYSIS_MONTHS: tuple[int, int] = (3, 11)

# --------------------------------------------------------------------------- #
# Spatial domain
# --------------------------------------------------------------------------- #
DOMAIN_LAT: tuple[float, float] = (32.0, 53.0)
DOMAIN_LON: tuple[float, float] = (-107.0, -64.0)

LAND_FRACTION_MIN: float = 0.50

# --------------------------------------------------------------------------- #
# Forecast hours
# --------------------------------------------------------------------------- #

OVERPASS_SLOT: int = 0
OVERPASS_HOURS_UTC: tuple[int, ...] = (17, 18, 19) # Check
FORECAST_SLOTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
FORECAST_HOURS_UTC: tuple[int, ...] = (21, 22, 23, 0, 1, 2)

#: Slot-0 variables replicated across the forecast hours as ``<name>_overpass``
#: -- the paper's proximity-sounding baseline ("the values calculated at
#: overpass time are replicated for each of the 21-02 UTC forecast timesteps").
#: Generalized from the old single PREDICTOR_VAR: any listed variable present
#: in the data gets its overpass column.
OVERPASS_REPLICATED_VARS: tuple[str, ...] = ("FCST_MU_CAPE",)

# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
REQUIRED_VALID_VARS: tuple[str, ...] = (
    "FCST_MU_CAPE",
    "FCST_MU_EL",
    "FCST_MU_LCL",
    "FCST_MU_CIN",
)

MIN_PARCELS: int = 20
MIN_1KM_PARCELS: int = 1
PARCEL_COUNT_VAR: str = "FCST_N"

# --------------------------------------------------------------------------- #
# Target (precipitation)
# --------------------------------------------------------------------------- #
# Methods: "one-hour accumulations averaged over 1x1 grid cells".
#
# IMPORTANT (root cause of the 2026-07 Gini inflation bug): in our NetCDF files
# ``MRMS_GaugeCorrQPE01H_av`` is the mean over *precipitating* sub-cells only
# (0 when none are wet), NOT the paper's all-pixel 1x1 grid-cell mean. Evidence:
# (a) its pooled percentiles run 1.7-9x the paper's Fig. 2c ladder, worst for
# light rain where wet fractions are smallest; (b) ``.._cnt`` is exactly > 0
# wherever ``.._av`` > 0 and tops out at 81 at every latitude (a fully wet
# cell); (c) rescaling by the wet fraction cnt/81 reproduces the paper's
# thresholds and Fig. 1 case-study values. The paper itself notes (Supplementary
# Note 8) that Gini is *higher* against within-precipitating-area QPE -- which is
# exactly the uncorrected ``_av``. The loader therefore reconstructs
#
#     qpe (grid-cell mean, paper's target) = _av * _cnt / WET_CELL_MAX_CNT
#
# and keeps the conditional mean as the separate column ``qpe_wet``.
QPE_VAR: str = "MRMS_GaugeCorrQPE01H_av"
QPE_CNT_VAR: str = "MRMS_GaugeCorrQPE01H_cnt"
#: Value of ``_cnt`` for a fully precipitating 1x1 cell (dataset max, reached at
#: multiple latitudes; NOT cos(lat)-weighted -- verified in tests/test_paper_benchmarks).
WET_CELL_MAX_CNT: float = 81.0

# Results: percentile thresholds "derived from the entire sample of all times and
# locations". Fig. 2b sweeps X from the 95th to the 99.95th percentile.
QPE_PERCENTILES: tuple[float, ...] = (95.0, 99.0, 99.5, 99.9, 99.95)

# The headline threshold used for Figs. 2a and 3.
HEADLINE_PERCENTILE: float = 99.95
# Paper reports QPE99.95 ~= 5.1 mm/h for their 2019-2020 sample -- a sanity target,
# not a hard-coded cut (we always recompute the threshold from the data).
PAPER_QPE9995_MM_PER_H: float = 5.1

# --------------------------------------------------------------------------- #
# Gini / CDF construction
# --------------------------------------------------------------------------- #
# Methods > Gini and significance: "the normalised cumulative distribution
# function (CDF) is calculated for 100 equally sized bins, each of which contains
# 1% of the sample data."
N_BINS: int = 100

# Methods > Gini and significance: zero-valued predictors get "a small random
# perturbation added, of order +/-1e-10, to allow unambiguous sorting." This makes
# the CDF linear over the zero-predictor range and does not affect results.
TIEBREAK_EPS: float = 1e-10

# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #
# Methods > Gini and significance: bootstrap "resampling procedure is repeated 500
# times and the standard deviation of the 500 Gini coefficients is assumed to be
# the 1 sigma standard error."
BOOTSTRAP_REPS: int = 500




DIFF_SIGNIFICANCE_MULTIPLIER: float = 2.0 * math.sqrt(2.0)  # 2*sqrt(2)

# OLS trend: "twice the slope standard error was treated as significant at p<0.05."
TREND_SIGNIFICANCE_MULTIPLIER: float = 2.0

RANDOM_SEED: int = 20240813  # arbitrary; date-like for memorability


# =========================================================================== #
# Hypothesis-battery constants (pre-registered 2026-07-20; formerly
# hypothesis_tests/config.py -- every choice cited there: methods scan
# 2026-07-20 [Politis & White 2004; Wilks 2016; Guillod 2015; Tuttle &
# Salvucci 2016/17; Findell & Eltahir 2003] + repo data audit 2026-07-20)
# =========================================================================== #

# ---- Targets ---------------------------------------------------------------
#: Heavy-precip target percentile: P99.9 primary (enough events per stratum);
#: the paper's P99.95 headline runs as the sensitivity target.
HEAVY_PERCENTILE: float = 99.9
HEAVY_PERCENTILE_SENS: float = HEADLINE_PERCENTILE  # 99.95
#: "Any precip" target (A2 contrast): cell-mean QPE above this (mm/h).
ANY_PRECIP_MM: float = 0.1
#: Extreme sub-pixel target (S3): MRMS _max (audit: true sub-pixel max).
MAX_PERCENTILE: float = 99.9
#: Convective-character target (S3): _sk percentile among precipitating rows.
SK_PERCENTILE: float = 99.0
#: Convective-event filter default (MRMS Convection flag = strict CORES, so
#: shares run low even for heavy events; P99.9 mean share ~0.2 -> default well
#: below 0.5). Free parameter on AnalysisConfig.
CONVECTIVE_SHARE_MIN: float = 0.2
#: Post-rain screen: a cell counts as "already rained" above this (mm/h).
RAIN_SCREEN_MM: float = 0.1

# ---- Mark's gold-standard sample screens (email 2026-07-21; verbatim code in
# convection_skill.mark_screens, parity-tested) ------------------------------
#: Altitude screen: lowest parcel must stay below this (m) at ALL 7 hours.
ALT_MAX_M: float = 1000.0
#: Dry-start screen: grid-mean QPE at hours 0 AND 1 must be <= this (mm/h).
#: 100x stricter than RAIN_SCREEN_MM -- a hard "pre-convective initial state"
#: precondition, not a contamination trim.
DRY_START_MM: float = 0.001
SAMPLE_UNITS: tuple[str, ...] = ("slot", "cell_day")

# ---- Anomalies & timing (Tuttle & Salvucci endogeneity fix) ----------------
N_HARMONICS: int = 2          # annual + semiannual
ZSCORE_PER_CELL: bool = True  # standardized anomalies (GLACE/Guillod convention)
#: SM predictors use ONLY the pre-window L4 slots (16:30/19:30 UTC); the
#: 01:30/04:30 UTC "same-day" slots fall AFTER the 21-02 UTC window and would
#: leak post-rain soil moisture.
L4_PREWINDOW_SLOTS: tuple[int, ...] = (0, 1)
ANTECEDENT_LAGS: tuple[int, ...] = (1, 3, 7, 14, 30)
ANTE_PRECIP_WINDOW: tuple[int, int] = (1, 5)   # prior 1-5 day precip control
NEIGHBORHOOD_HALFWIDTH: int = 1                # Guillod 3x3-cell surroundings

# ---- Battery inference ------------------------------------------------------
#: Day-block bootstrap length (SM anomaly e-fold 2-3 d; 7 divides Mar-Nov
#: cleanly). Used when AnalysisConfig.inference == "block".
BLOCK_DAYS: int = 7
N_BOOT_REPS: int = 500
FDR_ALPHA: float = 0.10       # BH battery-wide; Wilks 2016 "2x" rule
N_CONTROL_BINS: int = 10      # conditional-Gini control bins
MIN_EVENTS_PER_BIN: int = 20
BATTERY_SEED: int = 20260720  # pre-registered battery seed (paper seed above)

# ---- Pre-registered strata --------------------------------------------------
EAST_WEST_SPLIT_LON: float = -95.0  # Tuttle-Salvucci sign flip
SEASONS: dict[str, tuple[int, ...]] = {
    "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),
}
EARLY_SLOTS: tuple[int, ...] = (1, 2, 3)   # 21-23 UTC
LATE_SLOTS: tuple[int, ...] = (4, 5, 6)    # 00-02 UTC

#: Stratifiers applied to EVERY hypothesis (on top of its own strata) so each
#: result carries the time/season/region breakdown. Override per run via
#: ``AnalysisConfig.default_strata``; the label -> variable definitions live in
#: :data:`convection_skill.hypotheses.STRATIFIERS`.
DEFAULT_STRATA: tuple[str, ...] = ("season", "eastwest", "slot_phase")

# ---- SMAP variable map (audit-corrected) ------------------------------------
#: Non-L4 SMAP_* fields are ALL-NaN in every year (audit item 6); FCST_N is
#: identically zero; ulay1 is wind SPEED; pflux is kg m-2 s-1.
SM_VAR = "SMAP_L4_smsfc_av"
SM_SD_VAR = "SMAP_L4_smsfc_sd"
SM_WEGRAD_VAR = "SMAP_L4_smsfc_wegrad"
SM_SNGRAD_VAR = "SMAP_L4_smsfc_sngrad"
SM_ABSGRAD_VAR = "SMAP_L4_smsfc_absgrad"
QLAY1_VAR = "SMAP_L4_qlay1_av"
TLAY1_VAR = "SMAP_L4_tlay1_av"
WIND_VAR = "SMAP_L4_ulay1_av"
PFLUX_VAR = "SMAP_L4_pflux_av"


# =========================================================================== #
# AnalysisConfig -- the one object that defines a comparable run
# =========================================================================== #
#: Table columns that must be finite for a row to count as "valid" for each
#: data product (the validity screen is compositional: pick any subset).
VALIDITY_COLUMNS: dict[str, tuple[str, ...]] = {
    "mrms": ("qpe",),                                    # reconstructed target
    "airs_fcst": ("mu_cape", "mu_el", "mu_lcl", "mu_cin"),  # paper QC set
    "airs": ("mu_cape_overpass",),                       # overpass retrieval
    "smap": ("sm_raw",),                                 # pre-window L4 SM
}

INFERENCE_METHODS: tuple[str, ...] = ("iid", "block")


@dataclass(frozen=True)
class AnalysisConfig:
    """Everything that defines one run of the suite.

    Fields fall into two groups, mirroring the two config files
    (``configs/data_table.yaml`` + ``configs/hypothesis_tests.yaml``; see
    :data:`DATA_TABLE_KEYS` / :data:`HYPOTHESIS_KEYS` and :meth:`from_files`):

    - DATA TABLE -- which rows exist: temporal/geographic scope and the
      product-validity rule.
    - HYPOTHESIS TESTS -- everything that governs the tests run on that
      table: event definitions (the QPE percentile the Gini events use),
      post-rain/convective row screens, which hypotheses run, what stratifies
      what, and the inference convention. ``inference`` picks the ONE
      uncertainty convention for every test in the run (decision 2026-07-22:
      never mix conventions within a run; default = the paper's iid row
      bootstrap, "block" = the day-block bootstrap that corrects its 2-3x
      overconfidence).

    Event thresholds always come from the BASE sample (all in-domain land
    rows -- the paper's "thresholds are based on all data"), never from the
    screened rows, so event definitions stay comparable across configs.
    """

    # ======================= DATA TABLE ======================================
    # ---- sample -------------------------------------------------------------
    years: tuple[int, ...] = ALL_YEARS
    months: tuple[int, int] = ANALYSIS_MONTHS
    lat_range: tuple[float, float] = DOMAIN_LAT
    lon_range: tuple[float, float] = DOMAIN_LON
    land_fraction_min: float = LAND_FRACTION_MIN   # 0.0 keeps ocean cells
    slots: tuple[int, ...] = FORECAST_SLOTS
    #: Products whose VALIDITY_COLUMNS must all be finite in a kept row.
    #: () keeps every row (the threshold-base superset).
    valid_datasets: tuple[str, ...] = ("airs_fcst", "mrms")
    #: Keep only cell-days whose critical variables (Mark's trio: MU_CAPE,
    #: MU_CIN, grid-mean QPE) are valid at ALL 7 hours -- Mark's completeness
    #: constraint, applied via the cached ``valid7`` column. ON by default
    #: (Mark's gold standard); the earlier audit note stands: it deletes a
    #: disproportionately wet slice of otherwise-valid rows.
    require_complete_days: bool = True

    # ---- Mark's sample screens (verbatim semantics; mark_screens.py) --------
    #: Drop cell-days where the lowest parcel exceeds alt_max_m at ANY of the
    #: 7 hours (NaN altitude fails, as in Mark's mask).
    screen_altitude: bool = True
    alt_max_m: float = ALT_MAX_M
    #: Drop cell-days that are not dry (grid-mean QPE > dry_start_mm) at
    #: hour 0 or hour 1 (NaN QPE fails, as in Mark's mask).
    screen_dry_start: bool = True
    dry_start_mm: float = DRY_START_MM
    #: "cell_day": Prepared also carries the Mark-style one-row-per-cell-day
    #: table (slot variables widened to <var>_h<slot>); "slot" leaves rows as
    #: individual (date, slot, cell) points only.
    sample_unit: str = "cell_day"

    # ==================== HYPOTHESIS TESTS ===================================
    # ---- event definitions (thresholds from the UNSCREENED base sample) ------
    #: THE QPE percentile the Gini "heavy" events use (battery default P99.9;
    #: the paper's headline is P99.95, kept as the sensitivity target).
    heavy_percentile: float = HEAVY_PERCENTILE
    heavy_sens_percentile: float = HEAVY_PERCENTILE_SENS
    any_precip_mm: float = ANY_PRECIP_MM     # "any precip" bar (mm/h)
    max_percentile: float = MAX_PERCENTILE   # sub-pixel _max target (S3)
    sk_percentile: float = SK_PERCENTILE     # _sk-among-wet target (S3)

    # ---- event screens -------------------------------------------------------
    screen_overpass_rain: bool = False   # drop cell-days with slot-0 rain
    screen_forecast_rain: bool = False   # drop slots after first in-window rain
    rain_screen_mm: float = RAIN_SCREEN_MM
    convective_min: Optional[float] = None      # e.g. CONVECTIVE_SHARE_MIN
    convective_col: str = "convective_share"

    # ---- which hypotheses & what stratifies what ------------------------------
    #: "all" | "topline" (the primary spec per hypothesis-table row) | an
    #: explicit tuple of registry ids.
    hypotheses: Union[str, tuple[str, ...]] = "all"
    #: Stratifiers applied to every hypothesis (names from
    #: hypotheses.STRATIFIERS: humidity, eastwest, aridity, wind, season,
    #: slot_phase).
    default_strata: tuple[str, ...] = DEFAULT_STRATA
    #: Per-hypothesis stratifier override, e.g. {"T1T2_sign": ("aridity",)}
    #: -- replaces that spec's own strata (default_strata still apply).
    strata: dict = field(default_factory=dict)
    #: Per-hypothesis conditional-Gini controls override, e.g.
    #: {"T1T2_sign": ("mu_cape",)} -- replaces that spec's own controls.
    controls: dict = field(default_factory=dict)
    run_controls: bool = True   # conditional Ginis within control bins
    run_strata: bool = True     # per-stratum Ginis
    #: Event-rate curves: True = every curve spec, False = none, or a list of
    #: ids to restrict further (e.g. ("A2_cin",)).
    run_curves: Union[bool, tuple[str, ...]] = True

    # ---- inference ------------------------------------------------------------
    inference: str = "iid"               # "iid" (paper) | "block"
    n_boot_reps: int = N_BOOT_REPS
    block_days: int = BLOCK_DAYS         # block length when inference="block"
    seed: int = BATTERY_SEED
    fdr_alpha: float = FDR_ALPHA         # BH-FDR across the whole run
    n_control_bins: int = N_CONTROL_BINS
    min_events_per_bin: int = MIN_EVENTS_PER_BIN

    #: Optional explicit run name; label() derives one otherwise.
    name: str = ""

    def __post_init__(self):
        if self.sample_unit not in SAMPLE_UNITS:
            raise ValueError(f"sample_unit must be one of {SAMPLE_UNITS}, "
                             f"got {self.sample_unit!r}")
        if self.inference not in INFERENCE_METHODS:
            raise ValueError(f"inference must be one of {INFERENCE_METHODS}, "
                             f"got {self.inference!r}")
        unknown = set(self.valid_datasets) - set(VALIDITY_COLUMNS)
        if unknown:
            raise ValueError(f"unknown valid_datasets {sorted(unknown)}; "
                             f"choose from {sorted(VALIDITY_COLUMNS)}")
        # normalize the test-selection / strata fields so YAML lists and
        # hand-written Python lists behave identically (frozen -> setattr)
        if isinstance(self.hypotheses, str):
            if self.hypotheses not in ("all", "topline"):
                raise ValueError("hypotheses must be 'all', 'topline', or a "
                                 f"list of registry ids, got {self.hypotheses!r}")
        else:
            object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "default_strata", tuple(self.default_strata))
        if not isinstance(self.run_curves, bool):
            object.__setattr__(self, "run_curves", tuple(self.run_curves))
        for f_name in ("strata", "controls"):
            mapping = getattr(self, f_name)
            if not isinstance(mapping, dict):
                raise ValueError(f"{f_name} must be a mapping of hypothesis id "
                                 f"-> list of names, got {mapping!r}")
            object.__setattr__(self, f_name,
                               {k: tuple(v) for k, v in mapping.items()})

    def label(self) -> str:
        """Compact run label for filenames and the results 'run' column."""
        if self.name:
            return self.name
        parts = [f"y{self.years[0]}-{self.years[-1]}" if len(self.years) > 1
                 else f"y{self.years[0]}"]
        parts.append("+".join(self.valid_datasets) or "norule")
        # Mark's screens are the defaults; tag the label when one is RELAXED
        if not self.require_complete_days:
            parts.append("nocompdays")
        if not self.screen_altitude:
            parts.append("noalt")
        if not self.screen_dry_start:
            parts.append("nodry")
        if self.sample_unit != "cell_day":
            parts.append("slotrows")
        if self.screen_overpass_rain or self.screen_forecast_rain:
            parts.append("scr" + ("O" if self.screen_overpass_rain else "")
                         + ("F" if self.screen_forecast_rain else ""))
        if self.convective_min is not None:
            parts.append(f"conv{self.convective_min:g}")
        if self.heavy_percentile != HEAVY_PERCENTILE:
            parts.append(f"q{self.heavy_percentile:g}")
        if self.hypotheses != "all":
            parts.append(self.hypotheses if isinstance(self.hypotheses, str)
                         else f"{len(self.hypotheses)}hyp")
        parts.append(self.inference)
        return "_".join(parts)

    def validity_columns(self) -> tuple[str, ...]:
        """The table columns the validity screen requires to be finite."""
        cols: list[str] = []
        for ds in self.valid_datasets:
            cols.extend(VALIDITY_COLUMNS[ds])
        return tuple(dict.fromkeys(cols))

    def wants_curve(self, spec_id: str) -> bool:
        """Whether this run produces an event-rate curve for a curve spec."""
        if isinstance(self.run_curves, bool):
            return self.run_curves
        return spec_id in self.run_curves

    # ---- construction from config files ----------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        """Build from a plain mapping (YAML/JSON-friendly).

        Lists become tuples (years, months, slots, ranges, valid_datasets).
        An optional ``preset`` key routes through the named-mode constructors
        (``all_products`` / ``all_products_complete_days`` / ``smap`` /
        ``paper``); a preset fixes ``valid_datasets``, so don't pass both.
        Unknown keys raise (typo guard).
        """
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in dict(d).items()}
        preset = d.pop("preset", None)
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown AnalysisConfig keys {sorted(unknown)}; "
                             f"choose from {sorted(known)} (+ optional 'preset')")
        if preset is not None:
            presets = ("all_products", "all_products_complete_days", "smap", "paper")
            if preset not in presets:
                raise ValueError(f"unknown preset {preset!r}; choose from {presets}")
            return getattr(cls, preset)(**d)
        return cls(**d)

    @classmethod
    def from_file(cls, path) -> "list[AnalysisConfig]":
        """Load config(s) from ONE combined YAML/JSON file; always a list.

        The file is either ONE mapping of AnalysisConfig fields, or a
        comparison spec: a ``runs:`` list of mappings, each merged over an
        optional shared ``defaults:`` mapping. For the two-file layout
        (data table + hypothesis tests) use :meth:`from_files`.
        """
        data = load_config_mapping(path)
        if "runs" in data:
            defaults = data.get("defaults", {})
            return [cls.from_dict({**defaults, **run}) for run in data["runs"]]
        return [cls.from_dict(data)]

    @classmethod
    def from_files(cls, data_path, hypothesis_path) -> "list[AnalysisConfig]":
        """Build run config(s) from the two-file layout; always returns a list.

        ``data_path`` defines the analysis table (DATA_TABLE_KEYS only:
        scope + validity); ``hypothesis_path`` defines the tests run on it
        (HYPOTHESIS_KEYS only: event thresholds, screens, hypotheses, strata,
        inference). A knob in the wrong file raises, naming where it belongs.
        The hypothesis file may be a comparison spec (``defaults:`` +
        ``runs:``); every run then shares the same data table.
        """
        data = load_config_mapping(data_path)
        _check_key_ownership(data, DATA_TABLE_KEYS | {"preset"},
                             "data-table", data_path)
        hyp = load_config_mapping(hypothesis_path)
        if "runs" in hyp:
            defaults = hyp.get("defaults", {})
            for m in [defaults, *hyp["runs"]]:
                _check_key_ownership(m, HYPOTHESIS_KEYS, "hypothesis",
                                     hypothesis_path)
            return [cls.from_dict({**data, **defaults, **run})
                    for run in hyp["runs"]]
        _check_key_ownership(hyp, HYPOTHESIS_KEYS, "hypothesis", hypothesis_path)
        return [cls.from_dict({**data, **hyp})]

    # ---- Zach's named validity modes (2026-07-22) -----------------------------
    @classmethod
    def all_products(cls, **kw) -> "AnalysisConfig":
        """Mode 1: every cell-hour valid for AIRS, AIRS-FCST, SMAP and MRMS."""
        kw.setdefault("require_complete_days", False)  # mode 2 is mode 1 + this
        return cls(valid_datasets=("airs", "airs_fcst", "smap", "mrms"), **kw)

    @classmethod
    def all_products_complete_days(cls, **kw) -> "AnalysisConfig":
        """Mode 2: mode 1, and only cell-days valid at every forecast hour."""
        return cls(valid_datasets=("airs", "airs_fcst", "smap", "mrms"),
                   require_complete_days=True, **kw)

    @classmethod
    def smap(cls, **kw) -> "AnalysisConfig":
        """Mode 3: SMAP-specific -- only SMAP and MRMS need be valid."""
        return cls(valid_datasets=("smap", "mrms"), **kw)

    @classmethod
    def paper(cls, **kw) -> "AnalysisConfig":
        """Richardson main-analysis sample (paper years, AIRS-FCST QC).

        Mark's screens are NOT part of the paper's stated sample rules, so the
        replication preset switches them off (and keeps the audit's
        complete-days finding: off).
        """
        kw.setdefault("years", PAPER_YEARS)
        kw.setdefault("require_complete_days", False)
        kw.setdefault("screen_altitude", False)
        kw.setdefault("screen_dry_start", False)
        return cls(valid_datasets=("airs_fcst", "mrms"), **kw)


# --------------------------------------------------------------------------- #
# The two-file config layout: which knob belongs in which file
# --------------------------------------------------------------------------- #
#: Fields that define WHICH ROWS EXIST (configs/data_table.yaml).
DATA_TABLE_KEYS: frozenset = frozenset({
    "years", "months", "lat_range", "lon_range", "land_fraction_min", "slots",
    "valid_datasets", "require_complete_days", "name",
    # Mark's sample screens + the sample-unit flag define WHICH ROWS EXIST
    "screen_altitude", "alt_max_m", "screen_dry_start", "dry_start_mm",
    "sample_unit",
})
#: Fields that govern THE TESTS run on that table (configs/hypothesis_tests.yaml):
#: everything else on AnalysisConfig, including the post-rain screens (an
#: analysis decision, not a property of the table).
HYPOTHESIS_KEYS: frozenset = (
    frozenset(AnalysisConfig.__dataclass_fields__) - DATA_TABLE_KEYS
) | {"name"}


def load_config_mapping(path) -> dict:
    """One YAML/JSON file -> its top-level mapping (shared loader)."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"config file must be .yaml/.yml/.json, got {path.name}")
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a mapping at top level")
    return data


def _check_key_ownership(mapping: dict, allowed: frozenset, kind: str, path):
    """Raise if a mapping holds keys that belong in the OTHER config file."""
    misplaced = set(mapping) - allowed
    if not misplaced:
        return
    other = "hypothesis" if kind == "data-table" else "data-table"
    other_keys = HYPOTHESIS_KEYS if kind == "data-table" else DATA_TABLE_KEYS
    belongs_elsewhere = sorted(misplaced & other_keys)
    unknown = sorted(misplaced - other_keys)
    problems = []
    if belongs_elsewhere:
        problems.append(f"{belongs_elsewhere} belong in the {other} config")
    if unknown:
        problems.append(f"{unknown} are not AnalysisConfig fields")
    raise ValueError(f"{Path(path).name} is the {kind} config: "
                     + "; ".join(problems))
