"""The hypothesis registry: what gets tested, declaratively.

A :class:`HypothesisSpec` is the "configuration of variables" for one
hypothesis: predictor column, target flag, expected sign, conditional-Gini
controls, strata, and -- per the user's mandate -- a ``kernel_extension`` note
recording how the HYSPLIT trajectory kernels (``trajectory_kernels/``) upgrade
the local-only version. Adding a hypothesis is adding a spec; the runner
(:mod:`convection_skill.suite`) executes every spec identically over the
prepared dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HypothesisSpec:
    id: str
    title: str
    predictor: str                 # column in the analysis table
    target: str = "heavy"          # key into the flag columns built by the runner
    expected: str = "+"            # "+", "-", "regime", "shape", "n/a"
    sample: str = "slot"           # "slot" | "onset"
    controls: tuple = ()           # conditional-Gini control columns
    strata: tuple = ()             # stratifier names (STRATIFIERS keys)
    curve: bool = False            # also produce an event-rate curve
    #: "" = test the raw predictor; "quadratic" | "binned" = test its
    #: cross-fitted rate-ordered index instead (inversion.crossfit_index) --
    #: the apples-to-apples Gini for non-monotone predictors (A2/A4).
    invert: str = ""
    testability: str = "full"      # "full" | "partial" | "untestable"
    #: "topline" = THE primary spec for one row of Zach's hypothesis table;
    #: "secondary" = a variant/robustness spec (inversions, extra lags, sign
    #: contrasts). ``AnalysisConfig(hypotheses="topline")`` selects the former.
    tier: str = "secondary"
    kernel_extension: str = ""     # how trajectory kernels extend this test
    notes: str = ""


# --------------------------------------------------------------------------- #
# Stratifiers (pre-registered in config; tercile cutpoints from the sample)
# --------------------------------------------------------------------------- #
def _terciles(x: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    ranks = x.rank(pct=True)
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    out[ranks <= 1 / 3] = labels[0]
    out[(ranks > 1 / 3) & (ranks <= 2 / 3)] = labels[1]
    out[ranks > 2 / 3] = labels[2]
    return out


def _front_env(df: pd.DataFrame) -> pd.Series:
    """Frontal vs non-frontal environment (CODSUS 3-wide any-front flag).

    NaN flag (years without front files, 2019+) -> NA, so those rows drop out
    of the stratum comparison instead of masquerading as non-frontal.
    """
    x = df["front_any_3w"]
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[x > 0] = "frontal"
    out[x == 0] = "nonfrontal"
    return out


STRATIFIERS = {
    "humidity": lambda df: _terciles(df["fcst_q"], ("q_dry", "q_mid", "q_moist")),
    "front": _front_env,
    "eastwest": lambda df: np.where(df["is_east"], "east", "west"),
    "aridity": lambda df: _terciles(df["sm_cell_clim"], ("arid", "mid", "humid_clim")),
    "wind": lambda df: _terciles(df["wind"], ("calm", "breezy", "windy")),
    "season": lambda df: df["season"],
    "slot_phase": lambda df: np.where(df["is_late_slot"], "late_00-02Z", "early_21-23Z"),
}

#: Every runnable hypothesis is additionally stratified by season, region and
#: time-of-window (unless the run's config overrides ``default_strata``).
#: Defined next to the other pre-registered strata constants; re-exported here
#: because this is where the stratifier definitions live.
DEFAULT_STRATA: tuple[str, ...] = config.DEFAULT_STRATA


# --------------------------------------------------------------------------- #
# Registry (T = thermodynamic SM, S = spatial SM, A = atmospheric)
# --------------------------------------------------------------------------- #
_KE_SM = ("Replace local SM with kernel-weighted UPSTREAM SM "
          "(apply_kernel(kernels, smap)); receptor-band knob already built.")

REGISTRY: tuple[HypothesisSpec, ...] = (
    # ---- A: baselines first (everything else is an increment over these) ----
    HypothesisSpec("A1_mu", "Baseline: advected MU CAPE predicts heavy precip",
                   "mu_cape", expected="+", tier="topline",
                   kernel_extension="Already the trajectory product (Richardson pipeline).",
                   notes="Replicates the paper's headline; sanity anchor for the battery."),
    HypothesisSpec("A1_mml", "Baseline: MML CAPE", "mml_cape", expected="+",
                   kernel_extension="As A1_mu."),
    HypothesisSpec("A2_cin", "CIN non-monotonic: moderate cap aids extremes",
                   "mu_cin", target="heavy", expected="shape", curve=True,
                   controls=("mu_cape",), tier="topline",
                   kernel_extension="CIN history along the inflow path (cap erosion en route)."),
    HypothesisSpec("A2_cin_any", "CIN vs ANY precip (sign contrast with A2_cin)",
                   "mu_cin", target="any", expected="+", curve=True,
                   kernel_extension="As A2_cin."),
    # Rate-ordered inversions: the Gini is a rank statistic, so A2/A4's
    # non-monotone responses score ~0 raw. Testing the cross-fitted index
    # eta(x) = P(event|x) recovers the FULL ordering skill on the same Gini
    # scale as the monotone predictors (see inversion.py for the math).
    HypothesisSpec("A2_cin_invq", "CIN rate-ordered index (theoretical: quadratic logit)",
                   "mu_cin", expected="+", invert="quadratic",
                   controls=("mu_cape",),
                   kernel_extension="As A2_cin.",
                   notes="eta(CIN) via quadratic-logistic MLE (exact Bayes index "
                         "under unequal-variance Gaussian class-conditionals), "
                         "cross-fitted over contiguous day blocks."),
    HypothesisSpec("A2_cin_invb", "CIN rate-ordered index (numerical: binned rates)",
                   "mu_cin", expected="+", invert="binned",
                   controls=("mu_cape",),
                   kernel_extension="As A2_cin.",
                   notes="eta(CIN) via out-of-fold empirical event-rate curve."),
    HypothesisSpec("A4_q_invq", "Humidity rate-ordered index (theoretical: quadratic logit)",
                   "fcst_q", expected="+", invert="quadratic",
                   controls=("mu_cape",), testability="partial",
                   kernel_extension="As A4_q.",
                   notes="As A2_cin_invq, for the parcel-humidity response."),
    HypothesisSpec("A4_q_invb", "Humidity rate-ordered index (numerical: binned rates)",
                   "fcst_q", expected="+", invert="binned",
                   controls=("mu_cape",), testability="partial",
                   kernel_extension="As A4_q.",
                   notes="As A2_cin_invb, for the parcel-humidity response."),
    HypothesisSpec("A3_elev", "Elevated instability (MU-MML) drives late-window precip",
                   "mu_minus_mml", expected="+", strata=("slot_phase",),
                   tier="topline",
                   kernel_extension="Elevated receptor band: rebuild kernels with "
                                    "RECEPTOR_BAND_M raised to the residual layer."),
    HypothesisSpec("A4_q", "Column-humidity veto (parcel q as proxy)",
                   "fcst_q", expected="+", controls=("mu_cape",),
                   testability="partial", tier="topline",
                   kernel_extension="Kernel-weighted upstream humidity; q is conserved "
                                    "along trajectories so upstream q IS arriving q.",
                   notes="Proper test needs 850-600 hPa RH profile (flagged). "
                         "Curve retired 2026-07-23: the response is monotone "
                         "(rate curve added nothing beyond the Gini)."),
    HypothesisSpec("A5_hcf", "Heated condensation framework", "n/a",
                   expected="n/a", testability="untestable", tier="topline",
                   kernel_extension="PBL model is pluggable in kernels when reanalysis "
                                    "PBL arrives (ReanalysisPBL stub).",
                   notes="Needs T,q profile + PBL depth (table flag confirmed)."),
    HypothesisSpec("A6_recycle", "Upwind SM raises locally recycled moisture share",
                   "n/a", expected="n/a", testability="untestable", tier="topline",
                   kernel_extension="THE flagship kernel application: discount-weighted "
                                    "(rain-out) upstream-SM kernels = recycling proxy; "
                                    "all pieces exist (apply_kernel + discount.py).",
                   notes="Untestable locally by construction."),

    # ---- T: point-value SM ----
    HypothesisSpec("T1T2_sign", "Wet-soil vs dry-soil advantage (sign by regime)",
                   "sm_anom", expected="regime", tier="topline",
                   controls=("mu_cape", "pflux_ante", "pflux_prewindow"),
                   strata=("eastwest", "aridity", "humidity", "season", "slot_phase"),
                   kernel_extension=_KE_SM,
                   notes="Sign test: T1 predicts +, T2 predicts -, by stratum. "
                         "pflux_ante = Tuttle&Salvucci prior-day guard; "
                         "pflux_prewindow = same-morning-rain guard (this one "
                         "collapses the naive +0.31 to ~-0.04: decisive). "
                         "slot_phase stratum doubles as the A3 falsification "
                         "(surface decoupling should weaken SM skill late)."),
    HypothesisSpec("T3_gate", "Humidity gates the T1/T2 sign (CTP-HI proxy)",
                   "sm_anom", expected="regime", strata=("humidity",),
                   testability="partial", tier="topline",
                   kernel_extension="Gate on kernel-weighted upstream humidity instead "
                                    "of local; CTP/HI_low need profiles (flagged).",
                   notes="Findell-Eltahir box approximated by q terciles."),
    HypothesisSpec("T4_lag1", "Antecedent SM (1 d)", "sm_anom_lag1", expected="regime",
                   controls=("pflux_ante",), tier="topline",
                   kernel_extension=_KE_SM),
    HypothesisSpec("T4_lag7", "Antecedent SM (7 d)", "sm_anom_lag7", expected="regime",
                   controls=("pflux_ante",), kernel_extension=_KE_SM),
    HypothesisSpec("T4_lag30", "Antecedent SM (30 d)", "sm_anom_lag30", expected="regime",
                   controls=("pflux_ante",), kernel_extension=_KE_SM),
    HypothesisSpec("T4_cin", "Dry antecedent soil builds inhibition (mediation)",
                   "sm_anom_lag7", target="inhibited", expected="-",
                   kernel_extension="Antecedent SM along the inflow path (lagged kernels).",
                   notes="Target = strongly capped rows (MU_CIN below pooled P10). "
                         "Negative Gini = dry soil (low SM) -> strong cap."),
    HypothesisSpec("T5_cape", "SM -> mixed-layer CAPE (competition, moisture side)",
                   "sm_anom", target="high_mml_cape", expected="+", tier="topline",
                   kernel_extension="Kernel-weighted SM -> adjust ADVECTED CAPE "
                                    "(full-column RECEPTOR_BAND mode; separate feature)."),
    HypothesisSpec("T5_lcl", "SM -> low LCL (competition, moisture side)",
                   "sm_anom", target="low_mml_lcl", expected="+",
                   testability="partial", tier="topline",
                   kernel_extension="As T5_cape.",
                   notes="Full lapse-rate decomposition needs the T,q profile."),

    # ---- S: spatial heterogeneity ----
    HypothesisSpec("S1_sd", "Sub-grid SM heterogeneity favors CI",
                   "smsd_anom", expected="+", tier="topline",
                   controls=("sm_anom", "mu_cape"), strata=("season",),
                   kernel_extension="Heterogeneity sampled along the inflow footprint."),
    HypothesisSpec("S1_grad", "Meso SM gradient magnitude favors CI",
                   "absgrad_anom", expected="+",
                   controls=("sm_anom", "mu_cape"),
                   kernel_extension="Gradient at the kernel's source cells (upwind), "
                                    "not under the storm."),
    HypothesisSpec("S2_wind", "Wind gates the heterogeneity effect",
                   "absgrad_anom", expected="regime", strata=("wind",),
                   testability="partial", tier="topline",
                   kernel_extension="The gate becomes trajectory SPEED itself -- "
                                    "read directly off parcel displacement.",
                   notes="Audit: ulay1 is a wind-speed magnitude (usable as the gate); "
                         "no vector -> no directional test."),
    HypothesisSpec("S3_we_max", "Signed W-E gradient vs sub-pixel extremes",
                   "wegrad_anom", target="max_extreme", expected="regime",
                   testability="partial", tier="topline",
                   kernel_extension="Alignment term grad(SM) . (inflow direction) from "
                                    "trajectories replaces the missing shear vector.",
                   notes="Directional (grad.shear) mediator needs ERA5 S06 (flagged)."),
    HypothesisSpec("S3_sn_max", "Signed S-N gradient vs sub-pixel extremes",
                   "sngrad_anom", target="max_extreme", expected="regime",
                   testability="partial", kernel_extension="As S3_we_max."),
    HypothesisSpec("S3_we_sk", "Signed W-E gradient vs convective character (_sk)",
                   "wegrad_anom", target="sk_high", expected="regime",
                   testability="partial", kernel_extension="As S3_we_max."),
    HypothesisSpec("S4_local", "Locally-dry patches favored (spatial signal)",
                   "sm_local", expected="-", controls=("mu_cape",), tier="topline",
                   kernel_extension="Local anomaly at the kernel source cells."),
    HypothesisSpec("S4_nonlocal", "Regionally-wet days favored (temporal signal)",
                   "sm_nonlocal", expected="+", controls=("mu_cape",), tier="topline",
                   kernel_extension="Regional wetness along the whole inflow corridor."),
    HypothesisSpec("S5_onset", "Drier soils shift onset earlier",
                   "sm_anom", target="early_onset", sample="onset", expected="-",
                   tier="topline",
                   kernel_extension="Onset hour vs the TIME the air last had surface "
                                    "contact (kernel temporal marginal)."),

    # ---- F: synoptic surface fronts (CODSUS analyst fronts; fronts.py) ----
    # Concurrent-with-window flags, 2016-2018 only (files end 2018; NaN rows
    # for later years drop out of the Gini automatically). The mu_cape control
    # asks the interesting question: does knowing a front is present add skill
    # BEYOND the instability the front itself creates?
    HypothesisSpec("F1_any", "Surface front in/near cell favors heavy precip",
                   "front_any_3w", expected="+", controls=("mu_cape",),
                   kernel_extension="Front position along the inflow trajectory "
                                    "(was the parcel lifted en route?).",
                   notes="3-wide (near-front neighborhood) mask; binary "
                         "predictor, Gini = scaled event-rate contrast."),
    HypothesisSpec("F1_any_1w", "As F1_any, strict 1-cell front line",
                   "front_any_1w", expected="+", controls=("mu_cape",),
                   kernel_extension="As F1_any."),
    HypothesisSpec("F2_cold", "Cold front in/near cell favors heavy precip",
                   "front_cold_3w", expected="+", controls=("mu_cape",),
                   kernel_extension="As F1_any."),
    HypothesisSpec("F3_stationary", "Stationary front (training rain) favors heavy precip",
                   "front_stationary_3w", expected="+", controls=("mu_cape",),
                   kernel_extension="As F1_any."),
    HypothesisSpec("F4_warm", "Warm front in/near cell vs heavy precip",
                   "front_warm_3w", expected="+", controls=("mu_cape",),
                   kernel_extension="As F1_any."),
    HypothesisSpec("F5_occluded", "Occluded front in/near cell vs heavy precip",
                   "front_occluded_3w", expected="+", controls=("mu_cape",),
                   kernel_extension="As F1_any."),
)


def select_specs(selection="all",
                 registry: tuple[HypothesisSpec, ...] = REGISTRY,
                 ) -> tuple[HypothesisSpec, ...]:
    """Resolve ``AnalysisConfig.hypotheses`` to registry specs.

    ``"all"`` -> the full registry; ``"topline"`` -> the primary spec per
    hypothesis-table row (``tier == "topline"``); an iterable of ids -> those
    specs in the given order (unknown ids raise, listing the valid ones).
    """
    if selection == "all":
        return registry
    if selection == "topline":
        return tuple(s for s in registry if s.tier == "topline")
    by_id = {s.id: s for s in registry}
    unknown = [i for i in selection if i not in by_id]
    if unknown:
        raise ValueError(f"unknown hypothesis ids {unknown}; "
                         f"choose from {sorted(by_id)}")
    return tuple(by_id[i] for i in selection)
