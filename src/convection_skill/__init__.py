"""convection_skill: replication of the Richardson et al. (2024) Gini-coefficient
analysis linking AIRS-FCST CAPE to heavy hourly MRMS precipitation.

Reference
---------
Richardson, M. T., B. H. Kahn, and P. M. Kalmus (2024).
"Mesoscale air motion and thermodynamics predict heavy hourly U.S. precipitation."
Communications Earth & Environment 5:472. doi:10.1038/s43247-024-01614-1

Design goal
-----------
Every function here is written to be *maximally interpretable* so this code can
serve as the foundation for extending the analysis to SMAP soil-moisture
predictors. The statistical core (:mod:`convection_skill.gini`) is deliberately
**predictor-agnostic**: it never mentions CAPE. Swapping in a SMAP field requires
changing only which column of the analysis table is passed in -- no changes to the
statistics code.

As of 2026-07-22 this package is ALSO the unified hypothesis suite (the former
``hypothesis_tests`` package merged in): one config-driven dataset preparation
(``dataset``) feeds both the paper replication and the hypothesis battery, and
one runner (``suite``) executes every registered hypothesis identically.

Module map
----------
- ``config``          : every cited constant + ``AnalysisConfig`` (the one
                        object defining a comparable run: sample validity,
                        screens, inference convention).
- ``data_loading``    : NetCDF files -> raw/uniform xarray stages.
- ``dataset``         : THE row-table builder (cached base superset,
                        base-sample thresholds, config-driven screens, flags).
- ``quality_control`` : the paper's sample-selection rules, one function per rule.
- ``gini``            : predictor-agnostic Gini / detection-CDF machinery.
- ``significance``    : the paper's bootstrap SE and hourly-trend tests.
- ``stats``           : suite inference -- iid AND day-block bootstrap Gini,
                        conditional Gini, event-rate curves, BH-FDR.
- ``predictors``      : anomaly engineering (deseasonalize, lags, Guillod split).
- ``hypotheses``      : the declarative HypothesisSpec registry + stratifiers.
- ``suite``           : ``test_hypothesis`` + ``run_suite`` (comparable results).
- ``analysis``        : paper-figure computations (Fig. 2b/3), table-agnostic.
- ``report``          : suite report + forest/curve figures.
- ``plotting``        : figure builders mirroring the paper's Figs. 2 and 3.
"""

__all__ = [
    "config",
    "data_loading",
    "dataset",
    "quality_control",
    "gini",
    "significance",
    "stats",
    "predictors",
    "hypotheses",
    "suite",
    "analysis",
    "report",
    "plotting",
]
