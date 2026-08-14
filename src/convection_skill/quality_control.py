"""The paper's sample-selection rules, one function per rule.

Every rule below is a direct transcription of a sentence from Richardson et al.
(2024), Methods > "Calculation of thermodynamic indices and selection of data for
analysis". Keeping each rule as its own named, testable function means the QC can
be read side-by-side with the paper and each rule can be toggled independently
when extending the analysis.

The rules are applied by :func:`apply_paper_qc`, which also returns a small report
of how many rows each step removed -- useful for confirming the sample size
against the paper (">160k per forecast hour").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

# Columns that uniquely identify a grid cell on a given day.
CELL_KEYS = ["year", "date", "lat", "lon"]


def restrict_domain(table: pd.DataFrame) -> pd.DataFrame:
    """Keep 32-53N, 107-64W.

    Methods: "the latitude-longitude range is limited to 32-53N, 107-64W" and the
    "main analysis was then restricted to north of 32N based on poorer performance
    near the Gulf of Mexico."
    """
    lat_lo, lat_hi = config.DOMAIN_LAT
    lon_lo, lon_hi = config.DOMAIN_LON
    keep = (
        table["lat"].between(lat_lo, lat_hi)
        & table["lon"].between(lon_lo, lon_hi)
    )
    return table[keep]


def require_land(table: pd.DataFrame) -> pd.DataFrame:
    """Keep cells with land fraction >= 50%.

    Methods: "grid cells are excluded if their land fraction is below 50%".
    """
    return table[table["land_frac"] >= config.LAND_FRACTION_MIN]


def require_valid_indices(table: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where MU_CAPE, MU_EL, MU_LCL and MU_CIN are all valid.

    Methods: cells are included only if "MU_CAPE, MU_EL, MU_LCL and MU_CIN have
    valid values calculated by SHARPpy (including 0)". Here "valid" = finite;
    zeros are kept (they are physically meaningful, e.g. zero CAPE / CIN).
    """
    finite = np.ones(len(table), dtype=bool)
    for col in ("mu_cape", "mu_el", "mu_lcl", "mu_cin"):
        finite &= np.isfinite(table[col].to_numpy())
    return table[finite]


def require_all_timesteps_valid(
    table: pd.DataFrame, n_slots: int
) -> pd.DataFrame:
    """Keep only cell-days that survive at *every* forecast timestep.

    Methods: "by requiring valid indices across all timesteps, the geographic
    sample is the same for all forecast hours and all panels of Fig. 3 contain
    consistent datasets." Operationally: a (year, date, lat, lon) is kept only if
    it retains all ``n_slots`` forecast rows after the per-row filters above.

    .. warning:: OFF by default in :func:`apply_paper_qc` (2026-07). In the
       paper's own data this rule was evidently near-non-binding, but in our
       regenerated files 23% of valid rows sit in partial cell-days, and those
       rows are systematically wetter (wet fraction 0.32 vs 0.24; 2.2x the
       QPE>5.1 mm/h event rate). Applying the rule here therefore deletes the
       "punishing" cases and inflates every skill score: POD@CAPE90 0.87 vs the
       paper's 0.80, Gini +0.03-0.09 across Figs. 2b/3. Without it, POD and the
       Fig. 2b curves land within figure-reading tolerance of the paper (see
       tests/test_paper_benchmarks.py). The cost is the paper's same-sample-
       across-hours property, which our files cannot deliver anyway.
    """
    counts = table.groupby(CELL_KEYS)["slot"].transform("count")
    return table[counts == n_slots]


def threshold_base(table: pd.DataFrame) -> pd.DataFrame:
    """The sample the pooled QPE-percentile thresholds are derived from.

    Methods: "In all cases, thresholds are based on ALL data, including all
    locations, all seasons, and both wet and dry hours" -- i.e. every in-domain
    land row, *including* rows without valid AIRS retrievals (data screening
    applies to the skill evaluation, not the event definition). Build the input
    with ``dataset.build_base_table`` (the unscreened superset).

    Empirically this reading is decisive: with thresholds from this base and
    skill evaluated on each predictor's valid rows, our Fig. 2b curves match the
    paper's to within ~0.02 everywhere (vs +0.03-0.09 with same-sample
    thresholds).
    """
    return require_land(restrict_domain(table))


# NOTE: the paper also requires ">20 AIRS-FCST parcels within the profile"
# (config.MIN_PARCELS via FCST_N). In this clean data cut FCST_N is 0 wherever
# MU_CAPE is valid (see notebooks/01_data_audit, Q2), which indicates the
# parcel-count screen was already applied upstream. We therefore do not re-apply
# it here; if a usable count is confirmed, add a require_min_parcels() rule.


@dataclass
class QCReport:
    """Row counts after each QC step, for comparison against the paper."""

    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, name: str, table: pd.DataFrame) -> None:
        self.steps.append((name, len(table)))

    def __str__(self) -> str:
        lines = ["QC step".ljust(28) + "rows remaining"]
        for name, n in self.steps:
            lines.append(f"{name.ljust(28)}{n:,}")
        return "\n".join(lines)


def apply_paper_qc(
    table: pd.DataFrame,
    n_slots: int = len(config.FORECAST_SLOTS),
    require_all_timesteps: bool = False,
    return_report: bool = False,
):
    """Apply every selection rule, in the paper's order.

    Parameters
    ----------
    table
        A tidy row table from :func:`convection_skill.dataset.build_base_table`.
    n_slots
        Expected number of forecast slots per cell-day (default 6).
    require_all_timesteps
        Whether to keep only complete cell-days. Default False: although the
        paper states this rule, on our regenerated files it deletes the wettest
        23% of valid rows and demonstrably breaks agreement with every paper
        statistic -- see :func:`require_all_timesteps_valid` and
        tests/test_paper_benchmarks.py. Set True to reproduce the stated rule.
    return_report
        If True, also return a :class:`QCReport` of per-step row counts.

    Returns
    -------
    pandas.DataFrame  (and optionally QCReport)
        The QC-passing analysis sample.
    """
    report = QCReport()
    report.add("start", table)

    table = restrict_domain(table)
    report.add("domain 32-53N,107-64W", table)

    table = require_land(table)
    report.add("land fraction >= 50%", table)

    table = require_valid_indices(table)
    report.add("valid MU indices", table)

    if require_all_timesteps:
        table = require_all_timesteps_valid(table, n_slots)
        report.add("all timesteps valid", table)

    table = table.reset_index(drop=True)
    if return_report:
        return table, report
    return table


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from . import dataset
    from .config import AnalysisConfig

    tbl = dataset.build_base_table(AnalysisConfig(years=(2019,)))
    qc, rep = apply_paper_qc(tbl, return_report=True)
    print(rep)
    per_hour = qc.groupby("hour_utc").size()
    print("\nrows per forecast hour:")
    print(per_hour)
