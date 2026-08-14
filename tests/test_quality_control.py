"""Tests for the paper's QC rules on a tiny hand-built table.

Each test isolates one rule so a reader can see exactly what it removes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from convection_skill import quality_control as qc
from convection_skill import config


def _cell_day(year, date, lat, lon, land_frac, capes, cin=0.0, el=1000.0, lcl=500.0):
    """Build the six forecast-hour rows for one (year, date, lat, lon) cell.

    ``capes`` is a length-6 list of MU_CAPE values (NaN allowed) for slots 1-6.
    """
    rows = []
    for slot, hour, cape in zip(config.FORECAST_SLOTS, config.FORECAST_HOURS_UTC, capes):
        rows.append(dict(
            year=year, date=pd.Timestamp(date), month=pd.Timestamp(date).month,
            slot=slot, hour_utc=hour, lat=lat, lon=lon, land_frac=land_frac,
            mu_cape=cape, mu_cin=cin, mu_el=el, mu_lcl=lcl,
            mml_cape=cape, qpe=0.0, mu_cape_overpass=cape,
        ))
    return rows


def test_restrict_domain_drops_south_of_32N():
    rows = _cell_day(2019, "2019-06-01", 30.5, -95.5, 1.0, [100] * 6)  # too far south
    rows += _cell_day(2019, "2019-06-01", 40.5, -95.5, 1.0, [100] * 6)  # in domain
    table = pd.DataFrame(rows)
    out = qc.restrict_domain(table)
    assert set(out["lat"]) == {40.5}


def test_require_land_drops_ocean():
    rows = _cell_day(2019, "2019-06-01", 40.5, -95.5, 0.2, [100] * 6)  # ocean-ish
    rows += _cell_day(2019, "2019-06-01", 41.5, -95.5, 0.8, [100] * 6)  # land
    table = pd.DataFrame(rows)
    out = qc.require_land(table)
    assert set(out["lat"]) == {41.5}


def test_require_valid_indices_keeps_zeros_but_drops_nan():
    rows = _cell_day(2019, "2019-06-01", 40.5, -95.5, 1.0, [0.0] * 6)  # zero CAPE = valid
    table = pd.DataFrame(rows)
    assert len(qc.require_valid_indices(table)) == 6  # zeros kept

    rows2 = _cell_day(2019, "2019-06-01", 41.5, -95.5, 1.0, [100] * 6)
    df2 = pd.DataFrame(rows2)
    df2.loc[2, "mu_el"] = np.nan  # one invalid companion index
    assert len(qc.require_valid_indices(df2)) == 5


def test_require_all_timesteps_valid_drops_incomplete_cell_days():
    """A cell-day missing any forecast hour is dropped entirely."""
    complete = _cell_day(2019, "2019-06-01", 40.5, -95.5, 1.0, [100] * 6)
    incomplete = _cell_day(2019, "2019-06-01", 41.5, -95.5, 1.0, [100, 100, np.nan, 100, 100, 100])
    table = pd.DataFrame(complete + incomplete)
    # emulate the pipeline: valid-index filter first, then all-timesteps rule
    table = qc.require_valid_indices(table)
    out = qc.require_all_timesteps_valid(table, n_slots=len(config.FORECAST_SLOTS))
    assert set(out["lat"]) == {40.5}
    assert len(out) == 6


def test_apply_paper_qc_report_counts_are_monotonic():
    """The QC report should be non-increasing at each step."""
    good = _cell_day(2019, "2019-06-01", 40.5, -95.5, 1.0, [100] * 6)
    ocean = _cell_day(2019, "2019-06-01", 41.5, -95.5, 0.1, [100] * 6)
    south = _cell_day(2019, "2019-06-01", 28.5, -95.5, 1.0, [100] * 6)
    table = pd.DataFrame(good + ocean + south)
    out, report = qc.apply_paper_qc(table, return_report=True)
    counts = [n for _, n in report.steps]
    assert counts == sorted(counts, reverse=True)
    assert len(out) == 6  # only the good cell-day survives
