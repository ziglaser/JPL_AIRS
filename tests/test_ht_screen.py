"""Tests for the post-rain screen (table.rain_screen_mask + run_battery wiring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from convection_skill import config
from convection_skill.dataset import build_onset_table, rain_screen_mask


def _screen_table():
    """Three cell-days x slots 1-6, hand-set rain histories.

    - cell A (lat 40.5, lon -95.5): rains at slot 3 and 5, dry overpass
    - cell B (lat 40.5, lon -94.5): fully dry window, WET overpass
    - cell C (lat 41.5, lon -95.5): rains at slot 1, NaN overpass
    """
    day = np.datetime64("2019-06-05")
    rows = []
    for lat, lon, qpe_by_slot, qpe_ovp in [
        (40.5, -95.5, {3: 1.0, 5: 2.0}, 0.0),
        (40.5, -94.5, {}, 3.0),
        (41.5, -95.5, {1: 0.5}, np.nan),
    ]:
        for slot in range(1, 7):
            rows.append({"day": day, "lat": lat, "lon": lon, "slot": slot,
                         "qpe": qpe_by_slot.get(slot, 0.0),
                         "qpe_overpass": qpe_ovp})
    return pd.DataFrame(rows)


def _kept_slots(df, keep, lat, lon):
    at_cell = (df["lat"] == lat) & (df["lon"] == lon)
    return sorted(df.loc[keep & at_cell.to_numpy(), "slot"])


def test_forecast_screen_drops_strictly_after_first_rain():
    df = _screen_table()
    keep = rain_screen_mask(df, overpass=False, forecast=True)
    # cell A: first rain slot 3 kept, 4-6 dropped (incl. the later wet slot 5)
    assert _kept_slots(df, keep, 40.5, -95.5) == [1, 2, 3]
    # cell B: dry window -> everything kept (overpass rain ignored here)
    assert _kept_slots(df, keep, 40.5, -94.5) == [1, 2, 3, 4, 5, 6]
    # cell C: rains at slot 1 -> only slot 1 survives
    assert _kept_slots(df, keep, 41.5, -95.5) == [1]


def test_overpass_screen_drops_whole_cell_day_and_keeps_nan():
    df = _screen_table()
    keep = rain_screen_mask(df, overpass=True, forecast=False)
    assert _kept_slots(df, keep, 40.5, -95.5) == [1, 2, 3, 4, 5, 6]  # dry ovp
    assert _kept_slots(df, keep, 40.5, -94.5) == []                  # wet ovp
    assert _kept_slots(df, keep, 41.5, -95.5) == [1, 2, 3, 4, 5, 6]  # NaN kept


def test_combined_screen_is_the_intersection():
    df = _screen_table()
    both = rain_screen_mask(df, overpass=True, forecast=True)
    ovp = rain_screen_mask(df, overpass=True, forecast=False)
    fcst = rain_screen_mask(df, overpass=False, forecast=True)
    assert np.array_equal(both, ovp & fcst)


def test_rain_mm_threshold_is_strict_and_respected():
    df = _screen_table()
    # raise the bar above every window value -> forecast screen keeps all
    keep = rain_screen_mask(df, overpass=False, forecast=True, rain_mm=5.0)
    assert keep.all()
    # exactly at the overpass value -> strict '>' keeps the cell-day
    keep = rain_screen_mask(df, overpass=True, forecast=False, rain_mm=3.0)
    assert _kept_slots(df, keep, 40.5, -94.5) == [1, 2, 3, 4, 5, 6]


def test_screens_are_independent_of_row_order():
    df = _screen_table()
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    keep = rain_screen_mask(shuffled, overpass=True, forecast=True)
    assert _kept_slots(shuffled, keep, 40.5, -95.5) == [1, 2, 3]


def test_onset_rows_survive_the_forecast_screen():
    """The onset sample IS the first-rain rows: the forecast screen keeps them."""
    df = _screen_table()
    keep = rain_screen_mask(df, overpass=False, forecast=True,
                            rain_mm=config.ANY_PRECIP_MM)
    onset = build_onset_table(df, {"any": config.ANY_PRECIP_MM})
    kept = df[keep]
    for _, row in onset.iterrows():
        match = ((kept["lat"] == row["lat"]) & (kept["lon"] == row["lon"])
                 & (kept["slot"] == row["onset_slot"]))
        assert match.any()
