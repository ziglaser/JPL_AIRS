"""Tests for the loader's non-trivial temporal logic.

Covers the three ``smap_time_policy`` mappings that place each SMAP L4 field on the
paper's 21-02 UTC forecast hours from the observation timestamps the file provides:
:func:`_l4_interp_weights` ("interp"), :func:`_previous_valid_order`
("previous_valid"), and :func:`_last_before_window_order` ("last_before_window"),
plus the per-cell :func:`_coalesce_slots` fallback and the :func:`_make_l4_time_mapper`
dispatch. Kept as isolated, data-free unit tests so the across-midnight logic can be
verified by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_skill import config
from convection_skill.data_loading import (
    SMAP_TIME_POLICIES,
    _coalesce_slots,
    _l4_interp_weights,
    _last_before_window_order,
    _make_l4_time_mapper,
    _previous_valid_order,
)

# The fixed 3-hourly SMAP L4 observation grid in the data cut (UTC hours), verified
# in notebooks/01_data_audit from SMAP_L4_hour.
L4_OBS_HOURS = np.array([16.0, 19.0, 22.0, 1.0, 4.0])
FORECAST_HOURS = np.array(config.FORECAST_HOURS_UTC, dtype="float64")  # 21..02


def _interp(obs, forecast, values):
    """Reference: evaluate the returned weights against a per-slot value vector."""
    lo, hi, w = _l4_interp_weights(obs, forecast)
    out = np.full(len(forecast), np.nan)
    for i in range(len(forecast)):
        if lo[i] >= 0:
            out[i] = (1.0 - w[i]) * values[lo[i]] + w[i] * values[hi[i]]
    return lo, hi, w, out


def test_bracketing_slots_and_weights():
    """Each forecast hour blends the two L4 observations that straddle it."""
    lo, hi, w = _l4_interp_weights(L4_OBS_HOURS, FORECAST_HOURS)
    # 21->(19,22), 22->exactly 22, 23->(22,01), 00->(22,01), 01->exactly 01, 02->(01,04)
    assert lo.tolist() == [1, 2, 2, 2, 3, 3]
    assert hi.tolist() == [2, 3, 3, 3, 4, 4]
    np.testing.assert_allclose(w, [2 / 3, 0.0, 1 / 3, 2 / 3, 0.0, 1 / 3], atol=1e-12)


def test_interpolates_a_linear_ramp_exactly():
    """A field rising linearly with obs time is recovered at the forecast hours.

    Values equal to each slot's linear-clock hour (16,19,22,25,28) must interpolate
    back to the forecast hours' own linear-clock time (21,22,23,24,25,26).
    """
    ramp = np.array([16.0, 19.0, 22.0, 25.0, 28.0])  # value == linearized obs hour
    _, _, _, out = _interp(L4_OBS_HOURS, FORECAST_HOURS, ramp)
    np.testing.assert_allclose(out, [21.0, 22.0, 23.0, 24.0, 25.0, 26.0], atol=1e-12)


def test_hour_on_observation_reproduces_that_slot():
    """22 and 01 UTC coincide with observations, so w == 0 and they copy the slot."""
    lo, hi, w = _l4_interp_weights(L4_OBS_HOURS, np.array([22.0, 1.0]))
    assert lo.tolist() == [2, 3]
    np.testing.assert_allclose(w, [0.0, 0.0])


def test_post_midnight_hours_use_monotone_clock():
    """01 UTC interpolates between the 22 UTC and 04 UTC obs, not the 16 UTC one."""
    lo, hi, w = _l4_interp_weights(L4_OBS_HOURS, np.array([2.0]))  # 02 UTC
    assert (lo[0], hi[0]) == (3, 4)  # between 01 and 04 UTC observations


def test_no_extrapolation_outside_observed_range():
    """Hours before the first / after the last observation are left NaN."""
    lo, hi, w = _l4_interp_weights(L4_OBS_HOURS, np.array([15.0, 5.0]))  # 15 UTC, 05 UTC
    assert lo.tolist() == [-1, -1]
    assert hi.tolist() == [-1, -1]


def test_undefined_or_missing_yields_sentinel():
    """Undefined forecast hour or all-NaN observation times return -1."""
    lo, _, _ = _l4_interp_weights(L4_OBS_HOURS, np.array([np.nan]))
    assert lo[0] == -1
    lo_all_nan, _, _ = _l4_interp_weights(np.full(5, np.nan), FORECAST_HOURS)
    assert (lo_all_nan == -1).all()


def test_unsorted_observation_times():
    """Observation times need not be pre-sorted; indices refer to original slots."""
    # same grid, slots shuffled: original index 2 is the 22 UTC obs.
    shuffled = np.array([22.0, 16.0, 4.0, 19.0, 1.0])  # slots ->  22,16,04,19,01
    lo, hi, w = _l4_interp_weights(shuffled, np.array([22.0]))
    assert lo[0] == 0  # the 22 UTC observation is original slot 0 here
    np.testing.assert_allclose(w, [0.0])


# --------------------------------------------------------------------------- #
# "previous_valid" policy
# --------------------------------------------------------------------------- #
def test_previous_valid_picks_most_recent_past_observation():
    """Each forecast hour reads the newest observation at or before it, never after."""
    orders = _previous_valid_order(L4_OBS_HOURS, FORECAST_HOURS)
    # first (preferred) slot per forecast hour -> its observation hour
    first_obs = [L4_OBS_HOURS[o[0]] for o in orders]
    assert first_obs == [19.0, 22.0, 22.0, 22.0, 1.0, 1.0]


def test_previous_valid_orders_candidates_newest_first():
    """The fallback order runs backward in time (for per-cell missing-data search)."""
    orders = _previous_valid_order(L4_OBS_HOURS, np.array([0.0]))  # 00 UTC
    # obs at/before 00 UTC are 22,19,16 UTC -> slots 2,1,0 newest-first.
    assert orders[0].tolist() == [2, 1, 0]


def test_previous_valid_before_any_observation_is_empty():
    """A forecast hour earlier than every observation has no candidate -> NaN."""
    orders = _previous_valid_order(np.array([22.0]), np.array([21.0]))  # obs after hour
    assert orders[0].size == 0


# --------------------------------------------------------------------------- #
# "last_before_window" policy
# --------------------------------------------------------------------------- #
def test_last_before_window_is_the_pre_window_snapshot():
    """One antecedent obs, strictly before the 21 UTC window start, newest first."""
    order = _last_before_window_order(L4_OBS_HOURS, FORECAST_HOURS)
    assert order.tolist() == [1, 0]           # 19 UTC then 16 UTC
    assert L4_OBS_HOURS[order[0]] == 19.0      # the 19 UTC state is applied everywhere


def test_last_before_window_excludes_observation_at_window_start():
    """An observation exactly at the window start is not 'before' the window."""
    obs = np.array([18.0, 21.0])  # 21 UTC coincides with the earliest forecast hour
    order = _last_before_window_order(obs, FORECAST_HOURS)
    assert order.tolist() == [0]  # only the 18 UTC obs qualifies


# --------------------------------------------------------------------------- #
# per-cell coalesce + policy dispatch
# --------------------------------------------------------------------------- #
def test_coalesce_falls_through_missing_samples_per_cell():
    """First finite sample along the order wins, independently per cell."""
    # cube: 1 date, 3 slots, 1x2 grid. Cell A valid only at slot 2; cell B at slot 0.
    cube = np.full((1, 3, 1, 2), np.nan, dtype="float32")
    cube[0, 2, 0, 0] = 5.0   # cell A: only the last-tried slot is valid
    cube[0, 0, 0, 1] = 9.0   # cell B: the first-tried slot is valid
    out = _coalesce_slots(cube, np.array([0, 1, 2]))  # try slot 0, then 1, then 2
    np.testing.assert_array_equal(out[0, 0], [5.0, 9.0])  # A fell through to slot 2; B took slot 0


def test_make_mapper_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown smap_time_policy"):
        _make_l4_time_mapper(L4_OBS_HOURS, FORECAST_HOURS, "bogus", (1, 6, 1, 1))


@pytest.mark.parametrize("policy", SMAP_TIME_POLICIES)
def test_mapper_shapes_and_coverage_agree_across_policies(policy):
    """Every policy returns (D,S,LA,LO) and is finite exactly where the cell-day is.

    Because a cell-day is valid at all L4 slots or none in this data, all three
    policies must produce identical finite/NaN masks -- they differ only in values.
    """
    obs = L4_OBS_HOURS
    fh = FORECAST_HOURS
    n_date, n_slot, n_lat, n_lon = 2, len(fh), 1, 2
    cube = np.arange(n_date * 5 * n_lat * n_lon, dtype="float32").reshape(n_date, 5, n_lat, n_lon)
    cube[1] = np.nan  # second date has no retrieval -> all NaN
    mapper = _make_l4_time_mapper(obs, fh, policy, (n_date, n_slot, n_lat, n_lon))
    out = mapper(cube)
    assert out.shape == (n_date, n_slot, n_lat, n_lon)
    assert np.isfinite(out[0]).all()   # present day: every forecast hour filled
    assert np.isnan(out[1]).all()      # absent day: all NaN
