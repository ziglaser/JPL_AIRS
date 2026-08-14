"""Tests for the rain-out discount -- analytic weights and footprint integration."""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import config
from trajectory_kernels import footprint as F
from trajectory_kernels.discount import condensation_discount
from trajectory_kernels.fuzz import StohlFuzz
from trajectory_kernels.pbl import ConstantPBL


class _DeltaFuzz(StohlFuzz):
    def sigma_km(self, distance_km):
        return np.zeros_like(np.asarray(distance_km, dtype=float))


def _all_land(lat, lon):
    return np.ones_like(np.asarray(lat, dtype=float))


def test_no_condensation_is_unity():
    assert np.allclose(condensation_discount(np.array([8.0, 8.0, 8.0])), 1.0)


def test_loss_discounts_upstream_only():
    """q: 10 -> 10 -> 5 -> 5. Points before the loss carry 5/10; after, 1."""
    w = condensation_discount(np.array([10.0, 10.0, 5.0, 5.0]))
    assert np.allclose(w, [0.5, 0.5, 1.0, 1.0])


def test_moistening_is_clipped_at_one():
    """q increasing (uptake; not in this dataset) must never amplify."""
    w = condensation_discount(np.array([5.0, 10.0]))
    assert w.max() <= 1.0


def test_bad_values_degrade_to_noop():
    w = condensation_discount(np.array([np.nan, 0.0, 8.0]))
    assert np.allclose(w, 1.0)


def _parcel(q):
    """Eastward hourly parcel onto (40.5, -90.5) at 200 m in a deep PBL."""
    return {
        "t_hours": np.array([0.0, 1.0, 2.0, 3.0]),
        "time_utc": np.array(
            ["2019-06-05T18:00", "2019-06-05T19:00", "2019-06-05T20:00", "2019-06-05T21:00"],
            dtype="datetime64[ns]"),
        "lat": np.full(4, 40.5),
        "lon": np.array([-93.5, -92.5, -91.5, -90.5]),
        "alt": np.full(4, 200.0),
        "q": np.asarray(q, dtype=float),
    }


def _run(traj, rainout_discount):
    sl, so = F.source_window(40.5, -90.5)
    lag_hours = np.arange(0, 4, dtype=float)
    return F.footprint_from_trajectories(
        [traj], sl, so, lag_hours, ConstantPBL(2000.0), _DeltaFuzz(), _all_land,
        resample_step_min=60.0, rainout_discount=rainout_discount,
    )


def test_footprint_unchanged_when_q_constant():
    on = _run(_parcel([8.0] * 4), rainout_discount=True)
    off = _run(_parcel([8.0] * 4), rainout_discount=False)
    assert np.allclose(on, off)


def test_footprint_halves_upstream_of_condensation():
    """q drops 10 -> 5 between hours 1 and 2: trapezoidal dt = [.5, 1, 1, .5] and
    lag = [3, 2, 1, 0], so undiscounted per-lag mass is [.5, 1, 1, .5] (arrival
    lag first when sorted ascending). Discount [0.5, 0.5, 1, 1] halves the two
    upstream points only."""
    acc = _run(_parcel([10.0, 10.0, 5.0, 5.0]), rainout_discount=True)
    per_lag = acc.sum(axis=(1, 2))  # index = lag 0..3
    assert per_lag == pytest.approx([0.5, 1.0, 0.5, 0.25])
    assert acc.sum() == pytest.approx(2.25)


def test_build_footprint_records_discount_attr():
    from trajectory_kernels import trajectories as T
    if not (config.TRAJ_DIR / config.NOGRID_TEMPLATE.format(granule=189)).exists():
        pytest.skip("trajectory data not present")
    day = T.load_day()
    ds = F.build_footprint(day, 37.5, -87.5, arrival_step=3, rainout_discount=True)
    assert ds.attrs["rainout_discount"] == 1
