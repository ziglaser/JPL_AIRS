"""Tests for the surface-contact weight -- analytic taper behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import config
from trajectory_kernels.contact import contact_weight


def test_surface_is_full_contact():
    assert contact_weight(0.0, 1500.0) == pytest.approx(1.0)


def test_above_layer_is_zero():
    assert contact_weight(2.0 * 1500.0, 1500.0) == pytest.approx(0.0)
    assert contact_weight(1500.0 * config.CONTACT_FRACTION + 1.0, 1500.0) == pytest.approx(0.0)


def test_taper_is_monotone_and_bounded():
    pbl = 2000.0
    alts = np.linspace(0.0, 1.2 * config.CONTACT_FRACTION * pbl, 200)
    w = contact_weight(alts, pbl)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert np.all(np.diff(w) <= 1e-12)  # non-increasing with altitude


def test_below_taper_start_is_one():
    pbl = 1000.0
    layer_top = config.CONTACT_FRACTION * pbl
    taper_start = (1.0 - config.CONTACT_TAPER_FRACTION) * layer_top
    assert contact_weight(taper_start - 1.0, pbl) == pytest.approx(1.0)


def test_halfway_through_taper_is_half():
    pbl = 1000.0
    layer_top = config.CONTACT_FRACTION * pbl
    taper_start = (1.0 - config.CONTACT_TAPER_FRACTION) * layer_top
    mid = 0.5 * (taper_start + layer_top)
    assert contact_weight(mid, pbl) == pytest.approx(0.5, abs=1e-6)


def test_fraction_presets_change_cutoff():
    """A parcel at 0.7*PBL is out for STILT (0.5) but in for Sodemann (1.5)."""
    pbl = 1000.0
    alt = 0.7 * pbl
    assert contact_weight(alt, pbl, fraction=config.CONTACT_FRACTION_STILT) == pytest.approx(0.0)
    assert contact_weight(alt, pbl, fraction=config.CONTACT_FRACTION_SODEMANN) == pytest.approx(1.0)


def test_nonfinite_and_zero_pbl_give_zero():
    assert contact_weight(np.nan, 1500.0) == pytest.approx(0.0)
    assert contact_weight(100.0, 0.0) == pytest.approx(0.0)
