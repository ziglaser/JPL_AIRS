"""dl_front.swath: swath geometry, gap decomposition, bank, projections.

Everything runs on synthetic masks and winds with hand-computable answers:
the 16-day cycle arithmetic, morphological envelope closing, the gap_type
value semantics, a tiny swath bank built through monkeypatched fullgrid
readers (roundtrip + read-time thresholding + undersampled cycle days),
and the two wind-advection footprint projectors against a rigid-shift
answer worked out by hand from the 1-degree grid geometry.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dl_front import airs_fcst, config, swath

SHAPE = config.GRID_SHAPE                     # (68, 141): lat 10..77, lon -171..-31


def _block(rows: slice, cols: slice) -> np.ndarray:
    out = np.zeros(SHAPE, bool)
    out[rows, cols] = True
    return out


# --------------------------------------------------------------------------- #
# cycle_day
# --------------------------------------------------------------------------- #

def test_cycle_day_sixteen_day_periodicity():
    d0 = pd.Timestamp("2019-06-05")
    base = swath.cycle_day(d0)
    assert 0 <= base < swath.CYCLE_DAYS
    for k in range(2 * swath.CYCLE_DAYS + 3):
        assert (swath.cycle_day(d0 + pd.Timedelta(days=k))
                == (base + k) % swath.CYCLE_DAYS)
    # a full cycle later maps to the same bin
    assert swath.cycle_day(d0 + pd.Timedelta(days=16)) == base


def test_cycle_day_input_type_consistency():
    # str / Timestamp / datetime.date all land in the same bin (bank build
    # and lookup may pass different types; both go through this function)
    ts = pd.Timestamp("2021-03-14")
    assert swath.cycle_day("2021-03-14") == swath.cycle_day(ts)
    assert swath.cycle_day(ts.date()) == swath.cycle_day(ts)


# --------------------------------------------------------------------------- #
# swath_envelope
# --------------------------------------------------------------------------- #

def _diagonal_band(col_offset: int, half_width: int = 4) -> np.ndarray:
    """A NE-SW swath-like band: |col - (row + col_offset)| <= half_width."""
    rows, cols = np.indices(SHAPE)
    return np.abs(cols - (rows + col_offset)) <= half_width


def test_swath_envelope_closes_holes_and_cuts():
    band = _diagonal_band(20)
    holey = band.copy()
    holey[19:22, 38:41] = False              # interior 3x3 cloud hole
    holey[39:42, 58:61] = False              # another one
    holey[30:32, :] = False                  # 2-row cut clean across the band

    env = swath.swath_envelope(holey)
    # both enclosed holes are recovered in full ...
    assert env[19:22, 38:41].all() and env[39:42, 58:61].all()
    # ... the cut is bridged along the band interior (its outermost edge
    # pixels may stay eroded -- morphology, not a bug) ...
    for row in (30, 31):
        assert env[row, row + 18:row + 23].all()
    # ... and the envelope never hallucinates far from the band
    assert not env[0, 100] and not env[60, 10]


def test_swath_envelope_keeps_separate_bands_unmerged():
    two = _diagonal_band(20) | _diagonal_band(75)   # ~50 degrees apart
    env = swath.swath_envelope(two)
    # midway pixels between the bands stay outside the envelope
    for row in (10, 30, 50):
        assert not env[row, row + 47]


def test_swath_envelope_empty_input():
    empty = np.zeros(SHAPE, bool)
    assert not swath.swath_envelope(empty).any()


# --------------------------------------------------------------------------- #
# classify_gaps
# --------------------------------------------------------------------------- #

def test_classify_gaps_value_semantics():
    observed = _block(slice(10, 20), slice(10, 30))
    observed[50, 100] = True                 # stray retrieval OUTSIDE envelope
    envelope = _block(slice(10, 20), slice(10, 40))   # cols 30:40 = clouds
    domain = np.ones(SHAPE, bool)
    domain[:, 130:] = False
    observed[5, 135] = True                  # retrieval outside the domain

    out = swath.classify_gaps(observed, envelope, domain=domain)
    assert out.dtype == np.int8
    assert out[15, 15] == config.GAP_OBSERVED
    assert out[15, 35] == config.GAP_CLOUD           # in swath, no retrieval
    assert out[60, 5] == config.GAP_OUT_OF_SWATH
    assert out[5, 135] == config.GAP_OUT_OF_DOMAIN   # domain wins over observed
    # an observed pixel is in-swath by definition, never GAP_OUT_OF_SWATH
    assert out[50, 100] == config.GAP_OBSERVED


def test_classify_gaps_default_domain_is_everywhere():
    observed = _block(slice(10, 20), slice(10, 30))
    out = swath.classify_gaps(observed, observed)
    assert (out != config.GAP_OUT_OF_DOMAIN).all()


# --------------------------------------------------------------------------- #
# bank build / load / expected_swath roundtrip (monkeypatched fullgrid IO)
# --------------------------------------------------------------------------- #

BASE = pd.Timestamp("2019-01-05")
#: 6 days sharing BASE's cycle day (enough) + 3 on the next one (undersampled)
GOOD_DATES = {BASE + pd.Timedelta(days=16 * k) for k in range(6)}
THIN_DATES = {BASE + pd.Timedelta(days=1 + 16 * k) for k in range(3)}


def _fake_valid_frac(date) -> np.ndarray:
    """Pixel (10,10) observed every GOOD day; (10,11) on exactly one."""
    vf = np.zeros(SHAPE, np.float32)
    if date in GOOD_DATES:
        vf[10, 10] = 1.0
        if date == BASE:
            vf[10, 11] = 1.0
    elif date in THIN_DATES:
        vf[20, 20] = 1.0
    return vf


@pytest.fixture()
def tiny_bank(tmp_path, monkeypatch):
    """Build a bank via build_swath_bank with the fullgrid readers faked."""
    dates = {f"{d:%Y%m%d}": d for d in GOOD_DATES | THIN_DATES}

    def fake_find(date, root=None):
        key = f"{pd.Timestamp(date):%Y%m%d}"
        return f"FAKE:{key}" if key in dates else None

    def fake_period(path, hour, ds=None):
        assert hour == 18
        # build_swath_bank must pass the ONE pre-loaded dataset per day
        # instead of re-reading the file per hour (review 2026-08-13)
        assert ds == f"DS:{path}"
        date = dates[str(path).split(":")[1]]
        return {"valid_frac": SimpleNamespace(values=_fake_valid_frac(date))}

    monkeypatch.setattr(airs_fcst, "find_fullgrid", fake_find)
    monkeypatch.setattr(airs_fcst, "load_fullgrid", lambda p: f"DS:{p}")
    monkeypatch.setattr(airs_fcst, "period_fields", fake_period)
    path = tmp_path / "swath_bank.npz"
    swath.build_swath_bank([2019], hours=(18,), path=path)
    return path


def test_bank_roundtrip_and_thresholding(tiny_bank, monkeypatch):
    bank = swath.load_swath_bank(tiny_bank)
    assert bank["freq"].shape == (swath.CYCLE_DAYS, 1, *SHAPE)
    assert bank["hours"].tolist() == [18]
    assert bank["years"].tolist() == [2019]
    cyc = swath.cycle_day(BASE)
    assert bank["n_days"][cyc, 0] == 6
    assert bank["n_days"][(cyc + 1) % swath.CYCLE_DAYS, 0] == 3
    np.testing.assert_allclose(bank["freq"][cyc, 0, 10, 10], 1.0)
    np.testing.assert_allclose(bank["freq"][cyc, 0, 10, 11], 1 / 6)

    # threshold is applied at READ time from config.SWATH_MIN_FRACTION:
    # at the default 0.075 both pixels pass (1/6 > 0.075) ...
    foot = swath.expected_swath(BASE, 18, path=tiny_bank)
    assert foot[10, 10] and foot[10, 11]
    assert foot.sum() == 2
    # ... raising the knob drops the 1-of-6 pixel without any rebuild
    monkeypatch.setattr(config, "SWATH_MIN_FRACTION", 0.5)
    foot = swath.expected_swath(BASE, 18, path=tiny_bank)
    assert foot[10, 10] and not foot[10, 11]


def test_expected_swath_undersampled_and_missing(tiny_bank, tmp_path):
    # 3 < MIN_DAYS_PER_CYCLE_DAY contributing days -> None (caller falls back)
    assert swath.expected_swath(BASE + pd.Timedelta(days=1), 18,
                                path=tiny_bank) is None
    # hour never composited -> None
    assert swath.expected_swath(BASE, 21, path=tiny_bank) is None
    # no bank on disk at all -> None
    nowhere = tmp_path / "does_not_exist.npz"
    assert swath.load_swath_bank(nowhere) is None
    assert swath.expected_swath(BASE, 18, path=nowhere) is None


# --------------------------------------------------------------------------- #
# gap_type_for: bank preferred, per-day envelope fallback
# --------------------------------------------------------------------------- #

def test_gap_type_for_falls_back_to_per_day_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SWATH_BANK_PATH", tmp_path / "no_bank.npz")
    valid_frac = np.zeros(SHAPE, np.float32)
    valid_frac[10:20, 10:30] = 1.0
    valid_frac[14:16, 18:21] = 0.0           # interior cloud hole
    domain = np.ones(SHAPE, bool)
    domain[:, 130:] = False

    out = swath.gap_type_for("2019-06-05", 18, valid_frac, domain=domain)
    assert out[12, 12] == config.GAP_OBSERVED
    assert out[15, 19] == config.GAP_CLOUD          # hole closed by envelope
    assert out[60, 60] == config.GAP_OUT_OF_SWATH
    assert out[10, 135] == config.GAP_OUT_OF_DOMAIN


def test_gap_type_for_prefers_bank_footprint(tmp_path, monkeypatch):
    # a fully overcast swath segment leaves nothing for morphological closing
    # to work with -- only the climatological footprint can call it CLOUD
    date = pd.Timestamp("2019-06-05")
    cyc = swath.cycle_day(date)
    freq = np.zeros((swath.CYCLE_DAYS, 1, *SHAPE), np.float32)
    freq[cyc, 0, 10:20, 10:60] = 1.0         # footprint far beyond today's obs
    n_days = np.zeros((swath.CYCLE_DAYS, 1), np.int32)
    n_days[cyc, 0] = 10
    path = tmp_path / "swath_bank.npz"
    np.savez_compressed(path, freq=freq, n_days=n_days,
                        hours=np.asarray([18]), years=np.asarray([2019]))
    monkeypatch.setattr(config, "SWATH_BANK_PATH", path)

    valid_frac = np.zeros(SHAPE, np.float32)
    valid_frac[10:20, 10:20] = 1.0           # observed in a corner only
    out = swath.gap_type_for(date, 18, valid_frac)
    assert out[15, 15] == config.GAP_OBSERVED
    assert out[15, 50] == config.GAP_CLOUD          # overcast but in-bank
    assert out[15, 100] == config.GAP_OUT_OF_SWATH


# --------------------------------------------------------------------------- #
# forward projections
# --------------------------------------------------------------------------- #

def _uniform_winds(u_ms: float, v_ms: float,
                   where: np.ndarray | None = None):
    """(u, v) fields; NaN outside ``where`` to mimic swath-limited retrievals."""
    u = np.full(SHAPE, u_ms)
    v = np.full(SHAPE, v_ms)
    if where is not None:
        u[~where] = np.nan
        v[~where] = np.nan
    return u, v


def test_project_shift_whole_degree_offsets():
    # envelope rows 30:35 -> lats 40..44 N, lat0 = 42.0
    env = _block(slice(30, 35), slice(60, 70))
    # dlat = v * dt * 3600 / 111000 = 10.2778 * 6 * 3600 / 111000 = 2.0002 -> 2
    u, v = _uniform_winds(0.0, 10.2778, where=env)
    pred = swath.project_shift(env, u, v, 6.0)
    np.testing.assert_array_equal(pred, np.roll(env, 2, axis=0))

    # dlon = u * dt * 3600 / (111000 * cos(42 deg))
    #      = 5 * 21600 / 82489 = 1.309 -> 1 column east
    u, v = _uniform_winds(5.0, 0.0, where=env)
    pred = swath.project_shift(env, u, v, 6.0)
    np.testing.assert_array_equal(pred, np.roll(env, 1, axis=1))

    # sub-degree displacement rounds to zero: the no-motion null hypothesis
    u, v = _uniform_winds(0.0, 2.0, where=env)   # dlat = 0.39 -> 0
    np.testing.assert_array_equal(swath.project_shift(env, u, v, 6.0), env)


def test_project_shift_blanks_wraparound():
    env = _block(slice(64, 68), slice(60, 70))   # against the north edge
    u, v = _uniform_winds(0.0, 10.2778, where=env)   # +2 rows, 2 of 4 wrap
    pred = swath.project_shift(env, u, v, 6.0)
    assert pred[66:68, 60:70].all()
    assert not pred[:2].any()                    # wrapped rows are nonsense
    assert pred.sum() == 2 * 10


def test_project_hull_plausible_superset():
    env = _block(slice(30, 35), slice(60, 70))
    u, v = _uniform_winds(0.0, 10.2778)          # finite everywhere
    pred = swath.project_hull(env, u, v, 6.0)
    shifted = np.roll(env, 2, axis=0)
    # strictly-interior points of the advected hull are all covered ...
    assert pred[33:36, 62:68].all()
    # ... it agrees with the rigid shift on most of the shifted area ...
    assert (pred & shifted).sum() >= 0.6 * shifted.sum()
    # ... and it does not balloon into the rest of the grid
    assert pred.sum() <= 3 * env.sum()
    assert not pred[10, 10] and not pred[60, 130]
