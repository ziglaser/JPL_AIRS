"""Tests for ``front_finder.ingest_hysplit``, ``front_finder.mask_bank``,
and the ``dataset.airs_x`` / ``dataset.airs_samples`` real-AIRS fine-tune path.

Everything runs against a small synthetic ``fullgrid`` HYSPLIT file built
in-memory and written under ``tmp_path`` -- no real data, no network, no
TensorFlow.  Follows the house style of ``test_front_dataset.py``: same
``tests/_stubs`` trick for the transitive tensorflow import via
``front_finder.derive`` (``ingest_hysplit`` imports ``derive`` for
``thermo_channels``), and ``config.CODSUS_DIR``/``config.MERRA2_DIR``
monkeypatched to ``tmp_path`` so the real ``data/`` tree is never touched.

Synthetic fullgrid geometry (mirrors the real schema, audited 2026-08-04 --
see ``ingest_hysplit`` module docstring): dims (time=7, level=33, lat=28,
lon=43); level = 115..1075 step 30 hPa (30-hPa bin centers); lat =
25.5..52.5 step 1 (half-degree cell centers); lon = -106.5..-64.5 step 1.
The synthetic ``t`` field is an AFFINE function of (level, lat, lon) --
``t = 300 - 0.05*level + 0.1*lat + 0.01*lon`` -- so that both the vertical
(linear-in-level) and horizontal (bilinear, no cross term) interpolation
steps in ``to_label_grid`` reproduce it EXACTLY at any point strictly inside
the observed sub-block; this makes the label-grid T channel hand-checkable
by plugging the label lat/lon and target hPa level straight into the
formula.

The first half of the native lat rows (25.5..38.5, i.e. native lat index
< OBS_LAT_SPLIT) is "observed": N > 0 and t/q/u/v hold valid analytic
values at every level. The remaining rows (39.5..52.5) are "unobserved":
N == 0 and t/q/u/v == -9999.0 (fill) at every level -- exactly the real
fill convention, and it naturally makes those cells (and points that mix
observed/unobserved corners) NaN after interpolation too.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_STUBS = Path(__file__).resolve().parent / "_stubs"
if str(_STUBS) not in sys.path:                    # see test_front_dataset.py
    sys.path.insert(0, str(_STUBS))

from front_finder import config, dataset, ingest_hysplit as ih  # noqa: E402
from front_finder import mask_bank  # noqa: E402
from front_finder import labels  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic fullgrid geometry
# --------------------------------------------------------------------------- #
N_TIME = 7
LEVEL = 115.0 + 30.0 * np.arange(33)               # 115..1075, 33 levels
LAT = 25.5 + np.arange(28)                         # 25.5..52.5, 28 points
LON = -106.5 + np.arange(43)                        # -106.5..-64.5, 43 points
assert LEVEL.size == 33 and LAT.size == 28 and LON.size == 43

OBS_LAT_SPLIT = 14                                  # lat idx < 14 -> observed
FILL = ih.FILL

# A point safe deep inside the observed sub-block and away from all domain
# edges, used for the hand-checkable interpolation assertion.
CHECK_LAT, CHECK_LON = 30.0, -90.0


def _affine_t(level, lat, lon):
    """The exact analytic field baked into the synthetic ``t`` variable."""
    return 300.0 - 0.05 * level + 0.1 * lat + 0.01 * lon


def _make_fullgrid_ds(q_override=None):
    """Build the synthetic fullgrid Dataset (time, level, lat, lon)."""
    lev_g, lat_g, lon_g = np.meshgrid(LEVEL, LAT, LON, indexing="ij")

    t_field = _affine_t(lev_g, lat_g, lon_g)
    q_field = 5.0 + 0.005 * lev_g                    # ~5.6..10.4 g/kg
    u_field = 2.0 + 0.01 * lat_g
    v_field = -1.0 + 0.01 * lon_g
    pres_field = lev_g * 100.0
    alt_field = 44300.0 * (1.0 - (lev_g / 1013.25) ** 0.190)
    w_field = np.full_like(lev_g, 0.01)
    parceltime_field = np.zeros_like(lev_g)
    q_excess_field = np.zeros_like(lev_g)
    n_field = np.where(lat_g < LAT[OBS_LAT_SPLIT], 5.0, 0.0)

    if q_override is not None:
        q_field = q_field.copy()
        q_field[q_override] = 10.0

    unobserved = lat_g >= LAT[OBS_LAT_SPLIT]
    for field in (t_field, q_field, u_field, v_field, pres_field, alt_field,
                 w_field, parceltime_field, q_excess_field):
        field[unobserved] = FILL

    def _tile(a):
        return np.broadcast_to(a, (N_TIME,) + a.shape).copy()

    data = {}
    for name, field in (
        ("t", t_field), ("q", q_field), ("u", u_field), ("v", v_field),
        ("pres", pres_field), ("alt", alt_field), ("w", w_field),
        ("parceltime", parceltime_field), ("q_excess", q_excess_field),
        ("N", n_field),
    ):
        data[name] = (("time", "level", "lat", "lon"), _tile(field))

    ds = xr.Dataset(
        data,
        coords={"time": np.arange(N_TIME), "level": LEVEL, "lat": LAT, "lon": LON},
    )
    ds["q"].attrs["units"] = "g/kg"
    ds["q"].attrs["long_name"] = "mixing_ratio"
    return ds


def _write_fullgrid(tmp_path, date="20180605", window="1700-2059",
                    q_override=None) -> Path:
    ds = _make_fullgrid_ds(q_override=q_override)
    path = tmp_path / f"fullgrid_wrf27km_GOOD_1p00deg_{date}_{window}.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def fullgrid_path(tmp_path) -> Path:
    return _write_fullgrid(tmp_path)


# --------------------------------------------------------------------------- #
# 1. load_fullgrid
# --------------------------------------------------------------------------- #

def test_load_fullgrid_fill_to_nan_everywhere(fullgrid_path):
    ds = ih.load_fullgrid(fullgrid_path)
    unobserved_lat = LAT >= LAT[OBS_LAT_SPLIT]
    for var in ("t", "u", "v", "pres", "alt", "w", "parceltime", "q_excess"):
        sub = ds[var].sel(lat=unobserved_lat)
        assert np.isnan(sub.values).all(), f"{var} should be all-NaN off-swath"
    # observed region has no leftover fill values
    observed_lat = LAT < LAT[OBS_LAT_SPLIT]
    sub_t = ds["t"].sel(lat=observed_lat)
    assert not np.any(sub_t.values == FILL)
    assert np.isfinite(sub_t.values).all()


def test_load_fullgrid_q_unit_conversion_analytic(tmp_path):
    override = (0, 5, 5)   # (level, lat, lon) index, observed region
    path = _write_fullgrid(tmp_path, q_override=override)
    ds = ih.load_fullgrid(path)
    q = ds["q"].isel(time=0, level=override[0], lat=override[1], lon=override[2])
    expected = 0.01 / 1.01   # r = 10 g/kg = 0.01 kg/kg mixing ratio
    assert float(q.values) == pytest.approx(expected)


def test_load_fullgrid_q_attrs_updated(fullgrid_path):
    ds = ih.load_fullgrid(fullgrid_path)
    assert ds["q"].attrs["units"] == "kg/kg"
    assert ds["q"].attrs["long_name"] == "specific_humidity"


# --------------------------------------------------------------------------- #
# 2. overpass_time / nearest_bulletin
# --------------------------------------------------------------------------- #

def test_overpass_time_is_window_midpoint(fullgrid_path):
    t = ih.overpass_time(fullgrid_path)
    assert t == pd.Timestamp("2018-06-05 18:59:30")


def test_nearest_bulletin_rounds_to_3hourly(fullgrid_path):
    t = ih.overpass_time(fullgrid_path)
    b = ih.nearest_bulletin(t)
    assert b == pd.Timestamp("2018-06-05 18:00:00")


# --------------------------------------------------------------------------- #
# 3. to_label_grid
# --------------------------------------------------------------------------- #

@pytest.fixture
def label_grid(fullgrid_path):
    ds = ih.load_fullgrid(fullgrid_path)
    return ih.to_label_grid(ds, slot=0, winds=True)


def test_to_label_grid_shape_and_vars(label_grid):
    ch = label_grid
    assert ch.sizes["lat"] == 68
    assert ch.sizes["lon"] == 141
    assert tuple(ch["lev"].values) == config.TARGET_LEVELS_HPA
    for var in config.THERMO_VARS + config.WIND_VARS:
        assert var in ch.data_vars
    assert "observed" in ch.data_vars
    assert "valid_fraction" in ch.data_vars


def test_to_label_grid_nan_outside_fullgrid_domain(label_grid):
    ch = label_grid
    t_val = ch["T"].sel(lat=60.0, lon=-160.0, lev=850)
    assert np.isnan(float(t_val.values))
    assert ch["observed"].sel(lat=60.0, lon=-160.0) == False  # noqa: E712


def test_to_label_grid_T_matches_analytic_interpolation(label_grid):
    ch = label_grid
    t_val = float(ch["T"].sel(lat=CHECK_LAT, lon=CHECK_LON, lev=850).values)
    expected = _affine_t(850.0, CHECK_LAT, CHECK_LON)
    assert t_val == pytest.approx(expected, rel=1e-6)


def test_to_label_grid_observed_true_in_interior_false_outside(label_grid):
    ch = label_grid
    assert bool(ch["observed"].sel(lat=CHECK_LAT, lon=CHECK_LON).values) is True
    assert bool(ch["observed"].sel(lat=60.0, lon=-160.0).values) is False
    # deep in the unobserved half of the swath (still inside the fullgrid box)
    assert bool(ch["observed"].sel(lat=45.0, lon=CHECK_LON).values) is False


def test_to_label_grid_valid_fraction_in_unit_interval(label_grid):
    vf = label_grid["valid_fraction"].values
    finite = vf[np.isfinite(vf)]
    assert finite.min() >= -1e-9
    assert finite.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# 4. dataset.airs_x
# --------------------------------------------------------------------------- #

def _norm_stats():
    ranges = {
        "T": (150.0, 350.0), "q": (0.0, 0.03), "r": (0.0, 0.03),
        "Td": (150.0, 350.0), "theta_e": (200.0, 500.0), "Tv": (150.0, 350.0),
        "RH": (0.0, 1.2), "u": (-60.0, 60.0), "v": (-60.0, 60.0),
    }
    stats = {}
    for var in config.THERMO_VARS + config.WIND_VARS:
        mn, mx = ranges[var]
        for lev in config.TARGET_LEVELS_HPA:
            stats[f"{var}_{lev}"] = [mn, mx]
    return stats


def test_airs_x_shape_finite_and_mask_matches_valid_fraction(fullgrid_path):
    stats = _norm_stats()
    x, observed, t = ih_airs_x(fullgrid_path, stats)
    n_ch = len(dataset.channel_names(winds=True))
    assert x.shape == (*config.PADDED_SHAPE, len(config.TARGET_LEVELS_HPA), n_ch)
    assert np.all(np.isfinite(x))

    ch = ih.to_label_grid(ih.load_fullgrid(fullgrid_path), slot=0, winds=True)
    vf = ch["valid_fraction"].transpose("lat", "lon", "lev").values
    expected_mask = dataset._pad(np.nan_to_num(vf.astype(np.float32)))
    assert np.allclose(x[..., -1], expected_mask)


def test_airs_x_unobserved_cells_are_imputed(fullgrid_path):
    stats = _norm_stats()
    x, observed, t = ih_airs_x(fullgrid_path, stats)
    # label lat 45 (unobserved half) -> padded index +2; label lon -90 -> +1
    lat_pad = int(45 - (-171)) + 2  # not used; index directly via padded coords
    # Simpler: locate via unpadded (lat, lon) index into the 68x141 grid.
    lat_idx = int(45 - 10)     # GRID lat runs 10..77
    lon_idx = int(-90 - (-171))
    pl, po = lat_idx + 2, lon_idx + 1
    assert not observed[lat_idx, lon_idx]
    assert np.all(x[pl, po, :, :-1] == dataset.IMPUTE_VALUE)


def ih_airs_x(path, stats, winds=True, slot=0):
    return dataset.airs_x(path, stats, winds, slot)


# --------------------------------------------------------------------------- #
# 5. dataset.airs_samples
# --------------------------------------------------------------------------- #

FRONT_TYPE_ORDER = ("cold", "warm", "stationary", "occluded", "none")
FILL_LAT_LABEL, FILL_LON_LABEL = CHECK_LAT, CHECK_LON  # inside observed block


@pytest.fixture
def patched_codsus_dir(tmp_path, monkeypatch):
    codsus_dir = tmp_path / "CODSUS"
    monkeypatch.setattr(config, "CODSUS_DIR", codsus_dir)
    return codsus_dir


def _write_codsus_2018(codsus_dir, bulletin: pd.Timestamp):
    n_f, n_lat, n_lon = 5, *config.GRID_SHAPE
    fronts = np.zeros((1, n_f, n_lat, n_lon), dtype=np.uint8)
    fill_lat_i = int(FILL_LAT_LABEL - 10)
    fill_lon_i = int(FILL_LON_LABEL - (-171))
    fronts[:, :, fill_lat_i, fill_lon_i] = config.LABEL_FILL  # invalid pixel

    ds = xr.Dataset(
        {
            "fronts": (("time", "front", "lat", "lon"), fronts),
            "front_type": (("front",), np.array(FRONT_TYPE_ORDER, dtype=object)),
        },
        coords={
            "time": pd.DatetimeIndex([bulletin]),
            "lat": np.arange(10, 78, 1, dtype=np.float64),
            "lon": np.arange(-171, -30, 1, dtype=np.float64),
        },
    )
    # Manifest reorg 2026-08-13: year files live in a {width}wide/ subdir.
    (codsus_dir / "1wide").mkdir(parents=True, exist_ok=True)
    path = codsus_dir / "1wide" / "codsus_masked_merra2-1deg_1wide_2018.nc"
    ds.to_netcdf(path)
    return path


def test_airs_samples_yields_one_pair_with_correct_weight(
        tmp_path, patched_codsus_dir, fullgrid_path):
    bulletin = ih.nearest_bulletin(ih.overpass_time(fullgrid_path))
    _write_codsus_2018(patched_codsus_dir, bulletin)
    stats = _norm_stats()

    pairs = list(dataset.airs_samples([fullgrid_path], stats, winds=True))
    assert len(pairs) == 1
    x, y = pairs[0]
    w = y[..., -1]

    fill_lat_i = int(FILL_LAT_LABEL - 10) + 2
    fill_lon_i = int(FILL_LON_LABEL - (-171)) + 1
    assert w[fill_lat_i, fill_lon_i] == 0.0   # label-invalid despite observed

    unobs_lat_i = int(45 - 10) + 2
    unobs_lon_i = int(-90 - (-171)) + 1
    assert w[unobs_lat_i, unobs_lon_i] == 0.0  # valid label but not observed

    obs_lat_i = int(32 - 10) + 2   # interior, observed, not the fill pixel
    obs_lon_i = int(-90 - (-171)) + 1
    assert w[obs_lat_i, obs_lon_i] == 1.0

    assert np.count_nonzero(w) > 0
    assert set(np.unique(w)) <= {0.0, 1.0}


def test_airs_samples_skips_unpaired_year_with_warning(
        tmp_path, patched_codsus_dir, fullgrid_path):
    bulletin = ih.nearest_bulletin(ih.overpass_time(fullgrid_path))
    _write_codsus_2018(patched_codsus_dir, bulletin)
    stats = _norm_stats()

    unpaired_path = _write_fullgrid(tmp_path, date="20190605")

    with pytest.warns(UserWarning):
        pairs = list(dataset.airs_samples(
            [fullgrid_path, unpaired_path], stats, winds=True))
    assert len(pairs) == 1


# --------------------------------------------------------------------------- #
# 6. mask_bank
# --------------------------------------------------------------------------- #

def test_harvest_and_load_bank_round_trip(tmp_path, fullgrid_path):
    out_path = tmp_path / "gap_bank.npz"
    result_path = mask_bank.harvest([fullgrid_path], out_path=out_path)
    assert result_path == out_path
    assert out_path.exists()

    with np.load(out_path) as z:
        assert z["vf"].shape == (1, 68, 141, 5)
        assert list(z["date"]) == ["2018-06-05"]

    vf, dates = mask_bank.load_bank(out_path)
    assert vf.shape == (1, 68, 141, 5)
    assert vf.dtype == np.float32
    assert list(dates) == ["2018-06-05"]


def test_sample_mask_returns_bank_member_and_month_fallback(tmp_path, fullgrid_path):
    out_path = tmp_path / "gap_bank.npz"
    mask_bank.harvest([fullgrid_path], out_path=out_path)
    vf, dates = mask_bank.load_bank(out_path)
    rng = np.random.default_rng(0)

    m = mask_bank.sample_mask(vf, rng)
    assert m.shape == (68, 141, 5)
    assert np.array_equal(m, vf[0])

    m_june = mask_bank.sample_mask(vf, rng, month=6, dates=dates)
    assert np.array_equal(m_june, vf[0])

    # December: no bank entry within +-1 month of the single June sample --
    # falls back to the whole bank rather than raising/returning empty.
    m_dec = mask_bank.sample_mask(vf, rng, month=12, dates=dates)
    assert m_dec.shape == (68, 141, 5)


def test_apply_mask_imputes_low_valid_fraction_cells():
    n_ch = 4
    x = np.ones((*config.PADDED_SHAPE, 5, n_ch), dtype=np.float32)
    vf = np.zeros((*config.GRID_SHAPE, 5), dtype=np.float32)
    vf[10:20, 10:20, :] = 1.0   # a well-observed patch, rest below threshold

    out = mask_bank.apply_mask(x, vf, impute_value=0.5)

    vf_p = dataset._pad(vf)
    assert np.array_equal(out[..., -1], vf_p)

    invalid = vf_p < ih.OBSERVED_MIN_FRACTION
    assert np.all(out[..., :-1][invalid] == 0.5)
    assert np.all(out[..., :-1][~invalid] == 1.0)
