"""Tests for ``front_finder.dataset`` and ``front_finder.derive``.

Everything runs against small synthetic MERRA-2/CODSUS files built in-memory
and written under ``tmp_path`` -- no real data, no network, no TensorFlow.
``config.MERRA2_DIR``/``config.CODSUS_DIR`` are monkeypatched per test so the
real ``data/`` tree is never touched.

Layouts mirror the real pipeline exactly:
  - MERRA-2 daily file: ``acquire_merra2.download_day``'s on-disk schema
    (T/QV/U/V on the 5 target levels + PS, label-grid lat/lon, 8 3-hourly
    steps), written to ``day_path(date)`` so ``dataset.year_samples`` finds
    it the same way it would find real data.
  - CODSUS year file: ``labels.load_codsus``'s schema (``fronts(time, front,
    lat, lon)`` ubyte with ``front_type`` naming the front axis), written to
    the masked-variant filename ``labels.load_codsus`` builds.

A real bug found while writing these tests (see the bottom of this file):
``fronts/utils/data_utils.py`` and ``fronts/utils/variables.py`` (vendored,
third-party) unconditionally ``import tensorflow`` at module scope even
though ``front_finder.derive`` only ever calls their pure-numpy code
paths. That forces a tensorflow install onto every consumer. We work around
it here with a minimal stub module (``tests/_stubs/tensorflow.py``) rather
than pull in the real (huge) package for tests that never touch a
tf.Tensor -- see that file's docstring for exactly what it fakes and why
it's safe to do so for these tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_STUBS = Path(__file__).resolve().parent / "_stubs"
if str(_STUBS) not in sys.path:                    # see module docstring
    sys.path.insert(0, str(_STUBS))

from front_finder import config, dataset, derive, labels  # noqa: E402
from front_finder.acquire_merra2 import day_path  # noqa: E402

V = derive._variables()  # fronts/utils/variables.py (lazy since 2026-08-12)

# --------------------------------------------------------------------------- #
# Shared synthetic-grid geometry (matches config.GRID_SHAPE == (68, 141))
# --------------------------------------------------------------------------- #
LAT = np.arange(10, 78, 1, dtype=np.float64)      # 68: 10..77 N
LON = np.arange(-171, -30, 1, dtype=np.float64)   # 141: -171..-31 E
LEVS = list(config.TARGET_LEVELS_HPA)             # [1000, 925, 850, 700, 500]
assert (len(LAT), len(LON)) == config.GRID_SHAPE

# Below-ground NaN corner injected into the synthetic MERRA-2 file, lev=1000 only.
NAN_LAT_SLICE = slice(0, 3)
NAN_LON_SLICE = slice(0, 3)

# CODSUS front-axis ordering as stored on disk (the four physical classes
# per config.FRONT_TYPES, plus "none" -- matches the real archive schema).
FRONT_TYPE_ORDER = ("cold", "warm", "stationary", "occluded", "none")
FILL_LAT, FILL_LON = 5, 5                          # outside-mask pixel
LINE_PIXELS = [(10, 10), (11, 11), (12, 12)]        # a short "cold" front line
OVERLAP_LAT, OVERLAP_LON = 20, 20                   # cold+warm overlap pixel


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Point config.MERRA2_DIR / config.CODSUS_DIR at tmp_path subdirs."""
    merra2_dir = tmp_path / "MERRA2"
    codsus_dir = tmp_path / "CODSUS"
    monkeypatch.setattr(config, "MERRA2_DIR", merra2_dir)
    monkeypatch.setattr(config, "CODSUS_DIR", codsus_dir)
    return merra2_dir, codsus_dir


# --------------------------------------------------------------------------- #
# Synthetic-file builders
# --------------------------------------------------------------------------- #

def _make_day_dataset(date, rng, corner_nan=True) -> xr.Dataset:
    """One day's worth of the compact MERRA-2 label-grid file (8 steps)."""
    times = pd.date_range(date, periods=8, freq="3h")
    shape = (len(times), len(LEVS), len(LAT), len(LON))
    T = rng.uniform(250.0, 300.0, size=shape)
    rh = rng.uniform(0.2, 0.8, size=shape)          # physical: keeps Td <= T
    p_pa = np.broadcast_to(
        np.array(LEVS, dtype=np.float64)[None, :, None, None] * 100.0, shape)
    QV = V.specific_humidity_from_relative_humidity(p_pa, T, rh)
    U = rng.uniform(-20.0, 20.0, size=shape)
    Vw = rng.uniform(-20.0, 20.0, size=shape)
    PS = np.full((len(times), len(LAT), len(LON)), 101000.0)

    T, QV, U, Vw = (a.astype(np.float32) for a in (T, QV, U, Vw))
    if corner_nan:
        # Below-ground fill (MERRA-2 convention): NaN at lev=1000 in a corner.
        T[:, 0, NAN_LAT_SLICE, NAN_LON_SLICE] = np.nan
        QV[:, 0, NAN_LAT_SLICE, NAN_LON_SLICE] = np.nan
        U[:, 0, NAN_LAT_SLICE, NAN_LON_SLICE] = np.nan
        Vw[:, 0, NAN_LAT_SLICE, NAN_LON_SLICE] = np.nan

    return xr.Dataset(
        {
            "T": (("time", "lev", "lat", "lon"), T),
            "QV": (("time", "lev", "lat", "lon"), QV),
            "U": (("time", "lev", "lat", "lon"), U),
            "V": (("time", "lev", "lat", "lon"), Vw),
            "PS": (("time", "lat", "lon"), PS.astype(np.float32)),
        },
        coords={"time": times, "lev": LEVS, "lat": LAT, "lon": LON},
    )


def _write_merra2_day(date, rng, corner_nan=True):
    """Write one daily file to the (patched) day_path(date) and return it."""
    ds = _make_day_dataset(date, rng, corner_nan=corner_nan)
    p = day_path(pd.Timestamp(date))
    p.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(p)
    return ds, p


def _make_codsus_dataset(times) -> xr.Dataset:
    n_t, n_f, n_lat, n_lon = len(times), 5, len(LAT), len(LON)
    fronts = np.zeros((n_t, n_f, n_lat, n_lon), dtype=np.uint8)
    cold_idx = FRONT_TYPE_ORDER.index("cold")
    warm_idx = FRONT_TYPE_ORDER.index("warm")
    for lat_i, lon_i in LINE_PIXELS:
        fronts[:, cold_idx, lat_i, lon_i] = 1
    # Overlapping classes: both cold and warm lit at one pixel, every time --
    # exercises make_y's overlap-resolution rule.
    fronts[:, cold_idx, OVERLAP_LAT, OVERLAP_LON] = 1
    fronts[:, warm_idx, OVERLAP_LAT, OVERLAP_LON] = 1
    fronts[:, :, FILL_LAT, FILL_LON] = config.LABEL_FILL

    return xr.Dataset(
        {
            "fronts": (("time", "front", "lat", "lon"), fronts),
            "front_type": (("front",), np.array(FRONT_TYPE_ORDER, dtype=object)),
        },
        coords={"time": pd.DatetimeIndex(times), "lat": LAT, "lon": LON},
    )


def _write_codsus_year(year, times, masked=True, width=1):
    # Manifest reorg 2026-08-13: only the masked variant exists, in {w}wide/.
    ds = _make_codsus_dataset(times)
    path = (config.CODSUS_DIR / f"{width}wide"
            / f"codsus_masked_merra2-1deg_{width}wide_{year}.nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds, path


# --------------------------------------------------------------------------- #
# 1. derive.thermo_channels analytic-answer checks
# --------------------------------------------------------------------------- #

def test_dewpoint_from_specific_humidity_matches_variables_doctest():
    """P=1e5 Pa, q=0.02 -> Td ~ 298.20035572272803 K (fronts/utils/variables.py
    docstring example -- and one of the few that still matches the code)."""
    td = V.dewpoint_from_specific_humidity(1e5, 0.02)
    assert td == pytest.approx(298.20035572272803)


def test_equivalent_potential_temperature_docstring_is_stale():
    """The variables.py docstring for this function claims
    theta_e(P=1e5, T=300, Td=290) == 326.52430009577137, but running the
    *actual current code* gives 329.6988179536086 -- confirmed independently
    via ``python3 -m doctest fronts/utils/variables.py`` (19 failures across
    10 functions, mostly this kind of drift). This test pins the real,
    reproducible behavior of the installed code rather than the stale
    docstring number, and exists to flag the drift rather than hide it.
    """
    theta_e = V.equivalent_potential_temperature(1e5, 300.0, 290.0)
    assert theta_e == pytest.approx(329.6988179536086)


def test_thermo_channels_reproduces_dewpoint_value():
    p_pa = xr.DataArray([1e5])
    q = xr.DataArray([0.02])
    t = xr.DataArray([300.0])
    ch = derive.thermo_channels(t, q, p_pa)
    assert float(ch["Td"].values[0]) == pytest.approx(298.20035572272803)


def test_thermo_channels_physical_bounds():
    """Td <= T, RH in (0, 1] (a *fraction*, not a percent -- see note below),
    r > 0, for physically-consistent (T, q, P) triples.

    Note: relative_humidity_from_dewpoint returns a decimal fraction (e.g.
    0.54), not a 0-100 percentage, so the physical upper bound checked here
    is 1, not 100.
    """
    rng = np.random.default_rng(1)
    n = 500
    t = xr.DataArray(rng.uniform(230.0, 310.0, size=n))
    p_pa = xr.DataArray(rng.uniform(500.0, 1050.0, size=n) * 100.0)
    rh_target = rng.uniform(0.05, 0.95, size=n)
    q = xr.DataArray(V.specific_humidity_from_relative_humidity(
        p_pa.values, t.values, rh_target))

    ch = derive.thermo_channels(t, q, p_pa)

    assert np.all(ch["Td"].values <= t.values + 1e-6)
    assert np.all(ch["r"].values > 0)
    assert np.all(ch["RH"].values > 0)
    assert np.all(ch["RH"].values <= 1.0 + 1e-6)


def test_thermo_channels_has_all_config_vars():
    t = xr.DataArray([300.0, 280.0])
    q = xr.DataArray([0.01, 0.005])
    p_pa = xr.DataArray([1e5, 9e4])
    ch = derive.thermo_channels(t, q, p_pa)
    assert set(ch.data_vars) == set(config.THERMO_VARS)


# --------------------------------------------------------------------------- #
# 2. dataset.compute_norm_stats
# --------------------------------------------------------------------------- #

def test_compute_norm_stats_writes_json_bracketing_actual_values(patched_dirs, tmp_path):
    year = 2003
    date = pd.Timestamp(f"{year}-01-01")
    rng = np.random.default_rng(2)
    day_ds, _ = _write_merra2_day(date, rng)

    stats_path = tmp_path / "norm_stats.json"
    stats = dataset.compute_norm_stats(years=[year], step_days=1, path=stats_path)

    assert stats_path.exists()
    on_disk = json.loads(stats_path.read_text())
    assert on_disk == stats

    ch = derive.merra2_channels(day_ds.load(), winds=True)
    expected_keys = {f"{var}_{lev}" for var in ch.data_vars for lev in LEVS}
    assert set(stats) == expected_keys

    for var in ch.data_vars:
        for lev in LEVS:
            mn, mx = stats[f"{var}_{lev}"]
            assert mn < mx
            field = ch[var].sel(lev=lev).values
            assert mn <= np.nanmin(field) + 1e-6
            assert mx >= np.nanmax(field) - 1e-6


def test_compute_norm_stats_skips_missing_days(patched_dirs, tmp_path):
    """A day with no file on disk is silently skipped, not an error."""
    year = 2004
    rng = np.random.default_rng(3)
    _write_merra2_day(pd.Timestamp(f"{year}-01-01"), rng)
    # step_days=3 walks Jan-01, Jan-04, Jan-07, ... ; only Jan-01 exists.
    stats = dataset.compute_norm_stats(years=[year], step_days=3,
                                       path=tmp_path / "stats.json")
    assert len(stats) == len(config.THERMO_VARS + config.WIND_VARS) * len(LEVS)


# --------------------------------------------------------------------------- #
# 3. dataset.make_x
# --------------------------------------------------------------------------- #

@pytest.fixture
def one_day_channels(patched_dirs, tmp_path):
    """A single synthetic day, loaded and run through merra2_channels, plus
    norm stats computed from that same file (so make_x's output is
    guaranteed to land inside [0, 1])."""
    year = 2003
    date = pd.Timestamp(f"{year}-01-01")
    rng = np.random.default_rng(4)
    day_ds, _ = _write_merra2_day(date, rng)
    stats = dataset.compute_norm_stats(years=[year], step_days=1,
                                       path=tmp_path / "stats.json")
    ch = derive.merra2_channels(day_ds.load(), winds=True)
    return ch, stats


def test_make_x_shape_and_channel_count(one_day_channels):
    ch, stats = one_day_channels
    names = dataset.channel_names(winds=True)
    x = dataset.make_x(ch.isel(time=0), stats, winds=True)
    assert x.shape == (*config.PADDED_SHAPE, len(LEVS), len(names))
    assert np.all(np.isfinite(x))


def test_make_x_mask_channel_zero_at_nan_corner_only_at_lev_1000(one_day_channels):
    ch, stats = one_day_channels
    x = dataset.make_x(ch.isel(time=0), stats, winds=True)
    mask = x[..., -1]  # (lat, lon, lev) padded

    # unpadded corner indices -> padded coordinates (pad offsets: +2 lat, +1 lon)
    lat_idx = [i + 2 for i in range(0, 3)]
    lon_idx = [j + 1 for j in range(0, 3)]
    lev0 = LEVS.index(1000)
    for li in lat_idx:
        for lj in lon_idx:
            assert mask[li, lj, lev0] == 0.0
            for k, lev in enumerate(LEVS):
                if lev != 1000:
                    assert mask[li, lj, k] == 1.0

    # a pixel far from the corner is valid at every level
    far_lat, far_lon = 30 + 2, 30 + 1
    assert np.all(mask[far_lat, far_lon, :] == 1.0)


def test_make_x_imputed_pixels_are_exactly_half(one_day_channels):
    ch, stats = one_day_channels
    x = dataset.make_x(ch.isel(time=0), stats, winds=True)
    lat_idx = [i + 2 for i in range(0, 3)]
    lon_idx = [j + 1 for j in range(0, 3)]
    lev0 = LEVS.index(1000)
    for li in lat_idx:
        for lj in lon_idx:
            assert np.all(x[li, lj, lev0, :-1] == dataset.IMPUTE_VALUE)


def test_make_x_padding_rows_and_cols_are_zero_in_mask_channel(one_day_channels):
    ch, stats = one_day_channels
    x = dataset.make_x(ch.isel(time=0), stats, winds=True)
    mask = x[..., -1]
    pad_lat = config.PADDED_SHAPE[0] - config.GRID_SHAPE[0]
    pad_lon = config.PADDED_SHAPE[1] - config.GRID_SHAPE[1]
    assert np.all(mask[:2, :, :] == 0.0)                       # top lat pad
    assert np.all(mask[2 + config.GRID_SHAPE[0]:, :, :] == 0.0)  # bottom lat pad
    assert np.all(mask[:, :1, :] == 0.0)                        # left lon pad
    assert np.all(mask[:, 1 + config.GRID_SHAPE[1]:, :] == 0.0)  # right lon pad
    assert pad_lat == 4 and pad_lon == 3  # sanity on the padding constants


def test_make_x_normalized_values_within_unit_interval(one_day_channels):
    ch, stats = one_day_channels
    x = dataset.make_x(ch.isel(time=0), stats, winds=True)
    non_mask = x[..., :-1]
    assert non_mask.min() >= -1e-6
    assert non_mask.max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# 4. dataset.make_y
# --------------------------------------------------------------------------- #

@pytest.fixture
def codsus_time0(patched_dirs):
    year = 2003
    times = pd.date_range(f"{year}-01-01", periods=8, freq="3h")
    ds, _ = _write_codsus_year(year, times, masked=True)
    fronts = labels.front_stack(ds).values          # (time, front, lat, lon)
    valid = labels.valid_mask(ds).values             # (time, lat, lon)
    return fronts[0], valid[0]                        # time index 0


def test_make_y_shape_and_one_hot_sums_to_one(codsus_time0):
    fronts_t, valid_t = codsus_time0
    y = dataset.make_y(fronts_t, valid_t)
    assert y.shape == (*config.PADDED_SHAPE, len(dataset.CLASS_NAMES) + 1)

    lat0, lat1 = 2, 2 + config.GRID_SHAPE[0]
    lon0, lon1 = 1, 1 + config.GRID_SHAPE[1]
    classes = y[lat0:lat1, lon0:lon1, :len(dataset.CLASS_NAMES)]
    sums = classes.sum(axis=-1)
    assert np.allclose(sums, 1.0)


def test_make_y_front_line_maps_to_correct_class_index(codsus_time0):
    fronts_t, valid_t = codsus_time0
    y = dataset.make_y(fronts_t, valid_t)
    cold_class = dataset.CLASS_NAMES.index("cold")
    for lat_i, lon_i in LINE_PIXELS:
        pixel = y[lat_i + 2, lon_i + 1, :len(dataset.CLASS_NAMES)]
        assert pixel[cold_class] == 1.0
        assert pixel.sum() == 1.0


def test_make_y_none_class_elsewhere(codsus_time0):
    fronts_t, valid_t = codsus_time0
    y = dataset.make_y(fronts_t, valid_t)
    none_class = dataset.CLASS_NAMES.index("none")
    assert none_class == 0
    # a pixel that is neither on the front line, the overlap pixel, nor fill.
    lat_i, lon_i = 40 + 2, 40 + 1
    pixel = y[lat_i, lon_i, :len(dataset.CLASS_NAMES)]
    assert pixel[none_class] == 1.0


def test_make_y_overlap_resolves_to_cold_by_front_types_priority(codsus_time0):
    """make_y resolves overlapping classes to the *first* True in
    config.FRONT_TYPES order -- i.e. the loop in make_y walks FRONT_TYPES
    from last to first and lets the earliest-indexed class win (it's written
    last). cold is index 0 in FRONT_TYPES, so cold beats warm here."""
    fronts_t, valid_t = codsus_time0
    y = dataset.make_y(fronts_t, valid_t)
    cold_class = dataset.CLASS_NAMES.index("cold")
    pixel = y[OVERLAP_LAT + 2, OVERLAP_LON + 1, :len(dataset.CLASS_NAMES)]
    assert pixel[cold_class] == 1.0
    assert pixel.sum() == 1.0


def test_make_y_weight_channel_zero_at_fill_and_padding(codsus_time0):
    fronts_t, valid_t = codsus_time0
    y = dataset.make_y(fronts_t, valid_t)
    w = y[..., -1]

    assert w[FILL_LAT + 2, FILL_LON + 1] == 0.0

    assert np.all(w[:2, :, ] == 0.0)
    assert np.all(w[2 + config.GRID_SHAPE[0]:, :] == 0.0)
    assert np.all(w[:, :1] == 0.0)
    assert np.all(w[:, 1 + config.GRID_SHAPE[1]:] == 0.0)

    # everywhere else (unpadded, non-fill) the weight is 1
    lat0, lat1 = 2, 2 + config.GRID_SHAPE[0]
    lon0, lon1 = 1, 1 + config.GRID_SHAPE[1]
    interior = w[lat0:lat1, lon0:lon1].copy()
    interior[FILL_LAT, FILL_LON] = 1.0  # exclude the one known-zero pixel
    assert np.all(interior == 1.0)


# --------------------------------------------------------------------------- #
# 5. dataset.year_samples
# --------------------------------------------------------------------------- #

def test_year_samples_yields_all_timesteps_and_skips_missing_day(patched_dirs, tmp_path):
    year = 2003
    day1 = pd.Timestamp(f"{year}-01-01")
    day1_times = pd.date_range(day1, periods=8, freq="3h")
    day2 = pd.Timestamp(f"{year}-01-02")
    day2_times = pd.date_range(day2, periods=1, freq="3h")  # no MERRA-2 file
    times = day1_times.append(day2_times)

    rng = np.random.default_rng(5)
    _write_merra2_day(day1, rng)
    # deliberately do NOT write day2's MERRA-2 file
    assert not day_path(day2).exists()
    _write_codsus_year(year, times, masked=True)

    stats = dataset.compute_norm_stats(years=[year], step_days=1,
                                       path=tmp_path / "stats.json")

    samples = list(dataset.year_samples(year, stats, winds=True))
    assert len(samples) == 8  # day2's lone timestep is skipped, not errored

    names = dataset.channel_names(winds=True)
    for x, y in samples:
        assert x.shape == (*config.PADDED_SHAPE, len(LEVS), len(names))
        assert y.shape == (*config.PADDED_SHAPE, len(dataset.CLASS_NAMES) + 1)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))


def test_year_samples_front_line_present_in_every_yielded_label(patched_dirs, tmp_path):
    year = 2005
    day1 = pd.Timestamp(f"{year}-01-01")
    times = pd.date_range(day1, periods=8, freq="3h")
    rng = np.random.default_rng(6)
    _write_merra2_day(day1, rng)
    _write_codsus_year(year, times, masked=True)
    stats = dataset.compute_norm_stats(years=[year], step_days=1,
                                       path=tmp_path / "stats.json")

    cold_class = dataset.CLASS_NAMES.index("cold")
    for _, y in dataset.year_samples(year, stats, winds=True):
        for lat_i, lon_i in LINE_PIXELS:
            assert y[lat_i + 2, lon_i + 1, cold_class] == 1.0
