"""dl_front dataset/config tests with analytic answers (no TF, no data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dl_front import config, dataset
from dl_front.acquire_merra2_sfc import PHYSICAL_BOUNDS, bicubic_to_label_grid, is_physical


# --------------------------------------------------------------------------- #
# Bicubic remap
# --------------------------------------------------------------------------- #

def _native_ds(fields: dict) -> xr.Dataset:
    lat = np.arange(8.0, 79.01, 0.5)
    lon = np.arange(-173.75, -28.74, 0.625)
    time = pd.date_range("2003-01-02", periods=2, freq="3h")
    data = {}
    for name, f in fields.items():
        vals = f(lat[:, None], lon[None, :])
        data[name] = (("time", "lat", "lon"),
                      np.broadcast_to(vals, (2, *vals.shape)).copy())
    return xr.Dataset(data, coords={"time": time, "lat": lat, "lon": lon})


def test_bicubic_exact_on_cubic_polynomial():
    """A bicubic spline reproduces any cubic polynomial exactly."""
    poly = lambda la, lo: (la ** 3 / 1e4 + 2 * la - 0.5 * lo ** 2
                           + lo ** 3 / 1e5 + la * lo / 10)
    ds = _native_ds({v: poly for v in config.SFC_VARS})
    out = bicubic_to_label_grid(ds)
    la = np.asarray(config.LABEL_LATS)[:, None]
    lo = np.asarray(config.LABEL_LONS)[None, :]
    np.testing.assert_allclose(out["T2M"].isel(time=0).values, poly(la, lo),
                               rtol=1e-5)
    assert out["T2M"].shape == (2, *config.GRID_SHAPE)


def test_label_grid_matches_codsus():
    assert config.LABEL_LATS[0] == 10.0 and config.LABEL_LATS[-1] == 77.0
    assert config.LABEL_LONS[0] == -171.0 and config.LABEL_LONS[-1] == -31.0
    assert (len(config.LABEL_LATS), len(config.LABEL_LONS)) == config.GRID_SHAPE


def test_native_slices_cover_label_domain_with_margin():
    """Spline support must extend >=2 native cells past every label edge."""
    la0, la1 = config.MERRA2_SFC_LAT_SLICE
    lo0, lo1 = config.MERRA2_SFC_LON_SLICE
    lat = -90 + 0.5 * np.array([la0, la1])
    lon = -180 + 0.625 * np.array([lo0, lo1])
    assert lat[0] <= 10.0 - 1.0 and lat[1] >= 77.0 + 1.0
    assert lon[0] <= -171.0 - 1.25 and lon[1] >= -31.0 + 1.25


def test_is_physical_rejects_corruption():
    ds = _native_ds({v: (lambda la, lo: np.full((la + lo).shape, 290.0))
                     for v in config.SFC_VARS})
    ds["SLP"] += 100_000.0 - 290.0
    ds["QV2M"] *= 0.01 / 290.0
    ds["U10M"] -= 285.0
    ds["V10M"] -= 285.0
    assert is_physical(ds)
    bad = ds.copy(deep=True)
    bad["T2M"][0, 0, 0] = np.float32(3.4e38)          # garbage-slab signature
    assert not is_physical(bad)
    nan = ds.copy(deep=True)
    nan["SLP"][0, 0, 0] = np.nan                      # single-level: no fill
    assert not is_physical(nan)


# --------------------------------------------------------------------------- #
# Labels -> class grid
# --------------------------------------------------------------------------- #

def _label_ds(n_classes: int, hits: dict) -> xr.Dataset:
    names = dataset.class_names(n_classes)
    k = len(names) if n_classes == 6 else 5           # CODSUS files carry none
    fronts = np.zeros((1, k, *config.GRID_SHAPE), dtype=np.float32)
    for name, (i, j) in hits.items():
        fronts[0, list(names).index(name), i, j] = 1
    return xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": [np.datetime64("2010-01-01")],
                "lat": list(config.LABEL_LATS), "lon": list(config.LABEL_LONS),
                "front_type": ("front", list(names)[:k])})


def test_class_grid_priority():
    """Overlapping channels resolve warm > occluded > stationary > cold."""
    ds = _label_ds(5, {"cold": (3, 4), "warm": (3, 4), "stationary": (5, 6)})
    cls = dataset.class_grid(ds, 5)
    names = dataset.class_names(5)
    assert cls[0, 3, 4] == names.index("warm")
    assert cls[0, 5, 6] == names.index("stationary")
    assert cls[0, 0, 0] == names.index("none")
    assert cls.dtype == np.uint8


def test_class_grid_dryline():
    ds = _label_ds(6, {"dryline": (10, 20), "cold": (10, 21)})
    cls = dataset.class_grid(ds, 6)
    names = dataset.class_names(6)
    assert cls[0, 10, 20] == names.index("dryline")
    assert cls[0, 10, 21] == names.index("cold")
    assert (cls == names.index("none")).sum() == 68 * 141 - 2


def test_valid_label_steps_drops_fill_padding():
    """NOAA year files pad missing analyses as all-LABEL_FILL steps; those
    must be excluded from training/scoring (fill != "no front")."""
    ds = _label_ds(6, {"cold": (3, 4)})
    good = ds["fronts"].values
    padded = np.full_like(good, 2)                    # LABEL_FILL everywhere
    outside_only = np.zeros_like(good)
    outside_only[..., 0, 0] = 2                       # (10N, 171W): off-mask
    fronts = np.concatenate([good, padded, outside_only])
    lab = xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": pd.date_range("2010-01-01", periods=3, freq="3h"),
                "front_type": ("front", list(dataset.class_names(6)))})
    keep = dataset.valid_label_steps(lab, 6)
    np.testing.assert_array_equal(keep, [True, False, True])
    np.testing.assert_array_equal(dataset.valid_label_steps(lab),
                                  [True, False, True])   # 5-class branch


def test_valid_label_steps_6class_guards_full_crop_domain():
    """Review 2026-08-13: 274 crop_domain() pixels (the 6-class stage-A
    loss mask) lie outside the region mask; fill confined to that band
    must invalidate a 6-class step even though the 5-class guard keeps it."""
    crop_only = dataset.crop_domain() & ~dataset.region_mask().astype(bool)
    assert crop_only.any()                     # the guard/mask gap is real
    i, j = map(int, np.argwhere(crop_only)[0])
    ds = _label_ds(6, {"cold": (3, 4)})
    fill_step = np.zeros_like(ds["fronts"].values)
    fill_step[..., i, j] = 2                   # LABEL_FILL in the gap band
    lab = xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fill_step)},
        coords={"time": pd.date_range("2010-01-01", periods=1, freq="3h"),
                "front_type": ("front", list(dataset.class_names(6)))})
    np.testing.assert_array_equal(dataset.valid_label_steps(lab, 6), [False])
    np.testing.assert_array_equal(dataset.valid_label_steps(lab, 5), [True])


def test_fold_split_paper_semantics():
    """3 folds: each validates on a disjoint third, trains on the rest."""
    all_val = []
    for fold in range(config.N_FOLDS):
        tr, va = dataset.fold_split(300, fold)
        assert len(tr) == 200 and len(va) == 100
        assert not set(tr) & set(va)
        all_val.append(set(va))
    assert set().union(*all_val) == set(range(300))
    with pytest.raises(ValueError):
        dataset.fold_split(300, 3)


def test_sfc_x_standardizes():
    time = pd.date_range("2003-01-02", periods=1, freq="3h")
    day = xr.Dataset(
        {v: (("time", "lat", "lon"),
             np.full((1, *config.GRID_SHAPE), 10.0 * (i + 1), np.float32))
         for i, v in enumerate(config.SFC_VARS)},
        coords={"time": time})
    stats = {v: [10.0 * (i + 1) - 2.0, 4.0]
             for i, v in enumerate(config.SFC_VARS)}
    x = dataset.sfc_x(day, stats)
    assert x.shape == (1, *config.GRID_SHAPE, 5)
    np.testing.assert_allclose(x, 0.5)


def test_region_mask_shape_binary():
    m = dataset.region_mask()
    assert m.shape == config.GRID_SHAPE
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert 0 < m.sum() < m.size          # nontrivial mask


# --------------------------------------------------------------------------- #
# Analysis / crop domain (6-class track, user decision 2026-08-13)
# --------------------------------------------------------------------------- #

def _grid_index(lat: float, lon: float) -> tuple[int, int]:
    return (list(config.LABEL_LATS).index(lat),
            list(config.LABEL_LONS).index(lon))


def _write_lsm(path, fill: float = 1.0, holes=()):
    """A synthetic global land mask on the real file's half-degree centers.

    ``holes``: (lat, lon, value) triples painted onto the constant fill --
    on a half-degree-center grid an integer label point sits exactly between
    4 cells, so setting all 4 neighbours makes the bilinear interp exact.
    """
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    lsm = np.full((len(lat), len(lon)), fill)
    for la, lo, val in holes:
        rows = np.abs(lat - la) < 1.0        # the 2 centers straddling la
        cols = np.abs(lon - lo) < 1.0
        lsm[np.ix_(rows, cols)] = val
    xr.Dataset({"lsm": (("lat", "lon"), lsm)},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)


def test_halo_px_derived_from_config(monkeypatch):
    """(N_CONV_LAYERS + 1) * (KERNEL_SIZE // 2); +1 = the head conv."""
    assert dataset.halo_px() == 8            # (3 + 1) * (5 // 2)
    monkeypatch.setattr(config, "N_CONV_LAYERS", 2)
    monkeypatch.setattr(config, "KERNEL_SIZE", 3)
    assert dataset.halo_px() == 3            # (2 + 1) * 1
    # even kernel (review 2026-08-13): TF 'same' pads k // 2 at the end,
    # so per-layer reach is k // 2, not (k - 1) // 2
    monkeypatch.setattr(config, "KERNEL_SIZE", 4)
    assert dataset.halo_px() == 6            # (2 + 1) * 2


def test_analysis_domain_box_bounds_inclusive(tmp_path, monkeypatch):
    """All-land lsm isolates the box: lat 32..53, lon -107..-64 INCLUSIVE."""
    _write_lsm(tmp_path / "lsm.nc", fill=1.0)
    monkeypatch.setattr(config, "LAND_MASK_PATH", tmp_path / "lsm.nc")
    dom = dataset.analysis_domain()
    assert dom.shape == config.GRID_SHAPE and dom.dtype == bool
    for lat, lon in [(32.0, -107.0), (32.0, -64.0),
                     (53.0, -107.0), (53.0, -64.0)]:       # corners in
        assert dom[_grid_index(lat, lon)]
    for lat, lon in [(31.0, -80.0), (54.0, -80.0),
                     (40.0, -108.0), (40.0, -63.0)]:       # 1 deg out
        assert not dom[_grid_index(lat, lon)]
    assert dom.sum() == (53 - 32 + 1) * (107 - 64 + 1)     # 22 x 44 all-land


def test_analysis_domain_land_threshold(tmp_path, monkeypatch):
    """Interpolated land fraction >= LAND_FRACTION_MIN gates in-box cells."""
    holes = [(40.0, -90.0, 0.4),             # below the 0.5 threshold: out
             (40.0, -80.0, 0.5),             # exactly at it: in (>=)
             (20.0, -90.0, 0.0)]             # outside the box: irrelevant
    _write_lsm(tmp_path / "lsm.nc", fill=1.0, holes=holes)
    monkeypatch.setattr(config, "LAND_MASK_PATH", tmp_path / "lsm.nc")
    dom = dataset.analysis_domain()
    assert not dom[_grid_index(40.0, -90.0)]
    assert dom[_grid_index(40.0, -80.0)]
    assert dom.sum() == (53 - 32 + 1) * (107 - 64 + 1) - 1


def test_crop_domain_is_box_plus_halo_no_land():
    """Box expanded by halo_px() on all sides; NO land/codsus intersection."""
    crop = dataset.crop_domain()
    h = dataset.halo_px()
    assert crop.shape == config.GRID_SHAPE and crop.dtype == bool
    # default: lat 24..61, lon -115..-56 -- inside the grid, no clipping
    assert crop[_grid_index(32.0 - h, -107.0 - h)]
    assert crop[_grid_index(53.0 + h, -64.0 + h)]
    assert not crop[_grid_index(32.0 - h - 1, -90.0)]
    assert not crop[_grid_index(40.0, -64.0 + h + 1)]
    assert crop.sum() == (22 + 2 * h) * (44 + 2 * h)       # 2280 at h=8
    # water pixels inside the box stay in the crop (no land intersection),
    # and the analysis domain is a strict subset of the crop
    assert (crop & ~dataset.analysis_domain()).any()
    assert (dataset.analysis_domain() <= crop).all()


def test_crop_domain_clips_to_grid(monkeypatch):
    """A box hugging the grid edge clips instead of wrapping or raising."""
    monkeypatch.setattr(config, "ANALYSIS_LAT_RANGE", (11.0, 20.0))
    monkeypatch.setattr(config, "ANALYSIS_LON_RANGE", (-170.0, -160.0))
    crop = dataset.crop_domain()
    h = dataset.halo_px()
    # southern/western halo hits the grid boundary: clipped, not wrapped
    assert crop[_grid_index(10.0, -171.0)]
    assert not crop[_grid_index(77.0, -31.0)]
    assert crop.sum() == (20.0 + h - 10.0 + 1) * (-160.0 + h + 171.0 + 1)


# --------------------------------------------------------------------------- #
# Kriged caches & hour filtering
# --------------------------------------------------------------------------- #

def test_filter_hours():
    """Only steps whose UTC hour is in ``hours`` survive, triple in sync."""
    times = pd.DatetimeIndex(["2010-01-01 18:00", "2010-01-01 21:00",
                              "2010-01-02 00:00", "2010-01-02 03:00"])
    x = np.arange(4, dtype=np.float16)[:, None, None, None]
    y = np.arange(4, dtype=np.uint8)[:, None, None]
    fx, fy, ft = dataset.filter_hours(x, y, times, config.AIRS_HOURS)
    np.testing.assert_array_equal(fx.ravel(), [0, 1, 2])
    np.testing.assert_array_equal(fy.ravel(), [0, 1, 2])
    assert list(ft) == list(times[:3])
    assert fx.dtype == np.float16 and fy.dtype == np.uint8


def _multi_time_label_ds(n_classes: int, times, hits: dict) -> xr.Dataset:
    """Like ``_label_ds`` but with several time steps: hits[t] = {name: (i,j)}."""
    names = dataset.class_names(n_classes)
    k = len(names) if n_classes == 6 else 5
    fronts = np.zeros((len(times), k, *config.GRID_SHAPE), dtype=np.float32)
    for t, front_hits in hits.items():
        for name, (i, j) in front_hits.items():
            fronts[list(times).index(t), list(names).index(name), i, j] = 1
    return xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": pd.DatetimeIndex(times),
                "lat": list(config.LABEL_LATS), "lon": list(config.LABEL_LONS),
                "front_type": ("front", list(names)[:k])})


def _write_kriged_year(dirpath, year, times, value: float):
    """A tiny kriged cache in the krige_fill schema (constant fields)."""
    shape = (len(times), *config.GRID_SHAPE)
    data = {v: (("time", "lat", "lon"), np.full(shape, value, np.float32))
            for v in config.SFC_VARS}
    data["valid_frac"] = (("time", "lat", "lon"), np.ones(shape, np.float32))
    ds = xr.Dataset(data, coords={"time": pd.DatetimeIndex(times),
                                  "lat": list(config.LABEL_LATS),
                                  "lon": list(config.LABEL_LONS)},
                    attrs={"source": "degraded_reanalysis",
                           "variogram_model": "linear",
                           "max_obs_points": 1500, "created": "test",
                           # the loader refuses caches without the v3 stamp
                           # or with mismatched domain provenance
                           # (domain decision 2026-08-13)
                           "schema_version": 3,
                           "domain_lat_range": list(config.ANALYSIS_LAT_RANGE),
                           "domain_lon_range": list(config.ANALYSIS_LON_RANGE),
                           "land_fraction_min": config.LAND_FRACTION_MIN,
                           "halo_px": dataset.halo_px(),
                           "swath_bank": "per-day-envelope"})
    dirpath.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dirpath / f"kriged_sfc_{year}.nc")


def test_kriged_year_arrays_inner_join_and_zscore(tmp_path, monkeypatch):
    """Kriged steps pair with labels by exact timestamp; frozen-stats z-score."""
    label_times = pd.DatetimeIndex(["2010-01-01 18:00", "2010-01-01 21:00",
                                    "2010-01-02 00:00"])
    lab = _multi_time_label_ds(6, label_times,
                               {label_times[0]: {"cold": (3, 4)},
                                label_times[2]: {"dryline": (10, 20)}})
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: lab.copy(deep=True))
    monkeypatch.setattr(   # manifest reorg 2026-08-13: full paths
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    # kriged file misses the 21Z label step and adds an unlabeled 03Z step
    kriged_times = pd.DatetimeIndex(["2010-01-01 18:00", "2010-01-02 00:00",
                                     "2010-01-02 03:00"])
    _write_kriged_year(tmp_path / "degraded_reanalysis", 2010, kriged_times,
                       value=12.0)
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}   # (12-10)/4 = 0.5

    x, y, times = dataset.kriged_year_arrays(2010, 6, stats,
                                             "kriged-degraded")
    assert list(times) == [label_times[0], label_times[2]]   # inner join
    assert x.shape == (2, *config.GRID_SHAPE, 5) and x.dtype == np.float16
    np.testing.assert_allclose(x.astype(np.float32), 0.5)
    assert y.dtype == np.uint8
    names = dataset.class_names(6)
    assert y[0, 3, 4] == names.index("cold")
    assert y[1, 10, 20] == names.index("dryline")
    assert (y[0] == names.index("none")).sum() == 68 * 141 - 1


def test_kriged_year_arrays_unknown_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(   # manifest reorg 2026-08-13: full paths
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    with pytest.raises(KeyError):
        dataset.kriged_year_arrays(2010, 6, {}, "reanalysis")


def test_physical_bounds_cover_observed_ranges():
    # Sea-level pressure record lows (~870 hPa, typhoons) and highs
    # (~1084 hPa, Siberian high) must not be flagged as corruption.
    lo, hi = PHYSICAL_BOUNDS["SLP"]
    assert lo <= 87_000.0 and hi >= 108_400.0
