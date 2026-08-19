"""dl_front.airs_fcst + dl_front.krige_fill: kriged AIRS-FCST pipeline tests.

Numeric pieces (kriging fill, degenerate fallbacks) use synthetic fields
with analytic answers; the reader/end-to-end pieces run against the real
demo fullgrid day (2019-06-05) and skip cleanly where the data is absent.
Kriging solves are O(n^3) in the obs count, so data-driven tests shrink
config.KRIGE_MAX_OBS via monkeypatch to keep the suite fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dl_front import airs_fcst, config, dataset, krige_fill
from dl_front.acquire_merra2_sfc import day_path

DEMO_ROOT = config.REPO_ROOT / "data/HYSPLIT_demo"
DEMO_DAY = pd.Timestamp("2019-06-05")

needs_demo = pytest.mark.skipif(
    airs_fcst.find_fullgrid(DEMO_DAY, root=DEMO_ROOT) is None,
    reason="demo fullgrid day not on disk")

#: ``dataset.crop_domain()`` interpolates the land-fraction mask off disk, so
#: even fully synthetic cache tests need the mask file from the data root.
needs_land_mask = pytest.mark.skipif(
    not config.LAND_MASK_PATH.exists(),
    reason=f"land mask {config.LAND_MASK_PATH} not on disk "
           f"(set JPL_AIRS_DATA to a populated data root)")


def _smooth_field(shape=config.GRID_SHAPE) -> np.ndarray:
    """A smooth synthetic 'weather field': large-scale gradient + waves.

    The dominant trend makes the empirical variogram near-linear, which is
    what pykrige's DEFAULT automatic fit (frozen spec) assumes; a purely
    periodic field would be mis-fit with a large nugget and reconstruct
    poorly regardless of implementation correctness.
    """
    la, lo = np.meshgrid(np.linspace(0, np.pi, shape[0]),
                         np.linspace(0, np.pi, shape[1]), indexing="ij")
    return la + lo + 0.3 * np.sin(2 * la) * np.cos(2 * lo)


# --------------------------------------------------------------------------- #
# krige_fill
# --------------------------------------------------------------------------- #

def test_krige_fill_reconstructs_smooth_field(monkeypatch):
    monkeypatch.setattr(config, "KRIGE_MAX_OBS", 300)
    truth = _smooth_field()
    rng = np.random.default_rng(0)
    gappy = np.where(rng.random(truth.shape) < 0.6, np.nan, truth)  # 60% gaps
    observed = np.isfinite(gappy)

    out = krige_fill.krige_fill(gappy, rng=np.random.default_rng(1))
    assert not np.isnan(out).any()
    # observed pixels pass through bit-identical
    np.testing.assert_array_equal(out[observed], truth[observed])
    rmse = np.sqrt(np.mean((out[~observed] - truth[~observed]) ** 2))
    assert rmse < 0.1 * truth.std()         # well below the field's variance


def test_krige_fill_degenerate_few_obs():
    field = np.full(config.GRID_SHAPE, np.nan)
    field[10, 10], field[20, 20], field[30, 30] = 1.0, 2.0, 3.0
    out = krige_fill.krige_fill(field)      # < 10 obs -> observed-mean fill
    assert out[10, 10] == 1.0 and out[40, 40] == 2.0
    assert not np.isnan(out).any()

    empty = np.full(config.GRID_SHAPE, np.nan)
    assert np.isnan(krige_fill.krige_fill(empty)).all()   # untouched


def test_krige_fill_deterministic(monkeypatch):
    monkeypatch.setattr(config, "KRIGE_MAX_OBS", 200)
    truth = _smooth_field()
    gappy = np.where(np.random.default_rng(2).random(truth.shape) < 0.8,
                     np.nan, truth)
    when = pd.Timestamp("2019-06-05")
    a = krige_fill.krige_fill(gappy, rng=krige_fill._step_rng(when, 18, "T2M"))
    b = krige_fill.krige_fill(gappy, rng=krige_fill._step_rng(when, 18, "T2M"))
    c = krige_fill.krige_fill(gappy, rng=krige_fill._step_rng(when, 18, "QV2M"))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)          # channel enters the seed


# --------------------------------------------------------------------------- #
# airs_fcst: discovery & period fields
# --------------------------------------------------------------------------- #

@needs_demo
def test_find_fullgrid():
    # the demo tree has no YYYY/ layer and a doubled nested dir: only the
    # recursive fallback can find it
    p = airs_fcst.find_fullgrid(DEMO_DAY, root=DEMO_ROOT)
    assert p is not None and "20190605" in p.name
    assert airs_fcst.find_fullgrid("2019-06-06", root=DEMO_ROOT) is None


@needs_demo
@pytest.mark.parametrize("hour,expect_time", [
    (21, "2019-06-05 21:00"),               # uniform forecast slots only
    (0, "2019-06-06 00:00"),                # hour 0 = next-day 00 UTC
])
def test_period_fields(hour, expect_time):
    path = airs_fcst.find_fullgrid(DEMO_DAY, root=DEMO_ROOT)
    per = airs_fcst.period_fields(path, hour)
    assert pd.Timestamp(per["time"].values) == pd.Timestamp(expect_time)

    vf = per["valid_frac"].values
    assert vf.shape == config.GRID_SHAPE
    assert vf.min() >= 0.0 and vf.max() <= 1.0
    observed = vf >= airs_fcst.OBSERVED_MIN_FRACTION

    # plausible coverage over the CONUS-east fullgrid window ...
    lat, lon = (np.asarray(config.LABEL_LATS), np.asarray(config.LABEL_LONS))
    window = np.ix_((lat >= 26) & (lat <= 52), (lon >= -106) & (lon <= -65))
    assert 0.05 <= observed[window].mean() <= 0.7
    # ... and none at all outside it
    outside = np.ones(config.GRID_SHAPE, bool)
    outside[window] = False
    assert not observed[outside].any()

    for var in ("T2M", "QV2M", "U10M", "V10M"):
        grid = per[var].values
        assert grid.shape == config.GRID_SHAPE and grid.dtype == np.float32
        # values exactly where observed, NaN elsewhere
        assert (np.isfinite(grid) == observed).all()
    assert (230 < per["T2M"].values[observed]).all()
    assert (per["T2M"].values[observed] < 330).all()
    assert (0 <= per["QV2M"].values[observed]).all()
    assert (per["QV2M"].values[observed] < 0.03).all()


# --------------------------------------------------------------------------- #
# End-to-end: build_airs on the demo day
# --------------------------------------------------------------------------- #

@needs_demo
def test_build_airs_demo_day(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AIRS_FCST_ROOT", DEMO_ROOT)
    monkeypatch.setattr(config, "KRIGE_MAX_OBS", 150)   # speed
    if not day_path(DEMO_DAY).exists():     # SLP source; real file preferred
        slp = xr.Dataset(
            {"SLP": (("time", "lat", "lon"),
                     np.full((2, *config.GRID_SHAPE), 101325.0, np.float32))},
            coords={"time": [DEMO_DAY + pd.Timedelta(hours=21),
                             DEMO_DAY + pd.Timedelta(hours=23)],
                    "lat": np.asarray(config.LABEL_LATS),
                    "lon": np.asarray(config.LABEL_LONS)})
        monkeypatch.setattr(
            krige_fill, "_reanalysis_step",
            lambda when: slp.sel(time=when) if when.hour != 0 else
            slp.isel(time=0).drop_vars("time"))

    written = krige_fill.build_airs([2019], workers=1, out_dir=tmp_path)
    assert written == [tmp_path / "kriged_airs_fcst/kriged_sfc_2019.nc"]

    with xr.open_dataset(written[0]) as ds:
        ds = ds.load()
    assert ds.attrs["source"] == "airs_fcst"
    assert ds.attrs["variogram_model"] == config.KRIGE_VARIOGRAM
    assert ds.attrs["max_obs_points"] == 150
    assert "created" in ds.attrs
    assert ds.attrs["schema_version"] == 4
    # the resolved domain decision travels in the cache (schema v4)
    assert list(ds.attrs["domain_lat_range"]) == list(config.ANALYSIS_LAT_RANGE)
    assert list(ds.attrs["domain_lon_range"]) == list(config.ANALYSIS_LON_RANGE)
    assert ds.attrs["land_fraction_min"] == config.LAND_FRACTION_MIN
    assert ds.attrs["halo_px"] == dataset.halo_px()
    assert ds.attrs["swath_bank"] in (str(config.SWATH_BANK_PATH),
                                      "per-day-envelope")
    assert list(ds["time"].values) == list(
        pd.to_datetime(["2019-06-05 21:00", "2019-06-06 00:00"]))
    assert tuple(ds.sizes[d] for d in ("time", "lat", "lon")) == (2,) + config.GRID_SHAPE
    # schema v4: SFC_VARS gap-free INSIDE the crop (box + halo), NaN outside
    crop = dataset.crop_domain()
    for var in config.SFC_VARS:
        assert ds[var].dtype == np.float32
        vals = ds[var].values
        assert not np.isnan(vals[:, crop]).any()        # in-crop gap-free
        assert np.isnan(vals[:, ~crop]).all()           # nothing out-of-crop
    vf = ds["valid_frac"].values
    assert vf.dtype == np.float32 and vf.min() >= 0.0 and vf.max() <= 1.0
    assert 0.0 < (vf >= airs_fcst.OBSERVED_MIN_FRACTION).mean() < 0.12
    # gap_type: int8, only the four config.GAP_* codes, -1 exactly
    # outside the crop, and observed pixels inside the crop flagged 0
    gt = ds["gap_type"].values
    assert gt.dtype == np.int8
    assert set(np.unique(gt)) <= {config.GAP_OUT_OF_DOMAIN,
                                  config.GAP_OBSERVED, config.GAP_CLOUD,
                                  config.GAP_OUT_OF_SWATH}
    assert (gt[:, ~crop] == config.GAP_OUT_OF_DOMAIN).all()
    observed = (vf >= airs_fcst.OBSERVED_MIN_FRACTION) & crop
    assert (gt[observed] == config.GAP_OBSERVED).all()
    assert (gt[:, crop] != config.GAP_OUT_OF_DOMAIN).all()
    # kriged fields keep the observed pixels' physical range (in-crop)
    t2m = ds["T2M"].values[:, crop]
    assert 230 < t2m.min() and t2m.max() < 330


@needs_demo
def test_terrain_following_surface_covers_elevated_west():
    """Regression (post-mortem 2026-08-16): the fixed 985-hPa surface target
    had ZERO coverage over all elevated terrain (the high plains never have
    985-hPa air), so the dryline region looked permanently out-of-swath.
    The terrain-following extraction must (a) still saturate valid_frac at
    1.0 where fully surrounded, and (b) actually observe the elevated west
    on the demo day, where per-level checks show 62-70% raw coverage."""
    from dl_front import dataset

    path = airs_fcst.find_fullgrid(DEMO_DAY, root=DEMO_ROOT)
    per = airs_fcst.period_fields(path, 21)
    vf = per["valid_frac"].values
    assert vf.max() == 1.0                     # fully-surrounded pixels exist
    assert (vf > airs_fcst.OBSERVED_MIN_FRACTION).any()
    domain = dataset.analysis_domain()
    lons = np.asarray(config.LABEL_LONS)
    west = domain & (lons[None, :] < -95)
    obs = vf >= airs_fcst.OBSERVED_MIN_FRACTION
    assert (obs & west).sum() / west.sum() > 0.3   # was exactly 0.0 pre-fix
    # extrapolated fields stay physical over the elevated terrain
    t = per["T2M"].values
    assert np.nanmin(t[west]) > 230 and np.nanmax(t[west]) < 330


@needs_demo
def test_select_slot_forecast_slots_only():
    """User decision 2026-08-15 (forecast window only): slot 0 -- the
    overpass, whose per-pixel obs times span ~2.6 h -- is never selected;
    hours resolve to the uniform forecast slots, and an hour with no
    forecast slot (18Z, the old overpass label hour) raises (callers
    skip-with-note)."""
    path = airs_fcst.find_fullgrid(DEMO_DAY, root=DEMO_ROOT)
    ds = airs_fcst.load_fullgrid(path)
    slot, when = airs_fcst._select_slot(ds, path, 21)
    assert slot >= 1                          # never the overpass slot
    assert when == pd.Timestamp("2019-06-05 21:00")
    slot0, when0 = airs_fcst._select_slot(ds, path, 0)
    assert slot0 >= 1
    assert when0 == pd.Timestamp("2019-06-06 00:00")
    with pytest.raises(ValueError, match="no forecast slot"):
        airs_fcst._select_slot(ds, path, 18)


def test_classify_step_gaps_bank_draw_uses_own_envelope(tmp_path,
                                                        monkeypatch):
    """Regression (review 2026-08-13): gap-bank draws are DONOR-date swaths
    at a different 16-day cycle position; classifying them against the
    requested date's climatological footprint labeled the drawn swath's
    cloud holes out-of-swath and the empty footprint cloud."""
    from dl_front import swath

    date = pd.Timestamp("2019-06-05")
    cyc = swath.cycle_day(date)
    # climatological footprint rows 10:20 cols 10:60 -- far from the draw
    freq = np.zeros((swath.CYCLE_DAYS, 1, *config.GRID_SHAPE), np.float32)
    freq[cyc, 0, 10:20, 10:60] = 1.0
    n_days = np.zeros((swath.CYCLE_DAYS, 1), np.int32)
    n_days[cyc, 0] = 10
    bank_path = tmp_path / "swath_bank.npz"
    np.savez_compressed(bank_path, freq=freq, n_days=n_days,
                        hours=np.asarray([18]), years=np.asarray([2019]))
    monkeypatch.setattr(config, "SWATH_BANK_PATH", bank_path)

    # the drawn (donor) mask: swath rows 40:50 cols 80:100, one cloud hole
    valid_frac = np.zeros(config.GRID_SHAPE, np.float32)
    valid_frac[40:50, 80:100] = 1.0
    valid_frac[44:46, 88:91] = 0.0
    domain = np.ones(config.GRID_SHAPE, bool)

    gt = krige_fill._classify_step_gaps(date, 18, valid_frac, domain,
                                        used_bank=True)
    assert gt[42, 85] == config.GAP_OBSERVED
    assert gt[45, 89] == config.GAP_CLOUD           # hole in the DRAWN swath
    assert gt[15, 30] == config.GAP_OUT_OF_SWATH    # empty climat. footprint
    # a real-fullgrid mask (used_bank=False) still prefers the climatology:
    # the same pixels come out scrambled, which is exactly why bank draws
    # must not take this path
    gt_clim = krige_fill._classify_step_gaps(date, 18, valid_frac, domain,
                                             used_bank=False)
    assert gt_clim[45, 89] == config.GAP_OUT_OF_SWATH
    assert gt_clim[15, 30] == config.GAP_CLOUD


def test_observed_min_fraction_is_the_yaml_tunable():
    """airs_fcst must consume degradation.observed_min_fraction, not a local
    hardcode, so a JPL_DLFRONT_CONFIG override changes every gap path."""
    assert airs_fcst.OBSERVED_MIN_FRACTION == config.OBSERVED_MIN_FRACTION


def test_build_airs_empty_year_writes_empty_cache(tmp_path, monkeypatch):
    """A year with no fullgrid coverage in an otherwise NON-empty archive
    still gets a (time=0) cache file: it is the resume/done marker, and
    downstream must not crash on it.  (A fully empty archive instead trips
    the misconfigured-root guard -- see test_build_refuses_empty_archive,
    review 2026-08-13.)"""
    empty_root = tmp_path / "no_archive"
    # decoy coverage for ANOTHER year: the archive is sparse, not empty
    decoy = empty_root / "2019/wrf27km_20190605"
    decoy.mkdir(parents=True)
    (decoy / "fullgrid_wrf27km_GOOD_1p00deg_20190605_0000-0000.nc").touch()
    monkeypatch.setattr(config, "AIRS_FCST_ROOT", empty_root)
    written = krige_fill.build_airs([2018], max_days=2, out_dir=tmp_path)
    assert written == [tmp_path / "kriged_airs_fcst/kriged_sfc_2018.nc"]
    with xr.open_dataset(written[0]) as ds:
        assert ds.sizes["time"] == 0
        assert set(config.SFC_VARS) | {"valid_frac", "gap_type"} \
            <= set(ds.data_vars)
        assert ds["gap_type"].dtype == np.int8      # empty year, v4 schema
        assert ds.attrs["source"] == "airs_fcst"
        assert ds.attrs["schema_version"] == 4

    # rerun resumes: the existing year cache is skipped, not rebuilt
    mtime = written[0].stat().st_mtime_ns
    again = krige_fill.build_airs([2018], max_days=2, out_dir=tmp_path)
    assert again == written
    assert written[0].stat().st_mtime_ns == mtime
    # --force rebuilds
    krige_fill.build_airs([2018], max_days=2, out_dir=tmp_path, force=True)
    assert written[0].stat().st_mtime_ns != mtime


def test_build_refuses_empty_archive(tmp_path, monkeypatch):
    """Zero fullgrid files under AIRS_FCST_ROOT means a misconfigured root
    (e.g. a typo'd JPL_AIRS_FCST export), not a sparse archive: both build
    flavors must raise BEFORE writing any empty done-marker cache that
    skip_krige would treat as permanently complete (review 2026-08-13)."""
    for root in (tmp_path / "empty_dir", tmp_path / "does_not_exist"):
        if root.name == "empty_dir":
            root.mkdir()
        monkeypatch.setattr(config, "AIRS_FCST_ROOT", root)
        for build in (krige_fill.build_airs, krige_fill.build_degraded):
            with pytest.raises(RuntimeError, match="no fullgrid"):
                build([2018], max_days=1, out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()      # nothing was written


def _write_sfc_gap_bank(path, n, rng_seed=0):
    """A synthetic sfc_gap_bank npz with n fields across months and hours."""
    import numpy as _np
    rng = _np.random.default_rng(rng_seed)
    vf = rng.random((n, *config.GRID_SHAPE)).astype(_np.float16)
    months = (_np.arange(n) % 12) + 1
    dates = _np.asarray([f"2016-{m:02d}-15" for m in months])
    hours = _np.asarray([(21, 0)[i % 2] for i in range(n)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        _np.savez_compressed(fh, vf=vf, date=dates, hour=hours)
    return vf, dates, hours


def test_build_degraded_refuses_tiny_gap_bank(tmp_path, monkeypatch):
    """A tiny surface gap bank would give every degraded step near-identical
    gap geometry; the builder must fail loudly unless --allow-small-bank."""
    from dl_front import swath

    bank_path = tmp_path / "sfc_gap_bank.npz"
    _write_sfc_gap_bank(bank_path, n=3)
    monkeypatch.setattr(config, "SFC_GAP_BANK_PATH", bank_path)
    monkeypatch.setattr(swath, "_SFC_GAP_CACHE", {})
    empty_root = tmp_path / "no_archive"
    empty_root.mkdir()
    monkeypatch.setattr(config, "AIRS_FCST_ROOT", empty_root)
    with pytest.raises(RuntimeError, match="SFC_GAP_MIN_BANK"):
        krige_fill._gap_valid_frac(pd.Timestamp("2010-06-01"), 21,
                                   allow_small_bank=False)
    mask, note, used_bank = krige_fill._gap_valid_frac(
        pd.Timestamp("2010-06-01"), 21, allow_small_bank=True)
    assert mask.shape == config.GRID_SHAPE
    # a missing fullgrid file is a bank draw and must SAY so (review
    # 2026-08-13): the note quantifies donor-geometry steps downstream
    assert used_bank and "bank mask used" in note


def test_column_lapse_recovers_linear_profile():
    """The LSQ lapse fit must recover a synthetic linear T(z) exactly, fall
    back on single-bin columns, and clip to the configured bounds."""
    lev = np.array([805.0, 865.0, 925.0, 985.0])     # ascending pressure
    alt = np.zeros((4, 1, 3))
    alt[:, 0, :] = np.array([1800.0, 1200.0, 600.0, 100.0])[:, None]
    true_lapse = np.array([0.008, 0.008, 0.030])     # K/m; col 2 super-steep
    t = 300.0 - true_lapse[None, None, :] * alt
    n = np.ones((4, 1, 3))
    n[:3, 0, 1] = 0                                  # col 1: deepest bin only
    one = xr.Dataset(
        {"alt": (("level", "lat", "lon"), alt),
         "t": (("level", "lat", "lon"), t),
         "N": (("level", "lat", "lon"), n)},
        coords={"level": lev, "lat": [40.0], "lon": [-100.0, -99.0, -98.0]})
    sub = (n > 0)
    idx = np.full((1, 3), 3)                          # deepest = last level
    at_deepest = lambda var: one[var].values[3]
    lapse = airs_fcst._column_lapse(one, sub, idx, at_deepest)
    np.testing.assert_allclose(lapse[0, 0], 0.008, rtol=1e-9)   # exact fit
    assert lapse[0, 1] == pytest.approx(
        config.AIRS_SURFACE_LAPSE_K_PER_KM / 1000.0)  # fallback: 1 bin
    hi = config.AIRS_SURFACE_LAPSE_CLIP_K_PER_KM[1] / 1000.0
    assert lapse[0, 2] == pytest.approx(hi)           # clipped at dry adiabat


def test_sfc_gap_bank_sampling_prefers_hour_and_season(tmp_path, monkeypatch):
    """sample_gap_field prefers same-hour entries within +-1 month, and the
    missing-bank case raises with the build command in the message."""
    from dl_front import swath

    bank_path = tmp_path / "sfc_gap_bank.npz"
    vf, dates, hours = _write_sfc_gap_bank(bank_path, n=48)
    monkeypatch.setattr(config, "SFC_GAP_BANK_PATH", bank_path)
    monkeypatch.setattr(swath, "_SFC_GAP_CACHE", {})
    rng = np.random.default_rng(1)
    months = np.asarray([int(d[5:7]) for d in dates])
    for _ in range(20):
        draw = swath.sample_gap_field(rng, month=6, hour=21)
        # the draw must be one of the same-hour, month 5-7 fields
        pool = np.flatnonzero((hours == 21)
                              & (np.minimum(abs(months - 6),
                                            12 - abs(months - 6)) <= 1))
        assert any(np.array_equal(draw, vf[i].astype(np.float32))
                   for i in pool)
    monkeypatch.setattr(config, "SFC_GAP_BANK_PATH",
                        tmp_path / "nope.npz")
    monkeypatch.setattr(swath, "_SFC_GAP_CACHE", {})
    with pytest.raises(FileNotFoundError, match="build-bank"):
        swath.sample_gap_field(rng)


# --------------------------------------------------------------------------- #
# Loader vs schema v4: out-of-crop imputation + in-crop corruption guard
# --------------------------------------------------------------------------- #

def _all_none_label_ds(times) -> xr.Dataset:
    """6-class label file with no fronts anywhere (every pixel 'none')."""
    names = dataset.class_names(6)
    fronts = np.zeros((len(times), len(names), *config.GRID_SHAPE),
                      np.float32)
    return xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": pd.DatetimeIndex(times),
                "lat": list(config.LABEL_LATS), "lon": list(config.LABEL_LONS),
                "front_type": ("front", list(names))})


def _write_v3_cache(dirpath, year, times, value, domain,
                    poison_pixel=None, schema_version=4,
                    domain_attrs=None, channel_values=None,
                    drop_kriged_attr=False) -> None:
    """A current-schema cache: constant in-crop fields, NaN out-of-crop.

    ``schema_version=None`` omits the attr entirely -- a v1 cache, for the
    loader's schema-guard regression test.  ``domain_attrs`` overrides the
    recorded domain provenance (default: the live config, i.e. a matching
    cache) for the loader's domain-mismatch regression test.
    ``channel_values`` overrides ``value`` for named channels, which is how
    the sourcing tests poison the cache's CLEAN channels with a sentinel the
    loader must never return (channel-sourcing decision 2026-08-18).
    ``drop_kriged_attr`` omits ``kriged_channels`` entirely, as the v3
    builders (written before that attr existed) did."""
    shape = (len(times), *config.GRID_SHAPE)
    grid = np.where(domain, np.float32(value), np.nan).astype(np.float32)
    if poison_pixel is not None:        # simulate a corrupt in-domain NaN
        grid[poison_pixel] = np.nan
    data = {v: (("time", "lat", "lon"), np.broadcast_to(grid, shape).copy())
            for v in config.SFC_VARS}
    for v, other in (channel_values or {}).items():
        data[v] = (("time", "lat", "lon"), np.broadcast_to(
            np.where(domain, np.float32(other), np.nan).astype(np.float32),
            shape).copy())
    data["valid_frac"] = (("time", "lat", "lon"), np.ones(shape, np.float32))
    data["gap_type"] = (("time", "lat", "lon"), np.broadcast_to(
        np.where(domain, config.GAP_OBSERVED,
                 config.GAP_OUT_OF_DOMAIN).astype(np.int8), shape).copy())
    attrs = {"source": "degraded_reanalysis",
             "variogram_model": "linear",
             "max_obs_points": 1500, "created": "test",
             "domain_lat_range": list(config.ANALYSIS_LAT_RANGE),
             "domain_lon_range": list(config.ANALYSIS_LON_RANGE),
             "land_fraction_min": config.LAND_FRACTION_MIN,
             "halo_px": dataset.halo_px(),
             "kriged_channels": [v for v in config.SFC_VARS
                                 if v in config.KRIGED_CHANNELS],
             "swath_bank": "per-day-envelope"}
    attrs.update(domain_attrs or {})
    if drop_kriged_attr:
        attrs.pop("kriged_channels")
    if schema_version is not None:
        attrs["schema_version"] = schema_version
    ds = xr.Dataset(data, coords={"time": pd.DatetimeIndex(times),
                                  "lat": list(config.LABEL_LATS),
                                  "lon": list(config.LABEL_LONS)},
                    attrs=attrs)
    dirpath.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dirpath / f"kriged_sfc_{year}.nc")


def test_loader_imputes_out_of_crop_to_standardized_zero(tmp_path,
                                                         monkeypatch):
    """Schema v4 NaN outside the crop -> x == 0.0 there after z-scoring
    (the standardized mean, degrade_sfc's gap-imputation convention).

    ``KRIGED_CHANNELS`` is widened to all five SFC_VARS for the duration so
    the whole of x comes from the synthetic cache: since the channel-sourcing
    decision 2026-08-18 the clean channels are read from the reanalysis day
    file instead, which would make this an assertion about sourcing (covered
    by test_loader_v3_cache_reads_clean_channels_from_reanalysis) rather than
    about the out-of-crop impute this test exists for.
    """
    monkeypatch.setattr(config, "KRIGED_CHANNELS", config.SFC_VARS)
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(   # manifest reorg 2026-08-13: full paths
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    crop = dataset.crop_domain()
    _write_v3_cache(tmp_path / "degraded_reanalysis", 2010, times,
                    value=12.0, domain=crop)
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}   # (12-10)/4 = 0.5

    x, y, out_times = dataset.kriged_year_arrays(2010, 6, stats,
                                                 "kriged-degraded")
    assert list(out_times) == list(times)
    x32 = x.astype(np.float32)
    assert np.isfinite(x32).all()
    np.testing.assert_allclose(x32[:, crop, :], 0.5)
    np.testing.assert_array_equal(x32[:, ~crop, :], 0.0)


@needs_land_mask
def test_loader_raises_on_in_crop_nan(tmp_path, monkeypatch):
    """NaN INSIDE the crop violates the schema -> loud ValueError."""
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(   # manifest reorg 2026-08-13: full paths
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    crop = dataset.crop_domain()
    iy, ix = map(int, np.argwhere(crop)[0])
    _write_v3_cache(tmp_path / "degraded_reanalysis", 2010, times,
                    value=12.0, domain=crop, poison_pixel=(iy, ix))
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}
    with pytest.raises(ValueError, match="INSIDE the crop domain"):
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-degraded")


@needs_land_mask
def test_loader_rejects_v1_and_v2_caches(tmp_path, monkeypatch):
    """Regression (review 2026-08-13) + domain decision 2026-08-13: a v1
    cache (full-grid fill, no schema_version attr) contains no NaN
    anywhere, so it would sail past the corrupt-cache guard; a v2 cache is
    kriged over the old region-mask domain, which does not cover the new
    box+halo crop.  The loader must refuse both and name the rebuild
    command.

    Still true after the channel-sourcing relaxation 2026-08-18: that made
    ``KRIGE_SCHEMA_READABLE == (3, 4)``, because v3 differs from v4 only in
    per-channel provenance, but v1 and v2 are genuine FORMAT breaks (grid
    extent) that no amount of per-channel re-sourcing can repair."""
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(   # manifest reorg 2026-08-13: full paths
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    # v1-like: values EVERYWHERE (domain=all-True -> no NaN), no attr
    _write_v3_cache(tmp_path / "degraded_reanalysis", 2010, times,
                    value=12.0, domain=np.ones(config.GRID_SHAPE, bool),
                    schema_version=None)
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}
    with pytest.raises(ValueError,
                       match=r"schema_version=None.*build-degraded"):
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-degraded")
    # a v2 (old region-mask domain) cache is refused just as loudly
    _write_v3_cache(tmp_path / "airs_fcst", 2010, times, value=12.0,
                    domain=dataset.region_mask().astype(bool),
                    schema_version=2)
    # (numpy >= 2 reprs the netCDF attr as np.int64(2), hence the loose match)
    with pytest.raises(ValueError, match=r"schema_version=.*2.*build-airs"):
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-airs")


#: A real MERRA-2 ``sfc_daily`` step, needed by the tests that exercise the
#: clean-channel sourcing path (the loader reads those channels off disk, so
#: a synthetic cache alone is not enough).  Skipped, never failed, where the
#: local data root does not carry the day.
REA_STEP = pd.Timestamp("2016-06-01 18:00")
needs_rea_day = pytest.mark.skipif(
    not day_path(REA_STEP).exists(),
    reason=f"MERRA-2 sfc_daily day file {day_path(REA_STEP)} not on disk")


def _rea_stats_and_step() -> tuple[dict, dict]:
    """Frozen-stats stand-in and the raw fields of :data:`REA_STEP`.

    Stats are derived from the day itself so the standardized reanalysis
    values land near N(0, 1): x is stored float16, whose ~1e-3 RELATIVE
    resolution would otherwise make an exact comparison against physical-unit
    SLP (~1e5 Pa over a unit-scale std) meaningless.
    """
    with xr.open_dataset(day_path(REA_STEP)) as day:
        day = day.load()
    j = int(np.flatnonzero(pd.DatetimeIndex(day["time"].values) == REA_STEP)[0])
    raw = {v: day[v].values[j].astype(np.float32) for v in config.SFC_VARS}
    stats = {v: [float(np.nanmean(raw[v])), float(np.nanstd(raw[v]))]
             for v in config.SFC_VARS}
    return stats, raw


@needs_rea_day
def test_loader_reads_v3_cache_at_full_width(tmp_path, monkeypatch):
    """Channel-sourcing decision 2026-08-18: a v3 cache is READABLE at the
    full five-channel width, with no rebuild.

    This reverses the previous guard (which refused every v3 cache outright).
    v3 differs from v4 only in that its U10M/V10M are kriged rather than
    clean reanalysis copies -- a per-CHANNEL provenance difference, not a
    format break -- and the loader no longer reads a cache's clean channels
    at all, so v3's wind copies are simply never touched.  Refusing the file
    would have forced a multi-hour rebuild of ~15 years of caches for a
    distinction that cannot reach the model.

    The v3 builders predate the ``kriged_channels`` attr, so the file here
    carries none: the split has to be recovered from
    ``config.KRIGE_LEGACY_KRIGED_CHANNELS`` rather than the file being
    refused for lack of provenance.
    """
    times = pd.DatetimeIndex([REA_STEP])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    assert config.INPUT_CHANNELS == config.SFC_VARS      # premise: full width
    assert 3 in config.KRIGE_SCHEMA_READABLE
    crop = dataset.crop_domain()
    stats, _ = _rea_stats_and_step()
    cache_value = {v: stats[v][0] + 2.0 * stats[v][1]    # z-scores to +2
                   for v in config.KRIGED_CHANNELS}
    _write_v3_cache(tmp_path / "airs_fcst", 2016, times, value=0.0,
                    domain=crop, schema_version=3, drop_kriged_attr=True,
                    channel_values=cache_value)

    x, y, out_times = dataset.kriged_year_arrays(2016, 6, stats, "kriged-airs")
    assert list(out_times) == list(times) and y.dtype == np.uint8
    assert x.shape == (1, *config.GRID_SHAPE, 5)
    x32 = x.astype(np.float32)
    assert np.isfinite(x32).all()
    for c in config.KRIGED_CHANNELS:                     # sourced from cache
        np.testing.assert_allclose(x32[:, crop, config.SFC_VARS.index(c)],
                                   2.0, rtol=1e-3)


@needs_rea_day
def test_loader_reads_clean_channels_from_the_reanalysis_not_the_cache(
        tmp_path, monkeypatch):
    """Channel-sourcing decision 2026-08-18: the CLEAN channels bypass the
    cache entirely and come from ``sfc_daily`` at the same timestamp.

    A cache's clean channels are by definition a copy of the reanalysis, so
    there is no reason to trust the copy: reading the source directly means
    the model sees the SAME SLP/winds at train and eval time no matter which
    cache generation produced its T2M/QV2M fills, and makes a cache's own
    clean copies irrelevant (which is what lets a v3 cache be read at all).

    The proof has to be positive, not just "it loaded": the cache's
    SLP/U10M/V10M are poisoned with an absurd sentinel here, and the returned
    values must match the z-scored reanalysis field, masked to the crop.
    """
    times = pd.DatetimeIndex([REA_STEP])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    crop = dataset.crop_domain()
    stats, raw = _rea_stats_and_step()
    sentinel = 1.0e6                       # finite (so the NaN guard is quiet)
    clean = [v for v in config.SFC_VARS if v not in config.KRIGED_CHANNELS]
    assert clean, "premise: the config keeps at least one channel clean"
    _write_v3_cache(tmp_path / "airs_fcst", 2016, times, value=12.0,
                    domain=crop,
                    channel_values={v: sentinel for v in clean})

    x, _, _ = dataset.kriged_year_arrays(2016, 6, stats, "kriged-airs")
    x32 = x.astype(np.float32)
    for c in clean:
        got = x32[0, ..., config.SFC_VARS.index(c)]
        want = (raw[c] - stats[c][0]) / stats[c][1]
        np.testing.assert_allclose(got[crop], want[crop], rtol=1e-2,
                                   atol=1e-2)
        assert np.abs(got[crop]).max() < 100.0     # nowhere near the sentinel
        np.testing.assert_array_equal(got[~crop], 0.0)   # crop-masked impute
    # the kriged channels still come from the cache: (12 - mean) / std
    for c in config.KRIGED_CHANNELS:
        want = (12.0 - stats[c][0]) / stats[c][1]
        np.testing.assert_allclose(
            x32[0, crop, config.SFC_VARS.index(c)], want, rtol=1e-2)


def test_loader_rejects_cache_holding_a_clean_copy_of_a_kriged_channel(
        tmp_path, monkeypatch):
    """The one provenance hazard left after the sourcing decision
    2026-08-18: a channel the config declares KRIGED that this cache holds as
    a CLEAN reanalysis copy.

    The cache then carries no satellite information for it, but the values
    look exactly like a successful fill, so training would quietly use
    reanalysis where the configuration promises AIRS information.  (The
    opposite direction -- a cache that kriged MORE than the config asks for
    -- is deliberately NOT an error any more: those channels are read from
    the reanalysis and the cache's copies are never touched.  That is the
    relaxation which made the existing v3/4-channel caches reusable; see
    test_loader_reads_v3_cache_at_full_width.)"""
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    # config kriges T2M/QV2M; this cache kriged only QV2M and copied T2M clean
    _write_v3_cache(
        tmp_path / "airs_fcst", 2010, times, value=12.0,
        domain=dataset.crop_domain(),
        domain_attrs={"kriged_channels": ["QV2M", "U10M", "V10M"]})
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}
    with pytest.raises(ValueError) as exc:
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-airs")
    msg = str(exc.value)
    assert "['T2M'] are kriged channels per config airs.kriged_channels=" in msg
    assert "holds a CLEAN reanalysis copy where a satellite-shaped gap fill " \
           "is expected" in msg
    assert "krige_fill build-airs --years 2010" in msg      # the rebuild fix


def test_loader_rejects_legacy_cache_with_no_recoverable_channel_split(
        tmp_path, monkeypatch):
    """A readable cache that records no ``kriged_channels`` and has no
    documented legacy split cannot be sourced at all.

    Since the sourcing decision 2026-08-18 the split IS the contract -- it
    decides which channels come from the file and which from the reanalysis
    -- so a file that cannot state it must be refused rather than guessed at.
    v3 is recoverable (``config.KRIGE_LEGACY_KRIGED_CHANNELS``); a future
    readable version without the attr would not be, and this pins that the
    fallback is a lookup, not a default."""
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    monkeypatch.setattr(config, "KRIGE_LEGACY_KRIGED_CHANNELS", {})
    _write_v3_cache(tmp_path / "airs_fcst", 2010, times, value=12.0,
                    domain=dataset.crop_domain(), schema_version=3,
                    drop_kriged_attr=True)
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}
    with pytest.raises(ValueError,
                       match=r"no 'kriged_channels' attr.*build-airs"):
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-airs")


def test_loader_rejects_v3_cache_from_different_domain(tmp_path,
                                                       monkeypatch):
    """Review 2026-08-13: a v3 cache built under a DIFFERENT box/halo
    (both tunables) passed the schema guard although its crop provenance
    no longer matches -- a smaller live crop silently took real kriged
    values where the contract promises the 0-impute; a larger one raised
    a misleading 'cache is corrupt' NaN error.  The recorded domain attrs
    must be compared against the live config; land_fraction_min alone
    (scoring-mask-only knob) must NOT invalidate a cache.

    ``KRIGED_CHANNELS`` is widened to all five SFC_VARS so the accepted-cache
    branch at the end stays a pure statement about the cache's values (see
    test_loader_imputes_out_of_crop_to_standardized_zero for why)."""
    monkeypatch.setattr(config, "KRIGED_CHANNELS", config.SFC_VARS)
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: _all_none_label_ds(times))
    monkeypatch.setattr(   # manifest reorg 2026-08-13: full paths
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})
    crop = dataset.crop_domain()
    stats = {v: [10.0, 4.0] for v in config.SFC_VARS}
    # stale halo (e.g. built at KERNEL_SIZE=3): superset crop, no NaN
    # inside the live crop, so only the attr check can catch it
    _write_v3_cache(tmp_path / "degraded_reanalysis", 2010, times,
                    value=12.0, domain=crop,
                    domain_attrs={"halo_px": dataset.halo_px() - 2})
    with pytest.raises(ValueError, match=r"DIFFERENT domain.*halo_px"):
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-degraded")
    # stale box
    _write_v3_cache(tmp_path / "airs_fcst", 2010, times, value=12.0,
                    domain=crop, domain_attrs={"domain_lat_range": [30.0,
                                                                    55.0]})
    with pytest.raises(ValueError,
                       match=r"DIFFERENT domain.*domain_lat_range"):
        dataset.kriged_year_arrays(2010, 6, stats, "kriged-airs")
    # a land_fraction_min change does not touch the crop: cache still loads
    _write_v3_cache(tmp_path / "degraded_reanalysis", 2010, times,
                    value=12.0, domain=crop,
                    domain_attrs={"land_fraction_min": 0.9})
    x, _, _ = dataset.kriged_year_arrays(2010, 6, stats, "kriged-degraded")
    np.testing.assert_allclose(x.astype(np.float32)[:, crop, :], 0.5)
