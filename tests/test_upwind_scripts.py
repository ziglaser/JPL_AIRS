"""Tests for the upwind-feature pipeline scripts, on synthetic fixtures only.

Covers, without touching any real data path (and without JPL_AIRS_DATA set):

1. ``trajectories.discover_granules`` / ``load_day_dir`` against a fabricated
   archive-day directory of tiny ``nogrid_*.nc`` granule files (same fabrication
   approach as ``_synthetic_day`` in test_footprint.py, but written to disk in
   the RAW granule schema the loader ingests);
2. ``scripts/build_pblh_3hrly_1deg.py`` -- the 0.25 deg -> 1 deg aggregation
   (fill values, out-of-range values, the 0..360 and stretched lon axes), the
   doubled-year directory recursion, and an end-to-end ``main`` run on two
   fabricated native files;
3. ``scripts/merge_upwind_features.py`` -- daily kernel features landing at the
   right (date, time) slots, honest NaN for absent days / absent assessed
   coverage, Gamma_gap at the TRUE slot datetime including the next-calendar-day
   00-02 UTC slots, and the feature-tier attrs;
4. ``scripts/build_upwind_features.py`` -- its non-trajectory helpers (SMAP
   slot selection, anomaly formation, idempotence skip, day-dir resolution).
   The full driver needs a real HYSPLIT day and is skipped with a reason.

The scripts are imported by path (scripts/ is not a package), the same idiom as
test_front_flags_injection.py.
"""

from __future__ import annotations

import importlib.util
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from trajectory_kernels import config
from trajectory_kernels import trajectories as T

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a scripts/ module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================================== #
# 1. trajectories.discover_granules + load_day_dir on a fabricated day dir
# =========================================================================== #
#: Three granules in Aqua acquisition order; the third releases 1.7 h after the
#: first (the real early/late overpass stagger), so the 1.0 h swath-label rule
#: must call the first two "early" and the third "late".
_GRANULE_RELEASES = {
    188: np.datetime64("2019-06-05T18:53"),
    190: np.datetime64("2019-06-05T19:00"),
    205: np.datetime64("2019-06-05T20:35"),
}


def _write_granule(path: Path, release: np.datetime64,
                   all_invalid: bool = False) -> None:
    """One tiny raw granule in the loader's expected schema.

    Dims (time=4, level=2, fieldx=1, fieldy=2) -> 4 release parcels, one of
    which (level 0, fieldy 1) is NaN at step 0 so the valid-at-release filter
    is exercised. Step 0 is the granule's own release time; steps 1-3 sit on
    the shared 21/22/23 UTC grid, as in the real files. All values are inside
    the config.RANGE_* unit guards.

    ``all_invalid=True`` NaNs every parcel's step-0 position (the
    entirely-cloud-voided/over-ocean granule case), so the granule contributes
    zero valid parcels and must be SKIPPED by load_day_dir, not crash it.
    """
    times = np.array([release,
                      np.datetime64("2019-06-05T21:00"),
                      np.datetime64("2019-06-05T22:00"),
                      np.datetime64("2019-06-05T23:00")], dtype="datetime64[ns]")
    shape = (4, 2, 1, 2)  # (time, level, fieldx, fieldy)

    lat = np.full(shape, 40.0)
    lat += 0.1 * np.arange(2)[None, None, None, :]      # distinct per fieldy
    lat[0, 0, 0, 1] = np.nan                            # invalid at release
    if all_invalid:
        lat[0] = np.nan                                 # NO parcel valid at release
    lon = np.full(shape, -95.0) - 0.5 * np.arange(4)[:, None, None, None]
    alt = np.empty(shape)
    alt[:, 0] = 5000.0                                   # level 0: free troposphere
    alt[:, 1] = 200.0                                    # level 1: near-surface
    pres = np.empty(shape)
    pres[:, 0] = 550.0
    pres[:, 1] = 990.0

    ds = xr.Dataset(
        {name: (("time", "level", "fieldx", "fieldy"), arr) for name, arr in {
            "lat": lat, "lon": lon, "alt": alt, "pres": pres,
            "t": np.full(shape, 285.0), "q": np.full(shape, 8.0),
            "q_excess": np.zeros(shape)}.items()},
        coords={"time": times},
    )
    ds.to_netcdf(path)


@pytest.fixture()
def synthetic_day_dir(tmp_path):
    day_dir = tmp_path / "wrf27km_20190605"
    day_dir.mkdir()
    for granule, release in _GRANULE_RELEASES.items():
        _write_granule(
            day_dir / f"nogrid_wrf27km_GOOD_20190605_{granule}.nc", release)
    # decoys the globber / filename regex must ignore
    (day_dir / "nogrid_readme.nc").write_bytes(b"")     # no trailing _<granule>
    (day_dir / "fullgrid_wrf27km_GOOD_20190605_188.nc").write_bytes(b"")
    return day_dir


def test_discover_granules_sorted_and_filtered(synthetic_day_dir):
    pairs = T.discover_granules(synthetic_day_dir)
    assert [g for g, _ in pairs] == [188, 190, 205]      # acquisition order
    assert all(p.name.startswith("nogrid_") for _, p in pairs)


def test_discover_granules_empty_dir_raises(tmp_path):
    (tmp_path / "not_a_granule.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="not_a_granule.txt"):
        T.discover_granules(tmp_path)


def test_load_day_dir_concatenation_and_swath_labels(synthetic_day_dir):
    day = T.load_day_dir(synthetic_day_dir)

    # 3 of 4 parcels per granule survive the valid-at-release filter
    assert day.sizes["parcel"] == 9
    assert np.array_equal(day["parcel"].values, np.arange(9))  # contiguous ids

    # swath labels from release times: within 1 h of the earliest -> early
    for granule, expected in [(188, "early"), (190, "early"), (205, "late")]:
        labels = day["swath"].values[day["granule"].values == granule]
        assert set(labels) == {expected}

    # step-0 clock is the per-granule release
    for granule, release in _GRANULE_RELEASES.items():
        t0 = day["time_utc"].values[day["granule"].values == granule, 0]
        assert (t0 == release).all()

    # near-surface flag matches the receptor band on release altitude
    lo, hi = config.RECEPTOR_BAND_M
    rel_alt = day["alt"].isel(step=0).values
    assert np.array_equal(day["is_near_surface"].values,
                          (rel_alt >= lo) & (rel_alt <= hi))
    assert day["is_near_surface"].values.sum() == 6      # the 200 m parcels


def test_load_day_dir_equals_per_granule_loads(synthetic_day_dir):
    """The concatenated day is exactly the per-granule loads, granule-major."""
    day = T.load_day_dir(synthetic_day_dir)
    for granule, path in T.discover_granules(synthetic_day_dir):
        part = T._load_granule_file(path, granule, "whatever")
        mask = day["granule"].values == granule
        for name in T.PARCEL_VARS:
            assert np.array_equal(day[name].values[mask], part[name].values,
                                  equal_nan=True)
        assert np.array_equal(day["level"].values[mask], part["level"].values)
        assert np.array_equal(day["time_utc"].values[mask],
                              part["time_utc"].values)


def test_load_day_dir_skips_zero_parcel_granule(synthetic_day_dir):
    """A granule with zero valid parcels at release is dropped with a warning
    naming the file (it used to IndexError on the empty release clock), and the
    surviving granules load exactly as before."""
    _write_granule(synthetic_day_dir / "nogrid_wrf27km_GOOD_20190605_206.nc",
                   np.datetime64("2019-06-05T20:41"), all_invalid=True)
    with pytest.warns(UserWarning, match="nogrid_wrf27km_GOOD_20190605_206.nc"):
        day = T.load_day_dir(synthetic_day_dir)
    assert 206 not in day["granule"].values
    assert day.sizes["parcel"] == 9                      # the 3 good granules
    assert set(day["swath"].values[day["granule"].values == 205]) == {"late"}


def test_load_day_dir_all_granules_empty_raises(tmp_path):
    """When NO granule has a valid parcel at release, that is a data problem
    the caller must see: a clear ValueError, never an empty (parcel=0) day."""
    day_dir = tmp_path / "wrf27km_20190605"
    day_dir.mkdir()
    for granule in (188, 205):
        _write_granule(day_dir / f"nogrid_wrf27km_GOOD_20190605_{granule}.nc",
                       _GRANULE_RELEASES[granule], all_invalid=True)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError,
                           match="no granule in .* contains any valid parcel"):
            T.load_day_dir(day_dir)


# =========================================================================== #
# 2. scripts/build_pblh_3hrly_1deg.py -- the 0.25 -> 1 deg aggregation
# =========================================================================== #
@pytest.fixture(scope="module")
def pblh_mod():
    return _load_script("build_pblh_3hrly_1deg")


def _write_guo_file(path: Path, values_2d: np.ndarray, stretched_lon: bool = False):
    """One raw Guo-style native file: (time=1, lat=600, lon=1440), fill -999.

    ``stretched_lon=True`` stores the real corpus's drifting
    ``linspace(0, 360, 1440)`` axis (step 0.250174) instead of the intended
    0.25 deg lattice, for the ``--lon-grid file`` path.
    """
    from netCDF4 import Dataset

    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("lat", 600)
        nc.createDimension("lon", 1440)
        v = nc.createVariable("lat", "f8", ("lat",))
        v[:] = 90.0 - 0.25 * np.arange(600)
        v = nc.createVariable("lon", "f8", ("lon",))
        v[:] = (np.linspace(0.0, 360.0, 1440) if stretched_lon
                else 0.25 * np.arange(1440))
        v = nc.createVariable("Merged Planetary Boundary Layer Height",
                              "f4", ("time", "lat", "lon"), fill_value=-999.0)
        v[0] = values_2d.astype(np.float32)


def _poisoned_native_field():
    """A native field of 1000 m everywhere, except the 4x4 native block of the
    1 deg cell (40.5, -90.5): 13 points at 1200 m, 2 fill (-999), 1 unphysical
    (9000 m > PBLH_MAX_M). The correct 1 deg answer there is mean 1200, n 13."""
    vals = np.full((600, 1440), 1000.0)
    ii = np.where(np.floor(90.0 - 0.25 * np.arange(600)) == 40.0)[0]
    lon_w = ((0.25 * np.arange(1440) + 180.0) % 360.0) - 180.0
    jj = np.where(np.floor(lon_w) == -91.0)[0]
    assert ii.size == 4 and jj.size == 4
    block = np.ix_(ii, jj)
    vals[block] = 1200.0
    vals[ii[0], jj[0]] = -999.0
    vals[ii[0], jj[1]] = -999.0
    vals[ii[1], jj[2]] = 9000.0
    return vals, ii, jj


def test_target_grid_is_the_padded_conus_box(pblh_mod):
    from build_pbl_climatology import target_grid

    lat_t, lon_t = target_grid(pblh_mod.PADDED_CONUS)
    assert lat_t.size == 40 and lon_t.size == 55
    assert lat_t[0] == 58.5 and lat_t[-1] == 19.5        # descending, centres X.5
    assert lon_t[0] == -112.5 and lon_t[-1] == -58.5


def test_reduce_file_aggregation_fill_and_range(pblh_mod, tmp_path):
    """-999 fill and out-of-range samples are excluded; means/counts are exact;
    every padded-CONUS cell holds exactly 16 native points (domain edges too)."""
    from build_pbl_climatology import (_init_worker, build_index, native_coords,
                                       reduce_file, target_grid)

    vals, _, _ = _poisoned_native_field()
    path = tmp_path / "2019060518.nc"
    _write_guo_file(path, vals)

    lat_n, lon_n = native_coords("uniform", path)
    lat_t, lon_t = target_grid(pblh_mod.PADDED_CONUS)
    keep, flat_idx, n_native = build_index(lat_n, lon_n, lat_t, lon_t)

    # the global native grid fully tiles the padded box: 16 points per cell,
    # including the first/last rows and columns of the domain
    assert (n_native == 16).all()
    assert n_native.sum() == 40 * 55 * 16

    _init_worker(keep, flat_idx, len(lat_t) * len(lon_t))
    total, _sq, count = reduce_file(path)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    mean = mean.reshape(40, 55)
    count = count.reshape(40, 55)

    i = int(np.where(lat_t == 40.5)[0][0])
    j = int(np.where(lon_t == -90.5)[0][0])
    assert count[i, j] == 13                             # 16 - 2 fill - 1 outlier
    assert mean[i, j] == pytest.approx(1200.0)
    other = np.ones((40, 55), dtype=bool)
    other[i, j] = False
    assert (count[other] == 16).all()
    assert np.allclose(mean[other], 1000.0)


def test_native_coords_stretched_lon_axis(pblh_mod, tmp_path):
    """--lon-grid file reads the stored drifting linspace(0,360,1440) axis and
    wraps it to [-180, 180); the drift vs the uniform lattice stays < 0.26 deg
    (the documented +0.25 writing artefact)."""
    from build_pbl_climatology import native_coords

    path = tmp_path / "2019060100.nc"
    _write_guo_file(path, np.full((600, 1440), 1000.0), stretched_lon=True)
    _, lon_file = native_coords("file", path)
    _, lon_uni = native_coords("uniform", path)
    assert lon_file.min() >= -180.0 and lon_file.max() < 180.0
    east = slice(0, 700)                                 # away from the wrap seam
    assert np.max(np.abs(lon_file[east] - lon_uni[east])) < 0.26


def test_find_files_recurses_doubled_year_tree(tmp_path):
    """The corpus nests inconsistently (2017/*.nc but 2019/2019/*.nc): both are
    found, non-matching names are ignored, and the year comes from the stamp."""
    from build_pbl_climatology import find_files

    (tmp_path / "2017").mkdir()
    (tmp_path / "2019" / "2019").mkdir(parents=True)
    (tmp_path / "2017" / "2017010100.nc").write_bytes(b"")
    (tmp_path / "2019" / "2019" / "2019010112.nc").write_bytes(b"")
    (tmp_path / "2017" / "notes.nc").write_bytes(b"")    # ignored: bad stamp

    files = find_files(tmp_path, None)
    assert [s for _, s in files] == [datetime(2017, 1, 1, 0),
                                     datetime(2019, 1, 1, 12)]
    assert [s for _, s in find_files(tmp_path, [2019])] == [datetime(2019, 1, 1, 12)]


def test_build_pblh_main_end_to_end(pblh_mod, tmp_path):
    """Two fabricated native files (one in a doubled-year dir) -> one output nc
    with a true datetime time axis, exact means/counts, and absent-row gaps
    recorded in the attrs rather than NaN rows."""
    root = tmp_path / "Guo2024_model"
    (root / "2017").mkdir(parents=True)
    (root / "2019" / "2019").mkdir(parents=True)
    vals, _, _ = _poisoned_native_field()
    _write_guo_file(root / "2017" / "2017010100.nc", vals)
    _write_guo_file(root / "2019" / "2019" / "2019010112.nc",
                    np.full((600, 1440), 1000.0))

    out_dir = tmp_path / "derived"
    out_path = pblh_mod.main(["--guo-root", str(root), "--out-dir", str(out_dir),
                              "--no-cache", "--jobs", "1"])
    assert out_path == out_dir / "PBLH_1deg_3hrly_2017-2019.nc"

    with xr.open_dataset(out_path) as ds:
        assert list(ds["time"].values.astype("datetime64[h]")) == [
            np.datetime64("2017-01-01T00"), np.datetime64("2019-01-01T12")]
        i = int(np.where(ds["lat"].values == 40.5)[0][0])
        j = int(np.where(ds["lon"].values == -90.5)[0][0])
        assert float(ds["pblh"][0, i, j]) == pytest.approx(1200.0)
        assert int(ds["n_obs"][0, i, j]) == 13
        assert float(ds["pblh"][1, i, j]) == pytest.approx(1000.0)
        assert int(ds["n_obs"][1, i, j]) == 16
        assert (ds["n_native"].values == 16).all()
        # 1 of 2920 files present per year -> absent rows, noted in attrs
        assert "2017: 2919 of 2920 times absent" in ds.attrs["missing_source_times"]


# =========================================================================== #
# 3. scripts/merge_upwind_features.py -- end to end on a fabricated world
# =========================================================================== #
#: The merge fixture's grid and clock. Two dates: one inside the assessed PBLH
#: record, one far outside it (the honest-NaN case).
_MERGE_LAT = np.array([40.5, 39.5])
_MERGE_LON = np.array([-90.5, -89.5])
_DATE_IN = np.datetime64("2019-06-05")
_DATE_OUT = np.datetime64("2019-06-20")
_CLIM_PBLH_M = 500.0
_MML_LFC_M, _MU_LFC_M = 3000.0, 2000.0

#: Assessed 3-hourly stamps: 2019-06-04T00 .. 2019-06-06T21 (24 stamps), so the
#: in-record date's slots -- INCLUDING the 00-02 UTC next-day slots -- resolve,
#: while 2019-06-20 misses by two weeks. Values are a distinct linear ramp per
#: (stamp, cell) so a lookup at the wrong time or cell cannot pass.
_PBLH_STAMPS = np.arange(np.datetime64("2019-06-04T00"),
                         np.datetime64("2019-06-07T00"),
                         np.timedelta64(3, "h")).astype("datetime64[ns]")


def _pblh_table() -> np.ndarray:
    k = np.arange(_PBLH_STAMPS.size)[:, None, None]
    i = np.arange(_MERGE_LAT.size)[None, :, None]
    j = np.arange(_MERGE_LON.size)[None, None, :]
    return (1000.0 + 7.0 * k + 10.0 * i + 1.0 * j).astype(np.float32)


def _expected_pblh(when: np.datetime64, i: int, j: int) -> float:
    """Independent nearest-stamp-within-1.5 h lookup for the test's oracle."""
    diffs = np.abs((_PBLH_STAMPS - when) / np.timedelta64(1, "s"))
    k = int(np.argmin(diffs))
    if diffs[k] > config.PBLH_TIME_TOLERANCE_H * 3600:
        return np.nan
    return float(_pblh_table()[k, i, j])


@pytest.fixture(scope="module")
def merge_world(tmp_path_factory):
    """Fabricate every merge input once; individual tests run main() on it."""
    root = tmp_path_factory.mktemp("merge_world")
    fcst_dir = root / "FCST_SMAP_MRMS"
    daily_dir = root / "daily"
    out_dir = root / "out"
    fcst_dir.mkdir(), daily_dir.mkdir()

    # --- the FCST_SMAP_MRMS-like yearly file: 2 dates x 7 slots x 2x2 cells
    dates = np.array([_DATE_IN, _DATE_OUT], dtype="datetime64[ns]")
    shape = (2, 7, 2, 2)
    parceltime = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    # one real per-cell overpass clock at slot 0 (the only way slot 0 gets a time)
    parceltime[0, 0, 0, 0] = np.datetime64("2019-06-05T18:54")
    fcst = xr.Dataset(
        {"FCST_parceltime": (("date", "time", "lat", "lon"), parceltime),
         "FCST_MML_LFC": (("date", "time", "lat", "lon"),
                          np.full(shape, _MML_LFC_M, dtype=np.float32)),
         "FCST_MU_LFC": (("date", "time", "lat", "lon"),
                         np.full(shape, _MU_LFC_M, dtype=np.float32))},
        coords={"date": dates, "time": np.arange(7),
                "lat": _MERGE_LAT, "lon": _MERGE_LON},
    )
    fcst.to_netcdf(fcst_dir / "FCST_SMAP_MRMS_2019.nc")

    # --- one daily kernel file for the in-record date only, steps 1 and 4
    psi = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    daily = xr.Dataset(
        {"psi_anom": (("arrival_step", "target_lat", "target_lon"), psi),
         "coverage": (("arrival_step", "target_lat", "target_lon"),
                      np.full((2, 2, 2), 0.5, dtype=np.float32))},
        coords={"arrival_step": np.array([1, 4]),
                "target_lat": _MERGE_LAT, "target_lon": _MERGE_LON},
    )
    daily.to_netcdf(daily_dir / "UPW_20190605.nc")

    # --- assessed 3-hourly PBLH + a flat monthly-diurnal climatology
    pblh_path = root / "PBLH_1deg_3hrly.nc"
    xr.Dataset(
        {"pblh": (("time", "lat", "lon"), _pblh_table())},
        coords={"time": _PBLH_STAMPS, "lat": _MERGE_LAT, "lon": _MERGE_LON},
    ).to_netcdf(pblh_path)
    clim_path = root / "PBL_climatology.nc"
    xr.Dataset(
        {"pblh_mean": (("month", "hour", "lat", "lon"),
                       np.full((12, 8, 2, 2), _CLIM_PBLH_M, dtype=np.float32))},
        coords={"month": np.arange(1, 13), "hour": np.arange(0, 24, 3),
                "lat": _MERGE_LAT, "lon": _MERGE_LON},
    ).to_netcdf(clim_path)

    return {"root": root, "fcst_dir": fcst_dir, "daily_dir": daily_dir,
            "out_dir": out_dir, "pblh": pblh_path, "clim": clim_path,
            "psi": psi, "mod": _load_script("merge_upwind_features")}


@pytest.fixture(scope="module")
def merged(merge_world):
    """Run merge_upwind_features.main on the fabricated world, open the result."""
    w = merge_world
    out_path = w["mod"].main(["--year", "2019",
                              "--daily-dir", str(w["daily_dir"]),
                              "--fcst-dir", str(w["fcst_dir"]),
                              "--pblh-3hrly", str(w["pblh"]),
                              "--pblh-clim", str(w["clim"]),
                              "--out-dir", str(w["out_dir"])])
    assert not list(w["out_dir"].glob("*.tmp"))          # atomic write cleaned up
    with xr.open_dataset(out_path) as ds:
        yield ds.load(), w["psi"]


def test_merge_kernel_features_land_at_their_slots(merged):
    ds, psi = merged
    # arrival step s -> time index s; slot 0 and un-supplied slots stay NaN
    assert np.array_equal(ds["UPW_psi_anom"].values[0, 1], psi[0])
    assert np.array_equal(ds["UPW_psi_anom"].values[0, 4], psi[1])
    for empty_slot in (0, 2, 3, 5, 6):
        assert np.isnan(ds["UPW_psi_anom"].values[0, empty_slot]).all()
    assert np.allclose(ds["UPW_coverage"].values[0, [1, 4]], 0.5)


def test_merge_absent_day_is_all_nan(merged):
    ds, _ = merged
    assert np.isnan(ds["UPW_psi_anom"].values[1]).all()   # no UPW_20190620.nc
    assert ds.attrs["n_dates_with_daily_file"] == 1
    assert ds.attrs["daily_coverage_fraction"] == pytest.approx(0.5)


def test_merge_gamma_gap_at_true_slot_datetime(merged):
    """Gamma_gap = LFC - assessed PBLH at the slot's TRUE datetime, including
    slot 4 = 00 UTC of the NEXT calendar day (the date-boundary case)."""
    ds, _ = merged
    slot_when = {1: "2019-06-05T21", 2: "2019-06-05T22",
                 3: "2019-06-05T23", 4: "2019-06-06T00",
                 5: "2019-06-06T01", 6: "2019-06-06T02"}
    for s, when in slot_when.items():
        when = np.datetime64(when)
        for i in range(2):
            for j in range(2):
                zi = _expected_pblh(when, i, j)
                assert ds["UPW_pblh"].values[0, s, i, j] == pytest.approx(zi)
                assert ds["UPW_gamma_gap_mml"].values[0, s, i, j] == pytest.approx(
                    _MML_LFC_M - zi)
                assert ds["UPW_gamma_gap_mu"].values[0, s, i, j] == pytest.approx(
                    _MU_LFC_M - zi)
    # the boundary slot really used the next day's field, not 21-23 UTC's
    assert (_expected_pblh(np.datetime64("2019-06-06T00"), 0, 0)
            != _expected_pblh(np.datetime64("2019-06-05T21"), 0, 0))


def test_merge_slot0_uses_parceltime_only(merged):
    """Slot 0 has no nominal clock: only the one cell with an FCST_parceltime
    gets a PBLH (at ~18:54 -> the 18 UTC stamp); every other slot-0 cell is NaN."""
    ds, _ = merged
    got = ds["UPW_pblh"].values[0, 0]
    assert got[0, 0] == pytest.approx(
        _expected_pblh(np.datetime64("2019-06-05T18:54"), 0, 0))
    assert np.isnan(got.ravel()[1:]).all()


def test_merge_pblh_anom_nan_where_coverage_missing(merged):
    """No assessed coverage (2019-06-20 is outside the record) -> pblh, anom and
    both gamma_gaps are NaN. The anomaly must be NaN, NOT 0: filling from
    climatology would fabricate an anomaly of exactly zero."""
    ds, _ = merged
    for name in ("UPW_pblh", "UPW_pblh_anom", "UPW_gamma_gap_mml",
                 "UPW_gamma_gap_mu"):
        assert np.isnan(ds[name].values[1]).all(), name
    # ... while inside coverage the anomaly is assessed minus the climatology
    expected = _expected_pblh(np.datetime64("2019-06-05T21"), 0, 0) - _CLIM_PBLH_M
    assert ds["UPW_pblh_anom"].values[0, 1, 0, 0] == pytest.approx(expected)
    assert not np.allclose(ds["UPW_pblh_anom"].values[0, 1], 0.0)


def test_merge_tier_attrs_present(merged):
    ds, _ = merged
    tiers = {name: ds[name].attrs.get("feature_tier") for name in ds.data_vars}
    assert tiers["UPW_psi_anom"] == "core"
    assert tiers["UPW_gamma_gap_mml"] == "core"
    assert tiers["UPW_gamma_gap_mu"] == "core"
    assert tiers["UPW_pblh_anom"] == "core"
    assert tiers["UPW_pblh"] == "honesty"
    assert tiers["UPW_coverage"] == "honesty"
    assert "nan_policy" in ds.attrs


def test_merge_daily_dir_default_matches_builder_out_dir(merge_world, upw_mod):
    """The merge's --daily-dir default and the builder's --out-dir default are
    the SAME path, so the handoff cannot silently miss (a None default used to
    mean 'trajectory-free' without anyone asking for it)."""
    merge_args = merge_world["mod"].parse_args(["--year", "2019"])
    build_args = upw_mod.parse_args(["--date", "2019-06-05", "--traj-root", "x"])
    assert merge_args.daily_dir == build_args.out_dir
    assert merge_args.no_daily is False


def test_merge_no_daily_is_explicit_trajectory_free(merge_world, recwarn):
    """--no-daily: kernel features intentionally absent, no missing-files
    warning, and the attrs say so (daily_files_found=0)."""
    w = merge_world
    out_dir = w["root"] / "out_no_daily"
    out_path = w["mod"].main(["--year", "2019", "--no-daily",
                              "--daily-dir", str(w["daily_dir"]),  # must be ignored
                              "--fcst-dir", str(w["fcst_dir"]),
                              "--pblh-3hrly", str(w["pblh"]),
                              "--pblh-clim", str(w["clim"]),
                              "--out-dir", str(out_dir)])
    assert not any("found none" in str(rec.message) for rec in recwarn.list)
    with xr.open_dataset(out_path) as ds:
        assert "UPW_psi_anom" not in ds.data_vars         # daily dir NOT read
        assert "UPW_pblh" in ds.data_vars                 # trajectory-free stream intact
        assert ds.attrs["daily_files_found"] == 0
        assert ds.attrs["daily_dir"] == "(none: --no-daily)"


def test_merge_empty_daily_dir_warns_loudly(merge_world):
    """A daily dir that was EXPECTED but holds zero UPW_*.nc for the year is a
    prominent warning plus daily_files_found=0 in attrs -- never silence."""
    w = merge_world
    empty_dir = w["root"] / "daily_empty"
    empty_dir.mkdir()
    out_dir = w["root"] / "out_empty_daily"
    with pytest.warns(UserWarning, match="found none"):
        out_path = w["mod"].main(["--year", "2019",
                                  "--daily-dir", str(empty_dir),
                                  "--fcst-dir", str(w["fcst_dir"]),
                                  "--pblh-3hrly", str(w["pblh"]),
                                  "--pblh-clim", str(w["clim"]),
                                  "--out-dir", str(out_dir)])
    with xr.open_dataset(out_path) as ds:
        assert ds.attrs["daily_files_found"] == 0
        assert ds.attrs["n_dates_with_daily_file"] == 0


@pytest.mark.parametrize("script", ["merge_upwind_features",
                                    "build_upwind_features"])
def test_atomic_to_netcdf_tmp_then_replace(script, tmp_path, monkeypatch):
    """Both scripts write via <name>.nc.tmp in the same dir and promote it with
    one os.replace (atomic on POSIX), so a preemption mid-write can never leave
    a truncated file at the final path for the skip-if-exists check to trust."""
    import os as _os

    mod = _load_script(script)
    out_path = tmp_path / "OUT_20190605.nc"
    ds = xr.Dataset({"x": (("i",), np.arange(3.0))})

    calls = []
    real_replace = _os.replace

    def observed_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        assert Path(src).exists()                        # fully written first
        return real_replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", observed_replace)
    mod.atomic_to_netcdf(ds, out_path)

    assert calls == [(out_path.with_suffix(".nc.tmp"), out_path)]
    assert out_path.exists()
    assert not list(tmp_path.glob("*.tmp"))              # nothing left behind
    with xr.open_dataset(out_path) as back:
        assert np.array_equal(back["x"].values, np.arange(3.0))


@pytest.mark.parametrize("script", ["merge_upwind_features",
                                    "build_upwind_features"])
def test_atomic_to_netcdf_failed_write_leaves_no_final_file(script, tmp_path):
    """A write that dies partway (unencodable data) leaves NEITHER a final file
    (so the resume check will retry) NOR a stray .tmp."""
    mod = _load_script(script)
    out_path = tmp_path / "OUT_20190605.nc"
    ds = xr.Dataset({"x": (("i",), np.arange(3.0))})
    with pytest.raises(Exception):
        mod.atomic_to_netcdf(ds, out_path, encoding={"nonexistent_var": {}})
    assert not out_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


# =========================================================================== #
# 4. scripts/build_upwind_features.py -- non-trajectory helpers
# =========================================================================== #
@pytest.fixture(scope="module")
def upw_mod():
    return _load_script("build_upwind_features")


def test_resolve_day_dir_precedence(upw_mod, tmp_path):
    root = tmp_path
    doubled = root / "wrf27km_20190605" / "wrf27km_20190605"
    doubled.mkdir(parents=True)
    (root / "20190605").mkdir()
    # doubled nesting wins over the flat layouts when both exist
    assert upw_mod.resolve_day_dir(root, "20190605") == doubled
    # single nesting: the wrf27km_<date> dir itself
    (root / "wrf27km_20190606").mkdir()
    assert upw_mod.resolve_day_dir(root, "20190606") == root / "wrf27km_20190606"
    # bare-date layout
    (root / "20190607").mkdir()
    assert upw_mod.resolve_day_dir(root, "20190607") == root / "20190607"
    # absent day: None (an expected gap, not an error)
    assert upw_mod.resolve_day_dir(root, "20190699") is None


def _write_smap_world(tmp_path):
    """A minimal FCST file with 5 L4 analysis slots and a monthly baseline."""
    slots = np.array([16.5, 19.5, 22.5, 25.5, 28.5])
    dates = np.array([np.datetime64("2019-06-05")], dtype="datetime64[ns]")
    # sm value encodes its slot index so the chosen slot is verifiable
    sm = np.zeros((1, 5, 2, 2), dtype=np.float32)
    for k in range(5):
        sm[0, k] = 0.10 + 0.01 * k
    fcst_dir = tmp_path / "fcst"
    fcst_dir.mkdir()
    xr.Dataset(
        {"SMAP_L4_smsfc_av": (("date", "L4_nhours", "lat", "lon"), sm)},
        coords={"date": dates, "L4_nhours": slots,
                "lat": _MERGE_LAT, "lon": _MERGE_LON},
    ).to_netcdf(fcst_dir / "FCST_SMAP_MRMS_2019.nc")

    baseline = np.full((12, 2, 2), 0.08, dtype=np.float32)
    baseline[5] = 0.09          # June (month index 6) distinct from the rest
    baseline_path = tmp_path / "sm_baseline.nc"
    xr.Dataset(
        {"sm_baseline": (("month", "lat", "lon"), baseline)},
        coords={"month": np.arange(1, 13), "lat": _MERGE_LAT, "lon": _MERGE_LON},
    ).to_netcdf(baseline_path)
    return fcst_dir, baseline_path


def test_select_smap_slot_nearest_mean_arrival(upw_mod, tmp_path):
    fcst_dir, baseline_path = _write_smap_world(tmp_path)
    arrivals = np.array(["2019-06-05T21:00", "2019-06-05T23:00"],
                        dtype="datetime64[s]")            # mean 22 h -> slot 22.5
    sm_raw, sm_anom, slot = upw_mod.select_smap_slot(
        fcst_dir, Date(2019, 6, 5), arrivals, baseline_path)
    assert slot == 22.5
    assert np.allclose(sm_raw.values, 0.12)               # slot index 2
    assert np.allclose(sm_anom.values, 0.12 - 0.09)       # June baseline
    assert sm_raw.name == "sm_raw" and sm_anom.name == "sm_anom"


def test_select_smap_slot_past_midnight(upw_mod, tmp_path):
    """Arrivals straddling midnight: nhours counts past 24, so the mean of
    23:00 and next-day 02:00 is 24.5 h -> the 25.5 h slot, not 22.5."""
    fcst_dir, baseline_path = _write_smap_world(tmp_path)
    arrivals = np.array(["2019-06-05T23:00", "2019-06-06T02:00"],
                        dtype="datetime64[s]")
    _, _, slot = upw_mod.select_smap_slot(
        fcst_dir, Date(2019, 6, 5), arrivals, baseline_path)
    assert slot == 25.5


def test_main_idempotence_skip(upw_mod, tmp_path, capsys):
    """An existing output file short-circuits main to exit 0 before any input
    is touched (the slurm-array resume contract)."""
    out_dir = tmp_path / "daily"
    out_dir.mkdir()
    (out_dir / "UPW_20190605.nc").write_bytes(b"already built")
    rc = upw_mod.main(["--date", "2019-06-05",
                       "--traj-root", str(tmp_path / "does_not_exist"),
                       "--out-dir", str(out_dir)])
    assert rc == 0
    assert "skipping" in capsys.readouterr().out
    assert (out_dir / "UPW_20190605.nc").read_bytes() == b"already built"


def _builder_world(tmp_path, synthetic_day_dir, with_pblh: bool = True):
    """Args (list) for build_upwind_features.main with every input fabricated.

    ``with_pblh=False`` points --pblh-3hrly at a nonexistent file, the
    silently-degrades-to-analytic case the F2 guard must refuse.
    """
    fcst_dir, baseline_path = _write_smap_world(tmp_path)
    pblh_path = tmp_path / "PBLH_1deg_3hrly.nc"
    if with_pblh:
        xr.Dataset(
            {"pblh": (("time", "lat", "lon"), _pblh_table())},
            coords={"time": _PBLH_STAMPS, "lat": _MERGE_LAT, "lon": _MERGE_LON},
        ).to_netcdf(pblh_path)
    return ["--date", "2019-06-05",
            "--traj-root", str(synthetic_day_dir.parent),
            "--out-dir", str(tmp_path / "daily_out"),
            "--fcst-dir", str(fcst_dir),
            "--baseline", str(baseline_path),
            "--pblh-3hrly", str(pblh_path),
            "--pblh-clim", str(tmp_path / "no_clim.nc")]


def test_main_preflight_missing_fcst_exits(upw_mod, tmp_path, synthetic_day_dir):
    """A bad --fcst-dir must fail BEFORE the kernel build, naming the file."""
    args = _builder_world(tmp_path, synthetic_day_dir)
    i = args.index("--fcst-dir")
    args[i + 1] = str(tmp_path / "wrong_fcst_dir")
    with pytest.raises(SystemExit, match="FCST_SMAP_MRMS_2019.nc"):
        upw_mod.main(args)


def test_main_preflight_missing_baseline_exits(upw_mod, tmp_path, synthetic_day_dir):
    args = _builder_world(tmp_path, synthetic_day_dir)
    i = args.index("--baseline")
    args[i + 1] = str(tmp_path / "no_such_baseline.nc")
    with pytest.raises(SystemExit, match="build_smap_l4_baseline"):
        upw_mod.main(args)


def test_main_refuses_analytic_pblh_fallback(upw_mod, tmp_path, synthetic_day_dir):
    """No assessed 3-hourly PBLH -> SystemExit naming the missing path and the
    builder script/sbatch (review F2: the analytic fallback yields
    information-free m_star/omega while the output attrs look healthy)."""
    args = _builder_world(tmp_path, synthetic_day_dir, with_pblh=False)
    with pytest.raises(SystemExit) as exc:
        upw_mod.main(args)
    msg = str(exc.value)
    assert "PBLH_1deg_3hrly.nc" in msg
    assert "build_pblh_3hrly_1deg.py" in msg
    assert "upwind_pblh_3hrly.sbatch" in msg


def _stub_heavy_stages(upw_mod, monkeypatch):
    """Replace the footprint sweep and predictor stage with tiny fakes so main
    can run end-to-end in milliseconds; everything upstream (arg handling,
    preflight, PBL guard, SMAP slot selection, attrs, atomic write) stays real."""
    fake_kernels = xr.Dataset()
    fake_kernels.attrs["arrival_times_utc"] = ["2019-06-05T21:00", "2019-06-05T23:00"]

    def fake_build_all(day_ds, **kwargs):
        return fake_kernels

    def fake_build_features(kernels, sm_anom, sm_raw=None, pbl_model=None):
        shp = (_MERGE_LAT.size, _MERGE_LON.size)
        return xr.Dataset(
            {"n_parcels": (("target_lat", "target_lon"), np.ones(shp)),
             "psi_anom": (("target_lat", "target_lon"), np.zeros(shp)),
             "coverage": (("target_lat", "target_lon"), np.full(shp, 0.5))},
            coords={"target_lat": _MERGE_LAT, "target_lon": _MERGE_LON})

    monkeypatch.setattr(upw_mod.footprint, "build_all", fake_build_all)
    monkeypatch.setattr(upw_mod.predictors, "build_features", fake_build_features)
    monkeypatch.setattr(upw_mod.land, "make_land_lookup", lambda path: None)


def test_main_allow_pblh_fallback_escape(upw_mod, tmp_path, synthetic_day_dir,
                                         monkeypatch, capsys):
    """--allow-pblh-fallback lets the build proceed without the assessed layer,
    but the escape is stamped into the output (pblh_fallback='allowed') and the
    tallied source fractions are still recorded, assessed first."""
    _stub_heavy_stages(upw_mod, monkeypatch)
    args = _builder_world(tmp_path, synthetic_day_dir, with_pblh=False)
    rc = upw_mod.main(args + ["--allow-pblh-fallback"])
    assert rc == 0
    assert "--allow-pblh-fallback" in capsys.readouterr().out  # loud warning

    out_path = tmp_path / "daily_out" / "UPW_20190605.nc"
    assert not list((tmp_path / "daily_out").glob("*.tmp"))    # atomic write
    with xr.open_dataset(out_path) as ds:
        assert ds.attrs["pblh_fallback"] == "allowed"
        assert ds.attrs["pblh_source_fractions"].startswith("assessed=")


def test_main_production_run_not_stamped_fallback(upw_mod, tmp_path,
                                                  synthetic_day_dir, monkeypatch):
    """With the assessed layer present the guard passes silently and the output
    carries NO pblh_fallback stamp -- 'allowed' means someone typed the flag."""
    _stub_heavy_stages(upw_mod, monkeypatch)
    args = _builder_world(tmp_path, synthetic_day_dir, with_pblh=True)
    assert upw_mod.main(args) == 0
    with xr.open_dataset(tmp_path / "daily_out" / "UPW_20190605.nc") as ds:
        assert "pblh_fallback" not in ds.attrs
        assert ds.attrs["pblh_source_fractions"].startswith("assessed=")


@pytest.mark.skip(reason="full build_upwind_features driver needs a real HYSPLIT "
                         "trajectory day (footprint sweep over the CONUS grid); "
                         "covered by the dev smoke run on the demo day")
def test_build_upwind_features_full_driver():
    ...


# --------------------------------------------------------------------------- #
# merge pass 2: the multi-year Omega climatology and UPW_omega_anom
# --------------------------------------------------------------------------- #
def _write_companion(path, year, omega_value):
    """A minimal yearly companion: UPW_omega constant per file, two June days
    and one July day, 2 slots, a 1x1 grid -- enough for month grouping."""
    dates = np.array([f"{year}-06-10", f"{year}-06-20", f"{year}-07-05"],
                     dtype="datetime64[ns]")
    om = np.full((3, 2, 1, 1), float(omega_value))
    om[2] += 100.0  # July offset so the two months have distinct climatologies
    ds = xr.Dataset(
        {"UPW_omega": (("date", "time", "lat", "lon"), om)},
        coords={"date": dates, "time": [1, 2], "lat": [40.5], "lon": [-90.5]})
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def merge_mod():
    return _load_script("merge_upwind_features")


def test_omega_clim_requires_two_years(merge_mod, tmp_path):
    _write_companion(tmp_path / "UPWIND_FEATURES_2019.nc", 2019, 1000.0)
    with pytest.raises(SystemExit, match=">= 2 merged years"):
        merge_mod.build_omega_climatology(tmp_path, min_samples=1)


def test_omega_clim_and_anomaly_arithmetic(merge_mod, tmp_path):
    # two years, June omegas 1000 and 3000 -> June clim 2000; July +100 each
    _write_companion(tmp_path / "UPWIND_FEATURES_2019.nc", 2019, 1000.0)
    _write_companion(tmp_path / "UPWIND_FEATURES_2020.nc", 2020, 3000.0)
    clim_path = merge_mod.build_omega_climatology(tmp_path, min_samples=1)
    assert clim_path.name == "omega_clim_2019-2020.nc"
    clim = xr.open_dataset(clim_path)
    assert float(clim["omega_clim"].sel(month=6).isel(time=0)) == 2000.0
    assert float(clim["omega_clim"].sel(month=7).isel(time=0)) == 2100.0
    assert int(clim["omega_clim_n"].sel(month=6).isel(time=0)) == 4

    # min_samples guard: June pools 4 days across the two years, July only 2
    thin = merge_mod.build_omega_climatology(
        tmp_path, min_samples=3, out_path=tmp_path / "thin.nc")
    thin_ds = xr.open_dataset(thin)
    assert np.isfinite(thin_ds["omega_clim"].sel(month=6)).all()
    assert thin_ds["omega_clim"].sel(month=7).isnull().all()

    # pass 2 amend: 2019 June anomaly = 1000 - 2000 = -1000; July = -1000 too
    out = merge_mod.add_omega_anom(tmp_path / "UPWIND_FEATURES_2019.nc", clim_path)
    amended = xr.open_dataset(out)
    anom = amended["UPW_omega_anom"]
    assert anom.attrs["feature_tier"] == "ablation"
    np.testing.assert_allclose(anom.values, -1000.0)
    assert "omega_clim" in amended.attrs
    # the amend is idempotent-safe: original variables untouched
    np.testing.assert_allclose(
        amended["UPW_omega"].sel(date="2019-06-10").values, 1000.0)


def test_add_omega_anom_missing_companion_exits(merge_mod, tmp_path):
    _write_companion(tmp_path / "UPWIND_FEATURES_2019.nc", 2019, 1000.0)
    _write_companion(tmp_path / "UPWIND_FEATURES_2020.nc", 2020, 3000.0)
    clim_path = merge_mod.build_omega_climatology(tmp_path, min_samples=1)
    with pytest.raises(SystemExit, match="merge that year first"):
        merge_mod.add_omega_anom(tmp_path / "UPWIND_FEATURES_2021.nc", clim_path)
