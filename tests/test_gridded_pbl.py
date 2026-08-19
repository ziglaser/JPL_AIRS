"""Tests for :class:`trajectory_kernels.pbl.GriddedPBL`.

The three-layer fallback chain (assessed 3-hourly file -> monthly-diurnal
climatology -> analytic :class:`ClimatologicalPBL`) is exercised end to end on
tiny synthetic netCDF files built in ``tmp_path`` -- no real data paths, and no
``JPL_AIRS_DATA`` needed. Every grid value is a deterministic function of its
coordinates, so each test can recompute the expected answer independently and
identify *which* layer answered a query by its value alone (the
``last_source_fractions`` bookkeeping is then cross-checked against that).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from trajectory_kernels import config
from trajectory_kernels import pbl as P


# ---------------------------------------------------------------- fixtures --
# Small 1-deg grid; the assessed and climatology files share lat/lon coverage
# so a fall-through between them is driven by *time* holes, not geography.
LATS = np.array([39.5, 40.5, 41.5])
LONS = np.array([-92.5, -91.5, -90.5])

T0 = np.datetime64("2019-06-04T00:00", "ns")          # start of the 3-day record
MISSING_TIME = np.datetime64("2019-06-05T12:00", "ns")  # dropped timestamp (gap)
NAN_TIME = np.datetime64("2019-06-05T18:00", "ns")      # cell holding the NaN
NAN_LAT, NAN_LON = 40.5, -91.5


def assessed_value(t, lat, lon) -> float:
    """The synthetic assessed field: unique per (3-h slot, cell)."""
    slots = (np.datetime64(t, "ns") - T0) / np.timedelta64(3, "h")
    return 500.0 + 40.0 * float(slots) + 10.0 * float(lat) + float(lon)


def clim_value(month: int, hour: int, lat: float, lon: float) -> float:
    """The synthetic climatology: unique per (month, 3-h UTC slot, cell)."""
    return 100.0 + 30.0 * month + 4.0 * hour + 6.0 * lat + 2.0 * lon


def make_three_hourly(path) -> None:
    """3-hourly ``pblh (time, lat, lon)`` over 3 days, minus one timestamp,
    with one NaN cell (a no-retrieval hole, e.g. persistent cloud)."""
    times = T0 + np.arange(24) * np.timedelta64(3, "h")   # 2019-06-04 .. 06-06
    times = times[times != MISSING_TIME]
    vals = np.empty((times.size, LATS.size, LONS.size))
    for it, t in enumerate(times):
        for iy, la in enumerate(LATS):
            for ix, lo in enumerate(LONS):
                vals[it, iy, ix] = assessed_value(t, la, lo)
    ds = xr.Dataset({"pblh": (("time", "lat", "lon"), vals)},
                    coords={"time": times, "lat": LATS, "lon": LONS})
    ds["pblh"].loc[dict(time=NAN_TIME, lat=NAN_LAT, lon=NAN_LON)] = np.nan
    ds.to_netcdf(path)


def make_climatology(path) -> None:
    """``pblh_mean (month, hour, lat, lon)`` matching the real file's schema:
    hour = UTC slot 0, 3, ..., 21 and lat written DESCENDING to exercise the
    descending-axis handling in ``_nearest_cell_index``."""
    months = np.arange(1, 13)
    hours = np.arange(0, 24, 3)
    lat_desc = LATS[::-1].copy()
    vals = np.empty((months.size, hours.size, lat_desc.size, LONS.size))
    for im, m in enumerate(months):
        for ih, h in enumerate(hours):
            for iy, la in enumerate(lat_desc):
                for ix, lo in enumerate(LONS):
                    vals[im, ih, iy, ix] = clim_value(m, h, la, lo)
    ds = xr.Dataset(
        {"pblh_mean": (("month", "hour", "lat", "lon"), vals)},
        coords={"month": months, "hour": hours, "lat": lat_desc, "lon": LONS})
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("pbl")
    three = d / "pblh_3hrly.nc"
    clim = d / "pblh_clim.nc"
    make_three_hourly(three)
    make_climatology(clim)
    return three, clim


@pytest.fixture(scope="module")
def gpbl(paths):
    return P.GriddedPBL(three_hourly_path=paths[0], clim_path=paths[1])


def _clim_expected(t, lat, lon) -> float:
    """What layer 2 should return: calendar month + nearest 3-h UTC slot."""
    t = np.datetime64(t, "ns")
    month = int(t.astype("datetime64[M]").astype(int) % 12) + 1
    hour = int(np.rint(P._utc_fractional_hour(t) / 3.0) % 8) * 3
    return clim_value(month, hour, lat, lon)


# ------------------------------------------------------- layer 1: assessed --
def test_exact_assessed_hit(gpbl):
    t = np.datetime64("2019-06-04T06:00", "ns")
    got = float(gpbl.depth(40.5, -91.5, t))
    assert got == pytest.approx(assessed_value(t, 40.5, -91.5))
    assert gpbl.last_source_fractions == {
        "assessed": 1.0, "climatology": 0.0, "analytic": 0.0}


def test_nearest_time_within_tolerance(gpbl):
    """A query 1 h off a sample snaps to it (tolerance is 1.5 h)."""
    on_grid = np.datetime64("2019-06-04T06:00", "ns")
    got = float(gpbl.depth(40.5, -91.5, on_grid + np.timedelta64(1, "h")))
    assert got == pytest.approx(assessed_value(on_grid, 40.5, -91.5))
    assert gpbl.last_source_fractions["assessed"] == 1.0


def test_beyond_tolerance_falls_to_climatology(gpbl):
    """The dropped timestamp leaves its neighbours 3 h away -- beyond the 1.5 h
    tolerance -- so a query there must fall through to the climatology."""
    got = float(gpbl.depth(40.5, -91.5, MISSING_TIME))
    assert got == pytest.approx(_clim_expected(MISSING_TIME, 40.5, -91.5))
    assert gpbl.last_source_fractions == {
        "assessed": 0.0, "climatology": 1.0, "analytic": 0.0}


def test_nan_cell_falls_to_climatology(gpbl):
    """A no-retrieval (NaN) assessed cell is a hole, not an answer."""
    got = float(gpbl.depth(NAN_LAT, NAN_LON, NAN_TIME))
    assert got == pytest.approx(_clim_expected(NAN_TIME, NAN_LAT, NAN_LON))
    assert gpbl.last_source_fractions["climatology"] == 1.0
    # the neighbouring cell at the same time is intact and answers as layer 1
    got_ok = float(gpbl.depth(NAN_LAT, -90.5, NAN_TIME))
    assert got_ok == pytest.approx(assessed_value(NAN_TIME, NAN_LAT, -90.5))
    assert gpbl.last_source_fractions["assessed"] == 1.0


def test_pre_record_date_falls_to_climatology(gpbl):
    """2016 predates the assessed record entirely -> layer 2."""
    t = np.datetime64("2016-06-05T18:00", "ns")
    got = float(gpbl.depth(40.5, -91.5, t))
    assert got == pytest.approx(_clim_expected(t, 40.5, -91.5))
    assert gpbl.last_source_fractions == {
        "assessed": 0.0, "climatology": 1.0, "analytic": 0.0}


# ----------------------------------------------------------- layer 3 + mix --
def test_off_domain_point_falls_to_analytic(gpbl):
    """A point off both grids (deep tropics here) reaches layer 3 and must
    equal the analytic model evaluated directly."""
    t = np.datetime64("2019-06-05T18:00", "ns")
    got = float(gpbl.depth(10.5, -75.5, t))
    assert got == pytest.approx(float(P.ClimatologicalPBL().depth(10.5, -75.5, t)))
    assert gpbl.last_source_fractions == {
        "assessed": 0.0, "climatology": 0.0, "analytic": 1.0}


def test_source_fractions_mixed_query(gpbl):
    """One point per layer: fractions are 1/3 each and sum to 1, and each
    point's value identifies the layer that answered it."""
    lat = np.array([40.5, 40.5, 10.5])
    lon = np.array([-91.5, -91.5, -75.5])
    t = np.array(["2019-06-04T06:00", "2016-06-05T18:00", "2019-06-05T18:00"],
                 dtype="datetime64[ns]")
    got = gpbl.depth(lat, lon, t)
    assert got[0] == pytest.approx(assessed_value(t[0], 40.5, -91.5))
    assert got[1] == pytest.approx(_clim_expected(t[1], 40.5, -91.5))
    assert got[2] == pytest.approx(float(P.ClimatologicalPBL().depth(10.5, -75.5, t[2])))
    frac = gpbl.last_source_fractions
    assert frac["assessed"] == pytest.approx(1 / 3)
    assert frac["climatology"] == pytest.approx(1 / 3)
    assert frac["analytic"] == pytest.approx(1 / 3)
    assert sum(frac.values()) == pytest.approx(1.0)


# -------------------------------------------------------- degraded operation --
def test_missing_three_hourly_file(paths, tmp_path):
    """No 3-hourly file: layer 1 is off but layers 2 and 3 still answer."""
    pbl = P.GriddedPBL(three_hourly_path=tmp_path / "nope.nc", clim_path=paths[1])
    assert pbl.available == {"assessed": False, "climatology": True, "analytic": True}
    t = np.datetime64("2019-06-04T06:00", "ns")  # would have been an assessed hit
    assert float(pbl.depth(40.5, -91.5, t)) == pytest.approx(
        _clim_expected(t, 40.5, -91.5))
    assert pbl.last_source_fractions["climatology"] == 1.0
    assert float(pbl.depth(10.5, -75.5, t)) == pytest.approx(
        float(P.ClimatologicalPBL().depth(10.5, -75.5, t)))
    assert pbl.last_source_fractions["analytic"] == 1.0


def test_both_files_missing_pure_analytic():
    pbl = P.GriddedPBL(three_hourly_path=None, clim_path=None)
    assert pbl.available == {"assessed": False, "climatology": False, "analytic": True}
    lat = np.array([40.5, 10.5])
    t = np.datetime64("2019-06-05T19:00", "ns")
    got = pbl.depth(lat, -90.5, t)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, P.ClimatologicalPBL().depth(lat, -90.5, t))
    assert pbl.last_source_fractions == {
        "assessed": 0.0, "climatology": 0.0, "analytic": 1.0}


# ------------------------------------------------------------- broadcasting --
def test_broadcasting_matches_scalar_loops(gpbl):
    """Scalar lat/lon against an array of times (spanning all three layers),
    and a full 2-D broadcast, agree elementwise with scalar calls."""
    times = np.array(["2019-06-04T06:00", "2016-06-05T18:00", MISSING_TIME],
                     dtype="datetime64[ns]")
    vec = gpbl.depth(40.5, -91.5, times)
    assert vec.shape == times.shape
    for k, t in enumerate(times):
        assert vec[k] == pytest.approx(float(gpbl.depth(40.5, -91.5, t)))

    lat = np.array([[40.5], [10.5]])            # (2, 1)
    grid = gpbl.depth(lat, -91.5, times[None, :])  # -> (2, 3)
    assert grid.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert grid[i, j] == pytest.approx(
                float(gpbl.depth(lat[i, 0], -91.5, times[j])))


# ----------------------------------------------------------- eager vs lazy --
def test_date_window_matches_lazy(paths):
    """``date=`` eagerly loads only [date-1 d, date+2 d] but must return the
    same values as the lazy full-record open for queries inside the window."""
    lazy = P.GriddedPBL(three_hourly_path=paths[0], clim_path=paths[1])
    eager = P.GriddedPBL(three_hourly_path=paths[0], clim_path=paths[1],
                         date="2019-06-05")
    assert eager.available["assessed"]
    lat = np.array([40.5, NAN_LAT, 41.5, 10.5])
    lon = np.array([-91.5, NAN_LON, -90.5, -75.5])
    t = np.array(["2019-06-05T06:00", str(NAN_TIME), "2019-06-04T21:00",
                  "2019-06-05T18:00"], dtype="datetime64[ns]")
    np.testing.assert_allclose(eager.depth(lat, lon, t), lazy.depth(lat, lon, t))
    assert eager.last_source_fractions == lazy.last_source_fractions


# -------------------------------------------------- public lookups directly --
def test_assessed_lookup_direct(gpbl):
    """Layer 1 alone: hits return the grid value, misses (time gap, off-grid
    lat) return NaN rather than falling through."""
    t_hit = np.datetime64("2019-06-06T09:00", "ns")
    lat = np.array([39.5, 40.5, 10.5])
    lon = np.array([-92.5, -91.5, -91.5])
    t_ns = np.array([t_hit, MISSING_TIME, t_hit]).astype("datetime64[ns]").astype(np.int64)
    got = gpbl.assessed_lookup(lat, lon, t_ns)
    assert got[0] == pytest.approx(assessed_value(t_hit, 39.5, -92.5))
    assert np.isnan(got[1])   # dropped timestamp: no sample within 1.5 h
    assert np.isnan(got[2])   # off the lat grid
    assert got.shape == lat.shape


def test_climatology_lookup_direct(gpbl):
    """Layer 2 alone, exercising the descending-lat axis: both the northmost
    and southmost cells resolve to their own values; off-grid gives NaN."""
    lat = np.array([41.5, 39.5, 10.5])
    lon = np.array([-90.5, -92.5, -90.5])
    t = np.array(["2019-06-05T18:00", "2019-01-15T02:00", "2019-06-05T18:00"],
                 dtype="datetime64[ns]")
    got = gpbl.climatology_lookup(lat, lon, t.astype(np.int64))
    assert got[0] == pytest.approx(clim_value(6, 18, 41.5, -90.5))
    assert got[1] == pytest.approx(clim_value(1, 3, 39.5, -92.5))  # 02:00 -> slot 3
    assert np.isnan(got[2])
    assert got.shape == lat.shape
