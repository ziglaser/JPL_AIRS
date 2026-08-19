"""scripts/add_front_flags.py: front flags written into FCST_SMAP_MRMS files.

The alignment itself is covered by tests/test_codsus_fronts.py; what is tested
here is what the injector adds on top:

* the post-2026-08-13 label-tree paths actually resolve (the stale
  ``data/fronts/`` paths made every front column silently all-NaN);
* the written variables are BIT-IDENTICAL to the in-memory base-table columns
  (``fronts.year_front_flags``), which is the whole point of reusing
  ``fronts.file_front_flags`` instead of reimplementing the pooling;
* every pre-existing variable, attribute, dtype and _FillValue survives, and
  the primary file is never touched;
* the bk19-schema prediction path turns the fill byte 2 into NaN (never 0) and
  keeps model flags under a ``pred_`` prefix that cannot collide with
  ``fronts.front_columns()``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import config
from convection_skill import fronts as fr

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    """Import the script by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "add_front_flags", REPO_ROOT / "scripts" / "add_front_flags.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


afl = _load_script()

# The synthetic grids: 3 half-degree cells in each direction need the 1-degree
# corners 30..32 / -100..-98, and two dates so the Dec-31-style edge (the last
# date's next-day 00 UTC bulletin) is exercised.
LATS = np.array([30.5, 31.5])
LONS = np.array([-99.5, -98.5])
FRONT_LATS = np.array([30.0, 31.0, 32.0])
FRONT_LONS = np.array([-100.0, -99.0, -98.0])
DATES = np.array(["2018-06-01", "2018-06-02"], dtype="datetime64[ns]")
SLOTS = (0,) + config.FORECAST_SLOTS


def _write_label_file(path: Path, types: tuple[str, ...], on_times) -> None:
    """A CODSUS/NOAA-schema year file: float32 0/1 with a trailing ``none``."""
    times = pd.date_range("2018-06-01", "2018-06-03 21:00", freq="3h")
    names = np.array(list(types) + ["none"], dtype="<U12")
    vals = np.zeros((len(times), len(names), len(FRONT_LATS), len(FRONT_LONS)),
                    dtype=np.float32)
    for stamp, channel in on_times:
        vals[list(times).index(np.datetime64(stamp)), types.index(channel)] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"fronts": (("time", "front", "lat", "lon"), vals)},
               coords={"time": times, "lat": FRONT_LATS, "lon": FRONT_LONS,
                       "front_type": ("front", names)}).to_netcdf(path)


def _write_prediction_file(path: Path, in_domain_lats) -> None:
    """A bk19-schema hard-class file: ubyte, _FillValue=2 outside the domain.

    Sparse time axis (21 UTC + next-day 00 UTC only, i.e. dl_front's
    AIRS_HOURS) and float64 fractional days-since-epoch, like a real export.
    """
    times = np.array(["2018-06-01T21", "2018-06-02T00", "2018-06-02T21",
                      "2018-06-03T00"], dtype="datetime64[ns]")
    days = ((times - np.datetime64("1970-01-01T00:00:00"))
            / np.timedelta64(1, "D")).astype("f8")
    names = list(fr.FRONT_TYPES) + [fr.DRYLINE_TYPE, "none"]
    chan = np.zeros((len(times), len(names), len(FRONT_LATS), len(FRONT_LONS)),
                    dtype=np.uint8)
    chan[:, -1] = 1                              # argmax = 'none' everywhere
    chan[0, 0, 1, 1], chan[0, -1, 1, 1] = 1, 0   # a cold front at (31, -99)
    outside = ~np.isin(FRONT_LATS, in_domain_lats)
    chan[:, :, outside, :] = 2                   # untrained region
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("front", len(names))
        nc.createDimension("time", None)
        nc.createDimension("lat", len(FRONT_LATS))
        nc.createDimension("lon", len(FRONT_LONS))
        ftype = nc.createVariable("front_type", str, ("front",))
        for i, name in enumerate(names):
            ftype[i] = name
        var = nc.createVariable("fronts", "u1", ("time", "front", "lat", "lon"),
                                fill_value=np.uint8(2))
        var.coordinates = "front_type lat lon"
        nc.createVariable("lat", "f8", ("lat",))[:] = FRONT_LATS
        nc.createVariable("lon", "f8", ("lon",))[:] = FRONT_LONS
        tvar = nc.createVariable("time", "f8", ("time",))
        tvar.units = "days since 1970-01-01 00:00:00"
        tvar.calendar = "gregorian"
        tvar[:] = days
        var[:] = chan


def _write_primary(path: Path) -> None:
    """A miniature FCST_SMAP_MRMS year file with the conventions we must keep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("date", len(DATES))
        nc.createDimension("time", len(SLOTS))
        nc.createDimension("lat", len(LATS))
        nc.createDimension("lon", len(LONS))
        date = nc.createVariable("date", "i8", ("date",))
        date.units = "days since 2018-01-01"
        date.calendar = "proleptic_gregorian"
        date[:] = [151, 152]
        nc.createVariable("time", "i8", ("time",))[:] = np.array(SLOTS)
        lat = nc.createVariable("lat", "f8", ("lat",), fill_value=np.nan)
        lat[:] = LATS
        lon = nc.createVariable("lon", "f8", ("lon",), fill_value=np.nan)
        lon[:] = LONS
        cape = nc.createVariable("FCST_MU_CAPE", "f4",
                                 ("date", "time", "lat", "lon"),
                                 fill_value=np.float32(np.nan))
        cape.units = "J/kg"
        cape.long_name = "MU CAPE"
        cape[:] = np.arange(cape.size, dtype=np.float32).reshape(cape.shape)


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A self-contained primary file + NOAA label tree + prediction tree."""
    monkeypatch.setattr(fr, "NOAA_FRONTS_DIR", tmp_path / "noaa")
    monkeypatch.setattr(fr, "FRONTS_DIR", tmp_path / "wpc")
    types = fr.FRONT_TYPES + (fr.DRYLINE_TYPE,)
    # a dryline-ONLY bulletin (day 2, 21 UTC) isolates it from the four
    # common types so the 'any' definition can be tested
    on = [("2018-06-01T21:00", "cold"), ("2018-06-02T00:00", "cold"),
          ("2018-06-02T21:00", fr.DRYLINE_TYPE)]
    for width in fr.FRONT_WIDTHS:
        _write_label_file(
            tmp_path / "noaa" / f"{width}wide"
            / fr.NOAA_FILE_TEMPLATE.format(width=width, year=2018), types, on)
    _write_prediction_file(
        tmp_path / "pred" / "1deg_3wide" / "3hr"
        / fr.PRED_FILE_TEMPLATE.format(width=3, year=2018),
        in_domain_lats=[31.0, 32.0])
    _write_primary(tmp_path / "primary" / "FCST_SMAP_MRMS_2018.nc")
    return tmp_path


def _run(tree, *extra):
    afl.main(["--years", "2018", "--label-source", "noaa",
              "--primary-dir", str(tree / "primary"),
              "--out-dir", str(tree / "out"), *extra])
    return tree / "out" / "FCST_SMAP_MRMS_2018.nc"


# --------------------------------------------------------------------------- #
# Path resolution (the 2026-08-18 stale-path fix)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (config.DATA_DIR / "front_id").exists(),
    reason=f"no populated data root at {config.DATA_DIR} (set JPL_AIRS_DATA); "
           f"this test asserts the REAL label trees exist on disk")
def test_label_dirs_point_at_the_post_reorg_tree():
    for source in fr.LABEL_FILE_TEMPLATES:
        root = fr.label_search_dirs(source, 3)[0]
        assert root.parent.name.endswith("1deg_gridded"), root
        assert root.exists(), f"{source} label tree missing: {root}"


def test_label_path_finds_both_years_and_reports_absence(tree):
    assert fr.label_path("noaa", 3, 2018) is not None
    assert fr.label_path("noaa", 3, 1999) is None


def test_prediction_path_uses_the_bk19_layout(tree):
    path = fr.prediction_path(tree / "pred", 3, 2018)
    assert path.parts[-3:] == ("1deg_3wide", "3hr",
                               "merra2_merra2-1deg_3wide_3hr_2018.nc")
    assert fr.prediction_path(tree / "pred", 3, 2017) is None


def test_file_front_types_drops_none_and_keeps_dryline(tree):
    path = fr.label_path("noaa", 3, 2018)
    assert fr.file_front_types(path) == fr.FRONT_TYPES + (fr.DRYLINE_TYPE,)


# --------------------------------------------------------------------------- #
# The written flags are the base table's columns
# --------------------------------------------------------------------------- #
def test_written_flags_equal_the_in_memory_columns(tree):
    out = _run(tree)
    with xr.open_dataset(out) as ds:
        memory = fr.year_front_flags(2018, DATES, config.FORECAST_SLOTS,
                                     LATS, LONS, source="noaa")
        for name in fr.front_columns():
            written = ds[name].isel(time=list(config.FORECAST_SLOTS)).values
            expected = memory[name].values
            np.testing.assert_array_equal(np.isnan(written), np.isnan(expected))
            np.testing.assert_array_equal(written[~np.isnan(written)],
                                          expected[~np.isnan(expected)])


def test_slot_zero_and_missing_bulletins_are_nan_not_zero(tree):
    with xr.open_dataset(_run(tree)) as ds:
        flag = ds["front_cold_3w"]
        assert np.isnan(flag.isel(time=0).values).all()      # overpass slot
        # the last date's slots 4-6 need a 00 UTC bulletin the file has, but
        # its cold channel is off -> 0, while slot 0 stays NaN
        assert (flag.isel(date=0, time=[1, 2, 3]).values == 1.0).all()
        assert (flag.isel(date=1, time=[1, 2, 3]).values == 0.0).all()
        assert set(np.unique(flag.values[np.isfinite(flag.values)])) <= {0.0, 1.0}


def test_any_excludes_dryline_so_it_matches_the_base_table(tree):
    with xr.open_dataset(_run(tree)) as ds:
        # day 2 slots 1-3 read the dryline-only 21 UTC bulletin: the dryline
        # flag fires while 'any' -- max over the four types every source has,
        # i.e. exactly the base table's front_any_* -- stays 0.
        assert (ds["front_dryline_3w"].isel(date=1, time=1).values == 1.0).all()
        assert (ds["front_any_3w"].isel(date=1, time=1).values == 0.0).all()
        assert set(fr.front_columns()) <= set(ds.data_vars)
        assert "front_dryline_3w" in ds.data_vars
        assert "front_dryline_3w" not in fr.front_columns()


# --------------------------------------------------------------------------- #
# Fidelity of the copy
# --------------------------------------------------------------------------- #
def test_original_variables_and_encoding_survive_untouched(tree):
    out = _run(tree)
    src = tree / "primary" / "FCST_SMAP_MRMS_2018.nc"
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(out) as b:
        assert set(a.variables) <= set(b.variables)
        for name in a.variables:
            va, vb = a[name], b[name]
            assert va.dtype == vb.dtype and va.dimensions == vb.dimensions
            assert ({k: str(va.getncattr(k)) for k in va.ncattrs()}
                    == {k: str(vb.getncattr(k)) for k in vb.ncattrs()})
            np.testing.assert_array_equal(va[:], vb[:])
        # no flag variable leaked into the primary
        assert not [n for n in a.variables if n.startswith(afl.FLAG_PREFIXES)]


def test_flag_variables_carry_their_provenance_attrs(tree):
    with netCDF4.Dataset(_run(tree)) as nc:
        var = nc["front_stationary_1w"]
        assert var.dtype == np.float32
        assert np.isnan(var._FillValue)
        assert var.width == 1
        assert var.front_source == "noaa"
        assert var.source.endswith("_1wide_2018.nc")
        assert "slot 0" in var.time_mapping and "21 UTC" in var.time_mapping
        assert "met-drawn" in var.long_name


# --------------------------------------------------------------------------- #
# Idempotency and missing inputs
# --------------------------------------------------------------------------- #
def test_rerun_refuses_without_force_then_replaces_with_force(tree):
    out = _run(tree)
    with pytest.raises(SystemExit, match="already carries flag variables"):
        _run(tree)
    _run(tree, "--force")                      # replaces, does not duplicate
    with netCDF4.Dataset(out) as nc:
        assert sum(n == "front_any_3w" for n in nc.variables) == 1


def test_in_place_appends_to_the_primary_only_when_asked(tree):
    src = tree / "primary" / "FCST_SMAP_MRMS_2018.nc"
    afl.main(["--years", "2018", "--label-source", "noaa",
              "--primary-dir", str(tree / "primary"), "--in-place"])
    with xr.open_dataset(src) as ds:
        assert set(fr.front_columns()) <= set(ds.data_vars)
        assert np.isfinite(ds["front_cold_3w"].isel(date=0, time=1)).all()
    assert not (tree / "out").exists()
    with pytest.raises(SystemExit, match="already carries flag variables"):
        afl.main(["--years", "2018", "--label-source", "noaa",
                  "--primary-dir", str(tree / "primary"), "--in-place"])


def test_missing_label_year_gives_schema_stable_all_nan(tree, monkeypatch):
    monkeypatch.setattr(fr, "NOAA_FRONTS_DIR", tree / "nonexistent")
    with xr.open_dataset(_run(tree)) as ds:
        for name in fr.front_columns():
            assert np.isnan(ds[name].values).all(), name


# --------------------------------------------------------------------------- #
# The bk19-schema prediction leg
# --------------------------------------------------------------------------- #
def test_prediction_flags_are_nan_outside_the_trained_domain(tree):
    out = _run(tree, "--pred-dir", str(tree / "pred"),
               "--pred-tag", "D6C-test")
    with xr.open_dataset(out) as ds:
        cold = ds["pred_front_cold_3w"]
        # lat 30.5 pools the 1-degree rows 30 and 31; row 30 is fill, row 31 is
        # in-domain, so the ANY-overlap rule keeps it (valid_frac 0.5).
        assert np.isfinite(cold.isel(date=0, time=1).values).all()
        np.testing.assert_array_equal(
            ds["pred_front_valid_frac"].sel(lat=30.5).values, [0.5, 0.5])
        np.testing.assert_array_equal(
            ds["pred_front_valid_frac"].sel(lat=31.5).values, [1.0, 1.0])
        # the single predicted cold cell at (31, -99) reaches all four
        # half-degree cells that overlap it
        assert (cold.isel(date=0, time=1).values == 1.0).all()
        assert cold.attrs["model_tag"] == "D6C-test"
        assert "MODEL-PREDICTED" in cold.attrs["long_name"]


def test_prediction_names_cannot_be_confused_with_met_drawn_columns(tree):
    out = _run(tree, "--pred-dir", str(tree / "pred"), "--pred-tag", "t")
    with xr.open_dataset(out) as ds:
        predicted = [n for n in ds.data_vars if n.startswith("pred_front_")]
        assert predicted
        assert not set(predicted) & set(fr.front_columns())


def test_bad_prediction_tree_fails_loudly(tree):
    with pytest.raises(SystemExit, match="no bk19-schema year file"):
        _run(tree, "--pred-dir", str(tree / "empty"), "--pred-tag", "t")
