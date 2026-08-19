"""Analytic checks on the 1 deg PBL-climatology gridding rules."""
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_pbl_climatology",
    Path(__file__).resolve().parents[1] / "scripts" / "build_pbl_climatology.py")
bpc = importlib.util.module_from_spec(_SPEC)
sys.modules["build_pbl_climatology"] = bpc
_SPEC.loader.exec_module(bpc)


def _native():
    return bpc.native_coords("uniform", None)


def test_native_axes_match_the_audited_corpus():
    lat, lon = _native()
    assert (lat[0], lat[-1], len(lat)) == (90.0, -59.75, 600)
    assert (lon[0], lon[-1], len(lon)) == (0.0, -0.25, 1440)  # wrapped to [-180,180)


def test_target_grid_centres_sit_on_the_FCST_conus_grid():
    lat, lon = bpc.target_grid(bpc.DOMAINS["conus"])
    assert lat[0] == 52.5 and lat[-1] == 25.5 and len(lat) == 28
    assert lon[0] == -106.5 and lon[-1] == -64.5 and len(lon) == 43


def test_interior_cells_collect_exactly_the_4x4_native_block():
    lat_n, lon_n = _native()
    lat_t, lon_t = bpc.target_grid(bpc.DOMAINS["conus"])
    keep, flat_idx, n_native = bpc.build_index(lat_n, lon_n, lat_t, lon_t)
    assert keep.sum() == 28 * 4 * 43 * 4
    assert set(np.unique(n_native)) == {16}
    # every cell index is used, none out of range
    assert flat_idx.min() == 0 and flat_idx.max() == 28 * 43 - 1


def test_global_domain_edges_are_partial_where_the_source_stops():
    lat_n, lon_n = _native()
    lat_t, lon_t = bpc.target_grid(bpc.DOMAINS["global"])
    _, _, n_native = bpc.build_index(lat_n, lon_n, lat_t, lon_t)
    n = n_native.reshape(len(lat_t), len(lon_t))
    assert lat_t[0] == 89.5 and lat_t[-1] == -59.5
    assert (n[1:-1] == 16).all()          # interior rows fully sampled
    assert (n[0] == 16).all()             # 89.0..89.75 (the +90.0 row is its own cell)
    assert (n[-1] == 12).all()            # -60.0 row absent from the source


def test_pole_row_is_dropped_rather_than_folded_into_89_5():
    lat_n, lon_n = _native()
    lat_t, lon_t = bpc.target_grid(bpc.DOMAINS["global"])
    keep, _, _ = bpc.build_index(lat_n, lon_n, lat_t, lon_t)
    assert not keep.reshape(600, 1440)[0].any()


def test_local_solar_offsets_are_whole_slots_and_signed_by_hemisphere():
    lon = np.array([-112.5, -67.5, -22.5, 0.0, 22.5, 112.5])
    off = bpc.lst_slot_offset(lon)
    assert off.tolist() == [-2, -1, 0, 0, 1, 3]  # floor(lon/45 + 1/2), half-up
    # a whole hemisphere of centres maps monotonically, one step per 45 deg
    lon_all = bpc.target_grid(bpc.DOMAINS["global"])[1]
    assert (np.diff(bpc.lst_slot_offset(lon_all)) >= 0).all()


def test_accumulator_pools_every_valid_sample_with_equal_weight():
    ncell = 4
    acc = bpc.Accumulator(ncell)
    groups = bpc.column_groups(ncell, 2, None)          # UTC: one group, shift 0
    values = np.array([10.0, 20.0, 0.0, 0.0])
    red = (values, values ** 2, np.array([2, 1, 0, 0]))  # cell0: 2 samples summing 10
    acc.add_file(datetime(2019, 7, 4, 21), red, groups)
    acc.add_file(datetime(2019, 7, 5, 21), red, groups)
    assert acc.total[6, 7, 0] == 20.0 and acc.n_obs[6, 7, 0] == 4
    assert acc.n_times[6, 7, 0] == 2 and acc.n_times[6, 7, 2] == 0
    assert acc.total[6, 6, 0] == 0.0                     # nothing leaked to 18 UTC


def test_lst_mode_shifts_a_western_column_into_an_earlier_slot():
    lat_t, lon_t = bpc.target_grid(dict(lat_min=30, lat_max=31, lon_min=-113, lon_max=-111))
    ncell = len(lon_t)
    groups = bpc.column_groups(ncell, ncell, bpc.lst_slot_offset(lon_t))
    acc = bpc.Accumulator(ncell)
    red = (np.full(ncell, 500.0), np.full(ncell, 250000.0), np.ones(ncell, dtype=np.int64))
    acc.add_file(datetime(2019, 7, 4, 0), red, groups)   # 00 UTC at ~112 W -> 16 LST
    assert acc.n_obs[6, :, 0].tolist() == [0, 0, 0, 0, 0, 0, 1, 0]  # slot 6 = 18 h
    assert acc.total[6, 6, 0] == 500.0


def test_years_parser():
    assert bpc.parse_years("2017,2019-2021") == [2017, 2019, 2020, 2021]
    assert bpc.parse_years(None) is None
