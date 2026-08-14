"""Analytic-answer tests for ``front_finder.labels``.

Each case is hand-checkable without touching real CODSUS/benchmark data:

- dilate(): a single pixel's 8-connected footprint, the iterations=0 identity,
  and independence across a leading (non-spatial) axis.
- front_stack()/valid_mask(): a tiny synthetic Dataset mimicking the CODSUS
  schema, checked for fill handling and correct reordering by front name
  (not by whatever order the file happens to store them in).
- align_times(): inner join on time, order preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from front_finder import config
from front_finder.labels import align_times, dilate, front_stack, valid_mask


# --------------------------------------------------------------------------- #
# dilate
# --------------------------------------------------------------------------- #

def test_dilate_single_pixel_is_3x3_block():
    """One True pixel, 8-connected, iterations=1 -> the surrounding 3x3 block."""
    a = np.zeros((8, 8), dtype=bool)
    a[3, 3] = True
    d = dilate(a, 1)
    expected = np.zeros((8, 8), dtype=bool)
    expected[2:5, 2:5] = True
    assert np.array_equal(d, expected)


def test_dilate_zero_iterations_is_identity():
    a = np.zeros((8, 8), dtype=bool)
    a[3, 3] = True
    a[0, 0] = True
    d = dilate(a, 0)
    assert np.array_equal(d, a)


def test_dilate_stack_no_bleed_across_leading_axis():
    """A (2, 8, 8) stack dilates each (lat, lon) slice independently."""
    stack = np.zeros((2, 8, 8), dtype=bool)
    stack[0, 3, 3] = True
    stack[1, 5, 5] = True
    d = dilate(stack, 1)

    expected0 = np.zeros((8, 8), dtype=bool)
    expected0[2:5, 2:5] = True
    expected1 = np.zeros((8, 8), dtype=bool)
    expected1[4:7, 4:7] = True

    assert np.array_equal(d[0], expected0)
    assert np.array_equal(d[1], expected1)
    # explicit "no bleed": neither slice's dilation touches the other's pixel
    assert not d[0, 5, 5]
    assert not d[1, 3, 3]


# --------------------------------------------------------------------------- #
# front_stack / valid_mask
# --------------------------------------------------------------------------- #

# Full CODSUS front axis (5 classes: the 4 physical types plus "none").
_ALL_TYPES = ("cold", "warm", "stationary", "occluded", "none")
_LAT = np.array([10.0, 20.0, 30.0, 40.0])
_LON = np.array([-170.0, -160.0, -150.0, -140.0, -130.0])
# Fill pixel shared by every class/time (outside mask, e.g. a coastline cell).
_FILL_LAT, _FILL_LON = 3, 4


def _make_ds(front_order: tuple[str, ...]) -> xr.Dataset:
    """Synthetic 2-time, 5-front, 4x5 grid Dataset with one marked pixel per
    class (indexed by *name*, not position) plus a shared fill pixel.
    """
    time = pd.date_range("2018-01-01", periods=2, freq="3h")
    fronts = np.zeros((2, 5, 4, 5), dtype=np.uint8)
    # One True pixel per physical class, at time index 0, keyed by class name.
    marks = {"cold": (0, 0), "warm": (1, 1), "stationary": (2, 2),
              "occluded": (0, 3), "none": (1, 3)}
    for name, (lat_i, lon_i) in marks.items():
        pos = front_order.index(name)
        fronts[0, pos, lat_i, lon_i] = 1
    # Fill value 2 at one pixel, every class, every time.
    fronts[:, :, _FILL_LAT, _FILL_LON] = config.LABEL_FILL

    return xr.Dataset(
        {
            "fronts": (("time", "front", "lat", "lon"), fronts),
            "front_type": (("front",), np.array(front_order, dtype=object)),
        },
        coords={"time": time, "lat": _LAT, "lon": _LON},
    )


def test_front_stack_is_boolean_and_ordered_per_config():
    ds = _make_ds(_ALL_TYPES)
    stack = front_stack(ds)
    assert stack.dtype == bool
    assert tuple(stack["front"].values) == config.FRONT_TYPES


def test_front_stack_matches_marked_pixels():
    ds = _make_ds(_ALL_TYPES)
    stack = front_stack(ds)
    for name in config.FRONT_TYPES:
        lat_i, lon_i = {"cold": (0, 0), "warm": (1, 1), "stationary": (2, 2),
                         "occluded": (0, 3)}[name]
        plane = stack.sel(front=name).isel(time=0).values
        assert plane[lat_i, lon_i]
        assert plane.sum() == 1  # nothing else lit up for this class


def test_front_stack_fill_pixel_is_false():
    ds = _make_ds(_ALL_TYPES)
    stack = front_stack(ds)
    assert not bool(stack.isel(time=0, lat=_FILL_LAT, lon=_FILL_LON).any())
    assert not bool(stack.isel(time=1, lat=_FILL_LAT, lon=_FILL_LON).any())


def test_valid_mask_false_at_fill_true_elsewhere():
    ds = _make_ds(_ALL_TYPES)
    vm = valid_mask(ds)
    assert vm.dtype == bool
    assert not bool(vm.isel(time=0, lat=_FILL_LAT, lon=_FILL_LON))
    assert not bool(vm.isel(time=1, lat=_FILL_LAT, lon=_FILL_LON))
    # every other pixel is valid (not the fill value on any class)
    other = vm.values.copy()
    other[:, _FILL_LAT, _FILL_LON] = True
    assert other.all()


def test_front_stack_reorders_permuted_front_type():
    """A file that stores front_type in a different order is still read
    correctly, because front_stack looks up by name, not position.
    """
    ds_canonical = _make_ds(_ALL_TYPES)
    permuted_order = ("none", "occluded", "stationary", "warm", "cold")
    ds_permuted = _make_ds(permuted_order)

    stack_canonical = front_stack(ds_canonical)
    stack_permuted = front_stack(ds_permuted)

    assert tuple(stack_permuted["front"].values) == config.FRONT_TYPES
    xr.testing.assert_equal(stack_canonical, stack_permuted)


# --------------------------------------------------------------------------- #
# align_times
# --------------------------------------------------------------------------- #

def test_align_times_inner_joins_and_preserves_order():
    truth_time = pd.date_range("2018-01-01", periods=6, freq="D")
    pred_time = pd.date_range("2018-01-03", periods=6, freq="D")  # overlap: 01-03..01-06

    truth = xr.Dataset({"x": (("time",), np.arange(6))}, coords={"time": truth_time})
    pred = xr.Dataset({"x": (("time",), np.arange(6) * 10)}, coords={"time": pred_time})

    truth_aligned, pred_aligned = align_times(truth, pred)

    expected = pd.date_range("2018-01-03", periods=4, freq="D")
    assert list(truth_aligned["time"].values) == list(expected.values)
    assert list(pred_aligned["time"].values) == list(expected.values)
    # ascending order preserved (not just set equality)
    assert np.all(np.diff(truth_aligned["time"].values.astype("datetime64[ns]").astype("int64")) > 0)
    # values track the correct rows (not just matching time labels)
    assert list(truth_aligned["x"].values) == [2, 3, 4, 5]
    assert list(pred_aligned["x"].values) == [0, 10, 20, 30]


# --------------------------------------------------------------------------- #
# Real-data path builders (manifest reorg 2026-08-13)
# --------------------------------------------------------------------------- #
# The loaders' path arithmetic must match the on-disk tree that mirrors the
# cluster manifest (gattaca2:/gpfs/scratch/smap-convection):
#   WPC:  front_id/met_drawn_fronts/WPC_CODSUS/WPC_1deg_gridded/{w}wide/
#   NOAA: front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded/{w}wide/
#   BK19: front_id/predicted_fronts/bk19/1deg_{w}wide/{1hr,3hr}/
# Skip-if-missing (repo style, cf. needs_demo in test_dlfront_airs_krige):
# CI checkouts without the data tree stay green.

_KNOWN_YEAR = 2016   # inside every archive: WPC 2003-2018+, NOAA 2007-2022,
                     # BK19 1980-2018

needs_local_data = pytest.mark.skipif(
    not config.DATA_ROOT.joinpath("front_id/met_drawn_fronts").is_dir(),
    reason="local front_id data tree not on disk")


@needs_local_data
@pytest.mark.parametrize("width", [1, 3])
def test_wpc_path_builder_hits_real_file(width):
    from front_finder import labels
    ds = labels.load_codsus(_KNOWN_YEAR, width=width)
    assert ds["time"].size > 0
    ds.close()


@needs_local_data
@pytest.mark.parametrize("width", [1, 3])
def test_noaa_path_builder_hits_real_file(width):
    from front_finder import labels
    ds = labels.load_noaa(_KNOWN_YEAR, width=width)
    assert ds["time"].size > 0
    ds.close()


@needs_local_data
def test_bk19_path_builder_hits_real_file():
    from front_finder import labels
    ds = labels.load_benchmark(_KNOWN_YEAR, width=1, freq="3hr")
    assert ds["time"].size > 0
    ds.close()
