"""Smoke tests for the trajectory-kernel plots: they build a figure without error.

Plot *logic* is tested elsewhere (footprint/apply); here we only guard that each
diagnostic renders on a synthetic kernel and on real data if present.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

from trajectory_kernels import config
from trajectory_kernels import plotting as P


def _single_receptor():
    lag = np.array([0.0, 1.0, 2.0])
    slat = np.arange(35.5, 40.5 + 0.01, 1.0)
    slon = np.arange(-92.5, -87.5 + 0.01, 1.0)
    fp = np.zeros((lag.size, slat.size, slon.size))
    fp[0, 2, 3] = 2.0
    fp[1, 2, 2] = 1.0
    # per-lag-hour normalization (empty lag 2 -> NaN), the builders' convention
    lag_totals = fp.sum(axis=(1, 2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(lag_totals > 0, fp / lag_totals, np.nan)
    ds = xr.Dataset(
        {"footprint": (("lag", "source_lat", "source_lon"), fp),
         "kernel": (("lag", "source_lat", "source_lon"), k),
         "member_parcel": ("member", np.array([0, 1], dtype="int32"))},
        coords={"lag": lag, "source_lat": slat, "source_lon": slon,
                "dlat": ("source_lat", slat - 37.5), "dlon": ("source_lon", slon - -89.5)},
        attrs={"target_lat": 37.5, "target_lon": -89.5, "n_parcels": 2, "arrival_step": 3},
    )
    return ds


def _matching_day():
    """Two parcels whose histories end at the receptor of _single_receptor."""
    times = np.array(["2019-06-05T18:00", "2019-06-05T21:00",
                      "2019-06-05T22:00", "2019-06-05T23:00"],
                     dtype="datetime64[ns]")
    lons = np.array([[-92.5, -91.5, -90.5, -89.5],
                     [-93.0, -91.8, -90.6, -89.5]])
    return xr.Dataset(
        {"time_utc": (("parcel", "step"), np.tile(times, (2, 1))),
         "lat": (("parcel", "step"), np.full((2, 4), 37.5)),
         "lon": (("parcel", "step"), lons),
         "alt": (("parcel", "step"), np.full((2, 4), 300.0))},
        coords={"parcel": [0, 1], "step": np.arange(4)},
    )


def test_kernel_and_marginal_plots_render():
    ds = _single_receptor()
    for fn in (lambda: P.plot_kernel_at_lag(ds, 1.0),
               lambda: P.plot_kernel_evolution(ds),
               lambda: P.plot_spatial_influence(ds),
               lambda: P.plot_temporal_influence(ds)):
        fig, _ = fn()
        assert fig is not None


def test_trajectory_overlay_renders():
    ds = _single_receptor()
    day = _matching_day()
    for fn in (lambda: P.plot_kernel_at_lag(ds, 1.0, day=day),
               lambda: P.plot_kernel_evolution(ds, day=day),
               lambda: P.plot_spatial_influence(ds, day=day)):
        fig, ax = fn()
        # the overlay adds trajectory lines beyond the base plot's artists
        assert len(ax.lines) >= 2


def test_region_outline_is_one_closed_loop():
    # a single True cell -> exactly its own cell box, closed; an L of 3 cells
    # -> ONE closed loop tracing the 8 boundary edges (never fragments)
    lat = np.array([36.5, 37.5, 38.5])
    lon = np.array([-89.5, -88.5, -87.5])
    one = np.zeros((3, 3), dtype=bool)
    one[1, 1] = True
    loops = P._region_outlines(one, lat, lon)
    assert len(loops) == 1
    loop = loops[0]
    assert np.allclose(loop[0], loop[-1])  # closed
    xs, ys = sorted(set(loop[:, 0])), sorted(set(loop[:, 1]))
    assert xs == [-89.0, -88.0] and ys == [37.0, 38.0]  # the cell's edges

    ell = one.copy()
    ell[0, 1] = ell[1, 2] = True
    loops = P._region_outlines(ell, lat, lon)
    assert len(loops) == 1                      # connected region, one outline
    assert np.allclose(loops[0][0], loops[0][-1])
    assert len(loops[0]) == 7                   # 6 corners + closing repeat


def test_region_outline_inset_shrinks_inward():
    lat = np.array([36.5, 37.5, 38.5])
    lon = np.array([-89.5, -88.5, -87.5])
    one = np.zeros((3, 3), dtype=bool)
    one[1, 1] = True
    (loop,) = P._region_outlines(one, lat, lon, inset=0.1)
    assert np.allclose(loop[0], loop[-1])       # still exactly closed
    assert sorted(set(loop[:, 0])) == [-88.9, -88.1]
    assert sorted(set(loop[:, 1])) == [37.1, 37.9]


def test_com_region_mask_grows_from_centroid():
    # weights 8/4/2/1 (total 15): the mass centroid sits nearest the 8-cell, so
    # the 50% region is that cell alone (53%); the 90% region adds the next two
    # cells outward (14/15) but never the far 1-cell -- and it is connected
    lat = np.array([37.5, 38.5])
    lon = np.array([-90.5, -89.5])
    w = np.array([[8.0, 4.0], [2.0, 1.0]])
    m50 = P._com_region_mask(w, lat, lon, 0.5)
    assert m50.sum() == 1 and m50[0, 0]
    m90 = P._com_region_mask(w, lat, lon, 0.9)
    assert m90.sum() == 3 and not m90[1, 1]
    assert P._com_region_mask(np.zeros((2, 2)), lat, lon, 0.5) is None


def test_overlay_skipped_without_member_list():
    """build_all output has no member_parcel: day-passing must not crash."""
    ds = _single_receptor().drop_vars("member_parcel")
    fig, ax = P.plot_spatial_influence(ds, day=_matching_day())
    assert fig is not None


@pytest.mark.skipif(
    not (config.TRAJ_DIR / config.NOGRID_TEMPLATE.format(granule=189)).exists(),
    reason="trajectory data not present",
)
def test_real_plots_render(tmp_path):
    from trajectory_kernels import trajectories as T
    day = T.load_day()
    fig, _ = P.plot_trajectories(day, max_parcels=50, save_path=tmp_path / "t.png")
    assert (tmp_path / "t.png").exists()
