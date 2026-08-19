"""dl_front.six_panel: sample-pool and colormap logic (no TF, no real data;
the actual figure rendering is exercised manually against cluster data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_front import config, dataset, evaluate_test, six_panel

#: ``dataset.analysis_domain()``/``crop_domain()``/``region_mask()``
#: interpolate the land-fraction mask off disk, so even synthetic tests that
#: reach them need the data root's mask file.  Skipped, never failed, on
#: checkouts without a populated data root.
needs_land_mask = pytest.mark.skipif(
    not config.LAND_MASK_PATH.exists(),
    reason=f"land mask {config.LAND_MASK_PATH} not on disk "
           f"(set JPL_AIRS_DATA to a populated data root)")

N_CLASSES = 6


def _synthetic_source(times, x_channels=1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(times)
    x = rng.random((n, *config.GRID_SHAPE, x_channels)).astype(np.float32)
    y = np.full((n, *config.GRID_SHAPE), N_CLASSES - 1, dtype=np.uint8)
    return x, y, pd.DatetimeIndex(times)


def test_sample_pool_intersects_three_sources_and_filters_months(monkeypatch):
    """Only timestamps present in ALL of reanalysis/kriged-airs/bk19, and
    only March-November, enter the pool."""
    year = 2016
    # reanalysis: every AIRS hour, all months (pre-filter_hours input)
    all_hours_times = pd.date_range(f"{year}-01-01", f"{year}-12-31",
                                    freq="3h")[:-1]
    rea_src = _synthetic_source(all_hours_times, seed=1)
    # kriged-airs: AIRS_HOURS only, but only Feb and June (sparse archive)
    krig_times = pd.DatetimeIndex(
        [t for t in all_hours_times
         if t.hour in config.AIRS_HOURS and t.month in (2, 6)])
    krig_src = _synthetic_source(krig_times, seed=2)
    # bk19: covers everything at AIRS_HOURS (published archive is dense)
    bk19_times = pd.DatetimeIndex(
        [t for t in all_hours_times if t.hour in config.AIRS_HOURS])
    bk19_src = _synthetic_source(bk19_times, seed=3)

    def fake_load_year(y, n_classes, stats, source):
        assert y == year and n_classes == N_CLASSES
        return {"reanalysis": rea_src, "kriged-airs": krig_src}[source]

    def fake_bk19_year_arrays(y, n_classes):
        assert y == year and n_classes == N_CLASSES
        return bk19_src

    monkeypatch.setattr(evaluate_test, "load_year", fake_load_year)
    monkeypatch.setattr(evaluate_test, "bk19_year_arrays",
                        fake_bk19_year_arrays)
    monkeypatch.setattr(six_panel, "YEARS", (year,))

    pool, year_sources = six_panel.sample_pool(N_CLASSES, stats={})
    assert set(pool["year"]) == {year}
    assert pool["time"].dt.month.isin(six_panel.MONTHS).all()
    assert pool["time"].dt.hour.isin(config.AIRS_HOURS).all()
    # June (in MONTHS) survives the sparse kriged-airs archive; February
    # (has kriged-airs coverage but outside MONTHS) does not
    assert (pool["time"].dt.month == 6).any()
    assert not (pool["time"].dt.month == 2).any()
    assert year in year_sources


def test_sample_pool_raises_when_no_common_timestamps(monkeypatch):
    year = 2016
    disjoint_a = pd.date_range(f"{year}-03-01", periods=2, freq="D")
    disjoint_b = pd.date_range(f"{year}-09-01", periods=2, freq="D")

    monkeypatch.setattr(
        evaluate_test, "load_year",
        lambda y, n, s, source: _synthetic_source(
            disjoint_a if source == "reanalysis" else disjoint_b))
    monkeypatch.setattr(evaluate_test, "bk19_year_arrays",
                        lambda y, n: _synthetic_source(disjoint_b))
    monkeypatch.setattr(six_panel, "YEARS", (year,))

    with pytest.raises(RuntimeError, match="no timestamps common"):
        six_panel.sample_pool(N_CLASSES, stats={})


def test_class_cmap_covers_every_class_with_a_distinct_color():
    cmap, norm, names = six_panel._class_cmap(N_CLASSES)
    assert names == dataset.class_names(N_CLASSES)
    colors = [cmap(norm(i)) for i in range(len(names))]
    assert len(set(colors)) == len(names)          # all distinct
    # 'none' (last class) is the near-white recessive color, not a hue
    assert colors[-1][:3] == tuple(
        int(six_panel.NONE_COLOR[i:i + 2], 16) / 255
        for i in (1, 3, 5))


@needs_land_mask
def test_mask_outside_nans_pixels_outside_analysis_domain():
    cls = np.zeros(config.GRID_SHAPE, dtype=np.uint8)
    out = six_panel._mask_outside(cls)
    domain = dataset.analysis_domain()
    assert np.isnan(out[~domain]).all()
    assert not np.isnan(out[domain]).any()
    np.testing.assert_array_equal(out[domain], 0.0)


@needs_land_mask
def test_render_places_a_marked_cell_at_its_true_lon_lat(tmp_path, monkeypatch):
    """Regression (2026-08-17): the panels are FULL 68x141 grids, so imshow
    must get the full-grid edge extent.  Passing the crop window instead
    squeezed the whole -171..-31 E grid into it and displaced every front by
    tens of degrees.  Mark one cell and check the drawn image puts it on its
    own lon/lat, with the crop window as the visible frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dl_front.quicklook import _window

    lat_i, lon_j = 42, 100                       # 52 N, 71 W on the label grid
    lat, lon = config.LABEL_LATS[lat_i], config.LABEL_LONS[lon_j]
    assert dataset.analysis_domain()[lat_i, lon_j], "pick an in-domain cell"

    cls = np.full(config.GRID_SHAPE, N_CLASSES - 1, dtype=np.uint8)
    cls[lat_i, lon_j] = 0                        # 'cold'
    blank = np.full(config.GRID_SHAPE, N_CLASSES - 1, dtype=np.uint8)

    captured = []
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    six_panel.render(pd.Timestamp("2016-04-14"), cls, blank, blank,
                     blank, blank, blank, N_CLASSES, 0, tmp_path)
    try:
        ax = captured[0].axes[0]                 # the met-drawn truth panel
        image, = ax.images
        # the cell's own lon/lat must index back to the cell it was set in
        lo, hi, la_lo, la_hi = image.get_extent()
        n_lat, n_lon = config.GRID_SHAPE
        assert int((lon - lo) / (hi - lo) * n_lon) == lon_j
        assert int((lat - la_lo) / (la_hi - la_lo) * n_lat) == lat_i
        # ... and the view is the crop window, not the whole label grid
        crop = _window(dataset.crop_domain())
        assert ax.get_xlim() == pytest.approx(crop[:2])
        assert ax.get_ylim() == pytest.approx(crop[2:])
    finally:
        for fig in captured:
            plt.Figure.clf(fig)
