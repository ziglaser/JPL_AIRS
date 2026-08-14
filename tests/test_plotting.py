"""Tests for the CONUS map plotting helper (synthetic data; no NetCDF needed).

Requires cartopy for the projection/outlines; skipped where it is unavailable.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cartopy")  # map helpers need cartopy; skip the module otherwise

from convection_skill.plotting import (
    _panel_layout,
    _resolve_per_field,
    animate_field_map,
    make_conus_axes,
    plot_field_map,
)


@pytest.fixture()
def table() -> pd.DataFrame:
    """Tiny tidy table: 1 day x 2 hours x 3x4 grid, two fields."""
    lats = np.array([33.5, 34.5, 35.5])
    lons = np.array([-100.5, -99.5, -98.5, -97.5])
    rows = []
    for hour in (21, 22):
        for la in lats:
            for lo in lons:
                rows.append(dict(date=pd.Timestamp("2019-06-01"), hour_utc=hour,
                                 lat=la, lon=lo, qpe=la + lo, cape=hour * 10 + la))
    return pd.DataFrame(rows)


def test_maps_selected_day_hour_and_box(table):
    ax = make_conus_axes(extent=(-101, -98, 33, 35))
    mesh = plot_field_map(ax, table, "qpe", "2019-06-01", 21,
                          lat_range=(33, 35), lon_range=(-101, -98))
    # 2 lat rows x 3 lon cols inside the box, values = lat + lon.
    grid = mesh.get_array().reshape(2, 3)
    expected = np.add.outer([33.5, 34.5], [-100.5, -99.5, -98.5])
    np.testing.assert_allclose(grid, expected)
    plt.close(ax.figure)


def test_builds_geoaxes_when_ax_is_none(table):
    """ax=None auto-creates a projected GeoAxes for the requested box."""
    from cartopy.mpl.geoaxes import GeoAxes

    mesh = plot_field_map(None, table, "qpe", "2019-06-01", 21,
                          lat_range=(33, 36), lon_range=(-101, -97))
    assert isinstance(mesh.axes, GeoAxes)
    plt.close(mesh.axes.figure)


def test_missing_cells_render_as_nan(table):
    holey = table[~((table["lat"] == 34.5) & (table["lon"] == -99.5))]
    ax = make_conus_axes(extent=(-101, -97, 33, 36))
    mesh = plot_field_map(ax, holey, "qpe", "2019-06-01", 22)
    grid = np.ma.getdata(mesh.get_array().reshape(3, 4))
    assert np.isnan(grid[1, 1]) or np.ma.is_masked(mesh.get_array().reshape(3, 4)[1, 1])
    assert np.isfinite(grid[0, 0])
    plt.close(ax.figure)


def test_rejects_plain_axes(table):
    """A non-projected Axes is a clear error, not a silently unprojected map."""
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="GeoAxes"):
        plot_field_map(ax, table, "qpe", "2019-06-01", 21)
    plt.close(fig)


def test_empty_selection_raises_with_context(table):
    ax = make_conus_axes()
    with pytest.raises(ValueError, match="Hours present"):
        plot_field_map(ax, table, "qpe", "2019-06-01", 2)  # hour not in table
    plt.close(ax.figure)


def test_unknown_field_lists_available_columns(table):
    with pytest.raises(KeyError, match="not in the table"):
        plot_field_map(None, table, "nope", "2019-06-01", 21)


# --------------------------------------------------------------------------- #
# multi-field panels
# --------------------------------------------------------------------------- #
def test_panel_layout_heuristic():
    assert _panel_layout(1) == (1, 1)
    assert _panel_layout(2) == (1, 2)
    assert _panel_layout(3) == (1, 3)     # up to 3 -> single row
    assert _panel_layout(4) == (2, 2)     # then near-square
    assert _panel_layout(6) == (2, 3)
    assert _panel_layout(5, ncols=5) == (1, 5)  # explicit override


def test_resolve_per_field_broadcasts_and_maps():
    assert _resolve_per_field("v", ["a", "b"]) == {"a": "v", "b": "v"}
    assert _resolve_per_field({"a": 1}, ["a", "b"], default=0) == {"a": 1, "b": 0}
    assert _resolve_per_field([1, 2], ["a", "b"]) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="one per field"):
        _resolve_per_field([1], ["a", "b"])


def test_multiple_fields_return_panel_figure(table):
    """A sequence of fields draws one GeoAxes panel each in a single figure."""
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.figure import Figure

    fig = plot_field_map(None, table, ["qpe", "cape"], "2019-06-01", 21,
                         cmap={"qpe": "Blues", "cape": "turbo"})
    assert isinstance(fig, Figure)
    panels = [a for a in fig.axes if isinstance(a, GeoAxes)]
    assert len(panels) == 2
    plt.close(fig)


def test_multiple_fields_reject_explicit_axes(table):
    ax = make_conus_axes()
    with pytest.raises(TypeError, match="ax=None"):
        plot_field_map(ax, table, ["qpe", "cape"], "2019-06-01", 21)
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# animation
# --------------------------------------------------------------------------- #
def test_animate_saves_gif_with_one_frame_per_hour(table, tmp_path):
    from matplotlib.animation import FuncAnimation

    out = tmp_path / "anim.gif"
    anim = animate_field_map(table, "qpe", "2019-06-01", hours=(21, 22),
                             save_path=out, fps=2)
    assert isinstance(anim, FuncAnimation)
    assert out.exists() and out.stat().st_size > 0

    from PIL import Image
    with Image.open(out) as im:
        assert im.n_frames == 2  # one frame per requested hour
    plt.close("all")


def test_animate_multi_field_fixed_color_scale(table):
    """Multi-field animation builds one panel per field with a stable scale."""
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.collections import QuadMesh

    anim = animate_field_map(table, ["qpe", "cape"], "2019-06-01", hours=(21, 22))
    panels = [a for a in anim._fig.axes if isinstance(a, GeoAxes)]
    assert len(panels) == 2
    # color limits are set once (not per frame); the data mesh (not the cartopy
    # feature collections) carries a finite, ordered vmin/vmax held across frames.
    for ax in panels:
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        vmin, vmax = mesh.get_clim()
        assert np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax
    plt.close("all")
