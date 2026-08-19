"""front_formats.xml_to_codsus: antimeridian handling of the NOAA polylines.

Regression guard for the 2026-08-17 label bug: raw [-180, 180] longitudes fed
to the plain-lat/lon stroker turned a front crossing the antimeridian into a
~352-degree segment, painting a full-width horizontal bar across the grid at
that front's latitude (inside the analysis domain in ~36 % of 2016-2018
analyses).
"""
from __future__ import annotations

import numpy as np
import pytest

import codsus_regen as cr
from front_formats import xml_to_codsus as x2c


def test_unwrap_lon_leaves_a_non_crossing_polyline_alone():
    lon = np.array([-95.0, -92.5, -88.0])
    assert np.allclose(x2c.unwrap_lon(lon), lon)


def test_unwrap_lon_takes_the_short_way_round_the_antimeridian():
    """-178.1 -> +174.5 is a 7.4-degree step west, not a 352-degree step east."""
    out = x2c.unwrap_lon(np.array([-174.3, -178.1, 174.5]))
    assert np.allclose(out, [-174.3, -178.1, -185.5])
    assert np.abs(np.diff(out)).max() < 180.0


def test_unwrap_lon_picks_the_branch_nearest_the_grid():
    """A front entering from the far side keeps its in-grid points in grid.

    Anchoring the branch on the first point alone would push the whole line to
    +178/+190/+200 and off the grid; the mean-based shift brings it back so
    only the genuinely out-of-grid first point sits west of the -171 edge.
    """
    out = x2c.unwrap_lon(np.array([178.0, -170.0, -160.0]))
    assert np.allclose(out, [-182.0, -170.0, -160.0])


def test_crossing_front_paints_no_bar_across_the_grid():
    """The 2017-07-25 00Z stationary front: one cell column at most, not a row
    spanning the domain."""
    pts = np.array([(36.5, -168.9), (34.6, -170.8), (33.2, -174.3),
                    (32.9, -178.1), (32.4, 174.5), (33.0, 167.2),
                    (34.2, 163.1), (35.4, 160.4)])
    raw = np.zeros((len(cr.GRID_LATS), len(cr.GRID_LONS)), dtype=np.float32)
    fixed = np.zeros_like(raw)
    cr.rasterize_polyline(pts, raw, 1)
    cr.rasterize_polyline(np.column_stack([pts[:, 0], x2c.unwrap_lon(pts[:, 1])]),
                          fixed, 1)
    assert raw.sum(axis=1).max() > 100      # the bug: a full-width bar
    assert fixed.sum(axis=1).max() <= 3     # only the genuine western tail
    assert (fixed <= raw).all()             # the fix only ever REMOVES cells


def test_parse_xml_unwraps(tmp_path):
    xml = tmp_path / "pres_pmsl_2017072500f000.xml"
    xml.write_text(
        '<Products><Product><Layer><DrawableElement><Line pgenType='
        '"STATIONARY_FRONT"><Point Lat="32.9" Lon="-178.1"/>'
        '<Point Lat="32.4" Lon="174.5"/></Line></DrawableElement>'
        '</Layer></Product></Products>')
    (ftype, pts), = x2c.parse_xml(str(xml))
    assert ftype == "stationary"
    assert pts[1, 1] == pytest.approx(-185.5)


def test_split_valid_cuts_the_line_at_a_sentinel_point():
    """2014-12-14 18Z carries Lat=-9999 Lon=10359 mid-polyline."""
    pts = np.array([(87.6, -79.3), (-9999.0, 10359.0), (88.3, -81.7)])
    runs = x2c.split_valid(pts)
    assert [r.tolist() for r in runs] == [[[87.6, -79.3]], [[88.3, -81.7]]]


def test_sentinel_point_paints_no_stripe_across_the_grid():
    pts = np.array([(85.2, -145.9), (87.6, -79.3), (-9999.0, 10359.0)])
    raw = np.zeros((len(cr.GRID_LATS), len(cr.GRID_LONS)), dtype=np.float32)
    cr.rasterize_polyline(pts, raw, 1)
    assert raw.sum() > 50                      # the bug: a line off the map

    clean = np.zeros_like(raw)
    for run in x2c.split_valid(pts):
        cr.rasterize_polyline(
            np.column_stack([run[:, 0], x2c.unwrap_lon(run[:, 1])]), clean, 1)
    assert clean.sum(axis=0).max() <= 3        # no grid-height stripe
