"""Before/after figures for the two front-label pipeline bugs (2026-08-17).

The six-panel evaluation figures showed long dead-straight HORIZONTAL bars in
the "met-drawn truth" panel and an analysis domain shrunk to a staircase blob
over the east coast.  Both came from the label/plot pipeline, not the model:

1. ``front_formats.xml_to_codsus.parse_xml`` fed raw [-180, 180] longitudes
   straight to the plain-lat/lon stroker.  A front crossing the antimeridian
   (e.g. -178.1 -> +174.5) therefore became a 352-degree segment and painted a
   full-width horizontal line at its latitude -- inside the analysis domain in
   ~36 % of all 2016-2018 analyses.  Fixed by unwrapping each polyline onto a
   single 360-degree branch (``xml_to_codsus.unwrap_lon``).
2. ``dl_front.six_panel.render`` passed ``_window(crop_domain())`` as the
   imshow extent for FULL 68x141 grids, squeezing the whole -171..-31 E /
   10..77 N grid into the crop box.  Fixed by using ``_edge_extent()`` and
   zooming with the axis limits.

Each figure is one analysis time, three panels: the old plot, the old labels
drawn correctly, and the fixed labels (removed cells ringed).

Usage::

    PYTHONPATH=src python scripts/plot_front_label_fixes.py \\
        --times 20160414_0000 20170725_0000 \\
        --out-dir results/dl_front/quicklook/label_fixes
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import xarray as xr                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import codsus_regen as cr                                   # noqa: E402
from dl_front import dataset as dl_dataset                  # noqa: E402
from dl_front.quicklook import INK, OUT_OF_DOMAIN_GRAY, _edge_extent, _window  # noqa: E402
from dl_front.six_panel import FRONT_COLORS, NONE_COLOR     # noqa: E402
from front_formats import xml_to_codsus as x2c              # noqa: E402

XML_TEMPLATE = "{year}/{month:02d}/{day:02d}/pres_pmsl_{stamp}f000.xml"


def xml_path(when: pd.Timestamp) -> Path:
    """The raw NOAA analysis backing one label timestep."""
    return x2c.XML_DIR / XML_TEMPLATE.format(
        year=when.year, month=when.month, day=when.day,
        stamp=f"{when:%Y%m%d%H}")


def parse_xml_unfixed(path: Path) -> list:
    """:func:`x2c.parse_xml` WITHOUT the longitude unwrap -- the shipped files."""
    root = ET.parse(path, parser=ET.XMLParser(encoding="utf-8")).getroot()
    fronts = []
    for line in root.iter("Line"):
        ftype = x2c.PGEN_TO_TYPE.get(line.get("pgenType"))
        if ftype is None:
            continue
        pts = np.array([(float(p.get("Lat")), float(p.get("Lon")))
                        for p in line.iter("Point")])
        if len(pts):
            fronts.append((ftype, pts))
    return fronts


def class_grid(fronts: list, width: int, mask: np.ndarray) -> np.ndarray:
    """Rasterize one analysis to a class-index grid ('none' = last class)."""
    grids = x2c.rasterize_analysis(fronts, width, mask)
    names = dl_dataset.class_names(6)
    cls = np.full(grids.shape[1:], len(names) - 1, dtype=np.uint8)
    for i, name in enumerate(x2c.TYPES):
        cls[grids[i] > 0] = names.index(name)
    return cls


def _cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    names = dl_dataset.class_names(6)
    cmap = ListedColormap([FRONT_COLORS[n] for n in names[:-1]] + [NONE_COLOR])
    cmap.set_bad(OUT_OF_DOMAIN_GRAY)
    return cmap, BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names)), names


def _grayed(cls: np.ndarray, domain: np.ndarray) -> np.ndarray:
    out = cls.astype(np.float32)
    out[~domain] = np.nan
    return out


def render(when: pd.Timestamp, old: np.ndarray, new: np.ndarray,
           out_dir: Path) -> Path:
    cmap, norm, names = _cmap()
    domain = dl_dataset.analysis_domain()
    crop = _window(dl_dataset.crop_domain())
    box = _window(domain)
    full = _edge_extent()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panels = [
        ("BEFORE: shipped labels, buggy plot extent", old, crop, crop),
        ("bug 2 fixed: same labels, true geometry", old, full, crop),
        ("bugs 1+2 fixed: unwrapped labels", new, full, crop),
    ]
    for ax, (title, cls, extent, window) in zip(axes, panels):
        ax.imshow(_grayed(cls, domain), origin="lower", extent=extent,
                  cmap=cmap, norm=norm, interpolation="nearest")
        ax.plot([box[0], box[1], box[1], box[0], box[0]],
                [box[2], box[2], box[3], box[3], box[2]],
                color=INK, lw=0.6, ls="--")
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        ax.set_title(title, color=INK, fontsize=9)
        ax.tick_params(colors=INK, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("0.6")

    # ring the cells the unwrap removed, on the true-geometry "before" panel
    dropped = (old != new) & domain
    if dropped.any():
        rows, cols = np.nonzero(dropped)
        axes[1].scatter(np.asarray(config_lons())[cols],
                        np.asarray(config_lats())[rows],
                        s=6, facecolors="none", edgecolors=INK, linewidths=0.4)
        axes[1].set_title(f"bug 2 fixed: true geometry "
                          f"({dropped.sum()} spurious cells ringed)",
                          color=INK, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FRONT_COLORS[n])
               for n in names[:-1]] + \
              [plt.Rectangle((0, 0), 1, 1, color=NONE_COLOR, ec="0.6", lw=0.5)]
    fig.legend(handles, list(names), loc="lower center", ncol=len(names),
               frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{when:%Y-%m-%d %H:%MZ} met-drawn front labels "
                 f"(dashed = analysis domain, gray = outside it)",
                 color=INK, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"label_fix_{when:%Y%m%d_%H%M}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def config_lats():
    from dl_front import config
    return config.LABEL_LATS


def config_lons():
    from dl_front import config
    return config.LABEL_LONS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", nargs="+", required=True,
                    help="analysis times as YYYYmmdd_HHMM")
    ap.add_argument("--width", type=int, default=None,
                    help="stroke width (default: dl_front config.LABEL_WIDTH)")
    ap.add_argument("--out-dir", default="results/dl_front/quicklook/label_fixes")
    a = ap.parse_args(argv)

    from dl_front import config
    width = a.width if a.width is not None else config.LABEL_WIDTH
    with xr.open_dataset(cr.MASK_PATH) as m:
        mask = (m["codsus_mask"].values > 0).astype(np.float32)

    out_dir = Path(a.out_dir)
    for stamp in a.times:
        when = pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} "
                            f"{stamp[9:11]}:{stamp[11:13]}")
        path = xml_path(when)
        if not path.exists():
            print(f"{when:%Y-%m-%d %H:%MZ}: no XML at {path} -- skipped")
            continue
        old = class_grid(parse_xml_unfixed(path), width, mask)
        new = class_grid(x2c.parse_xml(str(path)), width, mask)
        print(f"{when:%Y-%m-%d %H:%MZ}: wrote {render(when, old, new, out_dir)}")


if __name__ == "__main__":
    main()
