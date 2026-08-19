"""``dataset.label_digest``: the stale-metrics detector (C5, 2026-08-18).

Why this guard exists: the antimeridian-crossing polyline bug fixed
2026-08-17 (whole horizontal bars painted across the label grid) was
repaired by REGENERATING the front label files in place.  Every metrics CSV
written before that was silently invalidated, and nothing in the pipeline
could tell a number scored on the old labels from one scored on the new.
Each ``_run.json`` now carries this digest and both chain scripts refuse to
reuse a CSV whose digest differs from the labels on disk -- so the digest
has to be (a) stable when nothing moved and (b) sensitive when the scored
labels did.

Mostly synthetic: ``load_label_ds`` is monkeypatched.  The digest is
restricted to ``dataset.analysis_domain()``, though, and that mask
interpolates the land-fraction file off disk -- so even the synthetic tests
need the land mask from the data root and skip (via ``needs_land_mask``)
where it is absent.  The one test that touches real label files (pre-fix vs
regenerated trees) additionally skips when the 2026-08-17 backup tree is
absent, which is the normal state of a fresh checkout -- data/ is
gitignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dl_front import config, dataset

N_CLASSES = 6

#: ``dataset.analysis_domain()`` interpolates the land-fraction mask off
#: disk, so even the synthetic digest tests need the data root's mask file.
needs_land_mask = pytest.mark.skipif(
    not config.LAND_MASK_PATH.exists(),
    reason=f"land mask {config.LAND_MASK_PATH} not on disk "
           f"(set JPL_AIRS_DATA to a populated data root)")


def _label_year(hits) -> xr.Dataset:
    """One 6-class label step; ``hits`` = {(row, col): class name}."""
    names = dataset.class_names(N_CLASSES)
    fronts = np.zeros((1, len(names), *config.GRID_SHAPE), np.float32)
    for (i, j), name in hits.items():
        fronts[0, list(names).index(name), i, j] = 1.0
    return xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": pd.DatetimeIndex(["2016-06-01 21:00"]),
                "lat": list(config.LABEL_LATS),
                "lon": list(config.LABEL_LONS),
                "front_type": ("front", list(names))})


@pytest.fixture
def fake_labels(monkeypatch, tmp_path):
    """Factory: install synthetic labels for every year, return the digest.

    The label directory is faked too (it is hashed in), so a test can vary
    the labels and the tree independently.
    """
    def digest(hits, labels_dir=None, years=(2016,)):
        monkeypatch.setattr(dataset, "load_label_ds",
                            lambda year, n_classes: _label_year(hits))
        monkeypatch.setattr(dataset.fd_config, "NOAA_LABELS_DIR",
                            tmp_path / (labels_dir or "current"))
        return dataset.label_digest(years, N_CLASSES)
    return digest


def _inside_analysis_domain():
    """A (row, col) the 6-class evaluation actually scores, and one it does
    not -- the digest's whole point is that it tracks the SCORED cells."""
    dom = dataset.analysis_domain()
    inside = tuple(int(v) for v in np.argwhere(dom)[0])
    outside = tuple(int(v) for v in np.argwhere(~dom)[0])
    return inside, outside


@needs_land_mask
def test_label_digest_is_stable_and_short(fake_labels):
    """Repeat calls on unchanged labels must agree exactly -- a digest that
    drifted (dict ordering, timestamps, float formatting) would rerun all
    ~19 eval legs of every chain invocation forever."""
    inside, _ = _inside_analysis_domain()
    first = fake_labels({inside: "cold"})
    second = fake_labels({inside: "cold"})
    assert first == second
    assert len(first) == 40 and set(first) <= set("0123456789abcdef")


@needs_land_mask
def test_label_digest_moves_when_scored_labels_move(fake_labels):
    """A relabelled cell inside the scoring mask changes the digest; the
    same change outside it does not.

    Documented behaviour, not an accident: a label change the evaluation
    never reads cannot alter any reported metric, so forcing ~19 legs to
    rerun for it would be pure cost.  The mask is analysis_domain() for
    6-class, region_mask() for 5.
    """
    inside, outside = _inside_analysis_domain()
    base = fake_labels({inside: "cold"})
    assert fake_labels({inside: "warm"}) != base       # class changed
    assert fake_labels({}) != base                     # front removed
    assert fake_labels({inside: "cold", outside: "warm"}) == base


@needs_land_mask
def test_label_digest_tracks_the_label_tree_and_the_years(fake_labels):
    """The resolved label directory and the year set are hashed in.

    The directory matters because pointing the loader at a backup tree
    (the pre-2026-08-17 labels) MUST give a different digest even if the
    counts happened to coincide; the years matter because a CSV scored on
    2016 alone is not comparable to one scored on 2016-2018.

    Corollary the chain scripts rely on, and the reason this is asserted:
    digests are only comparable within one host/data-root, never across
    machines (integration 2026-08-18).
    """
    inside, _ = _inside_analysis_domain()
    base = fake_labels({inside: "cold"})
    assert fake_labels({inside: "cold"}, labels_dir="pre_datelinebug") != base
    assert fake_labels({inside: "cold"}, years=(2016, 2017)) != base


def test_label_digest_differs_for_the_pre_fix_label_tree():
    """The real check: the regenerated labels must NOT digest like the
    pre-fix ones kept at ``NOAA_1deg_gridded.pre_2026-08-17_datelinebug``.

    That backup is the ground truth for "the labels changed": the dateline
    bug painted full-width horizontal bars, which is a large change inside
    the analysis domain.  Skipped (not failed) when the backup is absent --
    data/ is gitignored, so most checkouts, including the cluster after an
    rsync of the fixed tree only, have just the current labels.
    """
    from front_finder import config as fd_config

    year = 2016
    backup = fd_config.NOAA_LABELS_DIR.with_name(
        fd_config.NOAA_LABELS_DIR.name + ".pre_2026-08-17_datelinebug")
    current_file = (fd_config.NOAA_LABELS_DIR / f"{config.LABEL_WIDTH}wide"
                    / f"noaa_fronts_merra2-1deg_{config.LABEL_WIDTH}wide_"
                      f"{year}.nc")
    backup_file = (backup / f"{config.LABEL_WIDTH}wide"
                   / f"noaa_fronts_merra2-1deg_{config.LABEL_WIDTH}wide_"
                     f"{year}.nc")
    if not current_file.exists():
        pytest.skip(f"no current label file at {current_file} (set "
                    f"JPL_AIRS_DATA to a populated data root)")
    if not backup_file.exists():
        pytest.skip(f"no pre-fix label backup at {backup} -- it is created "
                    f"by the 2026-08-17 dateline-bug regeneration and is "
                    f"not part of a fresh checkout (data/ is gitignored)")

    def scored_counts():
        """The per-class cell counts the digest is built from."""
        with dataset.load_label_ds(year, N_CLASSES) as lab:
            keep = dataset.valid_label_steps(lab, N_CLASSES)
            cls = dataset.class_grid(lab, N_CLASSES)[keep]
        return np.bincount(cls[:, dataset.analysis_domain()].ravel(),
                           minlength=N_CLASSES)

    current, current_counts = dataset.label_digest([year], N_CLASSES), \
        scored_counts()
    original = fd_config.NOAA_LABELS_DIR
    try:
        fd_config.NOAA_LABELS_DIR = backup
        pre_fix, pre_fix_counts = dataset.label_digest([year], N_CLASSES), \
            scored_counts()
    finally:
        fd_config.NOAA_LABELS_DIR = original
    # The directory string is part of the hash, so a digest difference alone
    # would prove nothing here; assert the CONTENT moved too -- that is what
    # the digest is claiming to have detected.
    assert not np.array_equal(current_counts, pre_fix_counts)
    assert current != pre_fix
    # and the current tree still digests to what it did before the swap:
    # the loader must read the directory at CALL time, not at import
    assert dataset.label_digest([year], N_CLASSES) == current
