"""dl_front.evaluate_test tests: synthetic in-memory data, stub model.

The numpy path (stub object with ``.predict``) runs everywhere; one extra
test exercises a real (untrained) keras model when TensorFlow is importable
(the fronts-tf env), and skips cleanly in the numpy-only .venv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dl_front import config, evaluate_test

#: ``dataset.analysis_domain()``/``crop_domain()``/``region_mask()``
#: interpolate the land-fraction mask off disk, so even synthetic tests that
#: reach them need the data root's mask file.  Skipped, never failed, on
#: checkouts without a populated data root.
needs_land_mask = pytest.mark.skipif(
    not config.LAND_MASK_PATH.exists(),
    reason=f"land mask {config.LAND_MASK_PATH} not on disk "
           f"(set JPL_AIRS_DATA to a populated data root)")

N_CLASSES = 6
NONE = N_CLASSES - 1
#: dummy norm stats: evaluate_ckpt never touches them when a loader is
#: injected, but passing them avoids reading the frozen JSON in tests.
STATS = {v: [0.0, 1.0] for v in config.SFC_VARS}


def make_loader(hours=(18, 21, 0, 12)):
    """loader(year) -> synthetic (x, y, times), one step per hour.

    Every step is all-'none' except a cold-front line at row 30 (lat 40 N),
    cols 75:85 (96-87 W): central-US land, fully inside the 6-class
    analysis domain that evaluate_ckpt scores under (user decision
    2026-08-13).
    """
    def loader(year):
        n = len(hours)
        x = np.zeros((n, *config.GRID_SHAPE, 5), dtype=np.float16)
        y = np.full((n, *config.GRID_SHAPE), NONE, dtype=np.uint8)
        y[:, 30, 75:85] = 0                            # cold front line
        times = pd.DatetimeIndex(
            [pd.Timestamp(f"{year}-06-01") + pd.Timedelta(hours=h)
             for h in hours])
        return x, y, times
    return loader


class PerfectStub:
    """Duck-typed model: predicts the loader's truth with certainty."""

    def __init__(self):
        self.n_predicted = 0

    def predict(self, x, batch_size=64, verbose=0):
        self.n_predicted += len(x)
        probs = np.zeros((len(x), *config.GRID_SHAPE, N_CLASSES), np.float32)
        y = np.full((len(x), *config.GRID_SHAPE), NONE, dtype=np.uint8)
        y[:, 30, 75:85] = 0                            # mirror the loader
        np.put_along_axis(probs, y[..., None].astype(np.int64), 1.0, axis=-1)
        return probs


@needs_land_mask
def test_perfect_prediction_and_output_files(tmp_path):
    pm, scores = evaluate_test.evaluate_ckpt(
        PerfectStub(), [2016, 2017], N_CLASSES, "reanalysis",
        hours=(18, 21, 0), stats=STATS, loader=make_loader())

    assert pm.accuracy()["all_categories"] == 1.0
    np.testing.assert_allclose(scores.loc[("cold", 0), "csi"], 1.0)
    assert np.isnan(scores.loc[("warm", 0), "csi"])    # no warm pixels

    paths = evaluate_test.write_outputs(
        pm, scores, ckpt=tmp_path / "modelA.h5", source="kriged-airs",
        years=[2016, 2017], hours=(18, 21, 0), out_dir=tmp_path / "out")

    csv = pd.read_csv(paths["csv"])
    assert list(csv.columns) == ["front", "dilation", "km", "csi",
                                 "csi_lo", "csi_hi", "pod", "far", "fb"]
    assert paths["csv"].name == "modelA_kriged-airs.csv"
    cold0 = csv[(csv.front == "cold") & (csv.dilation == 0)]
    np.testing.assert_allclose(cold0["csi"].iloc[0], 1.0)

    paper = json.loads(paths["paper"].read_text())
    assert set(paper) == {"accuracy", "auc", "confusion_percent"}
    assert set(paper["accuracy"]) == {"all_categories", "front_no_front"}
    assert paper["accuracy"]["all_categories"] == 1.0
    assert "cold" in paper["confusion_percent"]

    run = json.loads(paths["run"].read_text())
    assert run["source"] == "kriged-airs"
    assert run["years"] == [2016, 2017] and run["hours"] == [18, 21, 0]
    assert "ckpt" in run and "git_rev" in run and "created" in run


@needs_land_mask
def test_scoring_mask_is_analysis_domain_for_6class():
    """User decision 2026-08-13: EVERY 6-class leg scores over
    dataset.analysis_domain() only.  A 'model' whose predictions are
    garbage outside the analysis domain (but perfect inside it) must
    still score perfectly -- under the old region-mask scoring the
    out-of-domain wrong pixels counted."""
    from dl_front import dataset

    outside = dataset.region_mask().astype(bool) \
        & ~dataset.analysis_domain()
    assert outside.any()          # region mask pixels the new mask drops

    class NoisyOutsideStub(PerfectStub):
        def predict(self, x, batch_size=64, verbose=0):
            probs = super().predict(x, batch_size, verbose)
            probs[:, outside] = 0.0
            probs[:, outside, 1] = 1.0             # bogus warm fronts
            return probs

    pm, scores = evaluate_test.evaluate_ckpt(
        NoisyOutsideStub(), [2016], N_CLASSES, "reanalysis",
        hours=(18, 21, 0), stats=STATS, loader=make_loader())
    assert pm.accuracy()["all_categories"] == 1.0    # garbage never scored
    np.testing.assert_allclose(scores.loc[("cold", 0), "csi"], 1.0)
    assert np.isnan(scores.loc[("warm", 0), "csi"])  # no warm pixels scored
    # PaperMetrics really accumulated over the analysis domain
    assert pm.confusion.sum() == \
        3 * int(dataset.analysis_domain().sum())


@needs_land_mask
def test_hours_filter_reduces_sample_count():
    """Default AIRS hours (21, 0) keep 2 of the 4 synthetic steps per year."""
    all_hours, airs = PerfectStub(), PerfectStub()
    evaluate_test.evaluate_ckpt(all_hours, [2016], N_CLASSES, "reanalysis",
                                hours=(18, 21, 0, 12), stats=STATS,
                                loader=make_loader())
    evaluate_test.evaluate_ckpt(airs, [2016], N_CLASSES, "reanalysis",
                                hours=None, stats=STATS,   # -> AIRS_HOURS
                                loader=make_loader())
    assert all_hours.n_predicted == 4
    assert airs.n_predicted == 2
    assert airs.n_predicted < all_hours.n_predicted


@needs_land_mask
def test_no_data_after_filter_raises():
    with pytest.raises(RuntimeError, match="no data"):
        evaluate_test.evaluate_ckpt(PerfectStub(), [2016], N_CLASSES,
                                    "reanalysis", hours=(6,), stats=STATS,
                                    loader=make_loader())


def test_missing_data_errors_name_the_fix():
    with pytest.raises(FileNotFoundError, match="acquire_merra2_sfc 1901"):
        evaluate_test.load_year(1901, N_CLASSES, STATS, "reanalysis")
    with pytest.raises(FileNotFoundError,
                       match="krige_fill build-airs --years 1901"):
        evaluate_test.load_year(1901, N_CLASSES, STATS, "kriged-airs")
    with pytest.raises(ValueError, match="unknown source"):
        evaluate_test.load_year(2016, N_CLASSES, STATS, "bogus")


def test_parse_years():
    assert evaluate_test.parse_years("2016-2021") == [2016, 2017, 2018,
                                                      2019, 2020, 2021]
    assert evaluate_test.parse_years("2016,2018,2020") == [2016, 2018, 2020]
    assert evaluate_test.parse_years("2016") == [2016]


@needs_land_mask
def test_keras_model_end_to_end(tmp_path):
    """Untrained 1x1-conv softmax model through the full pipeline (TF only)."""
    tf = pytest.importorskip("tensorflow")
    if not hasattr(tf, "keras"):  # tests/_stubs/tensorflow.py may be loaded
        pytest.skip("tensorflow is the tests/_stubs stand-in, not the real TF")
    model = tf.keras.Sequential([
        tf.keras.layers.Input((*config.GRID_SHAPE, 5)),
        tf.keras.layers.Conv2D(N_CLASSES, 1, activation="softmax")])
    pm, scores = evaluate_test.evaluate_ckpt(
        model, [2016], N_CLASSES, "reanalysis", hours=(18, 21, 0),
        stats=STATS, loader=make_loader())
    assert 0.0 <= pm.accuracy()["all_categories"] <= 1.0
    paths = evaluate_test.write_outputs(
        pm, scores, ckpt=tmp_path / "tiny.h5", source="reanalysis",
        years=[2016], hours=(18, 21, 0), out_dir=tmp_path / "out")
    assert list(pd.read_csv(paths["csv"]).columns) == [
        "front", "dilation", "km", "csi", "csi_lo", "csi_hi",
        "pod", "far", "fb"]


@needs_land_mask
def test_match_source_intersects_time_steps(monkeypatch):
    """Reanalysis runs must score ONLY steps present in the kriged-airs
    cache, so a sparse AIRS archive cannot skew the comparison."""
    ref = pd.DatetimeIndex([pd.Timestamp("2016-06-01 18:00"),
                            pd.Timestamp("2016-06-01 21:00")])
    monkeypatch.setattr(evaluate_test, "kriged_cache_times",
                        lambda year, source="kriged-airs": ref)
    stub, info = PerfectStub(), {}
    evaluate_test.evaluate_ckpt(stub, [2016], N_CLASSES, "reanalysis",
                                hours=(18, 21, 0), stats=STATS,
                                loader=make_loader(),
                                match_source="kriged-airs", info=info)
    assert stub.n_predicted == 2               # 00Z step dropped
    assert info["n_steps_per_year"] == {2016: 2}
    assert info["match_source"] == "kriged-airs"
    assert len(info["times_sha1"]) == 40

    # match_source == source is a no-op (kriged runs ARE the cache steps)
    stub2 = PerfectStub()
    evaluate_test.evaluate_ckpt(stub2, [2016], N_CLASSES, "kriged-airs",
                                hours=(18, 21, 0), stats=STATS,
                                loader=make_loader(),
                                match_source="kriged-airs")
    assert stub2.n_predicted == 3


@needs_land_mask
def test_missing_match_cache_fails_loudly(monkeypatch):
    with pytest.raises(FileNotFoundError, match="build-airs --years 1901"):
        evaluate_test.evaluate_ckpt(PerfectStub(), [1901], N_CLASSES,
                                    "reanalysis", hours=(18, 21, 0),
                                    stats=STATS, loader=make_loader(),
                                    match_source="kriged-airs")


# --------------------------------------------------------------------------- #
# BK19 published-prediction leg (three-way test, user decision 2026-08-13)
# --------------------------------------------------------------------------- #

def make_bk19_ds(n_time=2, n_lat=6, n_lon=9):
    """Tiny synthetic BK19 benchmark dataset (schema of the real files,
    with Capitalized front_type names to exercise case-insensitive match)."""
    xr = pytest.importorskip("xarray")
    types = ["Cold", "Warm", "Stationary", "Occluded", "None"]
    fronts = np.zeros((n_time, len(types), n_lat, n_lon), dtype=np.float32)
    fronts[:, 0, 2, 1:4] = 1.0            # cold line
    fronts[:, 1, 2, 3] = 1.0              # warm OVERLAPS cold at (2, 3)
    fronts[:, 3, 4, 0:2] = 1.0            # occluded segment
    times = pd.date_range("2016-06-01", periods=n_time, freq="3h")
    return xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": times, "front_type": ("front", types),
                "lat": np.arange(n_lat, dtype=float),
                "lon": np.arange(n_lon, dtype=float)})


def test_bk19_class_grid_painting():
    """Painter-priority overlap resolution, no dryline, none elsewhere."""
    ds = make_bk19_ds()
    cls = evaluate_test.bk19_class_grid(ds, N_CLASSES)
    assert cls.shape == (2, 6, 9) and cls.dtype == np.uint8
    names = config.CLASS_NAMES_6
    assert cls[0, 2, 1] == names.index("cold")
    assert cls[0, 2, 3] == names.index("warm")      # overlap: warm wins
    assert cls[0, 4, 0] == names.index("occluded")
    assert not (cls == names.index("dryline")).any()  # never predicted
    expected_front = np.zeros((6, 9), bool)
    expected_front[2, 1:4] = expected_front[4, 0:2] = True
    assert ((cls != NONE) == expected_front).all()  # none everywhere else


def test_bk19_missing_year_error_names_env_var(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BK19_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="JPL_BK19_DIR"):
        evaluate_test.bk19_path(2016)
    with pytest.raises(FileNotFoundError, match="JPL_BK19_DIR"):
        evaluate_test.load_year(2016, N_CLASSES, STATS, "bk19")


def test_bk19_predictions_stub_one_hot():
    """x carries class indices; predict must one-hot them exactly."""
    x = np.array([[[[0.0], [4.0], [5.0]]]], dtype=np.float32)  # (1, 1, 3, 1)
    probs = evaluate_test.BK19Predictions(N_CLASSES).predict(x)
    assert probs.shape == (1, 1, 3, N_CLASSES)
    assert probs.argmax(-1).tolist() == [[[0, 4, 5]]]
    np.testing.assert_allclose(probs.sum(-1), 1.0)


def test_bk19_ckpt_combination_rejected():
    with pytest.raises(SystemExit):
        evaluate_test.main(["--source", "bk19", "--ckpt", "x.h5"])
    with pytest.raises(SystemExit):        # and no-ckpt for a model source
        evaluate_test.main(["--source", "reanalysis"])


@needs_land_mask
def test_bk19_outputs_skip_paper_json(tmp_path):
    """bk19 leg: stem 'bk19', no paper json, run json notes the skip."""
    pm, scores = evaluate_test.evaluate_ckpt(
        PerfectStub(), [2016], N_CLASSES, "reanalysis",
        hours=(18, 21, 0), stats=STATS, loader=make_loader())
    paths = evaluate_test.write_outputs(
        pm, scores, ckpt=None, source="bk19", years=[2016],
        hours=(18, 21, 0), out_dir=tmp_path, write_paper=False,
        info={"dryline": "not predicted"})
    assert paths["csv"].name == "bk19.csv"
    assert "paper" not in paths
    assert not (tmp_path / "bk19_paper.json").exists()
    run = json.loads(paths["run"].read_text())
    assert run["ckpt"] is None
    assert run["paper_metrics"] == "skipped (binary baseline)"
    assert run["dryline"] == "not predicted"


# --------------------------------------------------------------------------- #
# Years default & leg comparison
# --------------------------------------------------------------------------- #

def test_years_default_comes_from_config():
    """No --years -> the yaml eval split governs (2016-2018 for 6-class,
    user decision 2026-08-13)."""
    assert evaluate_test.resolve_years(None, 6) == list(config.EVAL_YEARS_6)
    assert evaluate_test.resolve_years(None, 6) == [2016, 2017, 2018]
    assert evaluate_test.resolve_years(None, 5) == list(config.EVAL_YEARS_5)
    assert evaluate_test.resolve_years("2016-2017", 6) == [2016, 2017]


def leg_csv(out_dir, stem, rows):
    """rows: (front, dilation, km, csi) tuples -> a leg CSV like write_outputs'."""
    pd.DataFrame([{"front": f, "dilation": d, "km": km, "csi": csi,
                   "csi_lo": csi, "csi_hi": csi,
                   "pod": csi, "far": 0.0, "fb": 1.0}
                  for f, d, km, csi in rows]).to_csv(
        out_dir / f"{stem}.csv", index=False)


def test_compare_pivots_legs(tmp_path):
    leg_csv(tmp_path, "D6C-f0_kriged-airs",
            [("cold", 0, 0.0, 0.30), ("cold", 1, 111.0, 0.50),
             ("dryline", 1, 111.0, 0.10)])
    leg_csv(tmp_path, "bk19",
            [("cold", 0, 0.0, 0.25), ("cold", 1, 111.0, 0.40)])
    run_json(tmp_path, "D6C-f0_kriged-airs", "a" * 40)   # same-sample legs:
    run_json(tmp_path, "bk19", "a" * 40)                 # -> comparison.csv
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n")   # ignored: not a leg

    table = evaluate_test.compare(out_dir=tmp_path)
    assert list(table.columns) == [
        "D6C-f0_kriged-airs", "D6C-f0_kriged-airs_lo", "D6C-f0_kriged-airs_hi",
        "bk19", "bk19_lo", "bk19_hi"]
    assert table.index.names == ["front", "dilation_km"]
    assert table.loc[("cold", 111.0), "bk19"] == 0.40
    assert table.loc[("cold", 111.0), "D6C-f0_kriged-airs"] == 0.50
    assert np.isnan(table.loc[("dryline", 111.0), "bk19"])   # missing leg row

    saved = pd.read_csv(tmp_path / "comparison.csv")
    assert {"front", "dilation_km", "D6C-f0_kriged-airs", "bk19",
            "bk19_lo", "bk19_hi"} <= set(saved.columns)
    # rerunning must not pick up comparison.csv itself as a leg
    table2 = evaluate_test.compare(out_dir=tmp_path)
    assert list(table2.columns) == list(table.columns)


def test_compare_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no leg CSVs"):
        evaluate_test.compare(out_dir=tmp_path)


def test_compare_rejects_extra_arguments(capsys):
    """'compare' must error on trailing args, never silently drop them."""
    with pytest.raises(SystemExit, match="takes no arguments"):
        evaluate_test.main(["compare", "--years", "2016"])
    with pytest.raises(SystemExit, match="takes no arguments"):
        evaluate_test.main(["compare", "--help"])


def run_json(out_dir, stem, sha, years=(2016, 2017, 2018),
             match="kriged-airs"):
    (out_dir / f"{stem}_run.json").write_text(json.dumps(
        {"times_sha1": sha, "years": list(years), "match_source": match,
         "n_steps_per_year": {str(y): 10 for y in years}}))


def test_compare_cross_checks_times_sha1(tmp_path, capsys):
    """Legs with identical scored-timestamp SHA-1s pass silently; a
    mismatch (or a missing _run.json) is reported loudly -- the matching in
    evaluate_ckpt only DROPS steps, so this is the actual same-sample check."""
    rows = [("cold", 0, 0.0, 0.30)]
    leg_csv(tmp_path, "D6C-f0_kriged-airs", rows)
    leg_csv(tmp_path, "bk19", rows)
    run_json(tmp_path, "D6C-f0_kriged-airs", "a" * 40, match=None)
    run_json(tmp_path, "bk19", "a" * 40)

    evaluate_test.compare(out_dir=tmp_path)
    assert "WARNING" not in capsys.readouterr().out     # identical samples

    run_json(tmp_path, "bk19", "b" * 40, years=[2016], match=None)
    evaluate_test.compare(out_dir=tmp_path)
    out = capsys.readouterr().out
    assert "did NOT score identical time steps" in out
    assert ("b" * 40) in out and "years=[2016]" in out  # names the odd leg

    (tmp_path / "bk19_run.json").unlink()               # missing provenance
    evaluate_test.compare(out_dir=tmp_path)
    out = capsys.readouterr().out
    assert "no readable bk19_run.json" in out
