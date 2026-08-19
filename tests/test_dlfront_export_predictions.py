"""dl_front.export_predictions tests: synthetic probabilities, stub model.

Everything here runs in a numpy-only env (no TensorFlow, no GPU, no kriged
caches): the input leg is replaced by an injected ``loader`` and the model by
a stub with ``.predict``, exactly as in test_dlfront_evaluate_test.py.  The
BK19-schema test compares against the REAL published file when it is present
in the data root and degrades to a self-consistency check when it is not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dl_front import config, dataset, evaluate_test, export_predictions as ep

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
#: Cold-front line at row 30 (40 N), cols 75:85 (96-87 W): central-US land,
#: fully inside the 6-class analysis domain (user decision 2026-08-13).
ROW, COLS = 30, slice(75, 85)

xr = pytest.importorskip("xarray")
netCDF4 = pytest.importorskip("netCDF4")


def truth(n: int) -> np.ndarray:
    """(n, 68, 141) class grid: all 'none' plus one cold-front line."""
    y = np.full((n, *config.GRID_SHAPE), NONE, dtype=np.uint8)
    y[:, ROW, COLS] = 0
    return y


def make_loader(offsets=(12, 21, 24)):
    """loader(year) -> synthetic (x, y, times), one step per offset.

    ``offsets`` are hours after June-1 00 Z, so the default is 12 Z, 21 Z and
    NEXT-DAY 00 Z -- i.e. the two ``config.AIRS_HOURS`` steps (21, 0) three
    hours apart, plus a 12 Z step the exporter's hours filter must drop.
    """
    def loader(year):
        n = len(offsets)
        x = np.zeros((n, *config.GRID_SHAPE, 5), dtype=np.float16)
        times = pd.DatetimeIndex(
            [pd.Timestamp(f"{year}-06-01") + pd.Timedelta(hours=h)
             for h in offsets])
        return x, truth(n), times
    return loader


class Stub:
    """Duck-typed model: certainty softmax on :func:`truth`, warm-biased.

    ``warm_prob`` puts a fixed mass on the warm channel so the
    ``--class-scale`` knob has something to flip.
    """

    def __init__(self, warm_prob: float = 0.0):
        self.warm_prob = warm_prob
        self.n_predicted = 0

    def predict(self, x, batch_size=64, verbose=0):
        self.n_predicted += len(x)
        cls = truth(len(x))
        probs = np.zeros((len(x), *config.GRID_SHAPE, N_CLASSES), np.float32)
        np.put_along_axis(probs, cls[..., None].astype(np.int64),
                          1.0 - self.warm_prob, axis=-1)
        probs[..., 1] = self.warm_prob
        return probs


# --------------------------------------------------------------------------- #
# Pure pieces: decision rule, channel inversion, fill, tags, --years
# --------------------------------------------------------------------------- #

def test_class_channels_is_the_inverse_of_class_grid():
    cls = truth(3)
    chan = ep.class_channels(cls, N_CLASSES)
    assert chan.dtype == np.uint8
    assert chan.shape == (3, N_CLASSES, *config.GRID_SHAPE)
    # exclusive one-hot: exactly one channel on everywhere (BK19's channels
    # sum to 1 too), and the 'none' channel is the complement of the rest
    assert (chan.sum(axis=1) == 1).all()
    assert (chan[:, NONE] == (chan[:, :NONE].sum(axis=1) == 0)).all()
    assert (chan[:, 0, ROW, COLS] == 1).all()
    assert chan[:, 0, ROW + 1, COLS].sum() == 0
    # and it round-trips through the canonical BK19 consumer's painter
    np.testing.assert_array_equal(
        np.argmax(chan, axis=1).astype(np.uint8), cls)


@needs_land_mask
def test_class_channels_fills_outside_the_domain():
    domain = dataset.analysis_domain()
    chan = ep.class_channels(truth(2), N_CLASSES, valid=domain)
    # every channel is the fill byte outside, and untouched inside
    assert (chan[:, :, ~domain] == ep.FILL_BYTE).all()
    assert (chan[:, :, domain] <= 1).all()
    assert (chan[:, :, domain].sum(axis=1) == 1).all()
    # the fill region is time-invariant (labels.valid_mask's intent)
    assert (chan[0] == chan[1]).all()


def test_hard_classes_argmax_and_class_scale():
    probs = Stub(warm_prob=0.4).predict(np.zeros((2, *config.GRID_SHAPE, 5)))
    # plain argmax: 0.6 'none'/'cold' beats 0.4 warm
    np.testing.assert_array_equal(ep.hard_classes(probs, N_CLASSES), truth(2))
    # warm x2 -> 0.8 warm wins everywhere; the knob is off by default
    scaled = ep.hard_classes(probs, N_CLASSES, {"warm": 2.0})
    assert (scaled == 1).all()


def test_parse_class_scale_and_bad_input():
    assert ep.parse_class_scale(None) == {}
    assert ep.parse_class_scale("") == {}
    assert ep.parse_class_scale("warm=1.3,occluded=1.35") == {
        "warm": 1.3, "occluded": 1.35}
    with pytest.raises(ValueError):
        ep.parse_class_scale("warm")                  # no '='
    with pytest.raises(ValueError):
        ep.parse_class_scale("wrm=1.3")               # typo'd class
    with pytest.raises(ValueError):
        ep.parse_class_scale("warm=0")                # non-positive factor


def test_years_parsing_matches_the_other_clis():
    assert ep.parse_years("2016-2021") == [2016, 2017, 2018, 2019, 2020, 2021]
    assert ep.parse_years("2016,2018,2020") == [2016, 2018, 2020]


def test_tags_and_paths(tmp_path):
    assert ep.export_tag("D6C-f0", "kriged-airs") == "dlfront_D6C-f0_kriged-airs"
    assert ep.export_tag("D6C-f0", "reanalysis", {"warm": 1.3}) == \
        "dlfront_D6C-f0_reanalysis_scale-warm1.3"
    assert ep.ensemble_stem(["a/D6C-f0.h5", "b/D6C-f1.h5",
                             "c/D6C-f2.h5"]) == "D6C-ens3"
    # mixed families must NOT claim a fold ensemble
    assert ep.ensemble_stem(["a/D6A-f0.h5", "b/D6C-f0.h5"]) == "D6A-f0+D6C-f0"

    path = ep.output_path("dlfront_D6C-f0_kriged-airs", 2016, root=tmp_path)
    # byte-identical to what evaluate_test.bk19_path composes one level down,
    # which is what makes JPL_BK19_DIR=<root>/<tag> work with no code change
    assert path == (tmp_path / "dlfront_D6C-f0_kriged-airs" / "1deg_3wide"
                    / "3hr" / "merra2_merra2-1deg_3wide_3hr_2016.nc")
    assert ep.run_json_path(path).name == \
        "merra2_merra2-1deg_3wide_3hr_2016_run.json"


def test_write_bk19_netcdf_rejects_shape_mismatches(tmp_path):
    chan = ep.class_channels(truth(2), N_CLASSES)
    times = pd.DatetimeIndex(["2016-06-01T21", "2016-06-02T00"])
    with pytest.raises(ValueError):      # channels vs front_type disagree
        ep.write_bk19_netcdf(tmp_path / "a.nc", chan, times,
                             dataset.class_names(N_CLASSES)[:3], {})
    with pytest.raises(ValueError):      # channels vs the label grid
        ep.write_bk19_netcdf(tmp_path / "c.nc", chan[:, :, :10], times,
                             dataset.class_names(N_CLASSES), {})
    with pytest.raises(ValueError):
        ep.write_bk19_netcdf(tmp_path / "b.nc", chan, times[:1],
                             dataset.class_names(N_CLASSES), {})


# --------------------------------------------------------------------------- #
# The written file: BK19 schema fidelity + consumer round trip
# --------------------------------------------------------------------------- #

def export_one(tmp_path, **kw):
    """Export 2016 from the synthetic loader; return the .nc path."""
    kw.setdefault("loader", make_loader())
    path = ep.export_year([Stub()], ["fake/D6C-f0.h5"], 2016, "kriged-airs",
                          root=tmp_path, **kw)
    assert path is not None
    return path


@needs_land_mask
def test_written_file_matches_the_bk19_schema(tmp_path):
    path = export_one(tmp_path)
    assert path == ep.output_path("dlfront_D6C-f0_kriged-airs", 2016,
                                  root=tmp_path)

    with netCDF4.Dataset(path) as nc:
        assert nc.data_model == "NETCDF4"
        assert nc.dimensions["time"].isunlimited()
        assert [len(nc.dimensions[d]) for d in ("front", "lat", "lon")] == \
            [N_CLASSES, len(config.LABEL_LATS), len(config.LABEL_LONS)]
        # declaration order, verified against ncdump -h on the real file
        assert list(nc.variables) == ["crs", "front_type", "fronts", "lat",
                                      "lon", "time"]

        fronts = nc["fronts"]
        assert fronts.dtype == np.uint8
        assert fronts.dimensions == ("time", "front", "lat", "lon")
        assert fronts.ncattrs() == ["_FillValue", "long_name", "valid_min",
                                    "grid_mapping", "valid_max",
                                    "coordinates"]
        assert fronts.getncattr("_FillValue") == ep.FILL_BYTE
        assert fronts.long_name == "front line images"
        assert fronts.grid_mapping == "crs"
        assert fronts.coordinates == "front_type lat lon"
        # int64 (ncdump 0LL/1LL), NOT coerced to the variable's ubyte type
        assert fronts.getncattr("valid_min").dtype == np.dtype("int64")
        assert fronts.getncattr("valid_max").dtype == np.dtype("int64")
        assert fronts.filters()["zlib"] and fronts.filters()["shuffle"]
        assert fronts.filters()["complevel"] == ep.ZLIB_COMPLEVEL
        assert fronts.chunking() == [1, 1, len(config.LABEL_LATS),
                                     len(config.LABEL_LONS)]

        assert nc["crs"].grid_mapping_name == "latitude_longitude"
        nc.set_auto_mask(False)
        # crs is created but never written -> ncdump prints 'crs = _'
        assert nc["crs"][...] == netCDF4.default_fillvals["f8"]
        assert list(nc["front_type"][:]) == list(
            dataset.class_names(N_CLASSES))
        assert nc["front_type"].long_name == "kind of front"

        assert nc["time"].units == "days since 1970-01-01 00:00:00"
        assert nc["time"].calendar == "gregorian"
        assert nc["time"].dtype == np.dtype("float64")
        # 21 Z and next-day 00 Z only: 12 Z was filtered out by AIRS_HOURS
        assert len(nc.dimensions["time"]) == 2
        np.testing.assert_allclose(
            np.diff(nc["time"][:]), [0.125])          # exactly 3 h apart
        for name, values in (("lat", config.LABEL_LATS),
                             ("lon", config.LABEL_LONS)):
            var = nc[name]
            assert var.dtype == np.dtype("float64")
            assert "_FillValue" not in var.ncattrs()  # BK19 has none
            assert not var.filters()["zlib"]          # contiguous
            np.testing.assert_array_equal(var[:], np.asarray(values, "f8"))


@pytest.mark.skipif(
    not (config.BK19_DIR / "1deg_3wide" / "3hr").is_dir(),
    reason="no published BK19 archive in this data root")
def test_attribute_sets_agree_with_the_real_bk19_file(tmp_path):
    """Same variables/attrs/dtypes as the published product.

    Only the LEGITIMATE differences are allowed: the 'dryline' channel (front
    = 6, decided 2026-08-17), the sparse AIRS-hour time axis, and our own
    provenance globals instead of BK19's stale title + 817 KB NCO history.
    """
    real = evaluate_test.bk19_path(2016)
    mine = export_one(tmp_path)
    with netCDF4.Dataset(real) as a, netCDF4.Dataset(mine) as b:
        assert list(a.variables) == list(b.variables)
        for name in a.variables:
            va, vb = a[name], b[name]
            assert va.dtype == vb.dtype, name
            assert va.dimensions == vb.dimensions, name
            assert va.ncattrs() == vb.ncattrs(), name
            for attr in va.ncattrs():
                np.testing.assert_array_equal(
                    np.asarray(va.getncattr(attr)).dtype,
                    np.asarray(vb.getncattr(attr)).dtype)
            assert va.filters() == vb.filters(), name
            assert va.chunking() == vb.chunking(), name
        # our extensions, stated explicitly
        assert len(a.dimensions["front"]) == 5
        assert len(b.dimensions["front"]) == N_CLASSES
        assert "dryline" in list(b["front_type"][:])


@needs_land_mask
def test_round_trip_through_bk19_class_grid(tmp_path):
    """The canonical consumer reads back exactly the classes we exported."""
    path = export_one(tmp_path)
    domain = dataset.analysis_domain()
    with xr.open_dataset(path) as bk:
        cls = evaluate_test.bk19_class_grid(bk.load(), N_CLASSES)
    np.testing.assert_array_equal(cls[:, domain], truth(2)[:, domain])
    # documented caveat: xarray decodes the fill byte to NaN, so the consumer
    # paints out-of-domain cells as 'none' -- the raw byte still says fill
    assert (cls[:, ~domain] == NONE).all()
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        assert (nc["fronts"][:][:, :, ~domain] == ep.FILL_BYTE).all()


@needs_land_mask
def test_provenance_names_the_checkpoint_generation(tmp_path):
    path = export_one(tmp_path)
    run = json.loads(ep.run_json_path(path).read_text())
    assert run["checkpoints"] and run["checkpoints"][0].endswith("D6C-f0.h5")
    assert run["source"] == "kriged-airs"
    assert run["year"] == 2016 and run["n_steps"] == 2
    assert run["hours"] == list(config.AIRS_HOURS)
    assert run["decision_rule"] == "argmax" and run["class_scale"] == {}
    assert run["ensemble"] is False
    assert run["class_names"] == list(dataset.class_names(N_CLASSES))
    assert run["label_width"] == config.LABEL_WIDTH
    # the 2026-08-17 label bug: a reader must be able to identify the
    # checkpoint generation from the file alone
    assert "checkpoint_sha1" in run and "dlfront_config_sha1" in run
    assert run["times_sha1"] and run["created"]

    with netCDF4.Dataset(path) as nc:
        assert nc.model == "D6C-f0"
        assert nc.input_source == "kriged-airs"
        assert nc.decision_rule == "argmax"
        assert nc.class_names == " ".join(dataset.class_names(N_CLASSES))
        assert nc.times_sha1 == run["times_sha1"]
        assert nc.dlfront_config_sha1 == run["dlfront_config_sha1"]
        assert nc.hours_utc == " ".join(str(h) for h in config.AIRS_HOURS)
        # NOT BK19's stale title / NCO history
        assert "dl_front" in nc.title and "NCO" not in nc.ncattrs()


@needs_land_mask
def test_class_scale_lands_in_the_tag_and_the_provenance(tmp_path):
    scale = {"warm": 2.0}
    path = ep.export_year([Stub(warm_prob=0.4)], ["fake/D6C-f0.h5"], 2016,
                          "reanalysis", root=tmp_path, class_scale=scale,
                          loader=make_loader())
    assert path.parts[-4] == "dlfront_D6C-f0_reanalysis_scale-warm2"
    run = json.loads(ep.run_json_path(path).read_text())
    assert run["class_scale"] == {"warm": 2.0}
    assert run["decision_rule"] == "argmax after per-class softmax scaling"


@needs_land_mask
def test_ensemble_averages_softmax_before_the_argmax(tmp_path):
    """Two disagreeing members -> the MEAN decides, and the tag says ens2."""
    confident_warm = Stub(warm_prob=0.9)      # warm 0.9 vs truth 0.1
    mild_truth = Stub(warm_prob=0.2)          # truth 0.8 vs warm 0.2
    path = ep.export_year([confident_warm, mild_truth],
                          ["a/D6C-f0.h5", "b/D6C-f1.h5"], 2016,
                          "kriged-airs", root=tmp_path, loader=make_loader())
    assert path.parts[-4] == "dlfront_D6C-ens2_kriged-airs"
    run = json.loads(ep.run_json_path(path).read_text())
    assert run["ensemble"] is True
    assert run["combination"] == "mean softmax over checkpoints, then argmax"
    domain = dataset.analysis_domain()
    with xr.open_dataset(path) as bk:
        cls = evaluate_test.bk19_class_grid(bk.load(), N_CLASSES)
    # mean warm = 0.55 > mean truth 0.45 -> warm wins (a majority vote would
    # have kept the truth class, which is exactly the difference)
    assert (cls[:, domain] == 1).all()


# --------------------------------------------------------------------------- #
# Idempotency and per-year skipping
# --------------------------------------------------------------------------- #

@needs_land_mask
def test_existing_output_is_skipped_unless_forced(tmp_path):
    stub = Stub()
    kw = dict(root=tmp_path, loader=make_loader())
    first = ep.export_year([stub], ["fake/D6C-f0.h5"], 2016, "kriged-airs",
                           **kw)
    assert stub.n_predicted == 2
    assert ep.is_done(first)
    # second call: done-marker (.nc AND _run.json) -> no inference at all
    assert ep.export_year([stub], ["fake/D6C-f0.h5"], 2016, "kriged-airs",
                          **kw) is None
    assert stub.n_predicted == 2
    # a half-written pair is NOT accepted as done
    ep.run_json_path(first).unlink()
    assert not ep.is_done(first)
    assert ep.export_year([stub], ["fake/D6C-f0.h5"], 2016, "kriged-airs",
                          **kw) == first
    assert stub.n_predicted == 4
    # --force rewrites
    assert ep.export_year([stub], ["fake/D6C-f0.h5"], 2016, "kriged-airs",
                          force=True, **kw) == first
    assert stub.n_predicted == 6


def test_year_with_no_steps_at_the_airs_hours_is_skipped(tmp_path):
    path = ep.export_year([Stub()], ["fake/D6C-f0.h5"], 2016, "kriged-airs",
                          root=tmp_path, loader=make_loader(offsets=(6, 12)))
    assert path is None
    assert not ep.output_path("dlfront_D6C-f0_kriged-airs", 2016,
                              root=tmp_path).exists()


@needs_land_mask
def test_export_years_skips_the_unbuildable_years_loudly(tmp_path):
    def loader(year):
        if year == 2017:                       # e.g. a missing kriged cache
            raise FileNotFoundError(f"no kriged cache for {year}")
        return make_loader()(year)

    result = ep.export_years([Stub()], ["fake/D6C-f0.h5"],
                             [2016, 2017, 2018], "kriged-airs",
                             root=tmp_path, loader=loader)
    assert [p.parent.parent.parent.name for p in result["written"]] == \
        ["dlfront_D6C-f0_kriged-airs"] * 2
    assert [int(p.stem[-4:]) for p in result["written"]] == [2016, 2018]
    assert list(result["skipped"]) == [2017]
    assert "no kriged cache" in result["skipped"][2017]


# --------------------------------------------------------------------------- #
# CLI guards (no TF needed: they all fail before load_model)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("argv", [
    ["--ckpt", "x.h5", "--width", "1"],                 # 1wide is not ours
    ["--ckpt", "x.h5", "--class-scale", "wrm=1.3"],     # typo'd class
    ["--ckpt", "x.h5", "--ensemble-only"],              # needs >1 ckpt
    ["--ckpt", "definitely/missing.h5"],                # unreadable ckpt
])
def test_cli_rejects_bad_invocations(argv):
    with pytest.raises(SystemExit):
        ep.main(argv)


@needs_land_mask
def test_export_years_separates_already_complete_years_from_failures(tmp_path):
    """`present` is what lets main tell a requeue from a total failure."""
    kw = dict(root=tmp_path, loader=make_loader())
    first = ep.export_years([Stub()], ["fake/D6C-f0.h5"], [2016],
                            "kriged-airs", **kw)
    assert len(first["written"]) == 1 and first["present"] == []
    again = ep.export_years([Stub()], ["fake/D6C-f0.h5"], [2016],
                            "kriged-airs", **kw)
    assert again["written"] == [] and again["skipped"] == {}
    assert again["present"] == [ep.output_path(
        "dlfront_D6C-f0_kriged-airs", 2016, root=tmp_path)]


@needs_land_mask
def test_main_exits_nonzero_when_every_year_failed(tmp_path, monkeypatch):
    """Audit 2026-08-18: a fully-failed export must not report success.

    An `--dependency=afterok:` dependant (the flag-injection phase) would
    otherwise run against an empty archive behind a green job.
    """
    monkeypatch.setattr(ep.predict, "load_model", lambda p: Stub())
    monkeypatch.setattr(ep.dataset, "load_norm_stats", lambda: {})
    ckpt = tmp_path / "D6C-f0.h5"
    ckpt.write_bytes(b"stub")
    argv = ["--ckpt", str(ckpt), "--source", "kriged-airs",
            "--years", "2016,2017", "--root", str(tmp_path / "archive")]

    def missing(year, *a, **kw):
        raise FileNotFoundError(f"no kriged cache for {year}")

    monkeypatch.setattr(ep, "load_year", missing)
    assert ep.main(argv) == 1
    # one buildable year is enough to keep the job green (the other is still
    # reported on stdout as SKIPPED)
    monkeypatch.setattr(ep, "load_year", lambda year, *a, **kw:
                        missing(year) if year == 2017
                        else make_loader()(year))
    assert ep.main(argv) == 0
    # and so is a fully-complete idempotent requeue with a failing year
    monkeypatch.setattr(ep, "load_year", missing)
    assert ep.main(["--ckpt", str(ckpt), "--source", "kriged-airs",
                    "--years", "2016", "--root",
                    str(tmp_path / "archive")]) == 0
