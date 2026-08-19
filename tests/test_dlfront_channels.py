"""Model input-channel subsetting (channel ladder, user decision 2026-08-18).

Everything here guards the split between two sets that used to be one:

* ``config.SFC_VARS`` -- the ON-DISK schema of ``sfc_daily`` and the kriged
  caches (five variables, written by acquire_merra2_sfc / krige_fill), which
  must never change meaning;
* ``config.INPUT_CHANNELS`` -- the subset of them the MODEL consumes, which
  the stage-A ladder cuts to (T2M, QV2M, SLP) and (T2M, QV2M) to measure how
  much front skill survives on the channels AIRS actually provides (U10M and
  V10M are WRF-27km; SLP is copied clean from MERRA-2).

No TensorFlow: every model here is a duck-typed stub with an
``input_shape`` attribute, which is all evaluate_test's alignment check
reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dl_front import config, dataset, evaluate_test


@pytest.fixture(autouse=True)
def restore_input_channels():
    """Undo any ``set_input_channels`` a test performs.

    ``config.INPUT_CHANNELS`` is process-global module state (that is the
    point -- the CLIs install it once, before any loading), so a test that
    narrowed it would silently narrow every test that ran after it.
    """
    original = config.INPUT_CHANNELS
    yield
    config.INPUT_CHANNELS = original


# --------------------------------------------------------------------------- #
# config.set_input_channels (C1)
# --------------------------------------------------------------------------- #

def test_set_input_channels_normalises_to_sfc_vars_order():
    """Channel order must come from SFC_VARS, never from the flag's order.

    A checkpoint's weights are bound to the order of x's trailing axis, so
    "--channels QV2M,T2M" and "--channels T2M,QV2M" have to produce the same
    model; if they did not, an eval could feed a checkpoint its own channels
    transposed and report plausible-but-meaningless numbers.
    """
    assert config.set_input_channels(["QV2M", "T2M"]) == ("T2M", "QV2M")
    assert config.INPUT_CHANNELS == ("T2M", "QV2M")
    assert config.set_input_channels(["V10M", "SLP", "T2M"]) == \
        ("T2M", "SLP", "V10M")
    # accepts any iterable of names, and strips the whitespace a hand-typed
    # "--channels T2M, QV2M" leaves behind
    assert config.set_input_channels(("T2M", " QV2M")) == ("T2M", "QV2M")


def test_set_input_channels_rejects_bad_sets_by_name():
    """Unknown / duplicated / empty channel sets raise, naming the offender.

    House rule for errors: the message must name the offending value AND
    the knobs that fix it, because the alternative symptom (a model trained
    on the wrong inputs) only shows up as a slightly wrong CSI months later.
    """
    with pytest.raises(ValueError, match="RH2M"):
        config.set_input_channels(["T2M", "RH2M"])
    with pytest.raises(ValueError, match="duplicate"):
        config.set_input_channels(["T2M", "T2M"])
    with pytest.raises(ValueError, match="empty"):
        config.set_input_channels([])
    for spec in (["T2M", "RH2M"], ["T2M", "T2M"], []):
        try:
            config.set_input_channels(spec)
        except ValueError as exc:
            assert "--channels" in str(exc)          # names the CLI fix
            assert "channels" in str(exc)            # ... and the yaml key
    # a rejected set must not have been installed
    assert config.INPUT_CHANNELS == config.SFC_VARS


def test_sfc_vars_is_the_on_disk_schema_and_is_not_touched():
    """SFC_VARS keeps its five-variable meaning whatever the model consumes.

    acquire_merra2_sfc and krige_fill are written against it; narrowing it
    would silently change the files on disk rather than the model's inputs.
    """
    assert config.SFC_VARS == ("T2M", "QV2M", "SLP", "U10M", "V10M")
    config.set_input_channels(["T2M", "QV2M"])
    assert config.SFC_VARS == ("T2M", "QV2M", "SLP", "U10M", "V10M")
    assert set(config.INPUT_CHANNELS) <= set(config.SFC_VARS)


def test_yaml_default_is_all_five_channels():
    """configs/dl_front.yaml's inputs.channels documents the default: the
    5-channel paper replication, so the overnight main-chain run is
    unchanged by the ladder work."""
    assert ("inputs", "channels") in config.TUNABLES
    assert config.TUNABLES[("inputs", "channels")] == "INPUT_CHANNELS"
    assert config.load_tunables()["INPUT_CHANNELS"] == config.SFC_VARS


# --------------------------------------------------------------------------- #
# dataset.sfc_x honours INPUT_CHANNELS (C1)
# --------------------------------------------------------------------------- #

def _constant_day() -> tuple[xr.Dataset, dict]:
    """One surface step carrying all five on-disk vars, value 10*(i+1), and
    frozen stats keyed by NAME (mean = value - 2, sd = 4 -> z = 0.5)."""
    day = xr.Dataset(
        {v: (("time", "lat", "lon"),
             np.full((1, *config.GRID_SHAPE), 10.0 * (i + 1), np.float32))
         for i, v in enumerate(config.SFC_VARS)},
        coords={"time": pd.date_range("2003-01-02", periods=1, freq="3h")})
    stats = {v: [10.0 * (i + 1) - 2.0, 4.0]
             for i, v in enumerate(config.SFC_VARS)}
    return day, stats


def test_sfc_x_stacks_only_the_model_channels_in_order():
    """The file always has five vars; x has len(INPUT_CHANNELS), in
    SFC_VARS order, with each channel z-scored by ITS OWN frozen stats.

    The per-name stats lookup is what lets the frozen norm-stats JSON keep
    all five keys and stay untouched (user decision 2026-08-18): a
    positional lookup would z-score SLP with T2M's constants under
    --channels QV2M,SLP.
    """
    day, stats = _constant_day()
    # raw physical values, so a mis-ordered stack is visible in the values
    raw = {v: float(day[v].values[0, 0, 0]) for v in config.SFC_VARS}

    config.set_input_channels(["SLP", "T2M"])          # deliberately unsorted
    x = dataset.sfc_x(day, stats)
    assert x.shape == (1, *config.GRID_SHAPE, 2)
    # SFC_VARS order: T2M first, then SLP -- NOT the order typed above
    np.testing.assert_allclose(
        x[0, 0, 0], [(raw["T2M"] - stats["T2M"][0]) / stats["T2M"][1],
                     (raw["SLP"] - stats["SLP"][0]) / stats["SLP"][1]])

    config.set_input_channels(config.SFC_VARS)
    x5 = dataset.sfc_x(day, stats)
    assert x5.shape == (1, *config.GRID_SHAPE, 5)
    # subsetting must not disturb the surviving channels' values
    np.testing.assert_allclose(x[..., 0], x5[..., 0])          # T2M
    np.testing.assert_allclose(x[..., 1], x5[..., 2])          # SLP


# --------------------------------------------------------------------------- #
# Kriged-cache guard, restricted to the model's channels (C2)
# --------------------------------------------------------------------------- #

def _write_kriged_year(dirpath, year, times, kriged_channels,
                       value: float = 12.0):
    """A schema-v4 kriged cache whose recorded channel split is settable.

    Everything except ``kriged_channels`` matches the current config, so a
    failure in these tests can only come from the channel check.
    """
    shape = (len(times), *config.GRID_SHAPE)
    data = {v: (("time", "lat", "lon"), np.full(shape, value, np.float32))
            for v in config.SFC_VARS}
    data["valid_frac"] = (("time", "lat", "lon"), np.ones(shape, np.float32))
    ds = xr.Dataset(data, coords={"time": pd.DatetimeIndex(times),
                                  "lat": list(config.LABEL_LATS),
                                  "lon": list(config.LABEL_LONS)},
                    attrs={"source": "degraded_reanalysis",
                           "variogram_model": "linear",
                           "max_obs_points": 1500, "created": "test",
                           "schema_version": 4,
                           "domain_lat_range": list(config.ANALYSIS_LAT_RANGE),
                           "domain_lon_range": list(config.ANALYSIS_LON_RANGE),
                           "land_fraction_min": config.LAND_FRACTION_MIN,
                           "halo_px": dataset.halo_px(),
                           "kriged_channels": list(kriged_channels),
                           "swath_bank": "per-day-envelope"})
    dirpath.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dirpath / f"kriged_sfc_{year}.nc")


@pytest.fixture
def kriged_cache(tmp_path, monkeypatch):
    """Factory: build a cache with a given kriged_channels attr and load it.

    Labels are synthetic (one 6-class step) so no data root is needed.
    """
    times = pd.DatetimeIndex(["2010-01-01 18:00"])
    names = dataset.class_names(6)
    fronts = np.zeros((1, len(names), *config.GRID_SHAPE), np.float32)
    fronts[0, 0, 30, 75] = 1.0                        # one cold pixel
    lab = xr.Dataset(
        {"fronts": (("time", "front", "lat", "lon"), fronts)},
        coords={"time": times, "lat": list(config.LABEL_LATS),
                "lon": list(config.LABEL_LONS),
                "front_type": ("front", list(names))})
    monkeypatch.setattr(dataset, "load_label_ds",
                        lambda year, n_classes: lab.copy(deep=True))
    monkeypatch.setattr(
        config, "KRIGED_SOURCE_DIRS",
        {"kriged-degraded": tmp_path / "degraded_reanalysis",
         "kriged-airs": tmp_path / "airs_fcst"})

    def load(kriged_channels):
        _write_kriged_year(tmp_path / "degraded_reanalysis", 2010, times,
                           kriged_channels)
        stats = {v: [10.0, 4.0] for v in config.SFC_VARS}
        return dataset.kriged_year_arrays(2010, 6, stats, "kriged-degraded")
    return load


def test_kriged_guard_accepts_wider_cache_for_a_channel_subset(kriged_cache):
    """C2 relaxation: a 2-channel (T2M, QV2M) model may reuse the EXISTING
    4-channel-kriged caches.

    The old equality test compared the cache's whole kriged set against
    ``airs.kriged_channels``, which rejected those caches for the ladder's
    2-channel rung even though the channels they disagree about (U10M,
    V10M) are ones that model never reads.  Rebuilding ~15 years of caches
    to run an ablation would have been the alternative.
    """
    assert config.KRIGED_CHANNELS == ("T2M", "QV2M")   # premise of the test
    config.set_input_channels(["T2M", "QV2M"])
    x, y, times = kriged_cache(["T2M", "QV2M", "U10M", "V10M"])
    assert x.shape == (1, *config.GRID_SHAPE, 2)       # loaded, 2 channels
    assert len(times) == 1 and y.dtype == np.uint8


def test_kriged_guard_still_rejects_a_consumed_channel_mismatch(kriged_cache):
    """The ONE hazard the relaxed guard must not let through: a channel the
    config calls kriged that the cache holds as a CLEAN reanalysis copy.

    Under the channel-sourcing decision 2026-08-18 the loader reads only
    ``config.KRIGED_CHANNELS`` from the cache -- everything else comes from
    ``sfc_daily`` -- so a cache that kriged MORE than the config is now
    harmless (its extra fills are simply never read; that is what makes the
    v3 caches reusable).  The reverse is not: if the config promises an
    AIRS-shaped gap fill for a channel and the cache only has reanalysis in
    it, the model trains on reanalysis while the run's provenance claims
    satellite information -- and the values cannot tell the two apart.

    Both consumed channels are checked, and the message must name the two
    real fixes (drop the channel from ``airs.kriged_channels``, or rebuild).
    """
    config.set_input_channels(["T2M", "QV2M"])
    # QV2M is clean in the cache but kriged per the config
    with pytest.raises(ValueError) as exc:
        kriged_cache(["T2M", "U10M", "V10M"])
    msg = str(exc.value)
    assert "are kriged channels per config airs.kriged_channels=" in msg
    assert "holds a CLEAN reanalysis copy where a satellite-shaped gap " \
           "fill is expected" in msg
    assert "drop ['QV2M'] from airs.kriged_channels" in msg    # fix 1
    assert "krige_fill build-degraded --years 2010" in msg     # fix 2
    # ... and T2M, the other consumed channel, is guarded the same way
    with pytest.raises(ValueError, match=r"\['T2M'\] are kriged channels"):
        kriged_cache(["QV2M", "U10M", "V10M"])
    # A channel the config treats as CLEAN is no longer the guard's business:
    # a cache that kriged SLP is accepted for a (T2M, QV2M) model, because
    # its SLP copy is never read.  (The 5-channel width of the same relaxation
    # is exercised in test_dlfront_airs_krige.py, where the reanalysis day
    # files the clean channels are read from actually have to exist.)
    x, _, _ = kriged_cache(["T2M", "QV2M", "SLP"])
    assert x.shape == (1, *config.GRID_SHAPE, 2)


# --------------------------------------------------------------------------- #
# Checkpoint/config channel alignment (C4)
# --------------------------------------------------------------------------- #

class FakeModel:
    """Duck-typed stand-in for a loaded keras model: the alignment check
    reads only ``input_shape``'s last axis (the net is fully convolutional,
    so the spatial dims are None and deliberately not compared)."""

    def __init__(self, n_channels, spatial=(None, None)):
        self.input_shape = (None, *spatial, n_channels)


def _write_run_config(ckpt_dir, channels):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "run_config.yaml").write_text(
        "run_args:\n  channels:\n" +
        "".join(f"  - {c}\n" for c in channels))


def test_check_model_channels_refuses_a_mismatch(tmp_path):
    """NEVER score a model whose channel count is not the config's.

    This is the guard the whole 2026-08-18 cleanup exists for: feeding a
    5-channel checkpoint two channels (or vice versa) does not necessarily
    crash -- it produces plausible, comparable-looking, meaningless CSI.
    """
    ckpt = tmp_path / "D6A-f0.h5"
    config.set_input_channels(["T2M", "QV2M"])
    with pytest.raises(ValueError) as exc:
        evaluate_test.check_model_channels(FakeModel(5), ckpt)
    msg = str(exc.value)
    assert str(ckpt) in msg                       # names the checkpoint
    assert "5 input channel" in msg and "has 2" in msg    # names both counts
    assert "--channels" in msg                    # names the exact fix
    # matching counts pass silently, in both the subset and full cases
    evaluate_test.check_model_channels(FakeModel(2), ckpt)
    config.set_input_channels(config.SFC_VARS)
    evaluate_test.check_model_channels(FakeModel(5), ckpt)


def test_check_model_channels_quotes_the_checkpoints_own_list(tmp_path):
    """The suggested --channels comes from the checkpoint's run_config.yaml
    when it has one (integration 2026-08-18).

    The first version always suggested ``SFC_VARS[:n_model]``, which is
    right for the 5-channel and D6A3 rungs and wrong for any other subset --
    a (T2M, SLP) model would be told to pass T2M,QV2M.  Handing the operator
    a confidently wrong flag inside the error that exists to prevent
    silently wrong numbers is its own trap, so a guess is only offered when
    nothing was recorded, and is labelled as one.
    """
    ckpt = tmp_path / "odd" / "odd-f0.h5"
    _write_run_config(ckpt.parent, ["T2M", "SLP"])
    config.set_input_channels(config.SFC_VARS)
    with pytest.raises(ValueError, match=r"--channels T2M,SLP"):
        evaluate_test.check_model_channels(FakeModel(2), ckpt)

    bare = tmp_path / "bare" / "bare-f0.h5"        # no run_config.yaml
    bare.parent.mkdir()
    with pytest.raises(ValueError) as exc:
        evaluate_test.check_model_channels(FakeModel(2), bare)
    assert "GUESS" in str(exc.value)               # honest about guessing


def test_check_model_channels_skips_duck_typed_models(tmp_path):
    """The BK19 leg has no checkpoint and no input_shape; the check must be
    a no-op there rather than crashing the one leg it cannot apply to."""
    config.set_input_channels(["T2M", "QV2M"])

    class NoShape:
        pass

    evaluate_test.check_model_channels(NoShape(), None)
    evaluate_test.check_model_channels(FakeModel(None), tmp_path / "x.h5")


def test_resolve_channels_prefers_flag_then_run_config(tmp_path, capsys):
    """--channels wins; otherwise the checkpoint's recorded list is adopted
    (and PRINTED, so an eval never changes its own inputs quietly); with
    neither, the yaml default stands.  This is what makes the ablation
    chain's evals work without threading the flag through every leg."""
    ckpt = tmp_path / "D6A2-f0" / "D6A2-f0.h5"
    _write_run_config(ckpt.parent, ["T2M", "QV2M"])

    assert evaluate_test.resolve_channels(ckpt, "T2M,SLP") == ("T2M", "SLP")

    assert evaluate_test.resolve_channels(ckpt, None) == ("T2M", "QV2M")
    out = capsys.readouterr().out
    assert "adopted input channels" in out and "run_config.yaml" in out

    bare = tmp_path / "bare.h5"
    config.set_input_channels(config.SFC_VARS)
    assert evaluate_test.resolve_channels(bare, None) == config.SFC_VARS
