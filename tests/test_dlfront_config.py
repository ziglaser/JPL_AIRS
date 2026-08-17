"""The configs/dl_front.yaml tunables layer: strict, typo-proof, overridable."""
from __future__ import annotations

import numbers

import pytest
import yaml

from dl_front import config


def test_tracked_yaml_is_the_paper_replication():
    """The repo defaults must reproduce Biard & Kunkel's stated values."""
    vals = config.load_tunables()
    assert vals["N_CONV_LAYERS"] == 3
    assert vals["N_FILTERS"] == 80
    assert vals["KERNEL_SIZE"] == 5
    assert vals["DROPOUT"] == 0.5
    assert vals["NONE_WEIGHT"] == 0.35
    assert vals["LEARNING_RATE"] == 1e-4
    assert vals["PAPER_PATIENCE"] == 100
    assert vals["N_FOLDS"] == 3
    assert vals["TRAIN_YEARS_5"] == tuple(range(2003, 2008))
    assert vals["EVAL_YEARS_5"] == tuple(range(2008, 2016))
    assert vals["ROC_FACTORS"][0] == 0.0 and vals["ROC_FACTORS"][-1] == 4096.0


def test_dryline_splits_are_the_2026_08_12_decision():
    """All 6-class training stages share 2007-2015; 2016-2018 is held out
    (user decision 2026-08-13: the BK19 published predictions end 2018, so
    the three-way test uses identical years for every leg)."""
    vals = config.load_tunables()
    assert vals["TRAIN_YEARS_6"] == tuple(range(2007, 2016))
    assert vals["EVAL_YEARS_6"] == (2016, 2017, 2018)


def test_airs_and_kriging_tunables():
    """The frozen AIRS/kriging knobs (interface spec 2026-08-12)."""
    vals = config.load_tunables()
    assert vals["AIRS_HOURS"] == (21, 0)
    # terrain-following surface extraction (user decision 2026-08-16; the
    # old fixed 985-hPa target had zero coverage over all elevated terrain)
    assert vals["AIRS_SURFACE_SCAN_FLOOR_HPA"] == 600
    assert vals["AIRS_SURFACE_MAX_AGL_M"] == 1500
    assert vals["AIRS_SURFACE_LAPSE_K_PER_KM"] == 6.5
    assert vals["AIRS_SURFACE_LAPSE_DERIVED"] is True
    assert vals["AIRS_SURFACE_LAPSE_CLIP_K_PER_KM"] == (-3.0, 9.8)
    assert vals["KRIGED_CHANNELS"] == ("T2M", "QV2M", "U10M", "V10M")
    assert vals["KRIGE_VARIOGRAM"] == "linear"
    assert vals["KRIGE_MAX_OBS"] == 1500
    assert vals["KRIGE_SEED"] == 20260812


def test_module_constants_come_from_yaml():
    """Import-time load: every tunable is a module attribute, lists->tuples."""
    for const in config.TUNABLES.values():
        value = getattr(config, const)
        assert not isinstance(value, list), f"{const} should be a tuple"
        assert isinstance(value, (numbers.Number, tuple, str)), const
    assert config.MAX_EPOCHS == 600           # spot-check a train.py consumer
    assert config.LABEL_WIDTH == 3            # ...and a dataset.py consumer


def _write(tmp_path, mutate):
    raw = yaml.safe_load(config.CONFIG_YAML.read_text())
    mutate(raw)
    p = tmp_path / "override.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_override_file(tmp_path):
    """An experiment YAML changes exactly what it edits (env-var use case)."""
    p = _write(tmp_path, lambda raw: raw["training"].update(batch_size=256))
    vals = config.load_tunables(p)
    assert vals["BATCH_SIZE"] == 256
    assert vals["N_FILTERS"] == config.N_FILTERS


def test_unknown_key_raises(tmp_path):
    p = _write(tmp_path, lambda raw: raw["training"].update(batchsize=64))
    with pytest.raises(ValueError, match="batchsize"):
        config.load_tunables(p)


def test_new_sections_are_strict_too(tmp_path):
    """The airs/kriging sections obey the same typo-proofing."""
    p = _write(tmp_path, lambda raw: raw["kriging"].update(varigram="linear"))
    with pytest.raises(ValueError, match="varigram"):
        config.load_tunables(p)
    p = _write(tmp_path, lambda raw: raw["airs"].pop("hours"))
    with pytest.raises(ValueError, match="airs.hours"):
        config.load_tunables(p)


def test_unknown_section_raises(tmp_path):
    p = _write(tmp_path, lambda raw: raw.update(taining={"patience": 5}))
    with pytest.raises(ValueError, match="taining"):
        config.load_tunables(p)


def test_missing_key_raises(tmp_path):
    p = _write(tmp_path, lambda raw: raw["loss"].pop("none_weight"))
    with pytest.raises(ValueError, match="none_weight"):
        config.load_tunables(p)
