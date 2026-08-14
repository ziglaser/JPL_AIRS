"""Tests for AnalysisConfig and the config-driven screens (dataset.apply_screens).

Synthetic-table tests only -- the real-data path is exercised by the paper
benchmarks and the notebook-08 driver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from convection_skill import config
from convection_skill.config import AnalysisConfig
from convection_skill.dataset import apply_screens


# --------------------------------------------------------------------------- #
# AnalysisConfig semantics
# --------------------------------------------------------------------------- #
def test_config_validates_inference_and_datasets():
    with pytest.raises(ValueError, match="inference"):
        AnalysisConfig(inference="bayes")
    with pytest.raises(ValueError, match="valid_datasets"):
        AnalysisConfig(valid_datasets=("mrms", "goes"))


def test_named_modes_match_zachs_three():
    m1 = AnalysisConfig.all_products()
    m2 = AnalysisConfig.all_products_complete_days()
    m3 = AnalysisConfig.smap()
    assert set(m1.valid_datasets) == {"airs", "airs_fcst", "smap", "mrms"}
    assert not m1.require_complete_days
    assert m2.valid_datasets == m1.valid_datasets and m2.require_complete_days
    assert set(m3.valid_datasets) == {"smap", "mrms"}


def test_validity_columns_dedup_and_order():
    cfg = AnalysisConfig(valid_datasets=("airs_fcst", "mrms"))
    cols = cfg.validity_columns()
    assert set(cols) == {"mu_cape", "mu_el", "mu_lcl", "mu_cin", "qpe"}
    assert len(cols) == len(set(cols))


def test_label_distinguishes_runs():
    a = AnalysisConfig(inference="iid")
    b = AnalysisConfig(inference="block")
    c = AnalysisConfig(screen_forecast_rain=True)
    d = AnalysisConfig.smap()
    labels = {a.label(), b.label(), c.label(), d.label()}
    assert len(labels) == 4
    assert AnalysisConfig(name="custom").label() == "custom"


def test_default_inference_is_paper_iid():
    assert AnalysisConfig().inference == "iid"


# --------------------------------------------------------------------------- #
# YAML/JSON construction
# --------------------------------------------------------------------------- #
def test_from_dict_coerces_lists_and_rejects_typos():
    cfg = AnalysisConfig.from_dict(
        {"years": [2019], "valid_datasets": ["smap", "mrms"], "inference": "block"})
    assert cfg.years == (2019,) and cfg.valid_datasets == ("smap", "mrms")
    with pytest.raises(ValueError, match="unknown AnalysisConfig keys"):
        AnalysisConfig.from_dict({"yeers": [2019]})


def test_from_dict_preset_routes_to_named_mode():
    cfg = AnalysisConfig.from_dict({"preset": "smap", "years": [2019]})
    assert cfg.valid_datasets == ("smap", "mrms") and cfg.years == (2019,)
    with pytest.raises(ValueError, match="preset"):
        AnalysisConfig.from_dict({"preset": "everything"})


def test_from_file_yaml_single_and_runs(tmp_path):
    single = tmp_path / "one.yaml"
    single.write_text("years: [2019]\ninference: block\n")
    (cfg,) = AnalysisConfig.from_file(single)
    assert cfg.years == (2019,) and cfg.inference == "block"

    multi = tmp_path / "many.yaml"
    multi.write_text(
        "defaults:\n  years: [2019]\n"
        "runs:\n  - name: a\n  - name: b\n    inference: block\n")
    cfgs = AnalysisConfig.from_file(multi)
    assert [c.name for c in cfgs] == ["a", "b"]
    assert all(c.years == (2019,) for c in cfgs)
    assert cfgs[1].inference == "block" and cfgs[0].inference == "iid"


def test_from_file_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"years": [2019, 2020], "valid_datasets": ["mrms"]}')
    (cfg,) = AnalysisConfig.from_file(p)
    assert cfg.years == (2019, 2020) and cfg.valid_datasets == ("mrms",)
    with pytest.raises(ValueError, match="yaml"):
        AnalysisConfig.from_file(tmp_path / "cfg.toml")


# --------------------------------------------------------------------------- #
# The two-file layout (data table + hypothesis tests)
# --------------------------------------------------------------------------- #
def _write_pair(tmp_path, data_text="years: [2019]\n",
                hyp_text="heavy_percentile: 99.0\n"):
    data = tmp_path / "data.yaml"
    hyp = tmp_path / "hyp.yaml"
    data.write_text(data_text)
    hyp.write_text(hyp_text)
    return data, hyp


def test_from_files_merges_both_groups(tmp_path):
    data, hyp = _write_pair(
        tmp_path,
        hyp_text="heavy_percentile: 99.0\nhypotheses: topline\n"
                 "strata:\n  T1T2_sign: [aridity]\n")
    (cfg,) = AnalysisConfig.from_files(data, hyp)
    assert cfg.years == (2019,)
    assert cfg.heavy_percentile == 99.0
    assert cfg.hypotheses == "topline"
    assert cfg.strata == {"T1T2_sign": ("aridity",)}


def test_from_files_rejects_misplaced_keys(tmp_path):
    data, hyp = _write_pair(tmp_path, data_text="screen_forecast_rain: true\n")
    with pytest.raises(ValueError, match="belong in the hypothesis config"):
        AnalysisConfig.from_files(data, hyp)
    data, hyp = _write_pair(tmp_path, hyp_text="years: [2019]\n")
    with pytest.raises(ValueError, match="belong in the data-table config"):
        AnalysisConfig.from_files(data, hyp)


def test_from_files_runs_share_the_data_table(tmp_path):
    data, hyp = _write_pair(
        tmp_path,
        hyp_text="defaults:\n  heavy_percentile: 99.0\n"
                 "runs:\n  - name: a\n  - name: b\n    inference: block\n")
    cfgs = AnalysisConfig.from_files(data, hyp)
    assert [c.name for c in cfgs] == ["a", "b"]
    assert all(c.years == (2019,) and c.heavy_percentile == 99.0 for c in cfgs)
    assert cfgs[1].inference == "block"


def test_repo_config_files_load():
    repo = Path(__file__).resolve().parents[1]
    (full,) = AnalysisConfig.from_files(repo / "configs" / "data_table.yaml",
                                        repo / "configs" / "hypothesis_tests.yaml")
    assert full.hypotheses == "all" and full.heavy_percentile == 99.9
    assert full.screen_overpass_rain and full.screen_forecast_rain
    (sample,) = AnalysisConfig.from_files(
        repo / "configs" / "data_table.yaml",
        repo / "configs" / "hypothesis_topline_p99.yaml")
    assert sample.hypotheses == "topline" and sample.heavy_percentile == 99.0
    assert sample.label() == "topline_p99"


def test_label_reflects_percentile_and_selection():
    default = AnalysisConfig()
    assert AnalysisConfig(heavy_percentile=99.0).label() != default.label()
    assert "q99" in AnalysisConfig(heavy_percentile=99.0).label()
    assert "topline" in AnalysisConfig(hypotheses="topline").label()


def test_hypotheses_field_validates_and_normalizes():
    assert AnalysisConfig(hypotheses=["A1_mu"]).hypotheses == ("A1_mu",)
    with pytest.raises(ValueError, match="hypotheses"):
        AnalysisConfig(hypotheses="headline")


# --------------------------------------------------------------------------- #
# Spec selection and config-driven strata/controls in the runner
# --------------------------------------------------------------------------- #
def test_select_specs_all_topline_and_ids():
    from convection_skill.hypotheses import REGISTRY, select_specs
    assert select_specs("all") == REGISTRY
    topline = select_specs("topline")
    assert 0 < len(topline) < len(REGISTRY)
    assert all(s.tier == "topline" for s in topline)
    # one primary spec per hypothesis-table row, incl. the headline pair
    assert {"A1_mu", "T1T2_sign"} <= {s.id for s in topline}
    assert [s.id for s in select_specs(("T1T2_sign", "A1_mu"))] == \
        ["T1T2_sign", "A1_mu"]
    with pytest.raises(ValueError, match="unknown hypothesis ids"):
        select_specs(("A1_mu", "Z9_bogus"))


def _tiny_prepared(cfg):
    """200 synthetic rows over 20 days with every stratifier column."""
    from convection_skill.dataset import Prepared
    rng = np.random.default_rng(0)
    n = 200
    days = np.datetime64("2019-06-01") + (np.arange(n) % 20).astype("timedelta64[D]")
    table = pd.DataFrame({
        "mu_cape": rng.random(n), "day": days,
        "season": "JJA", "is_east": rng.random(n) > 0.5,
        "is_late_slot": rng.random(n) > 0.5,
        "fcst_q": rng.random(n), "sm_cell_clim": rng.random(n),
        "wind": rng.random(n),
    })
    flags = {"heavy": rng.random(n) > 0.8}
    return Prepared(cfg=cfg, table=table, thresholds={}, flags=flags, onset=None)


def test_config_governs_strata_controls_and_scopes():
    from convection_skill.hypotheses import HypothesisSpec
    from convection_skill.suite import test_hypothesis
    spec = HypothesisSpec("X", "toy", "mu_cape", controls=("fcst_q",),
                          strata=("wind",))
    kw = dict(n_boot_reps=50, valid_datasets=())

    rows, _ = test_hypothesis(spec, _tiny_prepared(AnalysisConfig(**kw)))
    scopes = {row["scope"] for row in rows}
    assert "ctrl:fcst_q" in scopes and any(s.startswith("wind=") for s in scopes)
    assert any(s.startswith("season=") for s in scopes)  # default strata ride along

    # strata override replaces the spec's own; run_controls kills the ctrl rows
    cfg = AnalysisConfig(strata={"X": ("humidity",)}, run_controls=False, **kw)
    scopes = {row["scope"] for row in test_hypothesis(spec, _tiny_prepared(cfg))[0]}
    assert not any(s.startswith("wind=") or s.startswith("ctrl:") for s in scopes)
    assert any(s.startswith("humidity=") for s in scopes)

    # run_strata off -> only the overall row remains
    cfg = AnalysisConfig(run_strata=False, run_controls=False, **kw)
    scopes = {row["scope"] for row in test_hypothesis(spec, _tiny_prepared(cfg))[0]}
    assert scopes == {"overall"}

    with pytest.raises(ValueError, match="unknown stratifiers"):
        test_hypothesis(spec, _tiny_prepared(
            AnalysisConfig(strata={"X": ("elevation",)}, **kw)))


def test_run_curves_accepts_bool_or_id_list():
    from convection_skill.hypotheses import HypothesisSpec
    from convection_skill.suite import test_hypothesis
    assert AnalysisConfig(run_curves=["A2_cin"]).run_curves == ("A2_cin",)
    assert AnalysisConfig(run_curves=["A2_cin"]).wants_curve("A2_cin")
    assert not AnalysisConfig(run_curves=["A2_cin"]).wants_curve("A4_q")
    assert AnalysisConfig().wants_curve("anything")

    spec = HypothesisSpec("X", "toy", "mu_cape", curve=True)
    kw = dict(n_boot_reps=50, valid_datasets=(), run_strata=False)
    _, curve = test_hypothesis(spec, _tiny_prepared(AnalysisConfig(**kw)))
    assert curve is not None
    _, curve = test_hypothesis(
        spec, _tiny_prepared(AnalysisConfig(run_curves=("other",), **kw)))
    assert curve is None


def test_event_thresholds_use_config_percentiles():
    from convection_skill.dataset import event_thresholds
    rng = np.random.default_rng(1)
    base = pd.DataFrame({
        "land_frac": 1.0, "qpe": rng.exponential(size=5000),
        "qpe_max": rng.exponential(size=5000), "qpe_sk": rng.normal(size=5000),
        "mml_cape": rng.random(5000), "mml_lcl": rng.random(5000),
        "mu_cin": -rng.random(5000),
    })
    lo = event_thresholds(base, AnalysisConfig(heavy_percentile=99.0))
    hi = event_thresholds(base, AnalysisConfig(heavy_percentile=99.9))
    assert lo["heavy"] == pytest.approx(np.percentile(base["qpe"], 99.0))
    assert lo["heavy"] < hi["heavy"]


# --------------------------------------------------------------------------- #
# apply_screens on a hand-built base table
# --------------------------------------------------------------------------- #
def _toy_base():
    """Two cells x two days x slots 1-2; one ocean cell; hand-set validity.

    Mark's screen columns default to passing values (low parcel, dry start,
    complete series) so each test flips exactly what it exercises.
    """
    rows = []
    for day in ("2019-06-05", "2019-06-06"):
        for lat, lon, land in [(40.5, -95.5, 1.0), (40.5, -94.5, 0.2)]:
            for slot in (1, 2):
                rows.append({
                    "day": np.datetime64(day), "lat": lat, "lon": lon,
                    "slot": slot, "land_frac": land,
                    "qpe": 0.0, "qpe_overpass": 0.0,
                    "mu_cape": 100.0, "mu_el": 1.0, "mu_lcl": 1.0, "mu_cin": -5.0,
                    "mu_cape_overpass": 50.0, "sm_raw": 0.3,
                    "alt_max": 500.0, "dry_start_qpe": 0.0, "valid7": True,
                })
    return pd.DataFrame(rows)


def test_land_screen_drops_ocean_cells():
    base = _toy_base()
    out = apply_screens(base, AnalysisConfig(valid_datasets=()))
    assert (out["land_frac"] >= config.LAND_FRACTION_MIN).all()
    assert len(out) == len(base) // 2


def test_validity_screen_is_compositional():
    base = _toy_base()
    base.loc[0, "sm_raw"] = np.nan  # first land row loses SMAP only
    n_mrms = len(apply_screens(base, AnalysisConfig(valid_datasets=("mrms",))))
    n_smap = len(apply_screens(base, AnalysisConfig.smap()))
    assert n_smap == n_mrms - 1  # the NaN-SMAP row drops only under mode 3


def test_complete_days_drops_partial_cell_days():
    # Mark's completeness: valid7 (CAPE/CIN/QPE finite at ALL 7 hours) is a
    # cell-day property; a False drops every slot of that cell-day.
    base = _toy_base()
    bad = (base["day"] == np.datetime64("2019-06-05")) & (base["lon"] == -95.5)
    base.loc[bad, "valid7"] = False
    cfg = AnalysisConfig(valid_datasets=("airs_fcst",), slots=(1, 2),
                         require_complete_days=True)
    out = apply_screens(base, cfg)
    assert not ((out["day"] == np.datetime64("2019-06-05"))
                & (out["lon"] == -95.5)).any()
    off = AnalysisConfig(valid_datasets=("airs_fcst",), slots=(1, 2),
                         require_complete_days=False)
    assert len(apply_screens(base, off)) == len(apply_screens(base, cfg)) + 2


def test_mark_screens_route_through_config():
    base = _toy_base()
    high = base["lon"] == -95.5
    base.loc[high, "alt_max"] = 1500.0   # parcel above 1000 m at some hour
    out = apply_screens(base, AnalysisConfig(valid_datasets=()))
    assert (out["lon"] != -95.5).all()
    out = apply_screens(base, AnalysisConfig(valid_datasets=(),
                                             screen_altitude=False))
    assert (out["lon"] == -95.5).any()

    base = _toy_base()
    base.loc[base["lon"] == -95.5, "dry_start_qpe"] = 0.5  # rained by hour 1
    out = apply_screens(base, AnalysisConfig(valid_datasets=()))
    assert (out["lon"] != -95.5).all()
    out = apply_screens(base, AnalysisConfig(valid_datasets=(),
                                             screen_dry_start=False))
    assert (out["lon"] == -95.5).any()

    # NaN in the deciding value FAILS the screen (Mark's semantics)
    base = _toy_base()
    base.loc[base["lon"] == -95.5, "alt_max"] = np.nan
    out = apply_screens(base, AnalysisConfig(valid_datasets=()))
    assert (out["lon"] != -95.5).all()


def test_rain_screens_route_through_config():
    base = _toy_base()
    base.loc[(base["lon"] == -95.5), "qpe_overpass"] = 5.0  # rained at overpass
    cfg = AnalysisConfig(valid_datasets=(), screen_overpass_rain=True)
    out = apply_screens(base, cfg)
    assert (out["lon"] != -95.5).all()  # whole cell dropped, other land rows kept
