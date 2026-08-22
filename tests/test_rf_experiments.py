"""Unit tests for convection_skill.rf_experiments -- synthetic frames only,
no real data files touched."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import models, rf_experiments as rfe


# --------------------------------------------------------------------------- #
# Experiment grid
# --------------------------------------------------------------------------- #
def test_grid_has_36_unique_well_formed_ids():
    grid = rfe.experiment_grid()
    assert len(grid) == 36
    ids = [e.id for e in grid]
    assert len(set(ids)) == 36
    pat = re.compile(r"^base-(airs|smap|both)_fronts-(none|met|pred)"
                     r"_smidx-[01]_pbl-[01]$")
    assert all(pat.match(i) for i in ids)


def test_grid_never_mixes_met_and_pred_fronts():
    for exp in rfe.experiment_grid():
        blocks = exp.blocks
        assert not ("FRONTS_MET" in blocks and "FRONTS_PRED" in blocks)
        # and the features reflect that structurally
        feats = exp.features
        has_met = any(f.startswith("met_front_") for f in feats)
        has_pred = any(f.startswith("pred_front_") for f in feats)
        assert not (has_met and has_pred)


def test_grid_features_match_axis_levels():
    for exp in rfe.experiment_grid():
        feats = set(exp.features)
        assert (set(rfe.AIRS_STEMS) <= feats) == (exp.base in ("airs", "both"))
        assert (set(rfe.SMAP_STEMS) <= feats) == (exp.base in ("smap", "both"))
        assert (set(rfe.SM_IDX_STEMS) <= feats) == bool(exp.smidx)
        assert (set(rfe.PBL_STEMS) <= feats) == bool(exp.pbl)
        assert (set(rfe.FRONTS_MET_STEMS) <= feats) == (exp.fronts == "met")
        assert (set(rfe.FRONTS_PRED_STEMS) <= feats) == (exp.fronts == "pred")
        # target-adjacent MRMS stems must never be features
        assert not {"qpe_max", "qpe_sk", "qpe"} & feats


# --------------------------------------------------------------------------- #
# Slot pivot + merge alignment
# --------------------------------------------------------------------------- #
def _synthetic_flag_dataset():
    """(date=2, time=0..6, lat=2, lon=1) flag cube, all zeros except a single
    known ON flag at (2019-06-02, slot 3, lat 31.5, lon -100.5)."""
    dates = np.array(["2019-06-01", "2019-06-02"], dtype="datetime64[ns]")
    lats, lons = np.array([30.5, 31.5]), np.array([-100.5])
    vals = np.zeros((2, 7, 2, 1), dtype=np.float32)
    vals[1, 3, 1, 0] = 1.0
    return xr.Dataset(
        {"front_cold_3w": (("date", "time", "lat", "lon"), vals)},
        coords={"date": dates, "time": np.arange(7), "lat": lats, "lon": lons})


def test_slot_wide_lands_flag_in_right_h_column():
    wide = rfe.slot_wide(_synthetic_flag_dataset(),
                         rename={"front_cold_3w": "met_front_cold_3w"})
    # 4 cell-days, slots 1..6 widened, met_ prefix applied
    assert len(wide) == 4
    assert {f"met_front_cold_3w_h{s}" for s in range(1, 7)} <= set(wide.columns)
    assert "met_front_cold_3w_h0" not in wide.columns  # overpass slot excluded
    row = wide[(wide["day"] == np.datetime64("2019-06-02"))
               & (wide["lat"] == 31.5)]
    assert float(row["met_front_cold_3w_h3"].iloc[0]) == 1.0
    assert float(row["met_front_cold_3w_h2"].iloc[0]) == 0.0
    # every other cell-day is all-zero
    others = wide.drop(row.index)
    hcols = [c for c in wide.columns if c.startswith("met_front")]
    assert (others[hcols].to_numpy() == 0).all()


def test_merge_on_cell_day_aligns_across_dtypes():
    """float32 lat/lon + datetime64[s] day on the left must still match the
    float64/ns enrichment keys (the real base table is float32)."""
    wide = rfe.slot_wide(_synthetic_flag_dataset(),
                         rename={"front_cold_3w": "met_front_cold_3w"})
    cell_days = pd.DataFrame({
        "day": np.array(["2019-06-02", "2019-06-01"], dtype="datetime64[s]"),
        "lat": np.array([31.5, 30.5], dtype=np.float32),
        "lon": np.array([-100.5, -100.5], dtype=np.float32),
        "qpe_h1": [0.3, 0.1],
    })
    merged = rfe.merge_on_cell_day(cell_days, wide)
    assert len(merged) == 2  # left join never adds/drops cell-days
    assert float(merged.loc[0, "met_front_cold_3w_h3"]) == 1.0
    assert float(merged.loc[1, "met_front_cold_3w_h3"]) == 0.0


def test_merge_on_cell_day_refuses_column_collision():
    wide = rfe.slot_wide(_synthetic_flag_dataset())  # no met_ rename
    cell_days = pd.DataFrame({
        "day": np.array(["2019-06-01"], dtype="datetime64[ns]"),
        "lat": [30.5], "lon": [-100.5],
        "front_cold_3w_h3": [0.0],  # base-table WPC column of the same name
    })
    with pytest.raises(ValueError, match="overwrite"):
        rfe.merge_on_cell_day(cell_days, wide)


# --------------------------------------------------------------------------- #
# Missing-stem assertion
# --------------------------------------------------------------------------- #
def test_missing_stem_assertion_names_the_absentees():
    df = pd.DataFrame({"day": [], "lat": [], "lon": [],
                       "mu_cape_h1": [], "sm_anom": []})
    assert rfe.missing_stems(df, ("mu_cape", "sm_anom")) == []
    missing = rfe.missing_stems(df, ("mu_cape", "UPW_pblh", "fcst_q"))
    assert missing == ["UPW_pblh", "fcst_q"]
    with pytest.raises(AssertionError) as err:
        rfe.assert_stems_present(df, ("mu_cape", "UPW_pblh", "fcst_q"))
    assert "UPW_pblh" in str(err.value) and "fcst_q" in str(err.value)


# --------------------------------------------------------------------------- #
# Idempotent skip / subset batching
# --------------------------------------------------------------------------- #
def test_experiments_to_run_skips_existing_unless_forced():
    grid = rfe.experiment_grid()
    done = {grid[0].id, grid[5].id}
    todo = rfe.experiments_to_run(grid, done)
    assert len(todo) == 34
    assert not {e.id for e in todo} & done
    assert len(rfe.experiments_to_run(grid, done, force=True)) == 36


def test_experiments_to_run_subset_regex():
    grid = rfe.experiment_grid()
    todo = rfe.experiments_to_run(grid, set(), subset=r"base-both_fronts-met")
    assert len(todo) == 4
    assert all(e.base == "both" and e.fronts == "met" for e in todo)
    # subset composes with the idempotent skip
    todo2 = rfe.experiments_to_run(grid, {todo[0].id},
                                   subset=r"base-both_fronts-met")
    assert len(todo2) == 3


def _seed_out_dir(tmp_path, years, months=(3, 11), n_rows=2,
                  write_facts=True, write_months=True):
    """A fake resumable out-dir: results.csv rows + matching per-experiment
    CSVs + (optionally) the sample_info.json recording their window."""
    grid = rfe.experiment_grid()
    (tmp_path / "importances").mkdir()
    (tmp_path / "tail_cdf").mkdir()
    rows = []
    for exp in grid[:n_rows]:
        rows.append({"id": exp.id, "r2_test": 0.5})
        (tmp_path / "importances" / f"{exp.id}.csv").write_text("f,i\na,1\n")
        (tmp_path / "tail_cdf" / f"{exp.id}_p99_5.csv").write_text(
            "bin,cum_freq\n")
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    if write_facts:
        facts = {"years": list(years), "n_samples": 10}
        if write_months:
            facts["months"] = list(months)
        import json
        (tmp_path / "sample_info.json").write_text(json.dumps(facts))


def test_window_guard_resumes_when_window_matches(tmp_path):
    _seed_out_dir(tmp_path, (2019,), months=(3, 11))
    results = rfe.check_window_consistency(tmp_path, (2019,), (3, 11))
    assert len(results) == 2  # retained rows come back for the skip logic


def test_window_guard_raises_on_years_mismatch(tmp_path):
    _seed_out_dir(tmp_path, (2019,))
    with pytest.raises(RuntimeError, match="2019"):
        rfe.check_window_consistency(tmp_path, (2018, 2019), (3, 11))


def test_window_guard_raises_on_months_mismatch(tmp_path):
    # same years, DIFFERENT month window: exactly as fatal as a years
    # mismatch -- the common sample would differ
    _seed_out_dir(tmp_path, (2019,), months=(3, 11))
    with pytest.raises(RuntimeError, match="months"):
        rfe.check_window_consistency(tmp_path, (2019,), (4, 10))


def test_window_guard_raises_on_pre_months_era_record(tmp_path):
    # sample_info.json from before the months field existed: window unknown
    _seed_out_dir(tmp_path, (2019,), write_months=False)
    with pytest.raises(RuntimeError, match="months"):
        rfe.check_window_consistency(tmp_path, (2019,), (3, 11))


def test_window_guard_raises_when_provenance_unknown(tmp_path):
    # results.csv rows with NO sample_info.json (e.g. interrupted old run)
    _seed_out_dir(tmp_path, (2019,), write_facts=False)
    with pytest.raises(RuntimeError, match="sample_info.json"):
        rfe.check_window_consistency(tmp_path, (2019,), (3, 11))


def test_window_guard_force_discards_all_stale_artifacts(tmp_path):
    _seed_out_dir(tmp_path, (2019,))
    results = rfe.check_window_consistency(tmp_path, (2020,), (3, 11),
                                           force=True)
    assert results.empty
    assert not (tmp_path / "results.csv").exists()
    assert not (tmp_path / "sample_info.json").exists()
    # per-experiment CSVs are gone too: figures read them by filename
    assert not list((tmp_path / "importances").glob("*.csv"))
    assert not list((tmp_path / "tail_cdf").glob("*.csv"))


def test_window_guard_force_discards_on_months_mismatch(tmp_path):
    _seed_out_dir(tmp_path, (2019,), months=(1, 12))
    results = rfe.check_window_consistency(tmp_path, (2019,), (3, 11),
                                           force=True)
    assert results.empty
    assert not (tmp_path / "results.csv").exists()


def test_window_guard_passes_empty_out_dir(tmp_path):
    assert rfe.check_window_consistency(tmp_path, (2019,), (3, 11)).empty


# --------------------------------------------------------------------------- #
# Month-window filtering
# --------------------------------------------------------------------------- #
def test_parse_months():
    assert rfe.parse_months("3-11") == (3, 11)
    assert rfe.parse_months("6") == (6, 6)
    with pytest.raises(ValueError):
        rfe.parse_months("0-11")
    with pytest.raises(ValueError):
        rfe.parse_months("11-3")


def test_filter_month_window_keeps_mar_to_nov():
    # one cell-day in each month of 2019: Mar-Nov survive, Dec-Feb drop
    days = pd.to_datetime([f"2019-{m:02d}-15" for m in range(1, 13)])
    df = pd.DataFrame({"day": days, "lat": 30.5, "lon": -100.5,
                       "qpe_h1": np.arange(12.0)})
    out = rfe.filter_month_window(df, (3, 11))
    assert len(out) == 9
    kept = pd.to_datetime(out["day"]).dt.month
    assert kept.min() == 3 and kept.max() == 11
    assert not set(kept) & {12, 1, 2}
    # inclusive at both edges, and a narrower window works too
    assert set(pd.to_datetime(
        rfe.filter_month_window(df, (6, 6))["day"]).dt.month) == {6}


def test_upsert_result_replaces_and_keeps_grid_order():
    grid = rfe.experiment_grid()
    results = pd.DataFrame(columns=list(rfe.RESULT_COLUMNS))
    r1 = {"id": grid[3].id, "r2_test": 0.1}
    r0 = {"id": grid[0].id, "r2_test": 0.2}
    results = rfe.upsert_result(results, r1)
    results = rfe.upsert_result(results, r0)
    assert list(results["id"]) == [grid[0].id, grid[3].id]  # grid order
    results = rfe.upsert_result(results, {"id": grid[3].id, "r2_test": 0.9})
    assert len(results) == 2  # replaced, not duplicated
    assert float(results.set_index("id").loc[grid[3].id, "r2_test"]) == 0.9


# --------------------------------------------------------------------------- #
# tail_top10_capture indexing (against models.make_binned_freqs verbatim)
# --------------------------------------------------------------------------- #
def _capture_for_event_positions(positions):
    """5 tail events placed at chosen positions of a perfectly-ordered
    pred = arange(1000); returns the top-decile capture."""
    pred = np.arange(1000, dtype=float)
    obs = np.zeros(1000)
    obs[list(positions)] = 100.0
    cdf = models.make_binned_freqs(pred, obs, obs_pct=99.5)
    assert len(cdf) == 100
    return rfe.tail_top10_capture(cdf)


def test_tail_top10_capture_indexing():
    # all 5 events at the very top predictions -> fully captured
    assert _capture_for_event_positions([995, 996, 997, 998, 999]) == \
        pytest.approx(1.0)
    # all 5 events at the very bottom predictions -> zero capture
    assert _capture_for_event_positions([0, 1, 2, 3, 4]) == pytest.approx(0.0)
    # 3 of 5 in the top decile (>= P90 = pred 899.1) -> capture 0.6
    assert _capture_for_event_positions([0, 1, 950, 960, 970]) == \
        pytest.approx(0.6)


def test_tail_top10_capture_rejects_wrong_bin_count():
    with pytest.raises(ValueError, match="100-bin"):
        rfe.tail_top10_capture(np.linspace(0, 1, 50))


# --------------------------------------------------------------------------- #
# End-to-end on a synthetic cell-day frame (no files): the common-sample +
# run_experiment plumbing produces finite metrics of the documented shape.
# --------------------------------------------------------------------------- #
def _synthetic_cell_days(n=240, seed=0):
    rng = np.random.default_rng(seed)
    days = np.repeat(
        pd.date_range("2019-06-01", periods=n // 4).to_numpy(), 4)[:n]
    df = pd.DataFrame({
        "day": days,
        "lat": np.tile([30.5, 31.5], n // 2),
        "lon": np.tile([-100.5, -99.5], n // 2),
    })
    for stem in rfe.ALL_STEMS + ("qpe",):
        if (stem in rfe.AIRS_STEMS or stem == "qpe"
                or stem.startswith(("met_front", "pred_front", "UPW_"))):
            for h in range(1, 7):
                df[f"{stem}_h{h}"] = rng.normal(size=n)
        else:
            df[stem] = rng.normal(size=n)
    return df


def test_run_experiment_shapes_on_synthetic_frame():
    df = _synthetic_cell_days()
    df.loc[0, "UPW_pblh_h2"] = np.nan  # one NaN row must fall out of the
    ds = rfe.build_common_sample(df)   # ONE common sample
    assert ds.sizes["sample"] == len(df) - 1
    exp = next(e for e in rfe.experiment_grid()
               if e.id == "base-airs_fronts-none_smidx-0_pbl-0")
    m = rfe.run_experiment(ds, exp)
    assert np.isfinite(m["r2_test"])
    assert sorted(m["r2_per_hour"]) == [f"r2_h{h}" for h in range(1, 7)]
    # one 100-bin capture curve per headline set point
    assert set(m["tail_cdfs"]) == {"p95", "p99_5"}
    assert all(len(c) == 100 for c in m["tail_cdfs"].values())
    assert 0.0 <= m["tail_top10_capture"] <= 1.0
    # importances named by expanded hourly features: 9 AIRS stems x 6 hours
    assert len(m["importances"]) == 54
    assert "mu_cape_h1" in m["importances"].index


def test_fit_paths_write_per_threshold_curve_files(tmp_path):
    """Every fit path must yield curves for BOTH set points, and
    write_tail_curves must persist them under the suffixed names the figures
    read (never the legacy single-file name)."""
    df = _synthetic_cell_days(n=160, seed=11)
    ds = rfe.build_common_sample(df)
    exp = next(e for e in rfe.experiment_grid()
               if e.id == "base-airs_fronts-none_smidx-0_pbl-0")
    thr = {"p95": 1.0, "p99_5": 2.0}
    for fit in (rfe.run_experiment, rfe.run_experiment_pooled,
                rfe.run_experiment_perhour):
        m = fit(ds, exp, thresholds=thr)
        assert set(m["tail_cdfs"]) == set(rfe.SET_POINTS)
        assert all(len(c) == 100 for c in m["tail_cdfs"].values())
    rfe.write_tail_curves(tmp_path, exp.id, m["tail_cdfs"])
    for label in rfe.SET_POINTS:
        path = tmp_path / "tail_cdf" / f"{exp.id}_{label}.csv"
        assert path.exists()
        curve = pd.read_csv(path)
        assert list(curve.columns) == ["bin", "cum_freq"]
        assert len(curve) == 100
    assert not (tmp_path / "tail_cdf" / f"{exp.id}.csv").exists()  # legacy


# --------------------------------------------------------------------------- #
# Hour-matched modes (pooled / perhour)
# --------------------------------------------------------------------------- #
def test_cell_day_split_reproduces_verbatim_partition():
    """cell_day_split must yield EXACTLY the row partition Mark's verbatim
    train_random_forest draws internally (same n, same seed) -- that identity
    is what makes the three modes' test sets one and the same."""
    df = _synthetic_cell_days(n=200, seed=3)
    ds = rfe.build_common_sample(df)
    rfr = models.train_random_forest(
        ds, ["mu_cape"], "qpe", hourly=False, seed=rfe.SEED,
        train_fraction=rfe.TRAIN_FRACTION,
        rfr_kwargs={"n_estimators": 2, "max_depth": 2})
    train_idx, test_idx = rfe.cell_day_split(ds.sizes["sample"])
    x_full = ds["mu_cape"].values.T  # (sample, time), his hourly=False layout
    np.testing.assert_array_equal(rfr["X_test"], x_full[test_idx])
    np.testing.assert_array_equal(rfr["X_train"], x_full[train_idx])


def test_hour_matched_matrices_layout():
    """Row (s, h) of the pooled X must hold sample s's hour-h value for
    hourly stems and sample s's single value for daily stems, sample-major."""
    df = _synthetic_cell_days(n=40, seed=4)
    ds = rfe.build_common_sample(df)
    keys = ["mu_cape", "sm_anom"]  # one hourly, one daily
    X, y, sample_of_row, hour_of_row = rfe.hour_matched_matrices(ds, keys)
    n, n_hours = ds.sizes["sample"], ds.sizes["time"]
    assert X.shape == (n * n_hours, 2) and y.shape == (n * n_hours,)
    hours = [int(t) for t in ds["time"].values]
    for s in (0, 7, n - 1):
        for i, h in enumerate(hours):
            row = s * n_hours + i
            assert sample_of_row[row] == s and hour_of_row[row] == h
            assert X[row, 0] == ds["mu_cape"].values[i, s]      # hour-matched
            assert X[row, 1] == ds["sm_anom"].values[s]         # repeated
            assert y[row] == ds["qpe"].values[i, s]


def test_pooled_groups_hours_by_cell_day():
    """No cell-day may straddle the pooled split, and metrics must have the
    common shapes (one importance per STEM, not per stem-hour)."""
    df = _synthetic_cell_days(n=160, seed=5)
    ds = rfe.build_common_sample(df)
    exp = next(e for e in rfe.experiment_grid()
               if e.id == "base-airs_fronts-none_smidx-1_pbl-0")
    m = rfe.run_experiment_pooled(ds, exp)
    assert np.isfinite(m["r2_test"])
    assert sorted(m["r2_per_hour"]) == [f"r2_h{h}" for h in range(1, 7)]
    assert all(len(c) == 100 for c in m["tail_cdfs"].values())
    # 9 AIRS stems + 13 SM_IDX stems, ONE importance each
    assert len(m["importances"]) == len(rfe.AIRS_STEMS) + len(rfe.SM_IDX_STEMS)
    assert "mu_cape" in m["importances"].index
    assert "mu_cape_h1" not in m["importances"].index


def test_perhour_uses_shared_split_and_stacks_importances():
    """Each hour's verbatim fit must test on the SAME cell-days as
    cell_day_split, and importances come back as a (stem x hour) frame."""
    df = _synthetic_cell_days(n=160, seed=6)
    ds = rfe.build_common_sample(df)
    exp = next(e for e in rfe.experiment_grid()
               if e.id == "base-airs_fronts-none_smidx-0_pbl-0")
    m = rfe.run_experiment_perhour(ds, exp)
    assert sorted(m["r2_per_hour"]) == [f"r2_h{h}" for h in range(1, 7)]
    imp = m["importances"]
    assert list(imp.columns) == [f"h{h}" for h in range(1, 7)]
    assert set(imp.index) == set(rfe.AIRS_STEMS)
    # the shared-split identity, checked through the verbatim path itself
    _, test_idx = rfe.cell_day_split(ds.sizes["sample"])
    rfr_h1 = models.train_random_forest(
        ds.isel(time=[0]), exp.features, "qpe", hourly=False, seed=rfe.SEED,
        train_fraction=rfe.TRAIN_FRACTION,
        rfr_kwargs={"n_estimators": 2, "max_depth": 2})
    np.testing.assert_array_equal(rfr_h1["Y_test"],
                                  ds["qpe"].values[0, test_idx])


def test_modes_registry_and_out_dirs():
    assert set(rfe.MODES) == {"wide", "pooled", "perhour"}
    assert rfe.MODES["wide"][1] == ""            # original runs stay in place
    assert rfe.MODES["pooled"][1] == "pooled_hourly"
    assert rfe.MODES["perhour"][1] == "per_hour"


# --------------------------------------------------------------------------- #
# Curated figure selection (worth_showing) + importance trim rule
# --------------------------------------------------------------------------- #
def _full_results_frame(gini=0.5):
    """All 36 experiments with constant metrics (overridable per-id after)."""
    rows = []
    for e in rfe.experiment_grid():
        rows.append({"id": e.id, "base": e.base, "fronts": e.fronts,
                     "smidx": e.smidx, "pbl": e.pbl,
                     "gini_p95": gini - 0.1, "gini_p99_5": gini,
                     "r2_test": 0.3})
    return pd.DataFrame(rows)


def test_worth_showing_curated_core():
    results = _full_results_frame()
    picked = rfe.worth_showing(results)
    # per base: bare baseline + its +pbl variant
    for base in ("airs", "smap", "both"):
        assert f"base-{base}_fronts-none_smidx-0_pbl-0" in picked
        assert f"base-{base}_fronts-none_smidx-0_pbl-1" in picked
    # fronts contrast at both+pbl and the everything-on run
    for fronts in ("none", "met", "pred"):
        assert f"base-both_fronts-{fronts}_smidx-0_pbl-1" in picked
    assert "base-both_fronts-met_smidx-1_pbl-1" in picked
    assert len(picked) == len(set(picked))  # no duplicate ids


def test_worth_showing_near_best_catch_all():
    """An uncurated experiment within NEAR_BEST_MARGIN of the global best
    headline Gini must be shown -- curation can never hide a surprise winner."""
    results = _full_results_frame(gini=0.5)
    surprise = "base-smap_fronts-pred_smidx-1_pbl-0"   # not in the curated core
    near = "base-airs_fronts-pred_smidx-1_pbl-0"       # within the margin
    far = "base-smap_fronts-met_smidx-1_pbl-0"         # just outside it
    idx = results.set_index("id")
    idx.loc[surprise, "gini_p99_5"] = 0.700            # the global best
    idx.loc[near, "gini_p99_5"] = 0.700 - rfe.NEAR_BEST_MARGIN + 1e-9
    idx.loc[far, "gini_p99_5"] = 0.700 - rfe.NEAR_BEST_MARGIN - 1e-6
    picked = rfe.worth_showing(idx.reset_index())
    assert surprise in picked
    assert near in picked
    assert far not in picked
    # only scored ids ever come back (partial runs stay valid)
    partial = idx.reset_index().iloc[:5]
    assert set(rfe.worth_showing(partial)) <= set(partial["id"])


def test_trim_importances_coverage_and_cap():
    # 90% coverage reached after 3 features -> exactly 3 kept
    imp = pd.Series([0.5, 0.3, 0.15, 0.04, 0.005, 0.005],
                    index=list("abcdef"), name="importance")
    kept = rfe.trim_importances(imp)
    assert list(kept.index) == ["a", "b", "c"]
    # 20 equal features would need 18 for 90% -> the cap (12) wins
    flat = pd.Series(np.full(20, 0.05), index=[f"f{i}" for i in range(20)])
    assert len(rfe.trim_importances(flat)) == rfe.IMPORTANCE_CAP
    # always at least one feature, even for degenerate all-zero importances
    assert len(rfe.trim_importances(pd.Series([1.0], index=["only"]))) == 1
    zeros = pd.Series([0.0, 0.0], index=["x", "y"])
    assert 1 <= len(rfe.trim_importances(zeros)) <= 2
    # unsorted input is ranked before trimming
    shuffled = imp.sample(frac=1.0, random_state=0)
    assert list(rfe.trim_importances(shuffled).index) == ["a", "b", "c"]
