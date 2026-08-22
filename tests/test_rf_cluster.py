"""Tests for the cluster RF pipeline: compile-script interpolators, the
block-experiment set, Gini wiring, and a synthetic end-to-end load."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convection_skill import config, models, rf_cluster as rfc
from convection_skill import rf_experiments as rfe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compile_rf_dataset as crd  # noqa: E402


# --------------------------------------------------------------------------- #
# Experiment set
# --------------------------------------------------------------------------- #
def test_experiment_set_shape_and_front_exclusivity():
    exps = rfc.experiment_set()
    assert len(exps) == 13
    assert len({e.id for e in exps}) == 13
    for e in exps:
        assert not {"FRONTS_MET", "FRONTS_PRED"} <= set(e.blocks)
    full = next(e for e in exps if e.id == "full-met")
    swap = next(e for e in exps if e.id == "full-pred")
    assert set(full.blocks) ^ set(swap.blocks) == {"FRONTS_MET", "FRONTS_PRED"}
    # thermo block is MU-only: no MML stem anywhere
    assert not any("mml" in s for e in exps for s in e.features)
    drop_pbl = next(e for e in exps if e.id == "drop-pbl")
    assert "UPW_pblh" not in drop_pbl.features
    assert "mu_cape" in drop_pbl.features


# --------------------------------------------------------------------------- #
# Gini wiring (rf_experiments.gini_metrics)
# --------------------------------------------------------------------------- #
def test_gini_metrics_orders_and_labels():
    rng = np.random.default_rng(0)
    pred = rng.random(20000)
    obs = pred + rng.normal(scale=0.05, size=pred.size)  # informative score
    thr = {"p95": float(np.percentile(obs, 95)),
           "p99_5": float(np.percentile(obs, 99.5))}
    m = rfe.gini_metrics(obs, pred, thr)
    assert set(m) == {"gini_p95", "gini_p99_5"}
    assert m["gini_p95"] > 0.5           # strongly informative
    shuffled = rfe.gini_metrics(obs, rng.permutation(pred), thr)
    assert abs(shuffled["gini_p95"]) < 0.1
    assert rfe.gini_metrics(obs, pred, None) == {}


def test_run_experiment_reports_gini_when_thresholds_given():
    # reuse the synthetic frame from the rf_experiments tests
    from tests.test_rf_experiments import _synthetic_cell_days
    df = _synthetic_cell_days(n=160, seed=9)
    ds = rfe.build_common_sample(df)
    exp = next(e for e in rfe.experiment_grid()
               if e.id == "base-airs_fronts-none_smidx-0_pbl-0")
    thr = {"p95": 1.0, "p99_5": 2.0}
    for fit in (rfe.run_experiment, rfe.run_experiment_pooled,
                rfe.run_experiment_perhour):
        m = fit(ds, exp, thresholds=thr)
        assert np.isfinite(m["gini_p95"]) and np.isfinite(m["gini_p99_5"])


# --------------------------------------------------------------------------- #
# Compile-script interpolators
# --------------------------------------------------------------------------- #
@pytest.fixture()
def pblh_files(tmp_path):
    """A tiny assessed 3-hourly file (linear-in-time field so interpolation is
    exact) and a matching monthly-diurnal climatology."""
    lats, lons = np.array([30.5, 31.5]), np.array([-100.5, -99.5])
    times = pd.date_range("2019-06-01", "2019-06-03", freq="3h")
    hours_since = ((times - times[0]) / pd.Timedelta("1h")).to_numpy()
    # pblh = 1000 + 10 * hours_since_start + 100 * cell_index  (linear in t)
    cell = np.arange(4).reshape(2, 2)
    vals = (1000.0 + 10.0 * hours_since[:, None, None]
            + 100.0 * cell[None, :, :])
    three = tmp_path / "pblh_3hrly.nc"
    xr.Dataset({"pblh": (("time", "lat", "lon"), vals)},
               coords={"time": times, "lat": lats, "lon": lons}
               ).to_netcdf(three)
    clim_hours = np.arange(0, 24, 3)
    cvals = np.zeros((12, 8, 2, 2)) + clim_hours[None, :, None, None] * 10.0
    clim = tmp_path / "pblh_clim.nc"
    xr.Dataset({"pblh_mean": (("month", "hour", "lat", "lon"), cvals)},
               coords={"month": np.arange(1, 13), "hour": clim_hours,
                       "lat": lats, "lon": lons}).to_netcdf(clim)
    return three, clim, lats, lons


def test_interp_pblh_is_exact_on_linear_field(pblh_files):
    three, _, lats, lons = pblh_files
    # one sample per cell at 22:30 UTC on June 1 = 22.5 h after the axis start
    times = np.full((1, 1, 2, 2), np.datetime64("2019-06-01T22:30"), "M8[ns]")
    out = crd.interp_pblh_to_times(times, lats, lons, three)
    expect = 1000.0 + 10.0 * 22.5 + 100.0 * np.arange(4).reshape(2, 2)
    np.testing.assert_allclose(out[0, 0], expect, rtol=1e-6)


def test_interp_pblh_refuses_wide_gaps_and_nat(pblh_files):
    three, _, lats, lons = pblh_files
    times = np.array([np.datetime64("2019-06-10T00:00"),   # beyond the axis
                      np.datetime64("NaT")],
                     dtype="M8[ns]").reshape(2, 1, 1, 1)
    times = np.broadcast_to(times, (2, 1, 2, 2)).copy()
    out = crd.interp_pblh_to_times(times, lats, lons, three)
    assert np.isnan(out).all()


def test_interp_clim_diurnal_wrap(pblh_files):
    _, clim, lats, lons = pblh_files
    # clim = 10 * slot_hour; at 22:30 UTC linear between h21 (210) and h0 (0)
    times = np.full((1, 1, 2, 2), np.datetime64("2019-06-01T22:30"), "M8[ns]")
    out = crd.interp_clim_to_times(times, lats, lons, clim)
    np.testing.assert_allclose(out[0, 0], 210.0 * (1 - 0.5) + 0.0 * 0.5)


# --------------------------------------------------------------------------- #
# Synthetic end-to-end load_cell_days
# --------------------------------------------------------------------------- #
def _synthetic_compiled(tmp_path, n_days=30):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2019-06-01", periods=n_days)
    lats, lons = np.array([30.5, 31.5]), np.array([-100.5, -99.5])
    shp4 = (n_days, 7, 2, 2)
    ds = xr.Dataset(coords={"date": dates, "time": np.arange(7),
                            "lat": lats, "lon": lons})
    for src in rfc.SLOT_RENAME:
        ds[src] = (("date", "time", "lat", "lon"),
                   rng.normal(1000, 100, shp4))
    for stem in ("UPW_psi_anom", "UPW_omega", "UPW_pblh", "UPW_pblh_anom",
                 "UPW_gamma_gap_mu"):
        ds[stem] = (("date", "time", "lat", "lon"), rng.normal(size=shp4))
    for p in ("met", "pred"):
        for t in rfc.FRONT_TYPES:
            ds[f"{p}_front_{t}_3w"] = (("date", "time", "lat", "lon"),
                                       rng.integers(0, 2, shp4).astype(float))
    # MRMS on the nhours axis, like the real files
    ds[config.QPE_VAR] = (("date", "nhours", "lat", "lon"),
                          rng.exponential(1.0, shp4))
    ds[config.QPE_CNT_VAR] = (("date", "nhours", "lat", "lon"),
                              np.full(shp4, 81.0))
    ds["FCST_alt"] = (("date", "time", "lat", "lon"),
                      np.full(shp4, 200.0))          # passes the 1000 m screen
    # dry start: zero QPE in hours 0-1 so the screen keeps everything
    ds[config.QPE_VAR][:, :2] = 0.0
    ds["sm_anom"] = (("date", "lat", "lon"), rng.normal(size=(n_days, 2, 2)))
    path = tmp_path / "compiled.nc"
    ds.to_netcdf(path)
    return path


def test_load_cell_days_end_to_end(tmp_path, monkeypatch):
    path = _synthetic_compiled(tmp_path)
    monkeypatch.setattr(rfc.dl, "load_land_fraction_grid",
                        lambda lats, lons: np.ones((lats.size, lons.size)))
    cell_days, thresholds = rfc.load_cell_days(path, years=(2019,),
                                               months=(3, 11))
    # every stem of every experiment resolves as an _h family or daily column
    all_stems = tuple(sorted({s for e in rfc.experiment_set()
                              for s in e.features}))
    assert rfe.missing_stems(cell_days, all_stems + ("qpe",)) == []
    # slot labels are h1..h6, never positional 0..5
    assert "mu_cape_h1" in cell_days and "mu_cape_h0" not in cell_days
    assert thresholds["p95"] < thresholds["p99_5"]
    # thresholds come from the exponential target, pre-screen
    assert thresholds["p99_5"] > 1.0
    # a known value survives the pivot in the right place
    with xr.open_dataset(path) as ds:
        want = float(ds["FCST_MU_CAPE"][3, 4, 0, 1])
        day = np.datetime64(ds["date"].values[3], "D")
    got = cell_days.loc[(cell_days["day"] == day)
                        & (cell_days["lat"] == 30.5)
                        & (cell_days["lon"] == -99.5), "mu_cape_h4"]
    assert float(got.iloc[0]) == pytest.approx(want)


def test_load_cell_days_names_missing_variables(tmp_path, monkeypatch):
    path = _synthetic_compiled(tmp_path)
    with xr.open_dataset(path) as ds:
        broken = ds.drop_vars("UPW_omega").load()
    broken_path = tmp_path / "broken.nc"
    broken.to_netcdf(broken_path)
    monkeypatch.setattr(rfc.dl, "load_land_fraction_grid",
                        lambda lats, lons: np.ones((lats.size, lons.size)))
    with pytest.raises(KeyError, match="UPW_omega"):
        rfc.load_cell_days(broken_path, years=(2019,), months=(3, 11))


# --------------------------------------------------------------------------- #
# Training-window filtering at experiment time (compiled file stays a superset)
# --------------------------------------------------------------------------- #
def _synthetic_compiled_yearlong(tmp_path):
    """A compiled file spanning Dec 2018 - Jan 2020 (every month of 2019
    present), so both the year and the month cut can be observed."""
    rng = np.random.default_rng(2)
    dates = pd.date_range("2018-12-15", "2020-01-15", freq="15D")
    lats, lons = np.array([30.5, 31.5]), np.array([-100.5, -99.5])
    shp4 = (dates.size, 7, 2, 2)
    ds = xr.Dataset(coords={"date": dates, "time": np.arange(7),
                            "lat": lats, "lon": lons})
    for src in rfc.SLOT_RENAME:
        ds[src] = (("date", "time", "lat", "lon"),
                   rng.normal(1000, 100, shp4))
    for stem in ("UPW_psi_anom", "UPW_omega", "UPW_pblh", "UPW_pblh_anom",
                 "UPW_gamma_gap_mu"):
        ds[stem] = (("date", "time", "lat", "lon"), rng.normal(size=shp4))
    for p in ("met", "pred"):
        for t in rfc.FRONT_TYPES:
            ds[f"{p}_front_{t}_3w"] = (("date", "time", "lat", "lon"),
                                       rng.integers(0, 2, shp4).astype(float))
    ds[config.QPE_VAR] = (("date", "nhours", "lat", "lon"),
                          rng.exponential(1.0, shp4))
    ds[config.QPE_CNT_VAR] = (("date", "nhours", "lat", "lon"),
                              np.full(shp4, 81.0))
    ds["FCST_alt"] = (("date", "time", "lat", "lon"), np.full(shp4, 200.0))
    ds[config.QPE_VAR][:, :2] = 0.0
    ds["sm_anom"] = (("date", "lat", "lon"),
                     rng.normal(size=(dates.size, 2, 2)))
    path = tmp_path / "compiled_yearlong.nc"
    ds.to_netcdf(path)
    return path


def test_load_cell_days_applies_training_window(tmp_path, monkeypatch):
    path = _synthetic_compiled_yearlong(tmp_path)
    monkeypatch.setattr(rfc.dl, "load_land_fraction_grid",
                        lambda lats, lons: np.ones((lats.size, lons.size)))
    cell_days, thresholds = rfc.load_cell_days(path, years=(2019,),
                                               months=(3, 11))
    days = pd.to_datetime(cell_days["day"])
    assert (days.dt.year == 2019).all()               # 2018/2020 cut
    assert days.dt.month.between(3, 11).all()          # Dec-Feb cut
    assert days.dt.month.min() == 3 and days.dt.month.max() == 11
    # thresholds come from the WINDOWED rows: a different window on the same
    # file gives (generically) different absolute set points
    _, thr_all = rfc.load_cell_days(path, years=(2018, 2019, 2020),
                                    months=(1, 12))
    assert thresholds["p99_5"] != pytest.approx(thr_all["p99_5"], rel=1e-6)
    # an empty window is a loud error, never an empty silent sample
    with pytest.raises(ValueError, match="no dates"):
        rfc.load_cell_days(path, years=(2016,), months=(3, 11))
