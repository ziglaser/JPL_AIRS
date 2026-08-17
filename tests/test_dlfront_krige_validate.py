"""dl_front.krige_validate: metric math, stratification, bounded real run.

The metric functions are pure numpy, so they are checked against arrays
whose RMSE / bias / gradient ratio can be computed by hand.  Stratification
uses a fully synthetic (domain, observed, gap_type) case with only the
baseline methods (no kriging solve).  One bounded end-to-end test runs the
real CLI over 2 sampled 2007 days (reanalysis on disk, gap bank draws with
--allow-small-bank) into a tmp dir; config.KRIGE_MAX_OBS is shrunk via
monkeypatch because kriging solves are O(n^3) in the obs count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_front import config, krige_validate as kv
from dl_front.acquire_merra2_sfc import day_path

needs_2007 = pytest.mark.skipif(
    not day_path(pd.Timestamp("2007-06-01")).exists(),
    reason="2007 sfc_daily reanalysis not on disk")


# --------------------------------------------------------------------------- #
# Metric functions on hand-checkable arrays
# --------------------------------------------------------------------------- #

def test_reconstruction_metrics_hand_values():
    # truth flat zero; fill errors on the two picked pixels are +1 and -3:
    # rmse = sqrt((1+9)/2), mae = 2, bias = -1.
    truth = np.zeros((5, 5))
    filled = truth.copy()
    filled[1, 1], filled[2, 3] = 1.0, -3.0
    pick = np.zeros((5, 5), bool)
    pick[1, 1] = pick[2, 3] = True
    m = kv.reconstruction_metrics(filled, truth, pick)
    assert m["n_pixels"] == 2
    assert m["rmse"] == pytest.approx(np.sqrt(5.0))
    assert m["mae"] == pytest.approx(2.0)
    assert m["bias"] == pytest.approx(-1.0)
    # flat truth has zero gradient -> ratio undefined, reported NaN
    assert np.isnan(m["gradient_ratio"])


def test_gradient_ratio_planes():
    # truth = 2x (|grad| = 2 everywhere), fill = x (|grad| = 1): ratio 0.5,
    # measured away from the fields' shared zero column so bias is nonzero.
    x = np.arange(8.0)[None, :].repeat(6, axis=0)
    truth, filled = 2 * x, x
    pick = np.zeros((6, 8), bool)
    pick[2:4, 3:6] = True
    m = kv.reconstruction_metrics(filled, truth, pick)
    assert m["gradient_ratio"] == pytest.approx(0.5)
    assert m["bias"] == pytest.approx(-x[pick].mean())


def test_reconstruction_metrics_empty_stratum_is_none():
    z = np.zeros((3, 3))
    assert kv.reconstruction_metrics(z, z, np.zeros((3, 3), bool)) is None


def test_gradient_magnitude_nan_isolation():
    # a NaN pixel NaNs its neighbours' central differences, but nanmean
    # over a pick away from it is unaffected
    field = np.arange(25.0).reshape(5, 5)
    field[0, 0] = np.nan
    gm = kv.gradient_magnitude(field)
    assert np.isnan(gm[0, 1]) and np.isnan(gm[1, 0])
    assert gm[3, 3] == pytest.approx(np.hypot(5.0, 1.0))


# --------------------------------------------------------------------------- #
# Distance bins
# --------------------------------------------------------------------------- #

def test_distance_bin_assignment():
    # one observed pixel at column 0 of a 1x10 strip: distance = column index
    observed = np.zeros((1, 10), bool)
    observed[0, 0] = True
    labels = kv.distance_stratum(kv.distance_to_observed(observed))
    assert labels[0, 0] == ""                       # observed: never scored
    assert list(labels[0, 1:3]) == ["1-2deg"] * 2   # d = 1, 2
    assert list(labels[0, 3:6]) == ["3-5deg"] * 3   # d = 3, 4, 5
    assert list(labels[0, 6:]) == [">5deg"] * 4     # d = 6..9


def test_distance_diagonal_neighbor_in_first_bin():
    observed = np.zeros((3, 3), bool)
    observed[0, 0] = True
    labels = kv.distance_stratum(kv.distance_to_observed(observed))
    assert labels[1, 1] == "1-2deg"                 # sqrt(2) ~ 1.41


# --------------------------------------------------------------------------- #
# Stratification (synthetic gap_type, baselines only -- no kriging solve)
# --------------------------------------------------------------------------- #

def _synthetic_case():
    """20x24 case: left half observed, right half held out, split into a
    'cloud' band (rows < 10) and 'out_of_swath' (rows >= 10).

    ``crop`` (the fill extent, box + halo in production) is the whole
    grid; ``domain`` (the SCORED analysis domain) excludes a right-hand
    strip -- filled but never scored, like the halo (user decision
    2026-08-13)."""
    shape = (20, 24)
    la, lo = np.meshgrid(np.arange(shape[0], dtype=float),
                         np.arange(shape[1], dtype=float), indexing="ij")
    truth = 280.0 + 0.5 * la + 0.2 * lo
    crop = np.ones(shape, bool)
    domain = np.ones(shape, bool)
    domain[:, -2:] = False                          # a strip out of domain
    observed = np.zeros(shape, bool)
    observed[:, :12] = True
    observed &= domain
    gap_type = np.full(shape, config.GAP_OUT_OF_SWATH, np.int8)
    gap_type[:10] = config.GAP_CLOUD
    gap_type[observed] = config.GAP_OBSERVED
    return truth, observed, crop, domain, gap_type


def test_evaluate_case_stratification():
    truth, observed, crop, domain, gap_type = _synthetic_case()
    rows = kv.evaluate_case(truth, observed, crop, domain, gap_type,
                            variograms=(), date="2020-01-15", hour=21,
                            channel="T2M")
    df = pd.DataFrame(rows)
    assert set(df.columns) == set(kv.METRIC_COLUMNS)
    assert set(df["method"]) == {"nearest", "mean"}          # no kriging
    target_n = int((domain & ~observed).sum())
    for method, g in df.groupby("method"):
        overall = g[g["stratum_kind"] == "overall"]
        assert overall["n_pixels"].item() == target_n
        # the strata partition the held-out pixels exactly
        assert g[g["stratum_kind"] == "gap_type"]["n_pixels"].sum() == target_n
        assert g[g["stratum_kind"] == "distance"]["n_pixels"].sum() == target_n
    assert set(df[df["stratum_kind"] == "gap_type"]["stratum"]) == \
        {"cloud", "out_of_swath"}


def test_evaluate_case_mean_method_hand_value():
    truth, observed, crop, domain, gap_type = _synthetic_case()
    rows = kv.evaluate_case(truth, observed, crop, domain, gap_type,
                            variograms=(), date="2020-01-15", hour=21,
                            channel="T2M")
    df = pd.DataFrame(rows)
    row = df[(df["method"] == "mean") &
             (df["stratum_kind"] == "overall")].iloc[0]
    target = domain & ~observed
    err = truth[observed].mean() - truth[target]
    assert row["bias"] == pytest.approx(err.mean())
    assert row["rmse"] == pytest.approx(np.sqrt((err ** 2).mean()))
    # a constant fill is flat except at the observed/domain boundary
    # (np.gradient central differences straddle it): strongly smoothed
    assert row["gradient_ratio"] < 0.5


def test_evaluate_case_nearest_copies_boundary_column():
    truth, observed, crop, domain, gap_type = _synthetic_case()
    rows = kv.evaluate_case(truth, observed, crop, domain, gap_type,
                            variograms=(), date="2020-01-15", hour=21,
                            channel="T2M")
    df = pd.DataFrame(rows)
    # truth rises 0.2/column rightwards; nearest copies column 11 across the
    # held-out half, so bias = -0.2 * mean horizontal distance (exact)
    row = df[(df["method"] == "nearest") &
             (df["stratum_kind"] == "overall")].iloc[0]
    target = domain & ~observed
    cols = np.nonzero(target)[1]
    assert row["bias"] == pytest.approx(-0.2 * (cols - 11).mean())


# --------------------------------------------------------------------------- #
# Bounded end-to-end run (real 2007 reanalysis, gap-bank masks)
# --------------------------------------------------------------------------- #

@needs_2007
def test_end_to_end_two_days(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KRIGE_MAX_OBS", 400)   # keep solves fast
    # 2007 has no local fullgrid files, so every sampled step draws its gap
    # mask from the surface gap bank -- provide a small synthetic one (the
    # real bank is harvested on the cluster by swath build-bank)
    from dl_front import swath as _swath
    rng_b = np.random.default_rng(7)
    bank_path = tmp_path / "sfc_gap_bank.npz"
    with open(bank_path, "wb") as fh:
        np.savez_compressed(
            fh,
            vf=rng_b.random((8, *config.GRID_SHAPE)).astype(np.float16),
            date=np.asarray([f"2007-{m:02d}-15" for m in
                             (1, 3, 5, 7, 9, 11, 6, 12)]),
            hour=np.asarray([21, 0, 21, 0, 21, 0, 21, 0]))
    monkeypatch.setattr(config, "SFC_GAP_BANK_PATH", bank_path)
    monkeypatch.setattr(_swath, "_SFC_GAP_CACHE", {})
    out = tmp_path / "krige_validation"
    # a leftover panel from a previous configuration must be cleared, and a
    # user's own file in panels/ must survive (review 2026-08-13)
    (out / "panels").mkdir(parents=True)
    (out / "panels" / "panel_1999-01-01_18Z_T2M.png").write_bytes(b"stale")
    (out / "panels" / "notes.txt").write_text("mine")
    kv.main(["--years", "2007", "--n-days", "2", "--allow-small-bank",
             "--variograms", "linear", "--panels", "2",
             "--out", str(out), "--seed", "20260813"])
    assert not (out / "panels" / "panel_1999-01-01_18Z_T2M.png").exists()
    assert (out / "panels" / "notes.txt").exists()

    metrics = pd.read_csv(out / "metrics.csv")
    assert list(metrics.columns) == list(kv.METRIC_COLUMNS)
    assert set(metrics["method"]) == {"krige_linear", "nearest", "mean"}
    assert set(metrics["channel"]) == set(config.KRIGED_CHANNELS)
    assert set(metrics["stratum_kind"]) == {"overall", "gap_type", "distance"}
    assert (metrics["n_pixels"] > 0).all()
    assert np.isfinite(metrics["rmse"]).all()
    assert metrics["date"].nunique() == 2

    assert (out / "projection_methods.csv").exists()
    assert (out / "summary.md").exists()
    text = (out / "summary.md").read_text()
    assert "Cloud vs out-of-swath" in text and "Variogram winner" in text
    # winner is picked on unitless per-channel skill vs the nearest
    # baseline, never on cross-channel RMSE means (review 2026-08-13)
    assert "skill_vs_nearest" in text
    assert "seed 20260813" in text and "--n-days 2" in text

    pngs = sorted((out / "panels").glob("panel_*.png"))
    assert len(pngs) == 2
    for png in pngs:
        assert png.stat().st_size > 20_000, f"{png.name} suspiciously small"
