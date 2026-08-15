"""Tests for dl_front.quicklook (spot-check PNGs for the CPU-built products).

Everything runs on tiny synthetic inputs written to tmp_path: a hand-built
swath-bank npz and hand-built kriged year caches, so the tests check the
sampling arithmetic and the rendering plumbing (files land where the chain
expects them) without any real archive.  Rendering is checked for existence
and non-emptiness only -- the maps themselves are for human eyes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("matplotlib")

from dl_front import config, quicklook, swath  # noqa: E402

SHAPE = config.GRID_SHAPE                       # (68, 141)


# --------------------------------------------------------------------------- #
# swath-bank rendering
# --------------------------------------------------------------------------- #

def _write_bank(path, hours=(18, 0), n_days_value=10):
    freq = np.zeros((swath.CYCLE_DAYS, len(hours), *SHAPE), np.float32)
    freq[:, :, 20:40, 30:80] = 0.5              # a block above the threshold
    n_days = np.full((swath.CYCLE_DAYS, len(hours)), n_days_value, np.int32)
    n_days[3, :] = 1                            # one undersampled cycle day
    np.savez_compressed(path, freq=freq, n_days=n_days,
                        hours=np.asarray(hours),
                        years=np.asarray([2016, 2017]))
    return path


def test_render_swath_bank_one_png_per_hour(tmp_path):
    bank = _write_bank(tmp_path / "bank.npz", hours=(18, 21, 0))
    out = quicklook.render_swath_bank(path=bank, out_dir=tmp_path / "ql")
    assert [p.name for p in out] == ["swath_bank_18Z.png",
                                     "swath_bank_21Z.png",
                                     "swath_bank_00Z.png"]
    assert all(p.stat().st_size > 0 for p in out)


def test_render_swath_bank_missing_bank_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        quicklook.render_swath_bank(path=tmp_path / "absent.npz",
                                    out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# kriged-cache sampling + rendering
# --------------------------------------------------------------------------- #

def _write_cache(cache_dir, year, times):
    """A schema-v3-shaped year cache with ``len(times)`` synthetic steps."""
    n = len(times)
    rng = np.random.default_rng(year)           # deterministic per year
    data = {}
    for var in config.SFC_VARS:
        field = rng.normal(size=(n, *SHAPE)).astype(np.float32)
        field[:, :5, :5] = np.nan               # out-of-crop corner (v3 NaN)
        data[var] = (("time", "lat", "lon"), field)
    data["valid_frac"] = (("time", "lat", "lon"),
                          np.full((n, *SHAPE), 0.5, np.float32))
    gap = np.full((n, *SHAPE), config.GAP_OUT_OF_SWATH, np.int8)
    gap[:, 20:40, 30:80] = config.GAP_OBSERVED
    gap[:, 25:30, 40:50] = config.GAP_CLOUD
    gap[:, :5, :5] = config.GAP_OUT_OF_DOMAIN
    data["gap_type"] = (("time", "lat", "lon"), gap)
    ds = xr.Dataset(
        data,
        coords={"time": pd.DatetimeIndex(times),
                "lat": np.asarray(config.LABEL_LATS),
                "lon": np.asarray(config.LABEL_LONS)})
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"kriged_sfc_{year}.nc"
    ds.to_netcdf(path)
    return path


def _hours(year, days, hour=18):
    return [pd.Timestamp(f"{year}-06-{d:02d} {hour}:00") for d in days]


def test_sample_steps_even_spacing_and_determinism(tmp_path):
    d = tmp_path / "caches"
    _write_cache(d, 2010, _hours(2010, [1, 2, 3]))
    _write_cache(d, 2011, _hours(2011, [1, 2]))
    picks = quicklook.sample_steps("kriged-airs", n=3, cache_dir=d)
    # 5 pooled steps, n=3 -> evenly spaced global indices 0, 2, 4
    assert [(p.name, t) for p, t in picks] == [
        ("kriged_sfc_2010.nc", 0),
        ("kriged_sfc_2010.nc", 2),
        ("kriged_sfc_2011.nc", 1)]
    # deterministic: a second call gives the identical sample
    assert picks == quicklook.sample_steps("kriged-airs", n=3, cache_dir=d)


def test_sample_steps_n_larger_than_pool_returns_everything(tmp_path):
    d = tmp_path / "caches"
    _write_cache(d, 2010, _hours(2010, [1, 2]))
    assert len(quicklook.sample_steps("kriged-airs", n=99, cache_dir=d)) == 2


def test_sample_steps_year_filter_and_empty_year(tmp_path):
    d = tmp_path / "caches"
    _write_cache(d, 2010, _hours(2010, [1, 2]))
    _write_cache(d, 2011, [])                   # zero-step done-marker year
    _write_cache(d, 2012, _hours(2012, [1]))
    picks = quicklook.sample_steps("kriged-airs", years=[2010, 2011],
                                   n=99, cache_dir=d)
    assert {p.name for p, _ in picks} == {"kriged_sfc_2010.nc"}
    with pytest.raises(FileNotFoundError):
        quicklook.sample_steps("kriged-airs", years=[1999], cache_dir=d)
    with pytest.raises(ValueError):             # only empty caches selected
        quicklook.sample_steps("kriged-airs", years=[2011], cache_dir=d)


def test_render_cache_writes_named_pngs(tmp_path):
    d = tmp_path / "caches"
    _write_cache(d, 2010, _hours(2010, [1, 2, 3]))
    out = quicklook.render_cache("kriged-degraded", n=2, cache_dir=d,
                                 out_dir=tmp_path / "ql")
    # deterministic sampling -> deterministic filenames (rerun overwrites)
    assert [p.name for p in out] == ["kriged-degraded_20100601_18Z.png",
                                     "kriged-degraded_20100603_18Z.png"]
    assert all(p.stat().st_size > 0 for p in out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_swath_bank(tmp_path):
    bank = _write_bank(tmp_path / "bank.npz", hours=(18,))
    quicklook.main(["swath-bank", "--path", str(bank),
                    "--out-dir", str(tmp_path / "ql")])
    assert (tmp_path / "ql/swath_bank_18Z.png").exists()


def test_cli_kriged_cache_with_years(tmp_path):
    d = tmp_path / "caches"
    _write_cache(d, 2010, _hours(2010, [1, 2]))
    quicklook.main(["kriged-airs", "--years", "2010", "--n", "1",
                    "--cache-dir", str(d), "--out-dir", str(tmp_path / "ql")])
    assert (tmp_path / "ql/kriged-airs_20100601_18Z.png").exists()
