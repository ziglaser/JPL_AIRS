"""Tests for kernel NetCDF I/O -- round-trip identity and provenance."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from trajectory_kernels import io


def _toy_kernels():
    """A tiny build_all-shaped dataset with two populated receptors."""
    n_step, n_lat, n_lon, n_lag, n_dl = 1, 3, 3, 2, 3
    shape = (n_step, n_lat, n_lon, n_lag, n_dl, n_dl)
    fp = np.zeros(shape, dtype="float32")
    fp[0, 1, 1, 0, 1, 1] = 2.0  # receptor (1,1): mass on its own cell at lag 0
    fp[0, 1, 1, 1, 1, 0] = 1.0  # and one cell west at lag 1
    fp[0, 0, 2, 0, 1, 1] = 4.0  # a second receptor
    totals = fp.sum(axis=(3, 4, 5), keepdims=True)
    kernel = np.where(totals > 0, fp / totals, np.nan).astype("float32")
    counts = (fp.sum(axis=(3, 4, 5)) > 0).astype("int32")
    return xr.Dataset(
        {
            "footprint": (("arrival_step", "target_lat", "target_lon", "lag", "dlat", "dlon"), fp),
            "kernel": (("arrival_step", "target_lat", "target_lon", "lag", "dlat", "dlon"), kernel),
            "n_parcels": (("arrival_step", "target_lat", "target_lon"), counts),
        },
        coords={
            "arrival_step": [3], "target_lat": [30.5, 31.5, 32.5],
            "target_lon": [-95.5, -94.5, -93.5], "lag": [0.0, 1.0],
            "dlat": [-1.0, 0.0, 1.0], "dlon": [-1.0, 0.0, 1.0],
        },
        attrs={"pbl_model": "ConstantPBL"},
    )


def test_dense_round_trip(tmp_path):
    ds = _toy_kernels()
    p = io.write_kernels(ds, tmp_path / "k.nc")
    back = io.read_kernels(p)
    assert np.allclose(back["footprint"].values, ds["footprint"].values)
    assert np.allclose(back["kernel"].values, ds["kernel"].values, equal_nan=True)
    assert back.attrs["pbl_model"] == "ConstantPBL"
    assert "source_granules" in back.attrs  # provenance attached


def test_sparse_round_trip_reconstructs_dense(tmp_path):
    ds = _toy_kernels()
    p = io.write_sparse(ds, tmp_path / "k_sparse.nc", which="footprint")
    dense = io.read_sparse(p)["footprint"].values
    assert np.allclose(dense, ds["footprint"].values)


def test_provenance_records_config(tmp_path):
    ds = _toy_kernels()
    back = io.read_kernels(io.write_kernels(ds, tmp_path / "k.nc"))
    assert back.attrs["contact_fraction"] == pytest.approx(
        __import__("trajectory_kernels").config.CONTACT_FRACTION
    )
