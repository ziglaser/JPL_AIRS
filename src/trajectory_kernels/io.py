"""NetCDF I/O for kernels: two self-describing on-disk representations, each
stamped with provenance attributes (source granules, config values).

- :func:`write_kernels` / :func:`read_kernels` -- the dense relative-window
  dataset, zlib-compressed.
- :func:`write_sparse` / :func:`read_sparse` -- a lossless COO list of nonzero
  entries, for stacking many days compactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import config


def _provenance(extra: dict | None = None) -> dict:
    """Config snapshot + source pointer, attached to every written file."""
    prov = {
        "tool": "trajectory_kernels",
        "source_traj_dir": str(config.TRAJ_DIR),
        "source_granules": list(config.ALL_GRANULES),
        "grid_lat": list(config.GRID_LAT),
        "grid_lon": list(config.GRID_LON),
        "source_step_deg": config.SOURCE_STEP_DEG,
        "source_window_halfwidth_deg": config.SOURCE_WINDOW_HALFWIDTH_DEG,
        "contact_fraction": config.CONTACT_FRACTION,
        "fuzz_sigma0_km": config.FUZZ_SIGMA0_KM,
        "fuzz_alpha": config.FUZZ_ALPHA,
        "fuzziness": config.FUZZINESS,
        "receptor_band_m": list(config.RECEPTOR_BAND_M),
        "resample_step_min": config.RESAMPLE_STEP_MIN,
    }
    if extra:
        prov.update(extra)
    return prov


def _encoding(ds: xr.Dataset, complevel: int = 4) -> dict:
    """zlib encoding for every numeric data variable."""
    numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
    return {v: {"zlib": True, "complevel": complevel} for v in numeric_vars}


def write_kernels(ds: xr.Dataset, path: str | Path, complevel: int = 4) -> Path:
    """Write the dense relative-window kernel dataset (zlib-compressed)."""
    path = Path(path)
    out = ds.copy()
    out.attrs = {**ds.attrs, **_provenance()}
    out.to_netcdf(path, encoding=_encoding(out, complevel))
    return path


def read_kernels(path: str | Path) -> xr.Dataset:
    """Read a dense relative-window kernel file written by :func:`write_kernels`."""
    return xr.open_dataset(path)


def write_sparse(ds: xr.Dataset, path: str | Path, which: str = "kernel") -> Path:
    """Write the nonzero ``which`` entries as a lossless COO table.

    One row per nonzero cell (integer index coordinates plus value); the axis
    coordinate values are kept as 1-D variables so the dense array can be
    reconstructed.

    Fidelity note: NaNs are treated as zero, so empty receptors' NaN ``kernel``
    reconstructs as 0. Use ``footprint`` as the sparse target (NaN-free by
    construction) or keep ``n_parcels`` alongside to re-mask.
    """
    path = Path(path)
    arr = ds[which].values
    nz = np.nonzero(np.nan_to_num(arr, nan=0.0))
    dims = ds[which].dims
    coo = xr.Dataset(
        {
            "value": ("entry", arr[nz].astype("float32")),
            **{f"{d}_idx": ("entry", nz[k].astype("int32")) for k, d in enumerate(dims)},
        },
        coords={d: (d, ds[d].values) for d in dims},
    )
    coo.attrs = {**ds.attrs, **_provenance({"sparse_of": which, "dense_dims": list(dims)})}
    coo.to_netcdf(path, encoding=_encoding(coo))
    return path


def read_sparse(path: str | Path) -> xr.Dataset:
    """Reconstruct the dense array from a COO file written by :func:`write_sparse`.

    Returns an ``xr.Dataset`` with the single reconstructed variable (named as in
    ``attrs['sparse_of']``) on its original dims/coords.
    """
    coo = xr.open_dataset(path)
    dims = list(coo.attrs["dense_dims"])
    name = coo.attrs["sparse_of"]
    shape = tuple(coo.sizes[d] for d in dims)
    dense = np.zeros(shape, dtype="float32")
    idx = tuple(coo[f"{d}_idx"].values for d in dims)
    dense[idx] = coo["value"].values
    out = xr.Dataset(
        {name: (dims, dense)},
        coords={d: (d, coo[d].values) for d in dims},
    )
    out.attrs = dict(coo.attrs)
    return out
