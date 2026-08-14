"""Ingest the HYSPLIT forward-trajectory granule files into one tidy
``(parcel, step)`` dataset.

A "parcel" is one released air mass = one ``(granule, level, fieldx, fieldy)``
valid at release. ``step`` is the trajectory index 0..6; actual clock time is
the ``time_utc(parcel, step)`` coordinate, because step 0 (release) is staggered
per granule (~18:53 UTC early swath, ~20:35 late) while steps 1-6 share the
21,22,23,00,01,02 UTC grid.

NOTE ON ``q`` (data audit): specific humidity (g/kg) is a *conserved Lagrangian
tracer* -- reduced only by condensation (removal logged in ``q_excess``), never
by surface moistening -- so trajectories are treated as geometry only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from . import config

#: Per-timestep parcel variables carried through from the raw files.
PARCEL_VARS: tuple[str, ...] = ("lat", "lon", "alt", "pres", "t", "q", "q_excess")

_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "lat": config.RANGE_LAT,
    "lon": config.RANGE_LON,
    "alt": config.RANGE_ALT_M,
    "pres": config.RANGE_PRES_HPA,
    "t": config.RANGE_T_K,
    "q": config.RANGE_Q_GKG,
}


def _swath_of(granule: int) -> str:
    for name, granules in config.SWATHS.items():
        if granule in granules:
            return name
    raise KeyError(f"granule {granule} is not in any swath in config.SWATHS")


def _assert_units(ds: xr.Dataset, source: str) -> None:
    """Fail loudly if any variable falls outside its expected physical range.

    A file in different units (kg/kg vs g/kg, km vs m) or a corrupt read trips
    this immediately, rather than silently poisoning every downstream kernel.
    """
    for name, (lo, hi) in _RANGE_CHECKS.items():
        vals = ds[name].values
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin < lo or vmax > hi:
            raise ValueError(
                f"{source}: {name} range [{vmin:.3g}, {vmax:.3g}] outside expected "
                f"[{lo:g}, {hi:g}] -- units or file mismatch?"
            )


def load_granule(granule: int, traj_dir: Path = config.TRAJ_DIR) -> xr.Dataset:
    """Load one granule file as a tidy ``(parcel, step)`` dataset.

    A parcel is kept if it is valid (finite ``lat``) at step 0. The five
    per-timestep fields become ``(parcel, step)`` data variables; per-parcel
    metadata (``granule``, ``swath``, ``level``, ``fieldx``, ``fieldy``, release
    position/pressure) and the actual ``time_utc(parcel, step)`` clock become
    coordinates.
    """
    path = traj_dir / config.NOGRID_TEMPLATE.format(granule=granule)
    raw = xr.open_dataset(path)

    # time_utc: the raw datetime axis, one value per step, shared by all parcels
    # in this granule (parceltime == time broadcast; only step 0 is granule-set).
    time_utc = raw["time"].values.astype("datetime64[ns]")
    n_step = time_utc.size

    stacked = raw.stack(parcel=("level", "fieldx", "fieldy"))
    valid = np.isfinite(stacked["lat"].isel(time=0).values)
    stacked = stacked.isel(parcel=np.where(valid)[0])

    out = xr.Dataset()
    for name in PARCEL_VARS:
        out[name] = (("parcel", "step"), stacked[name].transpose("parcel", "time").values)

    n_parcel = out.sizes["parcel"]
    levels = stacked["level"].values.astype("int16")
    fieldx = stacked["fieldx"].values.astype("int16")
    fieldy = stacked["fieldy"].values.astype("int16")

    out = out.assign_coords(
        step=("step", np.arange(n_step, dtype="int8")),
        parcel=("parcel", np.arange(n_parcel, dtype="int64")),
        granule=("parcel", np.full(n_parcel, granule, dtype="int16")),
        swath=("parcel", np.full(n_parcel, _swath_of(granule), dtype=object)),
        level=("parcel", levels),
        fieldx=("parcel", fieldx),
        fieldy=("parcel", fieldy),
        release_lat=("parcel", out["lat"].isel(step=0).values),
        release_lon=("parcel", out["lon"].isel(step=0).values),
        release_pres=("parcel", out["pres"].isel(step=0).values),
        time_utc=(("parcel", "step"), np.broadcast_to(time_utc, (n_parcel, n_step)).copy()),
    )
    raw.close()
    _assert_units(out, path.name)
    return out


def load_day(
    granules: Iterable[int] = config.ALL_GRANULES, traj_dir: Path = config.TRAJ_DIR
) -> xr.Dataset:
    """Concatenate all granules for the day into one ``(parcel, step)`` dataset.

    ``parcel`` is re-indexed to a contiguous global id across granules. Steps
    1-6 share the 21-02 UTC grid; step 0 is the per-granule release time (kept in
    ``time_utc``). Adds an ``is_near_surface`` parcel flag (release below the
    receptor band top) as a convenience for downstream selection.
    """
    parts = [load_granule(g, traj_dir) for g in granules]
    day = xr.concat(parts, dim="parcel")
    day = day.assign_coords(parcel=("parcel", np.arange(day.sizes["parcel"], dtype="int64")))

    # cheap "could be arriving-air" pre-filter; the footprint builder re-checks
    # the band at the actual arrival step
    band_lo, band_hi = config.RECEPTOR_BAND_M
    rel_alt = day["alt"].isel(step=0).values
    day = day.assign_coords(
        is_near_surface=("parcel", (rel_alt >= band_lo) & (rel_alt <= band_hi))
    )
    return day


def n_valid_steps(day: xr.Dataset) -> xr.DataArray:
    """Number of finite trajectory steps per parcel (for truncation checks).

    The audit found no mid-run dropout on this day, but downstream code must not
    assume it; this exposes the count so callers/tests can assert it.
    """
    return np.isfinite(day["lat"]).sum(dim="step")
