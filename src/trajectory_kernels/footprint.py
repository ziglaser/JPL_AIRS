"""The footprint / influence-kernel builder.

For a receptor (target cell, arrival step) each arriving parcel is walked
backward along its trajectory, depositing land- and PBL-contact-weighted
residence time onto a source grid, binned by lag -- the STILT source-receptor
construct (Lin+2003) specialized to surface coupling:

    f(x_s, tau) = sum over parcels/steps of  w_contact * land_frac * dt

Two products per receptor: the physical ``footprint`` (hours of land-surface
contact) and the normalized ``kernel`` (weights sum to 1 WITHIN each lag hour;
hour-to-hour weighting is deliberately left to a downstream step -- the
``footprint`` retains the physical contact-hours per lag if you need them).

:func:`footprint_from_trajectories` is the numeric core on plain arrays (unit-
testable with synthetic parcels); :func:`build_footprint` runs one receptor and
:func:`build_all` the whole grid.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import xarray as xr

from . import config, geo
from .contact import contact_weight
from .discount import condensation_discount
from .fuzz import FuzzKernel, StohlFuzz, deposit_gaussian
from .pbl import ClimatologicalPBL, PBLModel
from .resample import resample_trajectory

_EPOCH = np.datetime64("2019-06-05T00:00:00")


def output_grid() -> tuple[np.ndarray, np.ndarray]:
    """The (lat, lon) cell-centre axes of the TARGET grid (== lsm/CAPE grid)."""
    lat_min, lat_max, step = config.GRID_LAT
    lon_min, lon_max, _ = config.GRID_LON
    lat = np.arange(lat_min, lat_max + step / 2.0, step)
    lon = np.arange(lon_min, lon_max + step / 2.0, step)
    return lat, lon


def source_offsets() -> np.ndarray:
    """Source-cell centre offsets (degrees) from the receptor centre.

    Cells of ``SOURCE_STEP_DEG`` spanning +/- ``SOURCE_WINDOW_HALFWIDTH_DEG``,
    placed at odd multiples of step/2 so they NEST inside the 1-deg target
    cells (the receptor cell's edges fall exactly on source-cell edges).
    """
    hw = config.SOURCE_WINDOW_HALFWIDTH_DEG
    step = config.SOURCE_STEP_DEG
    return np.arange(-hw + step / 2.0, hw, step)


def source_window(target_lat: float, target_lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Absolute source (lat, lon) axes of one receptor's source window."""
    offsets = source_offsets()
    return target_lat + offsets, target_lon + offsets


def _utc_seconds(time_utc) -> np.ndarray:
    return (np.asarray(time_utc, dtype="datetime64[ns]") - _EPOCH) / np.timedelta64(1, "s")


def _residence_weights(fine_sec: np.ndarray) -> np.ndarray:
    """Trapezoidal per-point residence time (hours): weights sum to the segment
    duration, so a fully-in-contact parcel deposits exactly its in-PBL time."""
    dt = np.zeros_like(fine_sec)
    dt[1:-1] = (fine_sec[2:] - fine_sec[:-2]) / 2.0
    dt[0] = (fine_sec[1] - fine_sec[0]) / 2.0
    dt[-1] = (fine_sec[-1] - fine_sec[-2]) / 2.0
    return dt / 3600.0


# --------------------------------------------------------------------------- #
# Numeric core
# --------------------------------------------------------------------------- #
def footprint_from_trajectories(
    trajs: Sequence[dict],
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    lag_hours: np.ndarray,
    pbl_model: PBLModel,
    fuzz_kernel: FuzzKernel,
    land_fn: Callable,
    contact_fraction: float = config.CONTACT_FRACTION,
    resample_step_min: float = config.RESAMPLE_STEP_MIN,
    rainout_discount: bool = False,
) -> np.ndarray:
    """Accumulate the physical footprint ``f[lag, source_lat, source_lon]`` (hours).

    Each ``traj`` dict holds one parcel's history up to arrival (arrival last):
    ``t_hours`` (relative, increasing), ``time_utc``, ``lat``, ``lon``, ``alt``,
    and optionally ``q`` for the rain-out discount. Deposits go into the nearest
    integer-hour lag bin as Gaussians whose width grows with the along-track
    distance back from the receptor. With ``rainout_discount`` each point is
    down-weighted by the retained-humidity fraction (Sodemann+2008; exact here
    because q is loss-only).
    """
    n_lag = lag_hours.size
    footprint = np.zeros((n_lag, source_lat.size, source_lon.size))

    for traj in trajs:
        fine = resample_trajectory(traj["lat"], traj["lon"], traj["alt"],
                                   traj["t_hours"], resample_step_min)
        if fine["t_hours"].size < 2:
            continue

        # absolute clock at each fine point (seconds since a fixed epoch,
        # so midnight crossings are seamless)
        coarse_sec = _utc_seconds(traj["time_utc"])
        fine_sec = np.interp(fine["t_hours"], np.asarray(traj["t_hours"], dtype=float),
                             coarse_sec)
        fine_utc = _EPOCH + fine_sec.astype("timedelta64[s]")

        # lag back from arrival, and along-track distance back from arrival
        lag = (fine_sec[-1] - fine_sec) / 3600.0
        path_km = geo.cumulative_path_km(fine["lat"], fine["lon"])
        distance_back = path_km[-1] - path_km

        pbl_depth = np.asarray(pbl_model(fine["lat"], fine["lon"], fine_utc), dtype=float)
        contact = contact_weight(fine["alt"], pbl_depth, fraction=contact_fraction)
        land = np.asarray(land_fn(fine["lat"], fine["lon"]), dtype=float)
        mass = contact * land * _residence_weights(fine_sec)
        if rainout_discount and "q" in traj:
            fine_q = np.interp(fine["t_hours"], np.asarray(traj["t_hours"], dtype=float),
                               np.asarray(traj["q"], dtype=float))
            mass = mass * condensation_discount(fine_q)

        sigma = np.asarray(fuzz_kernel.sigma_km(distance_back), dtype=float)
        lag_bin = np.clip(np.round(lag).astype(int), 0, n_lag - 1)
        in_range = lag <= lag_hours[-1] + 0.5
        for i in np.where((mass > 0.0) & in_range)[0]:
            deposit_gaussian(source_lat, source_lon, fine["lat"][i], fine["lon"][i],
                             float(sigma[i]), float(mass[i]), footprint[lag_bin[i]])
    return footprint


# --------------------------------------------------------------------------- #
# Parcel selection helpers
# --------------------------------------------------------------------------- #
def _parcel_arrays(day: xr.Dataset) -> dict[str, np.ndarray]:
    """Pull the per-parcel arrays out of the dataset once (fast plain indexing)."""
    return {name: day[name].values
            for name in ("time_utc", "lat", "lon", "alt", "q", "swath")}


def _traj_dicts(arrays: dict, members: np.ndarray, arrival_step: int) -> list[dict]:
    """History-segment traj dicts (steps 0..arrival_step) for the given parcels."""
    steps = slice(0, arrival_step + 1)
    trajs = []
    for p in members:
        release_time = arrays["time_utc"][p, 0]
        t_hours = (arrays["time_utc"][p, steps] - release_time) / np.timedelta64(1, "h")
        trajs.append({
            "t_hours": t_hours.astype(float),
            "time_utc": arrays["time_utc"][p, steps],
            "lat": arrays["lat"][p, steps],
            "lon": arrays["lon"][p, steps],
            "alt": arrays["alt"][p, steps],
            "q": arrays["q"][p, steps],
        })
    return trajs


# --------------------------------------------------------------------------- #
# Kernel containment  (config.KERNEL_CONTAINMENT_FRAC)
# --------------------------------------------------------------------------- #
def _parcel_states_at_lags(trajs: Sequence[dict], lag_hours: np.ndarray):
    """Each parcel's (lat, lon, alt, time-seconds) at each lag before its own
    arrival, linearly interpolated along its trajectory (NaN before release)."""
    lag_hours = np.asarray(lag_hours, dtype=float)
    shape = (len(trajs), lag_hours.size)
    plat, plon, palt = (np.full(shape, np.nan) for _ in range(3))
    psec = np.full(shape, np.nan)
    for i, traj in enumerate(trajs):
        t_sec = np.asarray(_utc_seconds(traj["time_utc"]), dtype=float)
        t_query = t_sec[-1] - lag_hours * 3600.0
        plat[i] = np.interp(t_query, t_sec, np.asarray(traj["lat"], float), left=np.nan)
        plon[i] = np.interp(t_query, t_sec, np.asarray(traj["lon"], float), left=np.nan)
        palt[i] = np.interp(t_query, t_sec, np.asarray(traj["alt"], float), left=np.nan)
        psec[i] = t_query
    return plat, plon, palt, psec


def containment_mask(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    trajs: Sequence[dict],
    lag_hours: np.ndarray,
    pbl_model: PBLModel,
    contact_fraction: float = config.CONTACT_FRACTION,
    frac: float = config.KERNEL_CONTAINMENT_FRAC,
) -> np.ndarray:
    """Boolean (lag, source_lat, source_lon) kernel support: per lag, the cells
    inside the smallest circle -- grown outward from the member parcels' center
    of mass -- containing ``frac`` of the parcels in PBL contact then (the same
    gate the deposit uses; if none is in contact, all parcels present count).

    A half-cell-diagonal slack is added to the radius so the cells the contained
    parcels actually sit in are always kept (their centres can be up to half a
    diagonal from the parcel). Lags with no parcels stay fully open (they carry
    no mass anyway).
    """
    plat, plon, palt, psec = _parcel_states_at_lags(trajs, lag_hours)
    mask = np.ones((len(lag_hours), source_lat.size, source_lon.size), dtype=bool)
    grid_lat, grid_lon = np.meshgrid(source_lat, source_lon, indexing="ij")
    step = float(source_lat[1] - source_lat[0]) if source_lat.size > 1 else config.SOURCE_STEP_DEG

    for k in range(len(lag_hours)):
        present = np.isfinite(plat[:, k]) & np.isfinite(plon[:, k])
        if not present.any():
            continue
        la, lo, al = plat[present, k], plon[present, k], palt[present, k]
        when = _EPOCH + psec[present, k].astype("timedelta64[s]")
        pbl = np.asarray(pbl_model(la, lo, when), dtype=float)
        in_contact = contact_weight(al, pbl, fraction=contact_fraction) > 0.0
        if in_contact.any():
            la, lo = la[in_contact], lo[in_contact]
        com_lat, com_lon = float(la.mean()), float(lo.mean())

        # smallest radius (from the COM) containing >= frac of the parcels
        dist = np.sort(np.atleast_1d(geo.haversine_km(com_lat, com_lon, la, lo)))
        radius = float(dist[int(np.ceil(frac * dist.size)) - 1])
        half_diag = 0.5 * np.hypot(step * geo.km_per_deg_lat(),
                                   step * geo.km_per_deg_lon(com_lat))
        cell_dist = geo.haversine_km(com_lat, com_lon, grid_lat, grid_lon)
        mask[k] = cell_dist <= radius + half_diag
    return mask


def _normalize_per_lag(footprint: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Per-lag kernel: each populated lag slice of ``masked`` sums to 1 over the
    source cells (the last two axes). A lag whose containment mask removed ALL
    its mass falls back to the unmasked footprint (possible only when the
    deposits came from parcels outside the contact-gated cloud); lags with no
    mass at all are NaN, never fabricated."""
    totals = masked.sum(axis=(-2, -1), keepdims=True)
    raw_totals = footprint.sum(axis=(-2, -1), keepdims=True)
    fallback = (totals == 0) & (raw_totals > 0)
    if fallback.any():
        fb = np.broadcast_to(fallback, masked.shape)
        masked = np.where(fb, footprint, masked)
        totals = np.where(fallback, raw_totals, totals)
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = masked / totals  # 0/0 -> NaN exactly where a lag is empty
    return kernel


def _default_models(pbl_model, fuzz_kernel, land_fn):
    if pbl_model is None:
        pbl_model = ClimatologicalPBL()
    if fuzz_kernel is None:
        fuzz_kernel = StohlFuzz()
    if land_fn is None:
        from .land import make_land_lookup
        land_fn = make_land_lookup()
    return pbl_model, fuzz_kernel, land_fn


# --------------------------------------------------------------------------- #
# Single receptor
# --------------------------------------------------------------------------- #
def _wrap(footprint, source_lat, source_lon, lag_hours,
          target_lat, target_lon, meta, mask=None) -> xr.Dataset:
    """Package a footprint array + normalized kernel as an annotated Dataset.

    The kernel is the footprint truncated to the containment ``mask`` (parcel
    cloud support; None = no truncation) and normalized PER LAG HOUR (each lag
    slice sums to 1); lag hours with no deposited mass are NaN, never
    fabricated. The ``footprint`` itself is always the full physical field.
    """
    kernel = _normalize_per_lag(footprint,
                                footprint if mask is None else footprint * mask)
    ds = xr.Dataset(
        {
            "footprint": (("lag", "source_lat", "source_lon"), footprint),
            "kernel": (("lag", "source_lat", "source_lon"), kernel),
        },
        coords={
            "lag": ("lag", lag_hours.astype(float)),
            "source_lat": ("source_lat", source_lat),
            "source_lon": ("source_lon", source_lon),
            "dlat": ("source_lat", source_lat - target_lat),
            "dlon": ("source_lon", source_lon - target_lon),
        },
    )
    ds["footprint"].attrs["units"] = "hours of land-surface contact"
    ds["kernel"].attrs["long_name"] = (
        "influence kernel (truncated to the parcel-cloud containment region; "
        "weights sum to 1 within each lag hour; "
        "hour-to-hour weighting applied downstream)")
    ds.attrs.update({"target_lat": target_lat, "target_lon": target_lon, **meta})
    return ds


def build_footprint(
    day: xr.Dataset,
    target_lat: float,
    target_lon: float,
    arrival_step: int,
    pbl_model: Optional[PBLModel] = None,
    fuzz_kernel: Optional[FuzzKernel] = None,
    land_fn: Optional[Callable] = None,
    contact_fraction: float = config.CONTACT_FRACTION,
    catch_halfwidth_deg: float = config.RECEPTOR_CATCH_HALFWIDTH_DEG,
    receptor_band_m: tuple[float, float] = config.RECEPTOR_BAND_M,
    rainout_discount: bool = False,
    containment_frac: Optional[float] = config.KERNEL_CONTAINMENT_FRAC,
) -> xr.Dataset:
    """Influence kernel for one receptor (target cell, arrival step 1-6).

    Arriving parcels = within ``catch_halfwidth_deg`` of the target at
    ``arrival_step`` AND inside ``receptor_band_m`` (the RECEPTOR_BAND knob).
    Returns ``footprint``/``kernel`` on (lag, source_lat, source_lon) with
    provenance attrs (parcel counts, swath split, max available lag). The
    kernel's support is truncated per lag to the region containing
    ``containment_frac`` of the parcel cloud (None disables); the physical
    footprint is never truncated.
    """
    if arrival_step < 1:
        raise ValueError("arrival_step must be >= 1 (step 0 is the release)")
    pbl_model, fuzz_kernel, land_fn = _default_models(pbl_model, fuzz_kernel, land_fn)
    arrays = _parcel_arrays(day)

    arrive_lat = arrays["lat"][:, arrival_step]
    arrive_lon = arrays["lon"][:, arrival_step]
    arrive_alt = arrays["alt"][:, arrival_step]
    band_lo, band_hi = receptor_band_m
    caught = (
        (np.abs(arrive_lat - target_lat) <= catch_halfwidth_deg)
        & (np.abs(arrive_lon - target_lon) <= catch_halfwidth_deg)
        & (arrive_alt >= band_lo) & (arrive_alt <= band_hi)
    )
    members = np.where(caught)[0]
    source_lat, source_lon = source_window(target_lat, target_lon)

    if members.size == 0:
        meta = {"n_parcels": 0, "n_early": 0, "n_late": 0, "max_lag_hours": 0.0,
                "arrival_step": arrival_step, "rainout_discount": int(rainout_discount),
                "kernel_containment_frac": float(containment_frac or 0.0)}
        empty = np.zeros((1, source_lat.size, source_lon.size))
        ds = _wrap(empty, source_lat, source_lon, np.arange(1, dtype=float),
                   target_lat, target_lon, meta)
        ds["member_parcel"] = ("member", np.empty(0, dtype="int32"))
        return ds

    trajs = _traj_dicts(arrays, members, arrival_step)
    max_lag = min(max(t["t_hours"][-1] for t in trajs), config.MAX_LAG_HOURS)
    lag_hours = np.arange(0, np.ceil(max_lag) + 1, dtype=float)

    footprint = footprint_from_trajectories(
        trajs, source_lat, source_lon, lag_hours, pbl_model, fuzz_kernel, land_fn,
        contact_fraction=contact_fraction, rainout_discount=rainout_discount,
    )
    mask = None
    if containment_frac is not None:
        mask = containment_mask(source_lat, source_lon, trajs, lag_hours,
                                pbl_model, contact_fraction, containment_frac)
    swath = arrays["swath"][members]
    meta = {
        "n_parcels": int(members.size),
        "n_early": int((swath == "early").sum()),
        "n_late": int((swath == "late").sum()),
        "max_lag_hours": float(max_lag),
        "arrival_step": int(arrival_step),
        "arrival_time_utc": str(arrays["time_utc"][members[0], arrival_step]),
        "contact_fraction": float(contact_fraction),
        "rainout_discount": int(rainout_discount),
        "kernel_containment_frac": float(containment_frac or 0.0),  # 0 = off
    }
    ds = _wrap(footprint, source_lat, source_lon, lag_hours,
               target_lat, target_lon, meta, mask=mask)
    # arriving-parcel indices (into the day dataset's parcel dim), so plots can
    # overlay the actual HYSPLIT trajectories behind the kernel
    ds["member_parcel"] = ("member", members.astype("int32"))
    ds["member_parcel"].attrs["long_name"] = (
        "parcel indices (day dataset) of the arriving parcels")
    return ds


# --------------------------------------------------------------------------- #
# All receptors
# --------------------------------------------------------------------------- #
def _cells_with_arrivals(arrays: dict, arrival_step: int,
                         receptor_band_m: tuple[float, float],
                         grid_lat: np.ndarray, grid_lon: np.ndarray) -> dict:
    """Map (lat_index, lon_index) -> parcel indices arriving in that cell.

    Each parcel is assigned to its single nearest grid cell (no double
    counting); off-grid and out-of-band parcels are dropped.
    """
    lat0, _, step = config.GRID_LAT
    lon0 = config.GRID_LON[0]
    arrive_lat = arrays["lat"][:, arrival_step]
    arrive_lon = arrays["lon"][:, arrival_step]
    arrive_alt = arrays["alt"][:, arrival_step]

    lat_idx = np.round((arrive_lat - lat0) / step).astype(int)
    lon_idx = np.round((arrive_lon - lon0) / step).astype(int)
    band_lo, band_hi = receptor_band_m
    keep = (
        (arrive_alt >= band_lo) & (arrive_alt <= band_hi)
        & (lat_idx >= 0) & (lat_idx < grid_lat.size)
        & (lon_idx >= 0) & (lon_idx < grid_lon.size)
    )
    parcels = np.where(keep)[0]

    cells: dict[tuple[int, int], list[int]] = {}
    for p, i, j in zip(parcels, lat_idx[parcels], lon_idx[parcels]):
        cells.setdefault((int(i), int(j)), []).append(int(p))
    return cells


def build_all(
    day: xr.Dataset,
    arrival_steps: Sequence[int] = (1, 2, 3, 4, 5, 6),
    pbl_model: Optional[PBLModel] = None,
    fuzz_kernel: Optional[FuzzKernel] = None,
    land_fn: Optional[Callable] = None,
    contact_fraction: float = config.CONTACT_FRACTION,
    receptor_band_m: tuple[float, float] = config.RECEPTOR_BAND_M,
    max_lag_hours: float = config.MAX_LAG_HOURS,
    window_halfwidth_deg: float = config.SOURCE_WINDOW_HALFWIDTH_DEG,
    rainout_discount: bool = False,
    containment_frac: Optional[float] = config.KERNEL_CONTAINMENT_FRAC,
) -> xr.Dataset:
    """Kernels for every grid cell and arrival step, on a relative source window.

    Output dims: (arrival_step, target_lat, target_lon, lag, dlat, dlon) --
    targets on the 1-deg CAPE grid, the source window at SOURCE_STEP_DEG --
    the primary NetCDF schema. Empty receptors have ``n_parcels == 0``, zero
    footprint and NaN kernel -- honest gaps, never fabricated. Kernels (not
    footprints) are truncated per lag to the ``containment_frac`` parcel-cloud
    region (None disables).
    """
    pbl_model, fuzz_kernel, land_fn = _default_models(pbl_model, fuzz_kernel, land_fn)
    arrays = _parcel_arrays(day)
    grid_lat, grid_lon = output_grid()
    step_deg = config.SOURCE_STEP_DEG

    dlat = np.arange(-window_halfwidth_deg + step_deg / 2.0, window_halfwidth_deg, step_deg)
    dlon = np.arange(-window_halfwidth_deg + step_deg / 2.0, window_halfwidth_deg, step_deg)

    # lag axis sized to what the data supports, capped by max_lag_hours
    available = (arrays["time_utc"][:, list(arrival_steps)]
                 - arrays["time_utc"][:, [0]]) / np.timedelta64(1, "h")
    data_max_lag = float(np.nanmax(available)) if available.size else 0.0
    lag_hours = np.arange(0, np.ceil(min(data_max_lag, max_lag_hours)) + 1, dtype=float)

    shape = (len(arrival_steps), grid_lat.size, grid_lon.size,
             lag_hours.size, dlat.size, dlon.size)
    footprint = np.zeros(shape, dtype="float32")
    masked = np.zeros(shape, dtype="float32")  # footprint within the containment region
    counts = np.zeros(shape[:3], dtype="int32")

    for s, step in enumerate(arrival_steps):
        cells = _cells_with_arrivals(arrays, step, receptor_band_m, grid_lat, grid_lon)
        for (i, j), members in cells.items():
            trajs = _traj_dicts(arrays, np.asarray(members), step)
            source_lat, source_lon = grid_lat[i] + dlat, grid_lon[j] + dlon
            fp = footprint_from_trajectories(
                trajs, source_lat, source_lon, lag_hours,
                pbl_model, fuzz_kernel, land_fn,
                contact_fraction=contact_fraction, rainout_discount=rainout_discount,
            ).astype("float32")
            footprint[s, i, j] = fp
            if containment_frac is not None:
                fp = fp * containment_mask(source_lat, source_lon, trajs, lag_hours,
                                           pbl_model, contact_fraction, containment_frac)
            masked[s, i, j] = fp
            counts[s, i, j] = len(members)

    # per-lag-hour normalization of the containment-truncated footprint: each
    # (receptor, lag) slice sums to 1 over (dlat, dlon); hours with no
    # deposited mass are NaN
    kernel = _normalize_per_lag(footprint, masked)
    del masked

    dims6 = ("arrival_step", "target_lat", "target_lon", "lag", "dlat", "dlon")
    ds = xr.Dataset(
        {
            "footprint": (dims6, footprint),
            "kernel": (dims6, kernel),
            "n_parcels": (dims6[:3], counts),
        },
        coords={
            "arrival_step": ("arrival_step", np.asarray(arrival_steps, dtype="int8")),
            "target_lat": ("target_lat", grid_lat),
            "target_lon": ("target_lon", grid_lon),
            "lag": ("lag", lag_hours),
            "dlat": ("dlat", dlat),
            "dlon": ("dlon", dlon),
        },
    )
    ds["footprint"].attrs["units"] = "hours of land-surface contact"
    ds["kernel"].attrs["long_name"] = (
        "influence kernel (truncated to the parcel-cloud containment region; "
        "weights sum to 1 within each lag hour per receptor; "
        "hour-to-hour weighting applied downstream)")
    ds.attrs.update({
        "contact_fraction": float(contact_fraction),
        "receptor_band_m": list(receptor_band_m),
        "pbl_model": type(pbl_model).__name__,
        "fuzz_model": type(fuzz_kernel).__name__,
        "rainout_discount": int(rainout_discount),
        "kernel_containment_frac": float(containment_frac or 0.0),  # 0 = off
    })
    return ds
