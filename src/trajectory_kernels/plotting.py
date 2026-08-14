"""Diagnostic plots. Thin wrappers over matplotlib (no cartopy); geographic
context is a grey land outline from ``data/lsm.nc``. Every function returns
``(fig, ax)`` and optionally saves to ``save_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; callers can override before importing
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from . import config, geo  # noqa: E402
from .contact import contact_weight  # noqa: E402
from .pbl import ClimatologicalPBL  # noqa: E402


def _land_outline(ax, alpha: float = 0.25) -> None:
    """Draw a light land mask over the output domain for geographic context."""
    try:
        lsm = xr.open_dataset(config.LSM_PATH)["lsm"]
    except (FileNotFoundError, OSError):
        return
    lat0, lat1, _ = config.GRID_LAT
    lon0, lon1, _ = config.GRID_LON
    sub = lsm.sel(lat=slice(lat0 - 2, lat1 + 2), lon=slice(lon0 - 2, lon1 + 2))
    ax.contourf(
        sub["lon"], sub["lat"], (sub.values >= config.LAND_FRACTION_MIN).astype(float),
        levels=[0.5, 1.5], colors=["#cfcfcf"], alpha=alpha, zorder=0,
    )
    ax.set_xlim(lon0 - 1, lon1 + 1)
    ax.set_ylim(lat0 - 1, lat1 + 1)


def _finish(fig, ax, title: str, save_path: Optional[Path]):
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, ax


def plot_trajectories(
    day: xr.Dataset,
    parcel_ids: Optional[Sequence[int]] = None,
    color_by: str = "step",
    max_parcels: int = 300,
    title: str = "HYSPLIT forward trajectories",
    save_path: Optional[Path] = None,
):
    """Plot parcel paths on the domain, colored by ``step`` or ``alt``.

    ``parcel_ids`` selects parcels (default: an evenly spaced sample of up to
    ``max_parcels``). Each parcel is drawn as a faint line with markers at each
    stored step; the release point is a black dot.
    """
    if parcel_ids is None:
        n = day.sizes["parcel"]
        parcel_ids = np.linspace(0, n - 1, min(max_parcels, n)).astype(int)
    sel = day.isel(parcel=list(parcel_ids))
    lat = sel["lat"].values
    lon = sel["lon"].values

    fig, ax = plt.subplots(figsize=(9, 6))
    _land_outline(ax)
    if color_by == "step":
        for i in range(lat.shape[0]):
            ax.plot(lon[i], lat[i], "-", color="steelblue", lw=0.4, alpha=0.35, zorder=1)
        steps = np.broadcast_to(sel["step"].values, lat.shape)
        sc = ax.scatter(lon.ravel(), lat.ravel(), c=steps.ravel(), s=6, cmap="viridis", zorder=2)
        fig.colorbar(sc, ax=ax, label="trajectory step (0=release)")
    elif color_by == "alt":
        for i in range(lat.shape[0]):
            ax.plot(lon[i], lat[i], "-", color="grey", lw=0.3, alpha=0.3, zorder=1)
        sc = ax.scatter(lon.ravel(), lat.ravel(), c=sel["alt"].values.ravel(), s=6,
                        cmap="turbo", zorder=2)
        fig.colorbar(sc, ax=ax, label="altitude (m)")
    else:
        raise ValueError(f"color_by must be 'step' or 'alt', got {color_by!r}")
    ax.scatter(lon[:, 0], lat[:, 0], c="k", s=8, zorder=3, label="release")
    ax.legend(loc="upper right")
    return _finish(fig, ax, title, save_path)


def plot_contact_along_trajectory(
    day: xr.Dataset,
    parcel_id: int,
    pbl_model,
    contact_fn,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """For one parcel, plot altitude vs PBL depth and the resulting contact
    weight across the trajectory steps.
    """
    p = day.isel(parcel=int(parcel_id))
    alt = p["alt"].values
    lat = p["lat"].values
    lon = p["lon"].values
    tutc = p["time_utc"].values
    pbl = np.array([float(pbl_model(lat[i], lon[i], tutc[i])) for i in range(len(alt))])
    w = np.array([float(contact_fn(alt[i], pbl[i])) for i in range(len(alt))])
    hours = ((tutc - tutc[0]) / np.timedelta64(1, "h")).astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(hours, alt, "o-", color="firebrick", label="parcel altitude")
    ax1.plot(hours, pbl, "s--", color="steelblue", label="PBL depth")
    ax1.plot(hours, config.CONTACT_FRACTION * pbl, ":", color="steelblue",
             label=f"contact top ({config.CONTACT_FRACTION:g}xPBL)")
    ax1.set_ylabel("height (m)")
    ax1.legend(fontsize=8)
    ax1.set_title(title or f"parcel {parcel_id} (level {int(p['level'])}, granule {int(p['granule'])})")
    ax2.bar(hours, w, width=0.6, color="seagreen")
    ax2.set_ylabel("contact weight")
    ax2.set_xlabel("hours since release")
    ax2.set_ylim(0, 1.05)
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_coverage(kernels: xr.Dataset, arrival_step: Optional[int] = None,
                  save_path: Optional[Path] = None):
    """Map how many arriving parcels each receptor cell got (empty cells are the
    trajectory-coverage gaps)."""
    counts = kernels["n_parcels"]
    if arrival_step is not None:
        counts = counts.sel(arrival_step=arrival_step)
    else:
        counts = counts.max("arrival_step")
    masked = counts.where(counts > 0)
    fig, ax = plt.subplots(figsize=(8, 6))
    _land_outline(ax)
    pcm = ax.pcolormesh(kernels["target_lon"], kernels["target_lat"], masked.values,
                        cmap="viridis", shading="nearest")
    fig.colorbar(pcm, ax=ax, label="arriving near-surface parcels")
    lab = "any step" if arrival_step is None else f"step {arrival_step}"
    return _finish(fig, ax, f"Receptor coverage ({lab})", save_path)


def _receptor_marker(ax, ds):
    ax.scatter([ds.attrs["target_lon"]], [ds.attrs["target_lat"]], marker="*",
               s=220, c="red", edgecolor="k", zorder=5, label="receptor")


def _receptor_cell_box(ax, ds):
    """Outline the actual 1-deg receptor cell -- membership means being inside
    THIS box at the arrival hour, which the arrival-position dots verify."""
    half = config.GRID_LAT[2] / 2.0
    rect = mpatches.Rectangle(
        (ds.attrs["target_lon"] - half, ds.attrs["target_lat"] - half),
        2 * half, 2 * half, fill=False, edgecolor="red", lw=1.5, zorder=5,
    )
    ax.add_patch(rect)
    return rect


def _com_region_mask(values: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     frac: float) -> Optional[np.ndarray]:
    """Cells inside the smallest circle, grown outward from the lag's mass
    centroid, that holds ``frac`` of the total mass -- the plotted analog of the
    kernel's own COM-containment rule. Unlike a raw highest-density cell set
    (which can scatter into disconnected islands), a distance ball is always ONE
    connected region with a single closed outline."""
    w = np.nan_to_num(np.asarray(values, dtype=float))
    total = w.sum()
    if total <= 0:
        return None
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    com_lat = float((w * grid_lat).sum() / total)
    com_lon = float((w * grid_lon).sum() / total)
    dist = geo.haversine_km(com_lat, com_lon, grid_lat, grid_lon)
    order = np.argsort(dist, axis=None)
    csum = np.cumsum(w.ravel()[order])
    k = int(np.searchsorted(csum, frac * total - 1e-12))
    radius = dist.ravel()[order][min(k, order.size - 1)]
    return dist <= radius + 1e-9


def _region_outlines(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     inset: float = 0.0) -> list:
    """CLOSED boundary polygon(s) of the True cells of ``mask``, traced along
    the exact cell edges -- a staircase that never bulges into excluded cells,
    joined into loops (first vertex == last). Returns a list of (N, 2) arrays
    of (lon, lat) vertices.

    ``inset`` (degrees) offsets each whole loop uniformly inward, so outlines
    of overlapping regions (different lags claiming the same cells) nest
    cleanly instead of overdrawing each other -- and, unlike a per-cell inset,
    the loop stays fully connected across cell junctions.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    hlat = (lat[1] - lat[0]) / 2.0 if lat.size > 1 else config.SOURCE_STEP_DEG / 2.0
    hlon = (lon[1] - lon[0]) / 2.0 if lon.size > 1 else config.SOURCE_STEP_DEG / 2.0
    inset = float(min(inset, 0.8 * min(hlat, hlon)))  # never collapse a cell

    # Directed boundary edges on the cell-corner lattice, oriented with the
    # region on the LEFT (so "left of travel" == inward for the inset).
    # Vertex (vi, vj) sits at the corner lat[0]-hlat+vi*2*hlat etc.
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    out_edges: dict[tuple, list] = {}
    for i, j in zip(*np.where(mask)):
        if not padded[i + 2, j + 1]:  # open to the north: edge heads west
            out_edges.setdefault((i + 1, j + 1), []).append((i + 1, j))
        if not padded[i, j + 1]:      # open to the south: edge heads east
            out_edges.setdefault((i, j), []).append((i, j + 1))
        if not padded[i + 1, j + 2]:  # open to the east: edge heads north
            out_edges.setdefault((i, j + 1), []).append((i + 1, j + 1))
        if not padded[i + 1, j]:      # open to the west: edge heads south
            out_edges.setdefault((i + 1, j), []).append((i, j))

    # chain the edges into closed loops; at a corner where two loops touch
    # diagonally, the left-most turn keeps each region on its own loop
    used: set = set()
    loops = []
    for start in sorted(out_edges):
        for first in list(out_edges[start]):
            if (start, first) in used:
                continue
            loop = [start, first]
            used.add((start, first))
            prev, cur = start, first
            while cur != start:
                cands = [e for e in out_edges.get(cur, []) if (cur, e) not in used]
                if not cands:  # malformed mask; keep what we have
                    break
                dx, dy = cur[1] - prev[1], cur[0] - prev[0]
                nxt = max(cands, key=lambda e: dx * (e[0] - cur[0]) - dy * (e[1] - cur[1]))
                used.add((cur, nxt))
                loop.append(nxt)
                prev, cur = cur, nxt
            loops.append(loop)

    # geometry: shift every edge inward (left of travel) by the inset, merge
    # straight runs, and rebuild the vertices as intersections of consecutive
    # perpendicular edges -- an exactly closed rectilinear polygon
    lat0, lon0 = lat[0] - hlat, lon[0] - hlon
    polys = []
    for loop in loops:
        edges = []  # (is_horizontal, shifted constant coordinate)
        for a, b in zip(loop[:-1], loop[1:]):
            if a[0] == b[0]:  # horizontal edge: constant latitude
                y = lat0 + a[0] * 2.0 * hlat + (inset if b[1] > a[1] else -inset)
                e = (True, y)
            else:             # vertical edge: constant longitude
                x = lon0 + a[1] * 2.0 * hlon + (-inset if b[0] > a[0] else inset)
                e = (False, x)
            if not edges or edges[-1] != e:
                edges.append(e)
        if len(edges) > 1 and edges[0] == edges[-1]:
            edges.pop()
        pts = []
        for k in range(len(edges)):
            a, b = edges[k], edges[(k + 1) % len(edges)]
            pts.append((b[1], a[1]) if a[0] else (a[1], b[1]))  # (lon, lat)
        pts.append(pts[0])
        polys.append(np.asarray(pts))
    return polys


def _contour_lag_kernel(ax, ds, lag_field: np.ndarray, color,
                        hdr_fracs: Sequence[float], lw: float = 1.6,
                        inset: float = 0.0):
    """Outline one lag's mass regions (solid = first fraction, dashed/dotted =
    the next) as closed staircase polygons along the actual cell boundaries.
    Each region is the smallest centroid-outward circle of cells holding that
    fraction of the lag's mass, so it is always connected and its outline
    closed. A single-cell (delta) kernel gets exactly its own cell box."""
    vals = np.nan_to_num(np.asarray(lag_field, dtype=float))
    if vals.sum() <= 0:
        return False
    slat = ds["source_lat"].values
    slon = ds["source_lon"].values
    for frac, style in zip(hdr_fracs, ("-", "--", ":")):
        mask = _com_region_mask(vals, slat, slon, frac)
        if mask is None:
            continue
        loops = _region_outlines(mask, slat, slon, inset=inset)
        ax.add_collection(LineCollection(loops, colors=[color], linewidths=lw,
                                         linestyles=style, zorder=2))
    return True


def _zoom_to_source_window(ax, ds, pad: float = 0.5) -> None:
    """Frame the receptor's own source window (the land outline sets the full
    domain otherwise, which dwarfs the kernel and its trajectories)."""
    slat, slon = ds["source_lat"].values, ds["source_lon"].values
    ax.set_xlim(slon.min() - pad, slon.max() + pad)
    ax.set_ylim(slat.min() - pad, slat.max() + pad)


def _member_history(day: xr.Dataset, ds: xr.Dataset, max_trajectories: int):
    """(lat, lon, time_sec) history arrays for the receptor's arriving parcels.

    Uses the ``member_parcel`` indices recorded by ``build_footprint``; evenly
    subsamples to ``max_trajectories`` for readability. Returns None when the
    receptor carries no member list (e.g. build_all output) or is empty.
    """
    if "member_parcel" not in ds or ds.sizes.get("member", 0) == 0:
        return None
    members = ds["member_parcel"].values
    if members.size > max_trajectories:
        members = members[np.linspace(0, members.size - 1, max_trajectories).astype(int)]
    steps = slice(0, int(ds.attrs["arrival_step"]) + 1)
    return {
        "n_shown": members.size,
        "n_total": ds.sizes["member"],
        "lat": day["lat"].values[members, steps],
        "lon": day["lon"].values[members, steps],
        "alt": day["alt"].values[members, steps],
        "time_sec": day["time_utc"].values[members, steps]
                       .astype("datetime64[s]").astype(float),
    }


def _overlay_member_trajectories(ax, day, ds, max_trajectories: int):
    """Draw the arriving parcels' HYSPLIT paths (release -> receptor) on a map.

    Returns the history dict so callers can add lag-position markers without
    re-extracting (and re-subsampling) the trajectories.
    """
    hist = _member_history(day, ds, max_trajectories)
    if hist is None:
        return None
    for i in range(hist["lat"].shape[0]):
        ax.plot(hist["lon"][i], hist["lat"][i], "-", color="deepskyblue",
                lw=0.7, alpha=0.6, zorder=3)
    ax.scatter(hist["lon"][:, 0], hist["lat"][:, 0], c="k", s=10, zorder=4,
               edgecolor="w", linewidth=0.4,
               label=f"release ({hist['n_shown']}/{hist['n_total']} parcels)")
    return hist


def _positions_at_lag(hist: dict, lag_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Each member parcel's (lat, lon) ``lag_h`` hours before its own arrival
    (linear interpolation along its trajectory; NaN before release)."""
    t_query = hist["time_sec"][:, -1] - lag_h * 3600.0
    plat = np.array([np.interp(t, ts, la, left=np.nan) for t, ts, la in
                     zip(t_query, hist["time_sec"], hist["lat"])])
    plon = np.array([np.interp(t, ts, lo, left=np.nan) for t, ts, lo in
                     zip(t_query, hist["time_sec"], hist["lon"])])
    return plat, plon


def _contact_at_lag(hist: dict, lag_h: float, pbl_model,
                    contact_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions at ``lag_h`` plus whether each parcel is surface-coupled there
    -- the same gate as the footprint deposit (``contact_weight > 0``, i.e.
    below ``contact_fraction`` x the local PBL depth at that time)."""
    plat, plon = _positions_at_lag(hist, lag_h)
    t_query = hist["time_sec"][:, -1] - lag_h * 3600.0
    alt = np.array([np.interp(t, ts, al, left=np.nan) for t, ts, al in
                    zip(t_query, hist["time_sec"], hist["alt"])])
    when = t_query.astype("int64").astype("datetime64[s]")
    pbl = np.asarray(pbl_model(plat, plon, when), dtype=float)
    in_contact = contact_weight(alt, pbl, fraction=contact_fraction) > 0.0
    return plat, plon, in_contact


def _zoom_to_content(ax, ds, hist, pad: float = 1.0) -> None:
    """Frame the member trajectories plus the receptor cell (fall back to the
    source window when no trajectories are drawn)."""
    if hist is None:
        _zoom_to_source_window(ax, ds)
        return
    lat = np.concatenate([hist["lat"].ravel(), [ds.attrs["target_lat"]]])
    lon = np.concatenate([hist["lon"].ravel(), [ds.attrs["target_lon"]]])
    ax.set_xlim(np.nanmin(lon) - pad, np.nanmax(lon) + pad)
    ax.set_ylim(np.nanmin(lat) - pad, np.nanmax(lat) + pad)


def plot_kernel_at_lag(
    ds: xr.Dataset,
    lag: float,
    day: Optional[xr.Dataset] = None,
    max_trajectories: int = 150,
    hdr_fracs: Sequence[float] = (0.5, 0.9),
    pbl_model=None,
    save_path: Optional[Path] = None,
):
    """Map the influence kernel at a single lag -- where the arriving air was
    ``lag`` hours before it reached the receptor.

    Membership is defined at the ARRIVAL hour: a parcel belongs to the receptor
    if it is inside the red cell box then. With ``day`` given, the members'
    HYSPLIT trajectories are drawn with dots at arrival (inside the box, by
    construction) and at ``lag`` hours before arrival -- filled when the parcel
    is in PBL contact then (``contact_weight > 0`` under ``pbl_model``, default
    climatological -- the same gate the deposit uses), hollow when it is aloft
    and contributes nothing to the kernel, so every trajectory shows its
    position at the plotted lag. The kernel's mass regions enclosing
    ``hdr_fracs`` of the lag's mass (solid = first fraction, dashed = second;
    smallest centroid-outward circle of cells) are outlined as closed
    staircase polygons along the actual cell boundaries.
    """
    k = ds["kernel"].sel(lag=lag, method="nearest")
    lag_h = float(k["lag"])
    fig, ax = plt.subplots(figsize=(8, 6))
    _land_outline(ax)
    _contour_lag_kernel(ax, ds, k.values, "darkmagenta", hdr_fracs)
    hist = None
    if day is not None:
        hist = _overlay_member_trajectories(ax, day, ds, max_trajectories)
        if hist is not None:
            ax.scatter(hist["lon"][:, -1], hist["lat"][:, -1], s=16, c="darkorange",
                       edgecolor="k", linewidth=0.3, zorder=4,
                       label="parcels at arrival (define membership)")
            plat, plon, in_pbl = _contact_at_lag(
                hist, lag_h, pbl_model or ClimatologicalPBL(),
                float(ds.attrs.get("contact_fraction", config.CONTACT_FRACTION)))
            ax.scatter(plon[in_pbl], plat[in_pbl], s=14, c="deepskyblue",
                       edgecolor="k", linewidth=0.3, zorder=4,
                       label=f"parcels at lag {lag_h:.0f} h in PBL contact "
                             f"({int(in_pbl.sum())}/{in_pbl.size})")
            aloft = np.isfinite(plat) & np.isfinite(plon) & ~in_pbl
            if aloft.any():
                ax.scatter(plon[aloft], plat[aloft], s=14, facecolor="none",
                           edgecolor="deepskyblue", linewidth=0.8, zorder=4,
                           label=f"parcels at lag {lag_h:.0f} h aloft "
                                 f"(no deposit, {int(aloft.sum())}/{in_pbl.size})")
    _receptor_cell_box(ax, ds)
    _receptor_marker(ax, ds)
    _zoom_to_content(ax, ds, hist)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="darkmagenta", ls=s, label=f"kernel {f:.0%} mass")
                for f, s in zip(hdr_fracs, ("-", "--", ":"))]
    handles.append(mpatches.Patch(fill=False, edgecolor="red", label="receptor cell"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    return _finish(fig, ax, f"Kernel at lag {lag_h:.0f} h "
                            f"(receptor {ds.attrs['target_lat']:.1f}, {ds.attrs['target_lon']:.1f})",
                   save_path)


def plot_kernel_evolution(
    ds: xr.Dataset,
    day: Optional[xr.Dataset] = None,
    max_trajectories: int = 150,
    hdr_fracs: Sequence[float] = (0.9,),
    pbl_model=None,
    save_path: Optional[Path] = None,
):
    """The full backward evolution on one map: per-lag kernel outlines stepping
    upstream from the receptor cell, darker = further back in time.

    For each populated lag hour the kernel's mass regions (``hdr_fracs`` of
    that lag's mass; closed staircase polygons along the actual cell
    boundaries) are drawn in one light-to-dark sequential ramp, with dots at
    EVERY member parcel's position at that lag: filled when the parcel is in
    PBL contact then (the deposit's own gate), hollow when it is aloft and
    contributes nothing to the kernel (the legend counts the in-contact ones)
    -- so the dots always run the full length of the drawn trajectories. With
    ``day`` given, the members' trajectories run underneath in grey, so the
    outlines can be checked against the raw paths.
    """
    lags = ds["lag"].values
    populated = [float(l) for l in lags
                 if np.nansum(ds["footprint"].sel(lag=l).values) > 0]
    cmap = plt.get_cmap("Blues")
    deepest = max(populated[-1], 1.0) if populated else 1.0
    shade = {l: cmap(0.35 + 0.6 * (l / deepest)) for l in populated}

    fig, ax = plt.subplots(figsize=(9, 7))
    _land_outline(ax)
    hist = None
    if day is not None:
        hist = _member_history(day, ds, max_trajectories)
    if hist is not None:
        for i in range(hist["lat"].shape[0]):
            ax.plot(hist["lon"][i], hist["lat"][i], "-", color="0.45",
                    lw=0.6, alpha=0.5, zorder=1)
        ax.scatter(hist["lon"][:, 0], hist["lat"][:, 0], c="k", s=10, zorder=3,
                   edgecolor="w", linewidth=0.4)

    contact_fraction = float(ds.attrs.get("contact_fraction", config.CONTACT_FRACTION))
    pbl_model = pbl_model or ClimatologicalPBL()
    slat = ds["source_lat"].values
    cell = float(slat[1] - slat[0]) if slat.size > 1 else config.SOURCE_STEP_DEG
    handles = []
    for rank, l in enumerate(populated):
        # nest each lag's outline a little further inward (scaled to the source
        # cell size): consecutive lags often claim the same cells, and
        # coincident boundaries would otherwise overdraw each other
        drawn = _contour_lag_kernel(ax, ds, ds["kernel"].sel(lag=l).values,
                                    shade[l], hdr_fracs, inset=0.035 * cell * rank)
        label = f"lag {l:.0f} h"
        if hist is not None:
            plat, plon, in_pbl = _contact_at_lag(hist, l, pbl_model, contact_fraction)
            ax.scatter(plon[in_pbl], plat[in_pbl], s=16, color=shade[l],
                       edgecolor="k", linewidth=0.3, zorder=4)
            aloft = np.isfinite(plat) & np.isfinite(plon) & ~in_pbl
            ax.scatter(plon[aloft], plat[aloft], s=16, facecolor="none",
                       edgecolor=shade[l], linewidth=0.9, zorder=4)
            label += f" ({int(in_pbl.sum())}/{in_pbl.size} in PBL)"
        if drawn:
            handles.append(Line2D([], [], color=shade[l], marker="o", ms=5,
                                  label=label))
    _receptor_cell_box(ax, ds)
    _receptor_marker(ax, ds)
    _zoom_to_content(ax, ds, hist)

    if hist is not None:
        handles.insert(0, Line2D([], [], color="0.45", lw=1,
                                 label=f"member trajectories "
                                       f"({hist['n_shown']}/{hist['n_total']})"))
        handles.insert(1, Line2D([], [], color="k", marker="o", ms=4, ls="none",
                                 label="release"))
        handles.append(Line2D([], [], color="0.3", marker="o", ms=5, ls="none",
                              markerfacecolor="none",
                              label="hollow = aloft (no deposit)"))
    handles += [Line2D([], [], color="0.2", ls=s, label=f"{f:.0%} of lag mass")
                for f, s in zip(hdr_fracs, ("-", "--", ":"))]
    handles.append(Line2D([], [], color="red", marker="*", ms=12, ls="none",
                          markeredgecolor="k", label="receptor"))
    handles.append(mpatches.Patch(fill=False, edgecolor="red", label="receptor cell"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    return _finish(fig, ax,
                   "Kernel evolution: where the arriving air was, hour by hour "
                   f"(receptor {ds.attrs['target_lat']:.1f}, {ds.attrs['target_lon']:.1f}, "
                   f"arrival step {ds.attrs.get('arrival_step', '?')})",
                   save_path)


def plot_spatial_influence(
    ds: xr.Dataset,
    day: Optional[xr.Dataset] = None,
    max_trajectories: int = 150,
    save_path: Optional[Path] = None,
):
    """Lag-integrated source map (spatial marginal). Computed from the physical
    ``footprint`` (contact-hour weighting across lags), since the kernel is
    normalized per lag hour and no longer carries hour-to-hour weights.

    With ``day`` given, the arriving parcels' HYSPLIT trajectories are overlaid.
    """
    fp = ds["footprint"]
    spatial = fp.sum(dim="lag") / float(fp.sum())
    fig, ax = plt.subplots(figsize=(8, 6))
    _land_outline(ax)
    pcm = ax.pcolormesh(ds["source_lon"], ds["source_lat"], spatial.values,
                        cmap="magma_r", shading="nearest")
    fig.colorbar(pcm, ax=ax, label="lag-integrated influence (contact-hour weighted)")
    if day is not None:
        _overlay_member_trajectories(ax, day, ds, max_trajectories)
    _receptor_marker(ax, ds)
    _zoom_to_source_window(ax, ds)
    ax.legend(loc="upper right")
    return _finish(fig, ax, "Spatial influence (all lags) "
                            f"[{ds.attrs.get('n_parcels', '?')} parcels]", save_path)


def plot_temporal_influence(ds: xr.Dataset, save_path: Optional[Path] = None):
    """Influence vs lag (temporal marginal): how much of the arriving air's land
    contact came from each hour back. Computed from the physical ``footprint``
    (the per-lag-hour-normalized kernel is flat by construction)."""
    fp = ds["footprint"]
    temporal = fp.sum(dim=("source_lat", "source_lon")) / float(fp.sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ds["lag"].values, temporal.values, width=0.7, color="darkslateblue")
    ax.set_xlabel("lag (hours before arrival)")
    ax.set_ylabel("fraction of land-contact influence")
    ax.set_title(f"Temporal influence (receptor {ds.attrs['target_lat']:.1f}, "
                 f"{ds.attrs['target_lon']:.1f}, arrival step {ds.attrs.get('arrival_step','?')})")
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, ax
