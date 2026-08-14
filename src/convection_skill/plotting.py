from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .gini import detection_cdf, exceedance_flags, gini_from_cdf


def _label_for_percentile(p: float) -> str:
    """Pretty QPE-threshold label, e.g. 99.95 -> 'QPE$_{99.95}$'."""
    text = f"{p:g}"
    return f"QPE$_{{{text}}}$"


def plot_detection_cdf(
    ax,
    predictor: np.ndarray,
    flags: np.ndarray,
    label: str,
    rng: Optional[np.random.Generator] = None,
    **plot_kwargs,
) -> float:
    """Plot one event-capture curve on ``ax`` and return its Gini.

    The x-axis is the cumulative fraction of samples (predictor ascending); the
    y-axis is the cumulative fraction of events captured. The 1:1 diagonal (drawn
    once by the caller) is the uninformative reference.
    """
    x, y = detection_cdf(predictor, flags, rng=rng)
    g = gini_from_cdf(x, y)
    ax.plot(x, y, label=f"{label} (Gini={g:.3f})", **plot_kwargs)
    return g


def add_diagonal(ax) -> None:
    """Draw the uninformative 1:1 reference line."""
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=0)


def fig2a_single_threshold(
    ax,
    predictor: np.ndarray,
    target: np.ndarray,
    percentile: float = config.HEADLINE_PERCENTILE,
    rng: Optional[np.random.Generator] = None,
    threshold: Optional[float] = None,
) -> float:
    """Fig. 2a: the Gini derivation for one QPE threshold.

    Also marks the CAPE90 point to reproduce the paper's "80% of QPE99.95 events
    occur for CAPE > CAPE90" observation. ``threshold`` optionally supplies the
    absolute QPE cut from a broader base sample (paper: "thresholds are based on
    all data"); otherwise the percentile of ``target`` itself is used.
    """
    flags = (target > threshold) if threshold is not None \
        else exceedance_flags(target, percentile)
    add_diagonal(ax)
    g = plot_detection_cdf(ax, predictor, flags, _label_for_percentile(percentile),
                           rng=rng, color="C0", lw=2)
    # Fraction of events captured below the 90th predictor percentile.
    x, y = detection_cdf(predictor, flags, rng=rng)
    captured_above_p90 = 1.0 - float(np.interp(0.90, x, y))
    ax.axvline(0.90, color="C3", lw=1, ls=":")
    ax.annotate(
        f"{captured_above_p90*100:.0f}% of events\nat CAPE > CAPE$_{{90}}$",
        xy=(0.90, np.interp(0.90, x, y)), xytext=(0.45, 0.55),
        arrowprops=dict(arrowstyle="->", color="C3"), fontsize=9, color="C3",
    )
    ax.set_xlabel("cumulative fraction of samples (predictor ascending)")
    ax.set_ylabel("cumulative fraction of events")
    ax.set_title(f"Detection CDF, {_label_for_percentile(percentile)}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return g


def fig2c_multiple_thresholds(
    ax,
    predictor: np.ndarray,
    target: np.ndarray,
    percentiles: Sequence[float] = config.QPE_PERCENTILES,
    rng: Optional[np.random.Generator] = None,
    thresholds: Optional[dict] = None,
) -> dict[float, float]:
    """Fig. 2c: event-capture CDFs for several QPE thresholds on one predictor.

    ``thresholds`` optionally maps percentile -> absolute QPE cut from a broader
    base sample; otherwise percentiles of ``target`` itself are used.
    """
    add_diagonal(ax)
    ginis: dict[float, float] = {}
    for i, p in enumerate(percentiles):
        flags = (target > thresholds[p]) if thresholds is not None \
            else exceedance_flags(target, p)
        ginis[p] = plot_detection_cdf(ax, predictor, flags, _label_for_percentile(p),
                                      rng=rng, color=f"C{i}", lw=1.8)
    ax.set_xlabel("cumulative fraction of samples (predictor ascending)")
    ax.set_ylabel("cumulative fraction of events")
    ax.set_title("Detection CDFs by event rarity")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ginis


def plot_gini_vs_percentile(ax, results, labels: Optional[dict[str, str]] = None) -> None:
    """Fig. 2b: plot a *pre-computed* Gini-vs-percentile table.

    ``results`` is the DataFrame from
    :func:`convection_skill.analysis.gini_by_percentile` -- indexed by percentile
    with one column per predictor. ``labels`` optionally maps column names to
    prettier legend labels. Computation lives in ``analysis``; this only draws.
    """
    labels = labels or {}
    for i, col in enumerate(results.columns):
        ax.plot(results.index, results[col], "-o", color=f"C{i}",
                label=labels.get(col, col), lw=1.8)
    ax.set_xlabel("QPE threshold percentile")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Skill vs event rarity")
    ax.legend(loc="best", fontsize=8)


def plot_gini_vs_hour(ax, hourly_by_predictor: dict, percentile: float) -> None:
    """Fig. 3 summary: plot pre-computed per-hour Gini (with SE) for each predictor.

    ``hourly_by_predictor`` maps a display label to the DataFrame returned by
    :func:`convection_skill.analysis.hourly_gini` /
    :func:`convection_skill.analysis.hourly_significance` (columns step, hour_utc,
    gini, and optionally se). Computation lives in ``analysis``; this only draws.
    """
    for i, (name, d) in enumerate(hourly_by_predictor.items()):
        d = d.sort_values("step")
        yerr = d["se"] if "se" in d.columns else None
        ax.errorbar(d["step"], d["gini"], yerr=yerr, marker="o", capsize=3,
                    color=f"C{i}", label=name)
    steps = list(range(len(config.FORECAST_HOURS_UTC)))
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{h:02d}" for h in config.FORECAST_HOURS_UTC])
    ax.set_xlabel("forecast hour (UTC)")
    ax.set_ylabel(f"Gini coefficient (QPE$_{{{percentile:g}}}$)")
    ax.set_title("Detection skill vs forecast hour")
    ax.legend(loc="best")


def fig3_hourly_cdfs(
    ax,
    table,
    predictor_col: str,
    target_col: str = "qpe",
    percentile: float = config.HEADLINE_PERCENTILE,
    rng_seed: int = config.RANDOM_SEED,
    threshold: Optional[float] = None,
) -> dict[int, float]:
    """Fig. 3-style: one event-capture CDF per forecast hour, Gini in the legend.

    The QPE threshold is computed once on the pooled sample (all hours), then
    applied within each hour -- matching the paper's pooled-threshold rule.
    ``threshold`` optionally supplies the absolute cut from a broader base
    sample. Hours are ordered by forecast step (21,22,23,0,1,2 UTC).
    """
    add_diagonal(ax)
    threshold_flags = (table[target_col].to_numpy() > threshold) if threshold is not None \
        else exceedance_flags(table[target_col].to_numpy(), percentile)
    table = table.assign(_event=threshold_flags)
    ginis: dict[int, float] = {}
    hour_order = list(config.FORECAST_HOURS_UTC)
    for i, hour in enumerate(hour_order):
        sub = table[table["hour_utc"] == hour]
        if sub.empty:
            continue
        g = plot_detection_cdf(
            ax, sub[predictor_col].to_numpy(), sub["_event"].to_numpy(),
            f"{hour:02d} UTC", rng=np.random.default_rng(rng_seed),
            color=f"C{i}", lw=1.6,
        )
        ginis[hour] = g
    ax.set_xlabel("cumulative fraction of samples (predictor ascending)")
    ax.set_ylabel("cumulative fraction of events")
    ax.set_title(f"{_label_for_percentile(percentile)} detection by forecast hour")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ginis


# --------------------------------------------------------------------------- #
# CONUS map plotting (cartopy)
# --------------------------------------------------------------------------- #
# Cartopy is imported lazily so the rest of this module (the Gini/CDF plots) still
# imports in environments without it; only the map helpers below need it.
#
# Natural Earth "50m" is the right resolution for a continental map -- 110m is too
# coarse for clean state lines, 10m is needless weight -- and is the scale cached
# under ~/.local/share/cartopy, so the outlines render without a network fetch.
_NE_SCALE = "50m"


def _require_cartopy():
    """Import cartopy, or raise a message that says how to get it."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.geoaxes import GeoAxes
    except ImportError as exc:  # pragma: no cover - exercised only where cartopy absent
        raise ImportError(
            "The CONUS map helpers (make_conus_axes / plot_field_map) need cartopy. "
            "Install it with `conda install -c conda-forge cartopy` or `pip install cartopy`."
        ) from exc
    return ccrs, cfeature, GeoAxes


def conus_projection():
    """The standard CONUS Lambert Conformal Conic projection.

    Parallels 33/45N, central meridian 96W, central latitude 39N -- the same
    conic used by the NWS / HRRR for the continental U.S., so the domain looks
    like the maps in the paper rather than a stretched lat-lon rectangle.
    """
    ccrs, _, _ = _require_cartopy()
    return ccrs.LambertConformal(
        central_longitude=-96.0, central_latitude=39.0, standard_parallels=(33.0, 45.0)
    )


def _add_conus_features(ax, ocean: bool = True) -> None:
    """Draw land/ocean fill plus coastline, state/province and country outlines."""
    ccrs, cfeature, _ = _require_cartopy()
    ne = cfeature.NaturalEarthFeature
    if ocean:
        ax.add_feature(ne("physical", "land", _NE_SCALE), facecolor="#f5f4ef", zorder=0)
        ax.add_feature(ne("physical", "ocean", _NE_SCALE), facecolor="#dce9f5", zorder=0)
    ax.add_feature(
        ne("cultural", "admin_1_states_provinces_lakes", _NE_SCALE),
        edgecolor="0.55", facecolor="none", linewidth=0.5, zorder=4,
    )
    ax.add_feature(
        ne("cultural", "admin_0_boundary_lines_land", _NE_SCALE),
        edgecolor="0.2", facecolor="none", linewidth=0.9, zorder=4,
    )
    ax.add_feature(
        ne("physical", "coastline", _NE_SCALE),
        edgecolor="0.2", facecolor="none", linewidth=0.9, zorder=4,
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="0.8", alpha=0.6, linestyle=":", zorder=5
    )
    gl.top_labels = gl.right_labels = False
    gl.x_inline = gl.y_inline = False  # keep labels on the frame, not over the data


def make_conus_axes(
    fig=None,
    rect=111,
    extent: Optional[tuple[float, float, float, float]] = None,
    figsize: tuple[float, float] = (9.0, 6.0),
    ocean: bool = True,
):
    """Create a CONUS-projected GeoAxes with coastline, state and country outlines.

    Use this to build the axes that :func:`plot_field_map` draws onto (or just pass
    ``ax=None`` to that function and it calls this for you).

    Parameters
    ----------
    fig
        Figure to add the axes to; a new one (``figsize``) is created if None.
    rect
        Subplot spec passed to ``fig.add_subplot`` (e.g. ``111`` or ``(2, 2, 3)``).
    extent
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees; defaults to the whole
        analysis domain (``config.DOMAIN_LON`` x ``config.DOMAIN_LAT``).
    figsize
        Size of the new figure when ``fig`` is None.
    ocean
        Fill land/ocean for context (default True); set False for a bare frame.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
        The projected axes, with features already added.
    """
    ccrs, _, _ = _require_cartopy()
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=figsize)
    subplot = rect if isinstance(rect, tuple) else (rect,)
    ax = fig.add_subplot(*subplot, projection=conus_projection())
    if extent is None:
        lon_lo, lon_hi = config.DOMAIN_LON
        lat_lo, lat_hi = config.DOMAIN_LAT
        extent = (lon_lo, lon_hi, lat_lo, lat_hi)
    ax.set_extent(tuple(extent), crs=ccrs.PlateCarree())
    _add_conus_features(ax, ocean=ocean)
    return ax


def _panel_layout(n: int, ncols: Optional[int] = None) -> tuple[int, int]:
    """A (rows, cols) grid for ``n`` CONUS panels; ``ncols`` overrides the default.

    Up to 3 panels go in a single row (landscape maps tile better wide than
    stacked); beyond that it falls back to near-square: 4->2x2, 5/6->2x3, 7->3x3.
    """
    if ncols is None:
        ncols = n if n <= 3 else int(np.ceil(np.sqrt(n)))
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _resolve_per_field(value, fields: Sequence[str], default=None) -> dict:
    """Broadcast a scalar / per-field dict / per-field list to ``{field: value}``.

    Lets ``cmap``/``vmin``/``vmax`` be given once for all panels, or per field.
    """
    if isinstance(value, dict):
        return {f: value.get(f, default) for f in fields}
    if isinstance(value, (list, tuple)):
        if len(value) != len(fields):
            raise ValueError(
                f"expected {len(fields)} values (one per field), got {len(value)}"
            )
        return dict(zip(fields, value))
    return {f: value for f in fields}


def _require_fields(table, fields: Sequence[str]) -> None:
    """Fail early, with the available columns, if a requested field is absent."""
    missing = [f for f in fields if f not in table.columns]
    if missing:
        raise KeyError(
            f"field(s) {missing} not in the table. Available columns: "
            f"{sorted(table.columns)}"
        )


def _field_grid(
    table,
    field: str,
    day,
    hour_utc: int,
    lat_range,
    lon_range,
    lats=None,
    lons=None,
    strict: bool = True,
):
    """Pivot one field/day/hour to a lat x lon grid, optionally on fixed geometry.

    When ``lats``/``lons`` are given the grid is reindexed onto them (missing cells
    -> NaN), so successive hours share an identical mesh -- what animation needs to
    update in place. ``strict`` raises the informative "no rows" error used by the
    interactive maps; the animator passes ``strict=False`` to allow blank frames.
    """
    sel = table[
        (table["date"] == day)
        & (table["hour_utc"] == hour_utc)
        & table["lat"].between(*lat_range)
        & table["lon"].between(*lon_range)
    ]
    if sel.empty:
        if strict:
            available_hours = sorted(table["hour_utc"].unique())
            raise ValueError(
                f"No rows for date={pd.Timestamp(day).date()} hour_utc={hour_utc} in "
                f"lat {lat_range}, lon {lon_range}. "
                f"Hours present in table: {available_hours}."
            )
        return pd.DataFrame(
            np.nan, index=pd.Index(lats, name="lat"), columns=pd.Index(lons, name="lon")
        )
    # .pivot (not pivot_table) keeps a fully-missing field (e.g. SMAP on a
    # no-retrieval hour) as an all-NaN grid rather than dropping the column; cells
    # are unique per (date, hour, lat, lon), so no aggregation is needed.
    grid = sel.pivot(index="lat", columns="lon", values=field)
    if lats is not None and lons is not None:
        grid = grid.reindex(index=lats, columns=lons)
    return grid


def _draw_field_mesh(ax, grid, cmap, vmin, vmax):
    """pcolormesh one lat x lon grid onto a CONUS GeoAxes (data layer only)."""
    ccrs, _, _ = _require_cartopy()
    return ax.pcolormesh(
        grid.columns, grid.index, np.ma.masked_invalid(grid.values),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
        transform=ccrs.PlateCarree(), zorder=2,
    )


def plot_field_map(
    ax,
    table,
    field,
    date,
    hour_utc: int,
    lat_range: tuple[float, float] = config.DOMAIN_LAT,
    lon_range: tuple[float, float] = config.DOMAIN_LON,
    cmap="viridis",
    vmin=None,
    vmax=None,
    add_colorbar: bool = True,
    ncols: Optional[int] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Map one or more fields from the tidy table for a single day and forecast hour.

    Works for any column of the analysis/base tables (``qpe``, ``mu_cape``,
    ``smap_smsfc``, ...) -- a Fig.-1-style snapshot -- drawn on a proper CONUS
    Lambert Conformal projection with coastline, state/province and country
    outlines (see :func:`make_conus_axes`). Cells absent from the table (or NaN)
    render transparent, so the underlying map shows through and the plot doubles
    as a picture of the sample's spatial coverage.

    Pass a single ``field`` (str) to draw one map onto ``ax`` (returning its mesh),
    or a sequence of fields to lay them out as panels of one new figure (returning
    the figure); in the multi-field case ``ax`` must be None, since a fresh panel
    grid is created.

    Parameters
    ----------
    ax
        A cartopy GeoAxes to draw on (typically from :func:`make_conus_axes`). Pass
        ``None`` to have one built automatically. Must be None for multiple fields.
    table
        Tidy table with ``date``, ``hour_utc``, ``lat``, ``lon`` and the field
        column(s) (from ``data_loading.build_analysis_table`` or any QC'd subset).
    field
        Column name, or a sequence of column names for a multi-panel figure.
    date
        Day to plot; anything ``pandas.Timestamp`` accepts (e.g. "2020-07-26").
    hour_utc
        Forecast hour, one of ``config.FORECAST_HOURS_UTC`` (21,22,23,0,1,2).
        Note the 00-02 UTC slots of a given ``date`` are that evening's forecast
        hours (early UTC hours of the *following* calendar day).
    lat_range, lon_range
        Bounding box (inclusive, degrees N / degrees E); defaults to the whole
        analysis region, 32-53N, 107-64W. Sets the map extent.
    cmap, vmin, vmax
        Passed to ``pcolormesh``. With multiple fields each may be a single value
        (shared) or a ``{field: value}`` dict / per-field list, since fields like
        ``qpe`` and ``mu_cape`` want different scales.
    add_colorbar
        Attach a colorbar labelled with the field name (default True).
    ncols, figsize
        Multi-field only: panel-grid column count (default near-square) and figure
        size (default scales with the layout).

    Returns
    -------
    matplotlib.collections.QuadMesh (single field) or matplotlib.figure.Figure
    (multiple fields).
    """
    if not isinstance(field, str):
        return _plot_field_panels(
            list(field), table, date, hour_utc, lat_range, lon_range,
            cmap, vmin, vmax, add_colorbar, ncols, figsize, ax=ax,
        )

    _, _, GeoAxes = _require_cartopy()
    _require_fields(table, [field])
    day = pd.Timestamp(date)
    grid = _field_grid(table, field, day, hour_utc, lat_range, lon_range, strict=True)

    if ax is None:
        ax = make_conus_axes(
            extent=(lon_range[0], lon_range[1], lat_range[0], lat_range[1])
        )
    elif not isinstance(ax, GeoAxes):
        raise TypeError(
            "plot_field_map needs a cartopy GeoAxes; pass ax=None to build one "
            "automatically, or create it with make_conus_axes()."
        )

    mesh = _draw_field_mesh(ax, grid, cmap, vmin, vmax)
    ax.set_title(f"{field}  {day.date()}  {hour_utc:02d} UTC")
    if add_colorbar:
        ax.figure.colorbar(mesh, ax=ax, shrink=0.85, pad=0.03, label=field)
    return mesh


def _plot_field_panels(
    fields,
    table,
    date,
    hour_utc,
    lat_range,
    lon_range,
    cmap,
    vmin,
    vmax,
    add_colorbar,
    ncols,
    figsize,
    ax=None,
):
    """Draw several fields as CONUS panels of one figure (see :func:`plot_field_map`)."""
    if ax is not None:
        raise TypeError(
            "pass ax=None when plotting multiple fields; a panel figure is created."
        )
    if not fields:
        raise ValueError("field list is empty")
    _require_cartopy()
    _require_fields(table, fields)
    import matplotlib.pyplot as plt

    day = pd.Timestamp(date)
    cmaps = _resolve_per_field(cmap, fields, default="viridis")
    vmins = _resolve_per_field(vmin, fields)
    vmaxs = _resolve_per_field(vmax, fields)

    nrows, ncols = _panel_layout(len(fields), ncols)
    figsize = figsize or (ncols * 5.5, nrows * 3.7)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    extent = (lon_range[0], lon_range[1], lat_range[0], lat_range[1])
    for i, f in enumerate(fields):
        ax_i = make_conus_axes(fig=fig, rect=(nrows, ncols, i + 1), extent=extent)
        grid = _field_grid(table, f, day, hour_utc, lat_range, lon_range, strict=True)
        mesh = _draw_field_mesh(ax_i, grid, cmaps[f], vmins[f], vmaxs[f])
        ax_i.set_title(f)
        if add_colorbar:
            fig.colorbar(mesh, ax=ax_i, shrink=0.85, pad=0.03, label=f)
    fig.suptitle(f"{day.date()}  {hour_utc:02d} UTC", fontsize=13)
    return fig


def animate_field_map(
    table,
    field,
    date,
    hours: Sequence[int] = config.FORECAST_HOURS_UTC,
    save_path=None,
    fps: float = 2.0,
    lat_range: tuple[float, float] = config.DOMAIN_LAT,
    lon_range: tuple[float, float] = config.DOMAIN_LON,
    cmap="viridis",
    vmin=None,
    vmax=None,
    ncols: Optional[int] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Animate one or more fields across forecast hours; optionally save a GIF.

    Builds the same CONUS panel layout as :func:`plot_field_map` (one panel per
    field) and steps ``hours`` as animation frames -- a loop of the evening's
    21->02 UTC forecast, say. The colour scale is held *fixed* across frames (per
    field, from the 1st-99th percentile over all frames unless ``vmin``/``vmax``
    are given) so brightness changes mean real changes, not a moving scale. All
    frames share one mesh geometry (the union of cells present that day), updated
    in place, so missing cell-hours simply blink out.

    Parameters
    ----------
    table, field, date, lat_range, lon_range, cmap, vmin, vmax, ncols, figsize
        As in :func:`plot_field_map` (``field`` may be a str or a sequence).
    hours
        Forecast hours to step through, in order (default 21,22,23,0,1,2 UTC).
    save_path
        If given (e.g. ``"results/figures/qpe.gif"``), write a GIF there via
        Pillow. The parent directory must exist.
    fps
        Frames per second for the saved GIF.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation (keep a reference; in a notebook use ``HTML(anim.to_jshtml())``).
    """
    ccrs, _, _ = _require_cartopy()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fields = [field] if isinstance(field, str) else list(field)
    if not fields:
        raise ValueError("field list is empty")
    _require_fields(table, fields)
    hours = list(hours)
    if not hours:
        raise ValueError("hours is empty")
    day = pd.Timestamp(date)

    # Fixed mesh geometry: every cell present anywhere that day, in the box.
    box = table[
        (table["date"] == day)
        & table["lat"].between(*lat_range)
        & table["lon"].between(*lon_range)
        & table["hour_utc"].isin(hours)
    ]
    if box.empty:
        raise ValueError(
            f"No rows for date={day.date()} hours={hours} in "
            f"lat {lat_range}, lon {lon_range}."
        )
    lats = np.sort(box["lat"].unique())
    lons = np.sort(box["lon"].unique())

    # Per-field fixed colour scale across all frames.
    cmaps = _resolve_per_field(cmap, fields, default="viridis")
    vmins = _resolve_per_field(vmin, fields)
    vmaxs = _resolve_per_field(vmax, fields)
    for f in fields:
        if vmins[f] is None or vmaxs[f] is None:
            vals = box[f].to_numpy()
            finite = vals[np.isfinite(vals)]
            lo, hi = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) \
                if finite.size else (0.0, 1.0)
            if lo == hi:
                hi = lo + 1.0
            vmins[f] = lo if vmins[f] is None else vmins[f]
            vmaxs[f] = hi if vmaxs[f] is None else vmaxs[f]

    nrows, ncols = _panel_layout(len(fields), ncols)
    figsize = figsize or (ncols * 5.5, nrows * 3.9)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    extent = (lon_range[0], lon_range[1], lat_range[0], lat_range[1])

    meshes = []
    for i, f in enumerate(fields):
        ax_i = make_conus_axes(fig=fig, rect=(nrows, ncols, i + 1), extent=extent)
        grid = _field_grid(table, f, day, hours[0], lat_range, lon_range,
                           lats, lons, strict=False)
        mesh = _draw_field_mesh(ax_i, grid, cmaps[f], vmins[f], vmaxs[f])
        ax_i.set_title(f)
        fig.colorbar(mesh, ax=ax_i, shrink=0.85, pad=0.03, label=f)
        meshes.append((f, mesh))
    suptitle = fig.suptitle(f"{day.date()}  {hours[0]:02d} UTC", fontsize=13)

    def update(hour):
        for f, mesh in meshes:
            grid = _field_grid(table, f, day, hour, lat_range, lon_range,
                               lats, lons, strict=False)
            mesh.set_array(np.ma.masked_invalid(grid.values).ravel())
        suptitle.set_text(f"{day.date()}  {hour:02d} UTC")
        return [m for _, m in meshes] + [suptitle]

    anim = FuncAnimation(fig, update, frames=hours, blit=False)
    if save_path is not None:
        anim.save(str(save_path), writer=PillowWriter(fps=fps))
    return anim
