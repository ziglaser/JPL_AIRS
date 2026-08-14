from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


_SIGN = {True: "+", False: "-"}

def plot_convective_by_percentile(
    df: pd.DataFrame,
    save_path: Path,
    percentile_ticks: tuple = (90.0, 95.0, 99.0, 99.5, 99.9, 99.95),
) -> None:
    """2-D histograms: QPE percentile (log-tail axis) vs precip-mode shares.

    x = each row's percentile rank of cell-mean QPE in the pooled sample, on a
    -log10(1 - p) axis so the extreme tail is readable. Panels: convective
    (incl. hail, a convective precip mode) and stratiform shares of the
    PRECIPITATING sub-pixels (their residual is snow). Color = frequency,
    normalized WITHIN each
    percentile column; the line is the column mean, so "how convective is a
    P99.9 event" reads off directly. The convective panel is what the
    convective-event toggle thresholds.
    """
    from matplotlib.colors import LogNorm
    from scipy.stats import rankdata

    qpe = df["qpe"].to_numpy()
    rank_frac = rankdata(qpe) / (qpe.size + 1)   # percentile rank in (0, 1)
    x_all = -np.log10(1.0 - rank_frac)           # 90th -> 1, 99.9th -> 3, ...
    x_edges = np.linspace(1.0, 4.0, 46)          # percentiles 90 .. 99.99
    y_edges = np.linspace(0.0, 100.0, 26)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    panels = [
        ("convective_share", "% convective among precipitating sub-pixels"),
        ("stratiform_share", "% stratiform among precipitating sub-pixels"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (col, ylabel) in zip(axes, panels):
        y_all = df[col].to_numpy() * 100.0
        ok = np.isfinite(x_all) & np.isfinite(y_all)
        x, y = x_all[ok], y_all[ok]

        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        column_totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            column_freq = np.where(column_totals > 0, counts / column_totals, np.nan)

        in_range = (x >= x_edges[0]) & (x <= x_edges[-1])
        which_col = np.clip(np.digitize(x[in_range], x_edges) - 1, 0, x_centers.size - 1)
        col_sum = np.bincount(which_col, weights=y[in_range], minlength=x_centers.size)
        col_n = np.bincount(which_col, minlength=x_centers.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            col_mean = np.where(col_n > 0, col_sum / col_n, np.nan)

        pcm = ax.pcolormesh(x_edges, y_edges, column_freq.T, cmap="magma_r",
                            norm=LogNorm(vmin=1e-3, vmax=1.0))
        fig.colorbar(pcm, ax=ax, label="frequency within percentile column")
        ax.plot(x_centers, col_mean, color="deepskyblue", lw=2, label="column mean")
        tick_x = [-np.log10(1.0 - p / 100.0) for p in percentile_ticks]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{p:g}" for p in percentile_ticks])
        ax.set_xlabel("QPE percentile (pooled sample, log-tail axis)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left")
    fig.suptitle("How convective are heavy-QPE events?  (MRMS PrecipFlag)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


#: Convective-share thresholds (%) the heatmap animation sweeps
CONVECTIVE_THRESHOLDS_PCT: tuple = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0)


def animate_cape_qpe_convective_threshold(
    df: pd.DataFrame,
    save_path: Path,
    thresholds_pct: tuple = CONVECTIVE_THRESHOLDS_PCT,
    min_count: int = 20,
    cape_max: float = 4000.0,
    qpe_ticks: tuple = (80.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.95),
    fps: float = 1.0,
) -> None:
    """Animated 2-D heatmap: (raw CAPE x QPE percentile) -> % of cells whose
    convective-flag share exceeds a threshold, sweeping the threshold.

    x = raw AIRS-FCST MU CAPE (J/kg, linear bins; values above ``cape_max``
    clip into the last bin, labeled "+"); y = percentile rank of cell-mean QPE
    in the pooled sample on the -log10(1 - p) tail axis. Each frame colors the
    bins by the PERCENT OF CELLS with convective share (hail counted
    convective) above that frame's threshold; the color scale is fixed at
    0-100% across frames so the sweep is comparable. Bins with fewer than
    ``min_count`` rows are masked; the static right panel shows the bin
    counts. Writes an animated GIF.

    Reading it: at a loose threshold (1%) the map shows "any core at all";
    by 25% only strongly core-covered cells survive. How fast the high-CAPE /
    heavy-QPE corner fades with threshold is exactly the flag-dilution effect
    the convective-event toggle must reckon with.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm
    from scipy.stats import rankdata

    qpe = df["qpe"].to_numpy(dtype=float)
    cape = df["mu_cape"].to_numpy(dtype=float)
    share = df["convective_share"].to_numpy(dtype=float) * 100.0

    # QPE percentile rank among the FINITE rows of the pooled sample (dry rows
    # included), matching the suite's threshold convention
    y_all = np.full(qpe.shape, np.nan)
    finite_qpe = np.isfinite(qpe)
    y_all[finite_qpe] = -np.log10(
        1.0 - rankdata(qpe[finite_qpe]) / (finite_qpe.sum() + 1))

    x_edges = np.linspace(0.0, cape_max, 33)
    y_edges = np.linspace(0.7, 4.0, 34)  # QPE percentiles ~80 .. 99.99
    ok = (np.isfinite(cape) & np.isfinite(share)
          & (y_all >= y_edges[0]) & (y_all <= y_edges[-1]))
    x = np.minimum(cape[ok], cape_max - 1e-6)  # clip the tail into the last bin
    y, s = y_all[ok], share[ok]

    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    denominator = np.where(counts >= min_count, counts, np.nan)

    def pct_over(threshold_pct: float) -> np.ndarray:
        hits, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges],
                                    weights=(s > threshold_pct).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return hits / denominator * 100.0

    fig, (ax, ax_n) = plt.subplots(
        1, 2, figsize=(13, 5.2), sharey=True,
        gridspec_kw={"width_ratios": [1.6, 1.0]})
    pcm = ax.pcolormesh(x_edges, y_edges, pct_over(thresholds_pct[0]).T,
                        cmap="magma_r", vmin=0.0, vmax=100.0)
    fig.colorbar(pcm, ax=ax, label="% of cells with convective share > threshold")
    pcm_n = ax_n.pcolormesh(x_edges, y_edges,
                            np.where(counts > 0, counts, np.nan).T,
                            cmap="Greys", norm=LogNorm())
    fig.colorbar(pcm_n, ax=ax_n, label="rows per bin")

    tick_y = [-np.log10(1.0 - p / 100.0) for p in qpe_ticks]
    for a, title in ((ax, ""), (ax_n, "sample support")):
        a.set_yticks(tick_y)
        a.set_yticklabels([f"{p:g}" for p in qpe_ticks])
        a.set_xlabel("AIRS-FCST MU CAPE (J/kg)")
        a.set_title(title)
    last = ax.get_xticks()
    ax.set_xticklabels([f"{t:g}+" if t >= cape_max else f"{t:g}" for t in last])
    ax.set_ylabel("QPE percentile (pooled sample, log-tail axis)")
    title = ax.set_title(f"convective share > {thresholds_pct[0]:g}%  "
                         f"(bins with n >= {min_count})")
    fig.suptitle("Cells exceeding a convective-flag share, across CAPE x QPE "
                 "(hail counted convective)")

    def update(threshold_pct):
        pcm.set_array(pct_over(threshold_pct).T.ravel())
        title.set_text(f"convective share > {threshold_pct:g}%  "
                       f"(bins with n >= {min_count})")
        return pcm, title

    anim = FuncAnimation(fig, update, frames=thresholds_pct, blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)

def plot_high_cape_convective_fraction(
    df: pd.DataFrame,
    save_path: Path,
    thresholds_pct: tuple = CONVECTIVE_THRESHOLDS_PCT,
    cape_min: float = 500.0,
    qpe_thresholds: "dict[float, float] | None" = None,
    qpe_percentiles: tuple = (90.0, 95.0, 99.0, 99.5, 99.9, 99.95),
) -> None:
    """Line plot: % of high-CAPE cells flagged convective vs the share threshold.

    x = the convective-share threshold (%); y = among cells with AIRS-FCST MU
    CAPE > ``cape_min`` (J/kg) AND cell-mean QPE above a percentile threshold,
    the percent whose convective share (hail counted convective) exceeds x.
    One line per QPE percentile, so the flag-dilution effect reads off
    directly: how quickly "flagged convective" collapses as the threshold
    tightens, and whether heavier events resist the collapse. Legend carries
    each subset's mm/h cut and sample size.

    ``qpe_thresholds``: {percentile: mm/h} mapping, normally the BASE-sample
    replication ladder from ``convection_skill.dataset.qpe_percentile_thresholds``
    (the suite's "thresholds are based on all data" convention). If None, the
    percentiles are computed from ``df`` itself -- WARNING: on a rain-screened
    sample most rows are dry, so low percentiles collapse to 0 mm/h; prefer
    passing the base ladder.
    """
    qpe = df["qpe"].to_numpy(dtype=float)
    cape = df["mu_cape"].to_numpy(dtype=float)
    share = df["convective_share"].to_numpy(dtype=float) * 100.0
    high_cape = np.isfinite(cape) & (cape > cape_min) & np.isfinite(share)
    finite_qpe = qpe[np.isfinite(qpe)]
    if qpe_thresholds is not None:
        qpe_percentiles = tuple(sorted(qpe_thresholds))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(qpe_percentiles)))
    for p, color in zip(qpe_percentiles, colors):
        cut = (float(qpe_thresholds[p]) if qpe_thresholds is not None
               else float(np.nanpercentile(finite_qpe, p)))
        rows = high_cape & (qpe > cut)
        n = int(rows.sum())
        if n == 0:
            continue
        y = [100.0 * np.mean(share[rows] > t) for t in thresholds_pct]
        ax.plot(thresholds_pct, y, "-o", ms=4, color=color,
                label=f"QPE > P{p:g} ({cut:.1f} mm/h, n={n:,})")

    ax.set_xlabel("convective-share threshold (%)")
    ax.set_ylabel(f"% of cells with MU CAPE > {cape_min:g} J/kg flagged convective")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="QPE subset (pooled percentile)")
    ax.set_title("Flag dilution in high-CAPE environments "
                 "(hail counted convective)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
