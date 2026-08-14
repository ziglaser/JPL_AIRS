"""Comparison figures: CODSUS vs NOAA-XML front labels (1 deg, 2006-2018).

Colors follow the reference dataviz palette: sequential blue for magnitude,
blue<->red diverging with a neutral-gray midpoint for differences, fixed
per-entity categorical slots (front types / sources) everywhere.
"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, "src")
from front_finder import config, labels  # noqa: E402

OUT = Path("results/fronts/codsus_vs_noaa")
AGG = np.load(OUT / "aggregates.npz")
MONTHLY = pd.read_csv(OUT / "monthly_counts.csv")
TYPES = ("cold", "warm", "stationary", "occluded")
LATS = np.arange(10.0, 78.0)
LONS = np.arange(-171.0, -30.0)

# ---- palette (reference instance; see dataviz skill) ----------------------- #
TYPE_COLOR = {"cold": "#2a78d6", "warm": "#e34948", "stationary": "#008300",
              "occluded": "#4a3aa7", "dryline": "#eb6834"}
SRC_COLOR = {"CODSUS": "#2a78d6", "NOAA XML": "#eb6834"}
INK, INK2, MUTED, GRID_C, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                     "#e1e0d9", "#fcfcfb")
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
    "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", [
    "#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f2b8b7", "#e34948",
    "#8f2726"])
for cm in (SEQ, DIV):
    cm.set_bad("#f0efec")

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": GRID_C, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 150,
})

PC = ccrs.PlateCarree()


def geo_axes(ax, extent):
    ax.set_extent(extent, crs=PC)
    ax.coastlines("50m", lw=0.5, color=MUTED)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4,
                   edgecolor=MUTED, facecolor="none")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.4,
                   edgecolor=MUTED, facecolor="none")
    ax.spines["geo"].set_edgecolor(GRID_C)


with np.errstate(invalid="ignore", divide="ignore"):
    nv = AGG["n_valid"]
    pct = {("CODSUS", t): 100 * AGG[f"freq_codsus_{t}"] / np.where(nv, nv, np.nan)
           for t in TYPES}
    pct.update({("NOAA XML", t): 100 * AGG[f"freq_noaa_{t}"] / np.where(nv, nv, np.nan)
                for t in TYPES})

# --------------------------------------------------------------------------- #
# Figure 1: climatological frequency maps, per type x (CODSUS, NOAA, diff)
# --------------------------------------------------------------------------- #
EXTENT = (-130, -55, 20, 60)          # the WPC analysis envelope, roughly
fig, axes = plt.subplots(3, 4, figsize=(16, 7.6), layout="constrained",
                         subplot_kw={"projection": PC})
_la, _lo = (slice(*np.searchsorted(LATS, EXTENT[2:])),
            slice(*np.searchsorted(LONS, EXTENT[:2])))
vmax = max(np.nanpercentile(v[_la, _lo], 99.5) for v in pct.values())
dmax = max(np.nanpercentile(np.abs((pct[("NOAA XML", t)] - pct[("CODSUS", t)])[_la, _lo]),
                            99.5) for t in TYPES)
for j, t in enumerate(TYPES):
    for i, src in enumerate(("CODSUS", "NOAA XML")):
        ax = axes[i, j]
        geo_axes(ax, EXTENT)
        im = ax.pcolormesh(LONS, LATS, pct[(src, t)], cmap=SEQ, vmin=0,
                           vmax=vmax, transform=PC)
        if j == 0:
            ax.text(-0.08, 0.5, src, transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=11, color=INK)
        if i == 0:
            ax.set_title(t, fontsize=12, color=INK)
    ax = axes[2, j]
    geo_axes(ax, EXTENT)
    dm = ax.pcolormesh(LONS, LATS, pct[("NOAA XML", t)] - pct[("CODSUS", t)],
                       cmap=DIV, norm=TwoSlopeNorm(0, -dmax, dmax),
                       transform=PC)
    if j == 0:
        ax.text(-0.08, 0.5, "NOAA − CODSUS", transform=ax.transAxes,
                rotation=90, va="center", ha="center", fontsize=11, color=INK)
fig.suptitle("Front frequency, CODSUS vs NOAA XML analyses — 2006–2018, "
             "1° grid, same-hour joint-valid pixels", fontsize=14)
cb1 = fig.colorbar(im, ax=axes[:2, :], shrink=0.8, pad=0.01)
cb1.set_label("% of analyses with a front", color=INK2)
cb2 = fig.colorbar(dm, ax=axes[2, :], shrink=0.9, pad=0.01)
cb2.set_label("difference (pp)", color=INK2)
fig.savefig(OUT / "fig1_frequency_maps.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 2: the dryline (NOAA only -- what CODSUS lacks)
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 4.6))
DL_EXTENT = (-110, -93, 25, 42)
with np.errstate(invalid="ignore", divide="ignore"):
    panels = [("All year", 100 * AGG["freq_dryline"] / np.where(nv, nv, np.nan))]
    for s in ("MAM", "JJA"):
        nvs = AGG[f"n_valid_{s}"]
        panels.append((f"{s} only", 100 * AGG[f"freq_dryline_{s}"]
                       / np.where(nvs, nvs, np.nan)))
_dla, _dlo = (slice(*np.searchsorted(LATS, DL_EXTENT[2:])),
              slice(*np.searchsorted(LONS, DL_EXTENT[:2])))
vmax_dl = max(np.nanmax(f[_dla, _dlo]) for _, f in panels)
for k, (title, field) in enumerate(panels):
    ax = fig.add_subplot(1, 4, k + 1, projection=PC)
    geo_axes(ax, DL_EXTENT)
    im = ax.pcolormesh(LONS, LATS, field, cmap=SEQ, vmin=0, vmax=vmax_dl,
                       transform=PC)
    ax.set_title(title, fontsize=11, color=INK)
cb = fig.colorbar(im, ax=fig.axes, shrink=0.85, pad=0.01)
cb.set_label("% of analyses with a dryline", color=INK2)
ax = fig.add_subplot(1, 4, 4)
dl = MONTHLY[MONTHLY["type"] == "dryline"].groupby("month")["noaa"].sum()
ax.bar(dl.index, dl.values, color=TYPE_COLOR["dryline"], width=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_ylabel("dryline pixels, 2006–2018 total", color=INK2)
ax.set_title("Seasonality", fontsize=11, color=INK)
ax.grid(axis="y", color=GRID_C, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle("Drylines in the NOAA XML analyses — absent from CODSUS "
             "(2006–2018, 1° grid)", fontsize=13, y=1.04)
fig.savefig(OUT / "fig2_dryline.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 3: same-hour agreement (IoU) by year and type
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 4.6))
years = AGG["iou_years"]
for t in TYPES:
    iu = AGG[f"iou_{t}"]
    ax.plot(years, iu[:, 0] / iu[:, 1], color=TYPE_COLOR[t], lw=2,
            marker="o", ms=4, label=t)
iu = AGG["iou_any"]
ax.plot(years, iu[:, 0] / iu[:, 1], color=INK, lw=2, ls="--", marker="o",
        ms=4, label="any front")
ax.set_ylim(0, 0.7)
ax.set_ylabel("same-hour IoU (1° pixels)", color=INK2)
ax.set_title("CODSUS vs NOAA XML agreement is modest and steady — these are "
             "different analyst products", fontsize=12, loc="left")
ax.grid(color=GRID_C, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=5, loc="lower center",
          bbox_to_anchor=(0.5, -0.28))
fig.savefig(OUT / "fig3_agreement_iou.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 4: monthly climatology per type, CODSUS vs NOAA
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True)
month_lab = list("JFMAMJJASOND")
for k, t in enumerate((*TYPES, "dryline")):
    ax = axes[k]
    g = (MONTHLY[MONTHLY["type"] == t].groupby("month")
         [["codsus", "noaa", "valid"]].sum())
    if t != "dryline":
        ax.plot(g.index, 100 * g["codsus"] / g["valid"],
                color=SRC_COLOR["CODSUS"], lw=2, label="CODSUS")
    ax.plot(g.index, 100 * g["noaa"] / g["valid"],
            color=SRC_COLOR["NOAA XML"], lw=2, label="NOAA XML")
    ax.set_title(t + (" (NOAA only)" if t == "dryline" else ""),
                 fontsize=11, color=INK)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_lab, fontsize=8)
    ax.grid(color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("% of valid pixels", color=INK2)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Monthly front coverage by type — 2006–2018 climatology",
             fontsize=13, y=1.06)
fig.savefig(OUT / "fig4_seasonality.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 5: one synoptic case, side by side (max-dryline analysis)
# --------------------------------------------------------------------------- #
CASE = pd.Timestamp("2007-06-07 21:00")
CASE_EXTENT = (-130, -58, 24, 60)
fig, axes = plt.subplots(2, 1, figsize=(12, 10.5), layout="constrained",
                         subplot_kw={"projection": PC})
for ax, (src, loader, types) in zip(axes, (
        ("CODSUS", labels.load_codsus, TYPES),
        ("NOAA XML", labels.load_noaa, (*TYPES, "dryline")))):
    with loader(CASE.year) as ds:
        it = int(np.flatnonzero(pd.DatetimeIndex(ds["time"].values) == CASE)[0])
        ft = [str(s) for s in ds["front_type"].values]
        fr = ds["fronts"].values[it]
    geo_axes(ax, CASE_EXTENT)
    for t in types:
        m = np.ma.masked_where(fr[ft.index(t)] != 1,
                               np.ones(config.GRID_SHAPE))
        ax.pcolormesh(LONS, LATS, m, cmap=ListedColormap([TYPE_COLOR[t]]),
                      transform=PC)
    ax.set_title(src, fontsize=12, color=INK)
handles = [Patch(color=TYPE_COLOR[t], label=t)
           for t in (*TYPES, "dryline")]
fig.legend(handles=handles, frameon=False, ncol=5, loc="outside lower center")
fig.suptitle(f"Same analysis hour, two analyst products — {CASE:%Y-%m-%d %H} "
             "UTC (the strongest dryline analysis in the overlap)",
             fontsize=13)
fig.savefig(OUT / "fig5_case_20070607.png", bbox_inches="tight")
plt.close(fig)

print("wrote figures to", OUT)
