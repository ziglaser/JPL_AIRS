# %% [markdown]
# # Kernel exploration - three scientific questions (2019-06-05)
#
# - **Q1** ADVECTION: does the soil moisture the arriving air passed over differ
#   from the soil moisture under the cell right now? (If not, the trajectory
#   apparatus is redundant.)
# - **Q2** DIURNAL COUPLING: for near-surface (<1 km) air, how does land-surface
#   coupling vary with local time? (The 21-02 UTC window straddles the
#   afternoon mixed layer and its evening collapse.)
# - **Q3** A KERNEL BLOOMING: one receptor's kernel at each lag - watch the
#   footprint migrate upstream and fuzz out with age.
#
# Run headless with `PYTHONPATH=src python notebooks/07_kernel_exploration.py`.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_kernels import apply as A
from trajectory_kernels import config, contact, footprint as F
from trajectory_kernels import pbl as PBL
from trajectory_kernels import trajectories as T

FIG = config.RESULTS_DIR / "trajectory_kernels" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SLOT_HOUR = {1: 21, 2: 22, 3: 23, 4: 0, 5: 1, 6: 2}  # UTC


def smap_daily_mean() -> xr.DataArray:
    ds = xr.open_dataset(config.DATA_DIR / "FCST_SMAP_MRMS_2019.nc")
    i = int(np.where(ds["date"].values.astype("datetime64[D]") == np.datetime64("2019-06-05"))[0][0])
    return ds["SMAP_L4_smsfc_av"].isel(date=i).mean("L4_nhours", skipna=True)


# --------------------------------------------------------------------------- #
def q1_advection(day, smap) -> None:
    """Map upstream-minus-local SMAP influence at 23 UTC (step 3)."""
    kernels = F.build_all(day)
    upstream = A.apply_kernel(kernels, smap, which="kernel")  # (step, lat, lon)
    local = smap.values  # (lat, lon), same grid as target_lat/lon

    step = 3
    up = upstream.sel(arrival_step=step).values
    anom = up - local
    valid = np.isfinite(anom)
    print(f"[Q1] step {step}: {valid.sum()} receptors with kernels; "
          f"upstream-minus-local SMAP: mean {np.nanmean(anom):+.3f}, "
          f"std {np.nanstd(anom):.3f}, |anom|>0.05 at "
          f"{np.mean(np.abs(anom[valid]) > 0.05) * 100:.0f}% of them")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lon, lat = smap["lon"].values, smap["lat"].values
    for ax, field, title, cmap, kw in [
        (axes[0], local, "Local SMAP (under the cell)", "YlGnBu", dict(vmin=0, vmax=0.6)),
        (axes[1], up, "Upstream SMAP (what the air saw)", "YlGnBu", dict(vmin=0, vmax=0.6)),
        (axes[2], anom, "Upstream - Local (advective anomaly)", "RdBu",
         dict(vmin=-0.15, vmax=0.15)),
    ]:
        pcm = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="nearest", **kw)
        fig.colorbar(pcm, ax=ax, label="m3/m3", shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Q1  Does advection matter?  Air arriving over wetter (blue) / "
                 "drier (red) ground than lies beneath it  [23 UTC, 2019-06-05]",
                 fontsize=13)
    fig.savefig(FIG / "explore_advective_anomaly.png", dpi=130, bbox_inches="tight")
    return anom, valid


# --------------------------------------------------------------------------- #
def q2_diurnal_coupling(day) -> None:
    """Mean surface-contact weight of near-surface (<1 km) air vs local hour."""
    model = PBL.ClimatologicalPBL()
    lat = day["lat"].values.ravel()
    lon = day["lon"].values.ravel()
    alt = day["alt"].values.ravel()
    tutc = day["time_utc"].values.ravel()

    near = alt < 1000.0  # genuinely boundary-layer air
    lat, lon, alt, tutc = lat[near], lon[near], alt[near], tutc[near]
    pblh = model(lat, lon, tutc)
    w = contact.contact_weight(alt, pblh)
    local_h = PBL.local_solar_hour(lon, tutc)

    edges = np.arange(0, 25, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(local_h, edges) - 1
    mean_w = np.array([w[which == b].mean() if np.any(which == b) else np.nan
                       for b in range(len(centers))])
    n_pts = np.array([int(np.sum(which == b)) for b in range(len(centers))])
    mean_pblh = np.array([pblh[which == b].mean() if np.any(which == b) else np.nan
                          for b in range(len(centers))])

    print("[Q2] near-surface (<1km) air, mean contact weight by local hour:")
    for h, mw, n in zip(centers, mean_w, n_pts):
        if n > 50:
            print(f"     {int(h):02d}h local: contact {mw:.2f}  (n={n})")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    m = n_pts > 50
    ax1.plot(centers[m], mean_w[m], "o-", color="seagreen", lw=2, label="mean contact weight")
    ax1.fill_between(centers[m], 0, mean_w[m], color="seagreen", alpha=0.15)
    ax1.set_xlabel("local solar hour")
    ax1.set_ylabel("surface-contact weight of <1 km air", color="seagreen")
    ax1.set_ylim(0, 1.05)
    ax1.axvspan(19, 24, color="navy", alpha=0.06)
    ax1.text(21.3, 0.9, "evening\ncollapse", color="navy", ha="center", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(centers[m], mean_pblh[m], "s--", color="darkorange", alpha=0.7, label="mean PBL depth")
    ax2.set_ylabel("PBL depth (m)", color="darkorange")
    ax1.set_title("Q2  Near-surface air uncouples from the land as the PBL collapses "
                  "(2019-06-05 parcels)")
    fig.savefig(FIG / "explore_diurnal_coupling.png", dpi=130, bbox_inches="tight")


# --------------------------------------------------------------------------- #
def q3_kernel_bloom(day) -> None:
    """One receptor's kernel at each lag -- the footprint blooming upstream.

    Same well-populated Illinois receptor as the demo notebook, at the END of
    the forecast window (step 6 = 02 UTC) for the longest look-back.
    """
    rec = F.build_footprint(day, 40.5, -90.5, arrival_step=6)
    lags = rec["lag"].values
    lags = lags[lags <= rec.attrs["max_lag_hours"] + 0.5]
    n = len(lags)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    vmax = float(rec["kernel"].max())
    for ax, lg in zip(axes, lags):
        k = rec["kernel"].sel(lag=lg)
        ax.pcolormesh(rec["source_lon"], rec["source_lat"], k.values,
                      cmap="magma_r", shading="nearest", vmin=0, vmax=vmax)
        ax.scatter([-90.5], [40.5], marker="*", s=120, c="red", edgecolor="k")
        ax.set_title(f"lag {int(lg)} h")
        ax.set_xlim(-93, -89); ax.set_ylim(39, 43)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Q3  A kernel blooming: where the air over (40.5N, 90.5W) at 02 UTC "
                 "was, hour by hour back", fontsize=12)
    fig.savefig(FIG / "explore_kernel_bloom.png", dpi=130, bbox_inches="tight")
    print(f"[Q3] receptor 02 UTC: {rec.attrs['n_parcels']} parcels, "
          f"max lag {rec.attrs['max_lag_hours']:.1f} h, "
          f"temporal marginal {np.round(rec['kernel'].sum(('source_lat','source_lon')).values, 3)}")


if __name__ == "__main__":
    day = T.load_day()
    smap = smap_daily_mean()

    q1_advection(day=day, smap=smap)
    q2_diurnal_coupling(day=day)
    q3_kernel_bloom(day=day)
    print("figures ->", FIG)
