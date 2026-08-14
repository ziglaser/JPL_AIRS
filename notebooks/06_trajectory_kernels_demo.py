# %% [markdown]
# # Trajectory kernels - end-to-end demo (2019-06-05)
#
# - **Step 1** build all-receptor influence kernels and write them to NetCDF
# - **Step 2** the required single-receptor diagnostics (kernel-at-lag,
#   spatial/temporal marginals) for a well-populated Ohio-valley cell
# - **Step 3** apply the kernels to SMAP surface soil moisture -> the
#   `influence_smap_sfc` predictor (plus a land_frac geometry check)
# - **Step 4** extensibility: swap the PBL model / fuzz kernel in one line
#
# Run headless with `PYTHONPATH=src python notebooks/06_trajectory_kernels_demo.py`.

# %%
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_kernels import apply as A
from trajectory_kernels import config, footprint as F, io, plotting as P
from trajectory_kernels import pbl as PBL
from trajectory_kernels import trajectories as T
from trajectory_kernels.fuzz import StohlFuzz
from trajectory_kernels.land import make_land_lookup

FIG = config.RESULTS_DIR / "trajectory_kernels" / "figures"
OUT = config.RESULTS_DIR / "trajectory_kernels"
FIG.mkdir(parents=True, exist_ok=True)

# a well-populated receptor for the single-cell diagnostics, at the END of the
# forecast window (step 6 = 02 UTC) so the visualized trajectories carry the
# longest possible look-back (Zach, 2026-07-23). The old Ohio-valley cell
# (37.5, -87.5) thins to 3 parcels by step 6; this Illinois cell keeps 16.
DEMO_LAT, DEMO_LON, DEMO_STEP = 40.5, -90.5, 6
STEP_UTC = {1: 21, 2: 22, 3: 23, 4: 0, 5: 1, 6: 2}  # arrival step -> UTC hour

# %% [markdown]
# ## Step 1 - All-receptor kernels -> NetCDF (dense + sparse)

# %%
def build_kernels(day):
    kernels = F.build_all(day)
    dense = io.write_kernels(kernels, OUT / "kernels_20190605.nc")
    sparse = io.write_sparse(kernels, OUT / "kernels_20190605_sparse.nc", which="footprint")
    n_populated = int((kernels["n_parcels"] > 0).sum())
    print(f"kernels: {n_populated} populated receptors; "
          f"dense {dense.stat().st_size / 1e6:.1f} MB, "
          f"sparse {sparse.stat().st_size / 1e6:.1f} MB")

    return kernels

# %% [markdown]
# ## Step 2 - Single-receptor diagnostics

# %%
def receptor_diagnostics(day):
    rec = F.build_footprint(day, DEMO_LAT, DEMO_LON, arrival_step=DEMO_STEP)
    # snapshot near the receptor plus the deepest populated lag (the longest
    # look-back the trajectories support)
    lag_sums = rec["footprint"].sum(("source_lat", "source_lon")).values
    lag_deep = float(rec["lag"].values[np.where(lag_sums > 0)[0][-1]])
    P.plot_kernel_at_lag(rec, 1.0, day=day, save_path=FIG / "demo_kernel_lag1.png")
    P.plot_kernel_at_lag(rec, lag_deep, day=day,
                         save_path=FIG / f"demo_kernel_lag{lag_deep:.0f}.png")
    P.plot_kernel_evolution(rec, day=day, save_path=FIG / "demo_kernel_evolution.png")
    P.plot_spatial_influence(rec, day=day, save_path=FIG / "demo_spatial_influence.png")
    P.plot_temporal_influence(rec, save_path=FIG / "demo_temporal_influence.png")
    print(f"receptor {DEMO_LAT},{DEMO_LON} @{STEP_UTC[DEMO_STEP]:02d}UTC: "
          f"{rec.attrs['n_parcels']} parcels, "
          f"{float(rec['footprint'].sum()):.0f} h land contact, "
          f"deepest lag {lag_deep:.0f} h")

    return rec

# %% [markdown]
# ## Step 3 - Apply to SMAP surface soil moisture
#
# The convolution is predictor-agnostic; `land_frac` doubles as a geometry check
# (a normalized kernel over all-land upstream must return ~1).

# %%
def smap_smsfc_daily_mean():
    ds = xr.open_dataset(config.DATA_DIR / "FCST_SMAP_MRMS_2019.nc")
    dates = ds["date"].values.astype("datetime64[D]")
    day_index = int(np.where(dates == np.datetime64("2019-06-05"))[0][0])
    smap = ds["SMAP_L4_smsfc_av"].isel(date=day_index).mean("L4_nhours", skipna=True)

    return smap.rename("smap_smsfc").assign_attrs(units="m3/m3")

def apply_to_surface(kernels, smap):
    influence_smap = A.apply_kernel(kernels, smap, which="kernel")
    land_check = A.apply_kernel(kernels, make_land_lookup(), which="kernel")

    pred = xr.Dataset({"influence_smap_sfc": influence_smap,
                       "influence_land_frac": land_check})
    pred["influence_smap_sfc"].attrs["units"] = "m3/m3 (kernel-weighted upstream SMAP)"
    pred.to_netcdf(OUT / "influence_predictor_20190605.nc")

    n_valid = int(np.isfinite(influence_smap.values).sum())
    print(f"applied predictor: {n_valid} valid receptor-steps; "
          f"influence range {np.nanmin(influence_smap.values):.3f}"
          f"-{np.nanmax(influence_smap.values):.3f} m3/m3")
    print(f"land_frac geometry check: median = {np.nanmedian(land_check.values):.2f} "
          f"(expect ~1 inland)")

    return influence_smap

def plot_predictor_map(influence_smap, smap, step=DEMO_STEP):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    local = axes[0].pcolormesh(smap["lon"], smap["lat"], smap.values,
                               cmap="YlGnBu", shading="nearest")
    axes[0].set_title("SMAP surface soil moisture (daily mean)")
    fig.colorbar(local, ax=axes[0], label="m3/m3")

    infl = influence_smap.sel(arrival_step=step)
    upstream = axes[1].pcolormesh(infl["target_lon"], infl["target_lat"], infl.values,
                                  cmap="YlGnBu", shading="nearest")
    axes[1].set_title(f"Kernel-weighted upstream SMAP influence ({STEP_UTC[step]:02d} UTC)")
    fig.colorbar(upstream, ax=axes[1], label="m3/m3")
    for ax in axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(FIG / "demo_influence_predictor.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ## Step 4 - Extensibility: one-line model swaps
#
# A forced-shallow PBL excludes the ~1-1.6 km parcels; blurrier fuzz lowers the
# kernel peak. (The default climatological PBL is deep in this afternoon window,
# so ConstantPBL(300 m) is the illustrative contrast.)

# %%
def extensibility_demo(day, rec):
    rec_shallow = F.build_footprint(day, DEMO_LAT, DEMO_LON, arrival_step=DEMO_STEP,
                                    pbl_model=PBL.ConstantPBL(300.0))
    rec_blurry = F.build_footprint(day, DEMO_LAT, DEMO_LON, arrival_step=DEMO_STEP,
                                   fuzz_kernel=StohlFuzz(fuzziness=3.0))
    print(f"ConstantPBL(300 m) contact-h = {float(rec_shallow['footprint'].sum()):.0f} "
          f"vs Climatological = {float(rec['footprint'].sum()):.0f}")
    # compare at the deepest populated lag: with sigma0=0 the early lags are
    # near-delta deposits for any fuzziness, so the overall max hides the
    # difference (and the last lag bin on the axis can be empty -> NaN)
    lag_sums = rec["footprint"].sum(("source_lat", "source_lon"))
    lag_deep = float(rec["lag"].values[np.where(lag_sums.values > 0)[0][-1]])
    print(f"blurry-kernel peak at lag {lag_deep:.0f} h = "
          f"{float(rec_blurry['kernel'].sel(lag=lag_deep).max()):.3f} "
          f"vs default = {float(rec['kernel'].sel(lag=lag_deep).max()):.3f}")

    return None


if __name__ == "__main__":
    day = T.load_day()
    print(f"loaded {day.sizes['parcel']} parcels; "
          f"{int(day['is_near_surface'].sum())} near-surface")

    kernels = build_kernels(day)
    rec = receptor_diagnostics(day)

    smap = smap_smsfc_daily_mean()
    influence_smap = apply_to_surface(kernels, smap)
    plot_predictor_map(influence_smap=influence_smap, smap=smap)

    extensibility_demo(day=day, rec=rec)
    print("done ->", OUT)
