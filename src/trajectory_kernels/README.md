# trajectory_kernels

Turn HYSPLIT forward trajectories into **source–receptor soil-moisture influence
kernels**: for any grid cell and arrival time, *where* and *when* the arriving
near-surface air was in boundary-layer contact with the land surface. The
soil-moisture analog of the CAPE trajectory enhancement in `convection_skill`.

See `../../docs/TRAJECTORY_KERNEL_WORKPLAN.md` for the full design and the literature
it is built on (STILT footprints; Sodemann moisture-source diagnostics; Stohl
trajectory-error rule; Guillod soil-moisture/rain coupling scales).

## The one idea that shapes the code

In these files `q` is a **conserved Lagrangian tracer** (it only drops via
condensation, logged in `q_excess`) — so surface moistening is *not* in the
parcel humidity. The tool therefore builds a purely **geometric** residence-time
footprint; soil-moisture physics enters *only* by convolving that footprint with
an external surface field (`apply.py`). Geometry × surface state, kept separate,
is what makes it reusable for any predictor.

Two grids: receptors live on the **1° CAPE/lsm target grid**; each receptor's
source window is gridded at **0.25°** (`SOURCE_STEP_DEG`), with quarter-degree
cells nested exactly inside the 1° cells, so the kernel resolves the actual
parcel-cloud shape. Per lag, the **kernel keeps only the region containing 90%
of the parcel cloud** (`KERNEL_CONTAINMENT_FRAC`): the smallest circle grown
outward from the in-contact parcels' center of mass, which cuts the far-field
Gaussian fuzz tails; the physical `footprint` is never truncated.

## Pipeline (each stage usable and tested on its own)

| Module | Role |
|---|---|
| `config.py` | every constant, each cited to a data fact or paper |
| `trajectories.py` | ingest the 5 granule files → tidy `(parcel, step)` dataset |
| `pbl.py` | boundary-layer depth models (`ConstantPBL`, `ClimatologicalPBL`; ERA5 stub) |
| `contact.py` | STILT/Sodemann surface-coupling weight (taper at `f_c·PBLH`) |
| `resample.py` | sub-hourly trajectory interpolation |
| `fuzz.py` | trajectory-uncertainty Gaussian: `StohlFuzz` (σ ≈ 0.2·distance, Stohl 1998) or `EmpiricalFuzz.from_fullgrid()` (α measured from the day's own sub-box wind spread; 0.191 on 2019-06-05 vs Stohl's 0.2) |
| `discount.py` | optional Sodemann-style rain-out discount, exact here because `q` is loss-only: `w = q(arrival)/q(t)` (`rainout_discount=True` in the builders) |
| `footprint.py` | **the builder**: `build_footprint` (one receptor), `build_all` (grid) |
| `io.py` | NetCDF write/read — dense relative-window + sparse COO, with provenance |
| `apply.py` | predictor-agnostic convolution kernel × surface field → predictor |
| `plotting.py` | trajectory / kernel-at-lag / kernel-evolution (per-lag HDR contours stepping upstream) / spatial & temporal marginal / coverage |
| `land.py`, `geo.py` | land-fraction lookup, geodesy helpers |

## Quick start

```python
from trajectory_kernels import trajectories as T, footprint as F, apply as A, io
from trajectory_kernels.land import make_land_lookup

day = T.load_day()                                  # 94k parcels, 2019-06-05

# one receptor: an Ohio-valley cell (37.5N, 87.5W) arriving at 23 UTC (step 3)
rec = F.build_footprint(day, 37.5, -87.5, arrival_step=3)
rec["kernel"].sel(lag=1.0)                          # where the air was 1 h earlier

# all receptors, written to NetCDF
kernels = F.build_all(day)
io.write_kernels(kernels, "kernels_20190605.nc")

# apply to a surface field -> a drop-in predictor (arrival_step, lat, lon)
influence = A.apply_kernel(kernels, smap_smsfc_dataarray)   # any (lat,lon) field
```

Pluggability is by interface, not flags — swap `pbl_model=ConstantPBL(1500)`,
`fuzz_kernel=StohlFuzz(fuzziness=3.0)`, or a different surface field into `apply`,
each a one-line change. `notebooks/06_trajectory_kernels_demo.py` runs the whole
thing end to end and writes the figures in `results/trajectory_kernels/figures/demo_*.png`.

## Scope / honest limits (this dataset)

- One day, five granules, **two overpass swaths** (~19 and ~20:30 UTC), central +
  eastern CONUS only; **near-surface parcels are terrain-limited** (none over the
  high West). Cells with no arriving near-surface parcels get a **NaN kernel** —
  never a fabricated one; `plot_coverage` shows the gaps.
- Look-back lag is capped by the forward-trajectory length (≤ ~7 h here).
- Soil moisture is treated as static over the window (fine for one day; a
  lag/time-varying surface field is a documented extension in `apply.py`).
```
