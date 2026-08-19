# PBL depth: raw Guo et al. (2024) model output and the 1° gridded climatology

Two datasets are documented here:

1. **`data/PBL_depth/Guo2024_model/`** — the raw ML-merged planetary-boundary-layer
   height (PBLH) product as delivered: 0.25°, 3-hourly, global, land only, 2017–2021.
2. **`data/PBL_depth/derived/PBL_climatology_1deg_*.nc`** — what
   `scripts/build_pbl_climatology.py` makes out of it: a 1° **monthly diurnal
   climatology** (12 months × 8 slots of the day) for merging into
   `FCST_SMAP_MRMS` and for contextualising soil moisture during HYSPLIT
   accumulation of the soil-moisture index.

Everything below about the raw corpus was verified by reading it (all 14 587
files inventoried, a sample opened), not from the provider's notes.

---

## 1. Raw model output

### Layout on disk

```
data/PBL_depth/Guo2024_model/
├── 2017/YYYYMMDDHH.nc          2 920 files
├── 2018/YYYYMMDDHH.nc          2 920 files
├── 2019/2019/YYYYMMDDHH.nc     2 920 files   ← note the doubled year directory
├── 2020/YYYYMMDDHH.nc          2 928 files   (leap year)
└── 2021/2021/YYYYMMDDHH.nc     2 899 files   ← doubled, and short by 21 times
                                ─────────
                                14 587 files, 48 GB
```

The nesting is **inconsistent**: 2017/2018/2020 hold the files directly, 2019 and
2021 hold a second directory of the same name. Any reader must recurse (the build
script does) and take the timestamp from the *filename*, not the path.

One file = one synoptic time. The name is `YYYYMMDDHH.nc` in **UTC**, with
`HH ∈ {00, 03, 06, 09, 12, 15, 18, 21}` — 8 times per day. The filename stamp was
checked against the in-file `time` coordinate on a sample and always agreed.

**Completeness.** 2017–2020 are complete (every 3-hourly slot present, leap day
included). 2021 is missing 21 times, all in October: 2021-10-13 00Z through
2021-10-15 06Z (a 2.3-day outage), plus 2021-10-19 15Z and 18Z. No duplicate
timestamps anywhere. The gap is small (0.14% of the corpus) but it is in one
month of one year, so October 2021 carries slightly fewer samples — the gridded
product records this in its `missing_source_times` attribute and per-cell
`n_times` counts.

### File contents

```
Dimensions:  (time: 1, lat: 600, lon: 1440)
Coordinates:
  time  datetime64      the single valid time, e.g. 2017-01-01T00:00:00
  lat   float32 (600)   +90.00 → −59.75, step −0.25
  lon   float32 (1440)  0 → 360 (see the caveat below)
Data variables:
  "Merged Planetary Boundary Layer Height"  (time, lat, lon) float32
      description : "Merged Planetary Boundary Layer Height"
      units       : "m"
      _FillValue  : -999.0
```

Note the **variable name contains spaces** — `ds["Merged Planetary Boundary Layer
Height"]`, not a tidy identifier. There are no global attributes: no provenance,
no version, no grid description. ~3.5 MB per file (uncompressed float32).

**Coverage is land only.** 71.9% of every field is fill, essentially the ocean
fraction of the domain. The missing mask is static to within ~48 of 864 000 cells
(coastline/lake pixels that flicker), so treat it as a fixed land mask. Over the
CONUS box (25–53°N, 107–64°W) 30.7% of native cells are missing — the Gulf, the
Atlantic and Pacific margins, and the Great Lakes.

**Domain stops at 59.75°S**: Antarctica is absent by construction. The first
latitude row is exactly the North Pole (+90.0), which is a grid *point*, not a
1°-cell interior — the build script drops it rather than fold it into the 89.5°N
cell.

**Value range.** Sampled minima/maxima run ~20 m (stable nocturnal) to ~5 550 m
(deep summer desert convective PBL); physically sensible throughout. Diurnal
behaviour over CONUS in July: ~230 m at 09Z/12Z (night), ~1 520 m at 21Z
(mid-afternoon) — the expected order of magnitude.

### ⚠ The longitude-axis artefact

The stored `lon` axis is `linspace(0, 360, 1440)`, i.e. **step 0.250174**, ending
at exactly 360.0. A 1440-point 0.25° global grid should run 0 → 359.75. The
stored axis therefore drifts progressively eastward, reaching +0.125° of error at
the date line and +0.25° at the wrap point, and it duplicates longitude 0.

This is almost certainly an axis-writing artefact of an intended uniform 0.25°
grid, not a real grid. The build script's default `--lon-grid uniform` rebuilds
`lon = 0.25·j`; `--lon-grid file` uses the stored values verbatim if you ever
want to test the sensitivity. At 1° aggregation the difference moves at most one
native column per 1° cell in the eastern hemisphere, so it is a small effect —
but it is not zero, and it is worth knowing before comparing this product to any
other 0.25° field pixel-by-pixel.

### Provenance

Guo et al. (2024), a machine-learning **merged** PBLH product: reanalysis PBLH
fields corrected/blended against radiosonde- and lidar-derived boundary-layer
depths, delivered globally at 0.25°/3-hourly. The delivered files carry no
version or citation metadata, so record the delivery date and the source contact
alongside them; nothing in the corpus itself identifies the algorithm version.

---

## 2. The gridded 1° monthly diurnal climatology

### Building it

```bash
# global, local-solar-time frame (default) — 23 min on 10 cores (I/O-bound on drvfs)
python scripts/build_pbl_climatology.py --jobs 10

# the FCST_SMAP_MRMS CONUS grid, UTC frame — 32 min (re-reads the corpus:
# the per-year cache is keyed by domain and time reference)
python scripts/build_pbl_climatology.py --domain conus --time-ref utc --jobs 10

# smoke test
python scripts/build_pbl_climatology.py --limit 80 --no-cache --out /tmp/x.nc
```

Outputs land in `data/PBL_depth/derived/PBL_climatology_1deg_{domain}_{ref}_{y0}-{y1}.nc`.
Both are built as of 2026-08-18:

| file | grid | size | cells with data |
|---|---|---|---|
| `PBL_climatology_1deg_global_lst_2017-2021.nc` | 150 × 360, local solar | 12.9 MB | 32.0% (land) |
| `PBL_climatology_1deg_conus_utc_2017-2021.nc` | 28 × 43, UTC | 0.6 MB | 74.8% (land) |

Per-year partial sums are cached in `derived/_cache/`; the accumulators are plain
additive sums, so an interrupted run resumes for free and a cached year is simply
added back. Delete the cache after changing the domain, the time reference or the
validity filter (the cache key includes the first two, not the filter).

Unit tests: `pytest tests/test_pbl_climatology.py` (grid alignment, the 4×4 block
rule, the pole row, the local-solar shift, the accumulator).

### The aggregation rule

One rule, applied once:

> Every valid native 0.25° sample is pooled **with equal weight** into the 1° cell
> that contains it, where `cell = floor(coordinate)`. The reported mean is
> `sum(values) / count(values)`.

Consequences worth stating explicitly:

- **Cell centres land on X.5**, so `--domain conus` reproduces the
  `FCST_SMAP_MRMS` grid exactly (lat 25.5…52.5, 28 cells; lon −106.5…−64.5,
  43 cells) and the fields can be concatenated onto that product with no
  regridding.
- **It is a land mean.** Water is fill in the source, so a coastal cell's mean
  describes only its land part. `n_obs / (n_times × n_native)` is that cell's
  effective land fraction; a cell that is entirely water is all-NaN with
  `n_obs = 0`.
- Interior cells pool 4×4 = 16 native points per time. The southernmost row
  (−59.5°) pools 12, because the source stops at −59.75°. `n_native` records this
  per cell.
- **No area weighting.** Within one 1° cell the cos(lat) weight of the four native
  rows spreads by <1.5% even at 60°, which is far below the product's own
  uncertainty (~100 m); weighting would complicate the rule for no gain.
- **Validity filter:** `1 m ≤ PBLH ≤ 8000 m`. In practice this only removes the
  −999 fill; the observed corpus range sits well inside it.
- Months are the **calendar month of the UTC timestamp**, pooled across all
  available years (2017–2021).

### Diurnal reference frame — read this before using `hour`

`--time-ref` picks what the `hour` axis means:

| mode | `hour` means | use when |
|---|---|---|
| `lst` (default) | **local solar** hour, `slot = round((UTC + lon/15)/3) mod 8` | you want the physical diurnal cycle — night/morning/afternoon composites |
| `utc` | UTC hour, `slot = UTC/3` | you are joining to a UTC-indexed product (`FCST_SMAP_MRMS`, HYSPLIT arrival times) |

In `lst` mode the shift is a **whole** number of 3 h slots per longitude column
(`floor(lon/45 + ½)`, half-up so the step is monotone across all 360 columns), so
nothing is interpolated — each raw field is simply filed under a different slot
depending on the column's longitude. Two caveats: the shift is quantised to 3 h,
so a column's true local time can be up to 1.5 h from its slot label; and month
attribution still uses the UTC date, which mislabels the handful of samples that
fall on the far side of a month boundary in local time.

A UTC-hour diurnal composite is meaningless over a global domain (00Z is
mid-afternoon in one place and midnight in another), which is why `lst` is the
default. For CONUS-only work either is defensible — UTC keeps the join to
`FCST_SMAP_MRMS` exact, and CONUS spans only ~3 slots of solar time.

### Output file layout

```
Dimensions:  (month: 12, hour: 8, lat: N, lon: M)
Coordinates:
  month  int32  1…12                       calendar month
  hour   int32  0,3,6,…,21                 slot label; UTC or local solar per
                                           the `time_reference` attribute.
                                           Carries NO `units` attribute on
                                           purpose — "hours" would make CF
                                           readers decode it as a timedelta.
  lat    float32                           1° cell centres, descending (X.5)
  lon    float32                           1° cell centres, −180…180 (X.5)

Data variables:
  pblh_mean  (month, hour, lat, lon) float32 [m]   NaN where no samples
      Mean PBL depth over every valid native sample in the cell, pooled over
      all years and all days of that month at that slot.
  pblh_std   (month, hour, lat, lon) float32 [m]
      Sample standard deviation of the SAME pool. It mixes within-cell spatial
      spread with day-to-day and interannual spread — it is a variability
      measure, NOT a standard error of pblh_mean. Divide by √n_obs only if you
      are willing to pretend the samples are independent, which they are not.
  n_obs      (month, hour, lat, lon) int32
      Number of valid native samples in the mean (≈ n_times × 16 inland).
  n_times    (month, hour, lat, lon) int32
      Number of 3-hourly source files that contributed ≥1 valid sample.
      ≈ 5 years × days-in-month for a fully covered cell — use it to spot the
      October-2021 gap and any coastal cell that is only intermittently land.
      Measured: July peaks at 155 = 5 × 31; October runs 148–153, exactly the
      dent the 21 missing 2021 files predict.
  n_native   (lat, lon) int32
      Native 0.25° points inside the cell: 16 inland, 12 in the −59.5° row.

Global attributes: source, source_variable, source_resolution, time_reference,
domain, lon_grid, years, first_time, last_time, n_files, missing_source_times,
validity_filter, aggregation, created, command, history.
```

`lat` is **descending** (52.5 → 25.5), the source's orientation, whereas
`FCST_SMAP_MRMS` stores it ascending. The *values* are identical, and xarray
aligns on coordinate labels, so `xr.Dataset(dict(cape=..., pblh=...))` or
`xr.merge` just works (verified); use `ds.sortby("lat")` if you need the array
memory order to match too.

The `_cache/` directory holds ~120 MB of per-year partial sums per configuration
— safe to delete once the products are written.

### Using it

**Merging into `FCST_SMAP_MRMS`.** Build with `--domain conus --time-ref utc`.
The `lat`/`lon` axes are then identical to that product's; `FCST_SMAP_MRMS`
indexes `(day, UTChour)` while this indexes `(month, hour)`, so the join is
`month = month(day)`, `hour = 3·⌊UTChour/3⌋` (the product's evening window,
20–02Z, maps to slots 18, 21, 00). What you are attaching is a *climatology*, not
a per-day analysis: it tells you the typical boundary-layer depth for that cell,
month and time of day, which is what is needed to normalise a CAPE or
soil-moisture signal, and nothing about the specific day's weather.

**Contextualising soil moisture in HYSPLIT accumulation.** A parcel's surface
contact only matters relative to the boundary layer it sits in — the STILT
footprint and Sodemann PBL gate in `src/trajectory_kernels/` are both expressed
as fractions of PBLH. `trajectory_kernels.pbl` currently offers `ConstantPBL` and
the analytic `ClimatologicalPBL` (a cosine curve with a west–east gradient fitted
by hand from McGrath-Spangler & Denning 2012). This file is the data-driven
replacement: a `PBLModel` reading it looks up `(month, local-solar slot, cell)`
for each trajectory point, which is exactly the signature `PBLModel.depth(lat,
lon, time_utc)` already has. Build with the default `--time-ref lst` for that
use, since trajectories leave the CONUS box and the physical diurnal phase is
what the gate depends on. (That model class is not written yet — this documents
the intended consumer, not existing code.)

**Sanity values** (from the built files, useful as a regression check). Great
Plains cell 38.5°N/98.5°W, July, local solar: 297 m at 06 LST → 1 167 m at
12 → 1 574 m at 15 → 639 m at 21. Sahara (23.5°N/12.5°E) reaches 2 312 m at
15 LST; Amazon (3.5°S/62.5°W) 1 250 m. Global land mean at 15 LST: 937 m in
January, 1 445 m in July. In the CONUS/UTC file the same Plains cell reads
1 574 m at 21Z and 1 589 m at 00Z — i.e. the two products agree exactly once
the longitude's two-slot local shift is undone, which is the cross-check that
the `lst` bookkeeping is right.

**Known limits.**
- A climatology smooths away exactly the synoptic variability that drives extreme
  convection days. Use it for normalisation and context, not as a per-day PBLH.
- Only 5 years, and only 2017–2021 — no coverage of the 2016 season that appears
  elsewhere in this repo.
- Land only. Any trajectory segment over water gets no value from this file.
- The 3 h quantisation of local solar time means the afternoon-peak slot may be
  labelled 15 or 18 h locally depending on the column.
