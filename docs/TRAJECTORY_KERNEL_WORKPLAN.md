# Work Plan: Trajectory → Soil-Moisture Influence Kernels

**Goal.** Turn the HYSPLIT forward-trajectory NetCDF files in
`data/wrf27km_20190605/` into a tool that produces, for any grid cell and time, a
**source–receptor influence kernel**: a field saying *where* (geographically) and
*when* (at what lag) the air arriving at that cell was in surface contact over
land — so we can test the influence of upstream **soil moisture** on downstream
**precipitation**, alongside the existing CAPE analysis.

For CAPE the influence is a *state* carried forward (the parcel's overpass CAPE is
what it delivers downstream). For soil moisture the influence is a *path*: the
land a parcel passed over — while inside the boundary layer — could have modified
it, so the object we need is a residence-time-weighted **footprint**, the
source–receptor construct from Lagrangian dispersion/footprint models (STILT: Lin
et al. 2003; Fasoli et al. 2018) and Lagrangian moisture-source diagnostics
(Sodemann et al. 2008).

---

## 0. The finding that shapes everything: `q` is a conserved tracer

The data audit established that in these files **specific humidity `q` (g/kg) is
Lagrangian-conserved along each trajectory** — it is carried from the release
value and only ever *decreases*, via condensation, with the removed amount logged
in `q_excess` (g/kg, "mixing_ratio above 100 RH"; ≥ 0 always, nonzero for only
~6.6% of parcels, exactly the parcels whose `q` drops). The trajectories do **not**
sample environmental moistening, so **a soil-moisture → boundary-layer-moistening
signal is not observable in the parcels' own `q`.**

This is decisive, and it *validates* the predictor-agnostic architecture rather
than breaking it:

1. **The trajectory tool produces a purely geometric/kinematic footprint** —
   *where and when the arriving air was in PBL contact with the land surface.* It
   makes no claim about soil moisture itself.
2. **The soil-moisture influence is obtained entirely by convolving that
   footprint with an external surface field** (SMAP `smsfc`/`qlay1`, already in
   the project's `FCST_SMAP_MRMS_*.nc`; evaporative fraction; etc.). This is a
   one-line operation and is where all soil-moisture physics enters.
3. Sodemann-style along-trajectory *uptake* attribution (needs Δq > 0) is
   therefore **not applicable to this dataset** — there are no uptakes, only
   condensation losses. We keep the *geometry* half of the Sodemann/STILT method
   (PBL-gated residence time) and drop the *Δq-uptake* half. `q_excess` is still
   useful, as an optional Sodemann-style **rain-out discount** (a parcel that
   condensed en route has had its low-level history partly reset; §3.7).

So the split "trajectory geometry (from these files) × surface state (from SMAP)"
is not just house style — it is forced by the physics of the data, and it keeps
the tool reusable for any surface field.

---

## Design constraints (non-negotiable — inherited from the CAPE work)

1. **Predictor-agnostic core.** The kernel machinery never mentions "soil
   moisture"; it emits a footprint. Applying it to a surface field is a separate
   convolution. Swapping SMAP `smsfc` → `qlay1` → evaporative fraction requires
   zero changes to the trajectory/footprint code.
2. **One function = one physical rule, with a citation** in its docstring.
3. **All constants in one `config.py`**, each annotated with its source (mirrors
   `src/convection_skill/config.py`).
4. **Explicit sequential stages, each usable and inspectable alone** (Zach's
   staged style): `ingest → PBL & contact weight → footprint → NetCDF → apply`.
   No monolith-with-flags.
5. **Pluggable physics via interfaces, not flags.** `PBLModel`, `FuzzKernel`,
   `ContactWeight` are `Callable`s; the fixed default and future ERA5-`blh` /
   soil-moisture-derived versions are interchangeable implementations.
6. **Unit tests with analytic answers** (like `tests/test_gini.py`): straight
   constant-speed trajectory → known footprint line; stationary parcel →
   point-spread of known width; Gaussian fuzz integrates to 1; parcel above PBL →
   zero footprint.
7. **Thin notebooks** — one diagnostic figure each, logic in the library.
8. **Every sub-component independently verifiable and plottable** (explicit user
   ask): for any (lat, lon, arrival-time) I can plot (a) the kernel of where the
   air was a given hour earlier, and (b) the spatial- and temporal-marginal
   influence over that cell.

---

## 1. The data (confirmed by audit)

`data/wrf27km_20190605/wrf27km_20190605/`

### `nogrid_wrf27km_GOOD_20190605_{188,189,190,205,206}.nc` — one per AIRS granule
Forward HYSPLIT trajectories, dims `(time:7, level:57, fieldx:45, fieldy:30)`.
Variables (no units in file; confirmed from the fullgrid twin): `lat, lon,
alt`(m)`, pres`(hPa)`, t`(K)`, q`(g/kg)`, q_excess`(g/kg)`, parceltime`(UTC s), and
scalar `granule_mean_time`.

- **`level` = a fixed vertical release ladder** (not a trajectory index): level 0
  ≈ 102 hPa/16 km (top) → level 52 ≈ 987 hPa/256 m (near-surface), ~26 hPa/~255 m
  per step; levels 53–56 unused. `(fieldx, fieldy)` index horizontal overpass
  columns; at `time=0` every level in a column shares one lat/lon (a vertical
  stack), then each parcel (one per level) advects independently.
- **Two overpass swaths** (confirmed): granules **188/189/190 ≈ 18:50–19:02 UTC**
  (Aqua early-afternoon, ~13:53 local East) and **205/206 ≈ 20:32–20:38 UTC**
  (next orbit, ~1.7 h later). **189 is the main, near-full-field granule** (1307
  columns; 327 reaching the near-surface). 188 is small (18 columns).
- **Time axis:** `time[0]` = per-granule release (≈ `granule_mean_time`), then all
  granules snap to the shared grid **21, 22, 23, 00, 01, 02 UTC** — the same slots
  0–6 as the CAPE analysis (overpass + six forecast hours).
- **`parceltime` is NOT a per-parcel staggered clock** — it exactly equals the
  `time` coordinate broadcast to every parcel. The only stagger is the per-granule
  `time[0]` release moment (~107 min spread across the five granules); from 21 UTC
  on, everything is on one top-of-hour grid.
- **Combined release coverage: lat 25–53 N, lon −107 to −75 W = central + eastern
  CONUS only** (no west coast). **Near-surface (level-52) coverage is
  terrain-limited:** western/high-terrain granules (205/206, and mostly 190) have
  **~zero** parcels reaching the lowest levels (they'd be below ground), so
  near-surface receptors are only computable in the central/eastern plains and
  east.
- **Low parcels stay low:** levels 44–52 sit at median ~1.3 km, ~99% below 3 km
  through all 7 steps (a ~1% ascending tail reaches 4.5 km); none go below ground;
  they are essentially all over land, with a far-eastern minority drifting toward
  the Atlantic (→ needs ocean masking). Parcels advect ~50 hPa / few-hundred m
  over 6 h — genuinely 3-D, so the PBL-contact test is meaningful, not degenerate.
- **Clean:** no all-NaN variables, no out-of-range values, and **no
  mid-trajectory dropout** (every parcel valid at t0 is valid at t6); parcels are
  tracked freely (drift off the release footprint) rather than clipped to a grid.

### `fullgrid_wrf27km_GOOD_1p00deg_...nc` — the 1° box-average of the parcels
Dims `(time:7, level:33, lat:28, lon:43)`; grid lat 25.5–52.5, lon −106.5…−64.5 at
1°; 33 pressure levels 115–1075 hPa. `time` here is a bare 0..6 index that maps
exactly to the nogrid timesteps. Contains **u, v, w at all 33 levels** (trajectory
sanity-check / advection overlays), the CAPE family (`MU_/MML_ CAPE, CIN, LCL, EL,
LFC`), `N` = parcels per box, and a **`_std` spread for every field**
(`pres_std, alt_std, u_std, v_std, …`). **Two gifts for us:** `N` and the `_std`
spreads are an *empirical, data-driven* parcel-uncertainty estimate we can feed
the fuzz kernel (§3.5), independent of the theoretical Stohl rule.

### Not in either file — must come externally
**No PBL height, no land/sea mask, no terrain/geopotential, no surface pressure,
no soil moisture.** Sources: **land mask → the project's existing `data/lsm.nc`**
(same 1° grid, already used by the CAPE loader); PBL → climatology default now,
ERA5 `blh`/MERRA-2 `PBLTOP` later; terrain (for the below-ground check) → an
external DEM if needed; soil moisture → SMAP in `FCST_SMAP_MRMS_*.nc`.

**Coverage caveat (design-shaping).** One day, five granules, two swaths, central/
eastern CONUS, near-surface only off the high terrain. This is a **method-
development and verification** dataset, not a climatology. The tool degrades
honestly: a cell with no overlying near-surface parcels yields a NaN kernel, never
a fabricated one, and the pipeline is per-day so stacking days later is a loop.

---

## 2. The core object: a source–receptor footprint

For a **receptor** = (target cell `x_r`, arrival time `t_r`),

```
f(x_s, τ) = residence-time sensitivity of the air arriving at (x_r, t_r) to the
            land surface at source location x_s, at lag τ = t_r − t  (τ ≥ 0).
```

A parcel *p* **contributes to the receptor** if at `t_r` it is (i) horizontally
in the receptor cell (up to fuzz) and (ii) in the near-surface **inflow layer**
(precip is fed by low-level convergent inflow, so the receptor is the near-surface
air, not the whole column; the layer is a cited config band, §3.6/§5.5). For each
contributor we walk **backward along its own forward trajectory** and deposit, at
each (sub-hourly) step, a contact- and land-weighted, spatially fuzzed mass:

```
f(x_s, τ) = Σ_p Σ_{t: t_r−t=τ}  w_contact(p,t) · L(x_p(t)) · G_{σ(τ)}(x_s − x_p(t))
```

- `w_contact ∈ [0,1]` — STILT/Sodemann PBL coupling weight (§3.3).
- `L(x)` — land weight from `lsm.nc` (0 over ocean).
- `G_σ(τ)` — fuzzing kernel, 2-D Gaussian of width `σ(τ)` growing with lag (§3.5).
- Summed over the sub-hourly interpolated path (§3.4), so fast parcels don't skip
  source cells.

**Units / normalization — stated explicitly (a classic footprint bug).** Two
products, both written:
- **Footprint (physical, STILT form; Lin 2003, Fasoli 2018):** per source cell the
  density-weighted residence time below the contact layer,
  `S(x_s)=Σ Δt_below/(h*·ρ̄)`, so `footprint × surface flux` is a receptor
  response. Not normalized.
- **Influence kernel (probability):** the same field normalized so
  `Σ_{x_s,τ} K = 1` — "what fraction of the arriving air's land contact came from
  here/then." This is the user's "where the air it's seeing was"; the §6 marginals
  summarize it.

**Applied predictor** (the drop-in for the Gini table, parallel to `mu_cape`):
```
influence_S(x_r, t_r) = Σ_{x_s,τ} K(x_s,τ; x_r,t_r) · S(x_s, t_r − τ)
```
a scalar per receptor, for any surface field `S`.

---

## 3. Sub-components (each independently verifiable)

Package `src/trajectory_kernels/`, mirroring `src/convection_skill/`.

### 3.0 `config.py` — all constants, each cited
Grouped: grid, PBL climatology, contact fraction, fuzz growth, land threshold,
sub-hourly step, receptor inflow band, kernel spatial/temporal extents. Concrete
defaults from the literature scan:
- **Contact-layer fraction `f_c` (parcel counts as surface-coupled below
  `f_c·PBLH`): default 1.0**, with presets **0.5 (STILT, strong coupling)** and
  **1.5 (Sodemann)** as sensitivity bounds.
- **Climatological PBL:** daytime summer CONUS **~1.8 km**, west-to-east gradient
  ~3 km (Rockies) → ~1 km (East) (McGrath-Spangler & Denning 2012); **nocturnal
  collapse to ~200 m** (Seidel et al. 2012). *Decisive here*: our window runs to
  02 UTC (≈ 7–9 pm local), so the evening PBL collapse largely shuts off surface
  coupling for the late lags — a real physical effect the tool must reproduce.
- **Fuzz growth `σ(τ) = σ0 + α·D(τ)`, α ≈ 0.2** (Stohl 1998: single-trajectory
  error ≈ 20% of distance travelled, ~linear in age); `σ0` ~ one grid cell.
- **Kernel extents:** spatial **~150 km** effective radius (Guillod et al. 2015
  ~140 km event domain; mesoscale CI), temporal **up to ~24 h** in general — but
  **capped by our data at ≤ `t_r − t_overpass` (≤ ~7 h here)**.
- **Land threshold** 0.5 (reuse `config.LAND_FRACTION_MIN` from the CAPE work).
- **Receptor inflow band `RECEPTOR_BAND` — a config knob** (user decision). A
  `(z_bottom, z_top)` altitude (or pressure) band defining which arriving parcels
  count as "what the surface cell sees." **Default = near-surface** (e.g. below
  ~1 km / within the PBL), for the clean surface-coupling / STILT receptor story;
  widen it toward a full low-troposphere column when a broader inflow is wanted.
  A band-width sensitivity plot (§3.6) makes the choice's effect visible. *Note:*
  tracking property changes at **all** height levels (e.g. to adjust the advected
  CAPE/CIN rather than just build a surface footprint) is a **likely separate
  feature** — kept out of v1 unless it turns out cheap on top of the ingest layer.

### 3.1 `trajectories.py` — ingest (Stage 1)
`load_granule(path)` and `load_day(dir) -> tidy parcel object`: one record per
`(granule, column, level, time)` with `lat, lon, alt, pres, t, q, q_excess,
parceltime` + stable `parcel_id` and `swath` tag (early/late). Assert units at
ingest (q in g/kg, alt in m, pres hPa); keep `parceltime` as an audited column but
use one canonical UTC clock everywhere else.
- **Verify:** plot any parcel/granule path on a CONUS map colored by time;
  round-trip vs raw; assert `time=0` columns are single-lat/lon stacks; assert no
  mid-trajectory NaN.

### 3.2 `pbl.py` — boundary-layer depth (pluggable `PBLModel`)
`(lat, lon, time_utc) -> depth_m`. v1 `ClimatologicalPBL`: diurnal curve (deep
mid-afternoon → ~200 m nocturnal) with the west–east amplitude gradient.
`ConstantPBL(d)` for tests/ablation. Stubs with the same signature:
`ReanalysisPBL` (ERA5 `blh`), `SoilMoisturePBL` (drier soil → deeper PBL) —
documented, not built in v1.
- **Verify:** plot PBL(t) over a point across the 7 times (must show the evening
  collapse); unit-test curve endpoints.

### 3.3 `contact.py` — surface-coupling weight (`ContactWeight`)
`w(alt, pbl) -> [0,1]`: STILT-style, ~1 well inside `f_c·PBL`, smooth taper to 0
at the top, 0 above. `f_c` and taper from config.
- **Verify:** for one low trajectory, plot `alt`, `PBL`, `w_contact` over the 7
  times together; analytic tests (surface→1, 2·PBL→0, monotone taper). Report the
  **in-PBL fraction** as a diagnostic so the assumption is visible (defends
  against the parcels-not-actually-in-PBL risk).

### 3.4 `resample.py` — sub-hourly interpolation
Linear-in-time `(lat, lon, alt)` between the stored hourly points to a fine step
(config; ~10 min) so the footprint integral samples the whole path.
- **Verify:** interpolated endpoints equal stored points; path length ≥
  straight-line; constant-velocity parcel interpolates exactly.

### 3.5 `fuzz.py` — trajectory-uncertainty kernel (pluggable `FuzzKernel`)
Deposit a normalized 2-D Gaussian of width `σ(τ)` at each point. **Two backends:**
(a) `StohlFuzz` — `σ(τ)=σ0+α·D(τ)`, `α≈0.2` (theoretical); (b) `EmpiricalFuzz` —
drive `σ` from the fullgrid **`_std`/`N`** box-spread (data-driven). A single
`fuzziness` scalar multiplies `σ` so the user dials kernels sharper/blurrier.
- **Verify:** deposit sums to 1 (mass conservation); `σ=0`→nearest-cell delta;
  `σ(τ)` monotone in τ; symmetric input→symmetric blob.

### 3.6 `footprint.py` — the kernel builder (heart)
`build_footprint(parcels, receptor, pbl, contact, fuzz, land)` and `build_all(...)`
over every (target cell × arrival time) with contributors. Implements the §2 sum;
returns the physical footprint and the normalized kernel on a **relative source
window** (§4). Receptor inflow layer = the `RECEPTOR_BAND` config knob (§3.0):
default near-surface, widenable to a full low-troposphere column.
- **Verify:** (a) single straight constant-speed parcel in constant PBL → fuzzed
  line of known length/direction; (b) τ=0 mass sits on the receptor cell;
  (c) footprint lies **upwind** (mean displacement · local `u,v` from fullgrid <
  0 for backward lag); (d) normalized kernel sums to 1; (e) above-PBL parcels
  contribute nothing; (f) sensitivity of result to the inflow-band width.

### 3.7 `discount.py` — optional Sodemann-style rain-out discount
Using `q_excess` (condensation en route): down-weight a source cell's
contribution if the parcel **condensed after leaving it** (its low-level history
was partly reset by rain). This is the one piece of the Sodemann scheme this data
*can* support (Δq-uptake cannot). Off by default; a cited, testable add-on.
- **Verify:** a trajectory with no `q_excess` → identical to no-discount; a
  condensation event zeroes/reduces upstream weights only.

### 3.8 `io.py` — NetCDF writer/reader
Write per-day kernels + applied predictor (§4 schema) with CF-ish attrs, units,
and provenance (input granules, config hash, PBL/fuzz model names). Round-trip.
- **Verify:** write→read identity; attrs record the config used.

### 3.9 `apply.py` — convolve footprint with a surface field (predictor-agnostic)
`apply_kernel(kernel_ds, surface_da) -> influence(x_r, t_r)`; produces the scalar
predictor column for the analysis table.
- **Verify:** uniform surface field → returns that constant (kernel sums to 1);
  delta surface field → returns the kernel value at that cell.

### 3.10 `plotting.py` — the required diagnostics (thin)
- `plot_kernel_at_lag(lat, lon, arrival_time, lag)` — map of where the arriving
  air was `lag` hours earlier, receptor marked.
- `plot_spatial_influence(...)` — lag-integrated source map (spatial marginal).
- `plot_temporal_influence(...)` — influence vs lag, split land/ocean and
  in/out-PBL so the coupling (and the evening shut-off) is legible.
- `plot_trajectory_bundle(...)` — contributing parcels' paths.
- `plot_coverage(...)` — which cells got kernels, mean available lag.

---

## 4. NetCDF output schema

A dense 6-D sensitivity (target×arrival×source×lag) is ~10⁸ mostly-zero cells —
rejected. Provide:
- **(a) Relative-window kernel (primary):** source axis as *offsets* from the
  receptor — `dims=(arrival_time, target_lat, target_lon, lag, dlat, dlon)`,
  `dlat/dlon` a small window (~±10°). zlib-compresses well; "kernel over one cell"
  is a trivial slice (what the plots need).
- **(b) Sparse COO (fallback/exact):** one row per nonzero `(recep_lat, recep_lon,
  arrival, src_lat, src_lon, lag, weight)`. Lossless, good for stacking days.
- **Applied-predictor file:** `(arrival_time, lat, lon) -> influence_S` for a
  chosen surface field — tiny, the drop-in predictor the Gini analysis consumes.

All files carry provenance attrs (granules, config, PBL/fuzz names, code version).

---

## 5. Adversarial robustness — failure modes and guards

1. **`q` is a conserved tracer → no in-situ soil-moisture signal.** *This is the
   headline caveat.* Guard: the tool never infers soil moisture from parcel `q`;
   all SM physics enters via the external convolution (§0, §3.9). The plan states
   this in the module docstrings so no one later mistakes `q` for a moistening
   diagnostic.
2. **Forward-only, ≤~7 h trajectories → truncated history**, and **swath-dependent
   lag** (late-swath 205/206 parcels start at 20:35, so a 21 UTC receptor has ~25
   min of them vs ~2 h of early-swath parcels). Guard: `lag` capped at
   `t_r − t_overpass`; every kernel records **max available lag**, **fraction of
   parcels retained**, and **per-swath contribution**, so "no influence" is never
   confused with "short record."
3. **Coarse hourly sampling → skipped cells.** Guard: §3.4 sub-hourly interp;
   path-length test.
4. **Terrain-limited near-surface coverage.** Western/high-terrain granules have
   ~no near-surface parcels, so receptors there are uncomputable. Guard: coverage
   map is first-class; empty cells → NaN, never zero; documented as a data limit,
   not a bug.
5. **Central/eastern-only, no west coast.** Guard: domain of validity stated;
   kernels only where parcels exist.
6. **Ocean drift.** Far-eastern parcels drift over the Atlantic; SM is undefined
   there. Guard: `lsm.nc` land mask zeros ocean; temporal-marginal plot separates
   land/ocean so ocean legs are visible, not silently dropped.
7. **Nocturnal PBL shut-off.** Our window reaches evening; the PBL collapses to
   ~200 m, so late-lag surface coupling should nearly vanish. Guard: the diurnal
   `ClimatologicalPBL` produces this; a verification figure must show the
   daytime-peaked, evening-suppressed temporal marginal.
8. **Receptor vertical definition.** "Air over a cell" is a column; coupling is
   near-surface. Guard: receptor = the `RECEPTOR_BAND` config knob (§3.0, default
   near-surface, widenable to a full low-troposphere column); a band-width
   sensitivity plot is a verification step.
9. **Fuzz-bandwidth arbitrariness.** Guard: default from Stohl's 20%-of-distance
   rule *and* the data-driven `_std` option; `fuzziness` is the one documented
   knob, shown in a sensitivity figure.
10. **Mass conservation / double counting** across granules, levels, swaths, and
    the fuzz deposit. Guard: per-parcel weights normalized before summation; unit
    test that deposit conserves mass; global check total mass ≈ (contributing
    parcels × contact time).
11. **Clock correctness.** `parceltime` equals the `time` axis except the
    per-granule `time[0]` release; the only real offset is that release moment.
    Guard: one canonical UTC clock; a test that per-granule `time[0]` sits in its
    expected scan window and that 21–02 UTC align across granules.
12. **Units** (g/kg vs kg/kg, m vs km, hPa). Guard: asserted at ingest against the
    audited ranges; config carries units.
13. **Off-footprint but valid drift.** Parcels leave the release box yet stay
    non-NaN. Guard: integrate along the true path regardless of the release
    footprint; only the `lsm`/domain masks bound the source.

---

## 6. Verification & diagnostics (the user's explicit ask)

For a chosen receptor (e.g. a central-plains cell at 21–02 UTC, where near-surface
parcels exist) the notebook shows:
- **Kernel at a single previous hour** — `plot_kernel_at_lag`: where the arriving
  air was 1 h, 2 h, … earlier.
- **Spatial influence** — lag-integrated source map (spatial marginal).
- **Temporal influence** — influence vs lag, land/ocean and in/out-PBL split
  (should peak in daylight, fall off into the evening — §5.7).
- **Contributing trajectory bundle** — the actual parcel paths behind it.
- **Upwind sanity overlay** — footprint vs fullgrid low-level `u,v` (must point
  upwind).
- **Coverage / so-what report** per day: cells with kernels, mean available lag,
  land-contact fraction, per-swath split.

Plus the analytic unit-test suite (§3).

---

## 7. Phased build order

- **Phase 0 — Audit & config** *(done; results in §0–§1)*: lock data facts and
  literature parameters into `config.py`.
- **Phase 1 — Ingest + trajectory plot** (§3.1, §3.10 path plot). Round-trip test.
- **Phase 2 — PBL + contact weight** (§3.2, §3.3). `w_contact(t)` plotted; tests.
- **Phase 3 — Fuzz + sub-hourly resample** (§3.4, §3.5). Single-parcel deposit;
  mass-conservation tests.
- **Phase 4 — Footprint builder** (§3.6). The three required kernel plots for one
  receptor; upwind + normalization + inflow-band-sensitivity tests.
- **Phase 5 — NetCDF I/O** (§3.8, §4). Written kernel + round-trip; provenance.
- **Phase 6 — Apply to a surface field** (§3.9): the demo convolves with **SMAP
  surface soil moisture (`smsfc`)** (user decision — root-zone `qlay1` too deep
  for this coupling), producing an `influence_smap_sfc` predictor column ready to
  drop into the Gini table beside `mu_cape` (reuses SMAP from
  `FCST_SMAP_MRMS_*.nc`). The apply-step is predictor-agnostic, so any other
  surface field is a one-line swap. Recommend a `land_frac`-only run first as a
  pure geometry/plumbing check before wiring in real soil moisture.
- **Phase 7 — Extensibility demos**: `ConstantPBL`→`ClimatologicalPBL`; dial
  `fuzziness`; `StohlFuzz`→`EmpiricalFuzz` (fullgrid `_std`); show where ERA5
  `blh` and the `discount.py` rain-out module plug in — each a one-line change,
  proving the interfaces.

---

## References (from the literature scan; DOIs)

- Sodemann, Schwierz & Wernli (2008), *JGR* 113, D03107 — moisture-source
  diagnostic; PBL gate at 1.5·z_h, Δq thresholds, rain-out discounting.
  https://doi.org/10.1029/2007JD008503
- Lin et al. (2003), *JGR* 108(D16), 4493 — STILT footprint = near-surface
  residence-time sensitivity. https://doi.org/10.1029/2002JD003161
- Fasoli et al. (2018), *GMD* 11, 2813 — STILT-R v2 footprint below 0.5·z_PBL.
  https://doi.org/10.5194/gmd-11-2813-2018
- Stohl (1998), *Atmos. Environ.* 32(6), 947 — trajectory error ≈ 20% of distance,
  ~linear in age (fuzz rule). https://doi.org/10.1016/S1352-2310(97)00457-3
- Guillod et al. (2015), *Nat. Commun.* 6, 6443 — spatial(−)/temporal(+) SM–rain
  coupling; ~140 km, 3–15 h scales. https://doi.org/10.1038/ncomms7443
- Tuttle & Salvucci (2016), *Science* 352, 825 — CONUS SM–precip sign flips by
  region (stay sign-agnostic). https://doi.org/10.1126/science.aaa7185
- McGrath-Spangler & Denning (2012), *JGR* 117, D15101 — N. American summer PBL
  ~1–3 km, west–east gradient. https://doi.org/10.1029/2012JD017615
- Seidel et al. (2012), *JGR* 117, D17106 — PBL diurnal cycle, nocturnal collapse.
  https://doi.org/10.1029/2012JD018143
- Stohl & James (2005), *J. Hydrometeorol.* 6, 961 — column E−P budget (why the
  PBL-gated per-cell method is preferred for surface attribution).
  https://doi.org/10.1175/JHM470.1
