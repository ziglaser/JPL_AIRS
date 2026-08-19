# Upwind soil-moisture index: a primer, then an adversarial review of HANDOFF.md

**Author:** Claude session, 18–19 August 2026, for Zach Glaser.
**Scope:** (1) the upwind soil-moisture index rebuilt pedagogically from
scratch, defining everything before using it; (2) an adversarial evaluation of
`HANDOFF.md` (the 17–18 Aug session's brief, on D:); (3) the production
implementation on the data actually available, including the Guo et al. (2024)
PBL-depth product in `data/PBL_depth/` and the full HYSPLIT archive on the
compute cluster.

**Revision note (19 Aug).** This supersedes the 18 Aug draft, which (a) put
the evaluation before the definitions and used Φ without ever defining it,
(b) wrongly assumed HYSPLIT trajectories exist only for the demo day — the
full archive is on the cluster, so the kernel machinery *is* the production
path, and (c) recommended a climatological-anomaly soil-moisture input, which
Zach correctly objected to: the latent/sensible partition is set by
*absolute* wetness (§1.7).

---

## 0. Verdict in three sentences

The physics in HANDOFF.md is sound, well-grounded in the footprint
literature, and most of its self-criticisms are correct; its day-1 priorities
(wire the energy weight, fix the containment defect) are the right ones. Its
main omissions are the **scale-up** — nothing about compute, storage, QA, or
the per-day PBL/SMAP inputs needed to sweep the machinery over ~1 800 days —
and the **geometric pathway**, which it correctly identifies as the main
missing physics but leaves blocked, even though the new assessed PBLH product
plus the forecast LFC already on the grid unlock it with zero trajectory work
(§2, F2). Its confusing presentation is structural, not mathematical: it
interleaves the simple concept with machinery that exists only because demo
parcels are sparse (§2, F7); §1 below is the untangled version.

---

## 1. The idea, built up from nothing

*(Every symbol defined at first use. One worked example carried throughout:
5 June 2019, receptor cell (40.5°N, −90.5°E), arrival 00 UTC, mean
boundary-layer wind 7 m s⁻¹ from the southwest.)*

### 1.1 The question

The standard soil-moisture predictor asks: *how wet is the soil under the
cell where convection might fire?* But the air that fires spent all afternoon
moving. At 7 m s⁻¹, air arriving at 00 UTC was **126 km away at 19 UTC** —
more than a full 1° cell upwind. The soil that heated and moistened this air
is not the soil under the receptor. We want a predictor that reads the soil
**along the air's actual path**, weighted by *when the ground could actually
influence it and how strongly*.

### 1.2 Step 1 — Where was the air? (the trajectory)

HYSPLIT integrates parcel positions $\mathbf{x}_k$ forward through the WRF
wind field from the AIRS overpass ($k=0$) to the horizon end ($k=N$), at
step length $\Delta t_k$ (s; 600 in the resampled pipeline). The path
$\{(\mathbf{x}_k, t_k)\}$ is the entire geometric content of the method —
everything else is deciding how much each path point counts.

### 1.3 Step 2 — When did the ground matter? (contact)

Air feels the soil only while inside the **planetary boundary layer** (PBL) —
the lowest ~0.2–2 km that turbulence stirs top-to-bottom on a ~10-minute
timescale. Above the PBL top $z_i$ (m), the surface below is irrelevant.
Define a contact gate

$$\Lambda_k \in [0,1]$$

equal to 1 for a parcel well inside the mixed layer at step $k$, tapering to
0 through the entrainment zone near $z_i$ (implemented in
`contact.contact_weight` as a linear taper over the top 25 % of the contact
layer). The gate is the reason an elevated moist parcel that never touches
the PBL correctly gets *zero* soil influence, however extreme the soil under
its track — something no point-collocated predictor can express.

### 1.4 Step 3 — How hard did each contact hour count? (the energy weight)

Soil influences air through the surface heat fluxes: sensible $H$ and latent
$\mathrm{LE}$ (both W m⁻²). We have neither, but their **sum** is set by the
energy supply, not by the soil:

$$H + \mathrm{LE} = R_n - G \approx a\,\mathrm{DSWF}, \qquad a \approx 0.55$$

where $R_n$ is net radiation, $G$ ground heat flux, DSWF the downward
shortwave flux at the surface (W m⁻²; computable from pure solar geometry,
`insolation.py`), and $a$ a bulk coefficient absorbing albedo, net longwave,
and $G$. The *soil* only chooses the **split** of that sum — dry soil sends
the energy up as heat, wet soil as moisture. Hence the design:

- **weight** each path step by the energy the surface could deliver,
- **read** the soil wetness under it to learn how that energy was split.

Define the per-step weight (J m⁻² of available surface energy):

$$w_k \;=\; a\,\mathrm{DSWF}(\mathbf{x}_k, t_k)\;\Lambda_k\;\ell_k\;\Delta t_k$$

with $\ell_k \in [0,1]$ the land fraction under $\mathbf{x}_k$. Worked
numbers at 95°W in June: DSWF ≈ 890 W m⁻² at 19 UTC (≈ 12:45 local solar),
550 at 23 UTC, 220 at 01 UTC, 0 by 03 UTC — the weight dies through the
evening exactly as the surface decouples, so day/night gating comes mostly
for free.

### 1.5 Step 4 — The two accumulated numbers: Φ and m*

**Φ (capacity, J m⁻²)** is the total available surface energy delivered to
the air along its coupled path — the *opportunity* for soil influence,
regardless of what the soil was:

$$\boxed{\;\Phi \;=\; \sum_{k=1}^{N} w_k\;}$$

Worked: 3 h of full contact at path-mean DSWF 750 W m⁻² gives
$\Phi = 0.55 \times 750 \times 10\,800 \approx 4.5$ MJ m⁻². Φ carries **no
soil information** — it is set by insolation, PBL contact, and horizon.

**m\* (dilution mass, kg m⁻²)** answers *how much air that energy was spread
through*. Energy injected at noon into a 700 m-deep layer is stirred through
an 1 800 m layer by evening. Treating the surface input as a passive tracer
(entrainment adds mass carrying none of it), the total input is conserved
and ends up diluted through the **deepest** layer the air ever occupied:

$$m^* = \rho\,\max_k z_i(\mathbf{x}_k, t_k), \qquad
\Omega = \Phi / m^* \;\;[\mathrm{J\,kg^{-1}}]$$

with $\rho \approx 1.10$ kg m⁻³. Worked: $m^* = 1.10 \times 1800 = 1980$
kg m⁻², $\Omega \approx 2250$ J kg⁻¹ ≈ 2.2 K of warming or 0.9 g kg⁻¹ of
moistening if delivered purely one way. Divide once by the maximum — never
per step (the instantaneous $1/m_k$ weighting and later entrainment dilution
cancel; see F5 for the fine print).

**Which of Φ, m*, Ω is the feature? Ω** (19 Aug revision; Zach's point).
m* has exactly two predictor roles: a *geometric* one (deep PBL → top nearer
the LFC), which Γ_gap (§1.8) expresses better because it references $z_i$ to
the threshold that actually matters, and a *dilution* one, which is only
meaningful as the denominator of Φ — i.e. inside Ω. HANDOFF §7.2's argument
against Ω ("dividing by m* cancels the mechanism") was valid only when m*
was the sole carrier of PBL information in the feature set; once Γ_gap
exists, Ω's dependence on m* is physics (the surface imprint genuinely *is*
more dilute in a deeper layer), not cancelled signal. So the intensive
$\Omega$ — surviving surface energy per kg of arriving air — is the feature;
Φ and m* are retained as diagnostics/ablation columns only (and Φ's
independent variance is mostly time-of-place artifacts anyway, F3).

**Note (review round): Φ is a parcel-ensemble mean, not an ensemble sum.**
The receptor is one air column, and the parcels are Monte Carlo samples of
its history; the code therefore computes Φ as the footprint sum divided by
`n_parcels` — the Monte Carlo estimate of the *single-column* path integral
$\sum_k w_k$ above. Doubling the number of released parcels halves each
parcel's share and leaves Φ unchanged (only its sampling noise shrinks), so
Φ cannot scale with sample size, and receptors with different `n_parcels`
are directly comparable.

### 1.6 Step 5 — The soil-reading index: Ψ

The energy-weighted average of soil wetness $S$ along the path:

$$\boxed{\;\Psi[S] \;=\; \frac{\sum_k w_k\,S(\mathbf{x}_k, t_k)}
{\sum_k w_k}\;}$$

A weighted mean: bounded, unit-preserving (m³ m⁻³ for raw SMAP), and
comparable across days, horizons, and PBL depths. Equivalent **kernel** view:
collect the weights landing in each upwind map cell $x$ into $K(x)$; then
$\Psi = \langle K, S\rangle / \langle K, 1\rangle$ and
$\Phi = \langle K, a\,\mathrm{DSWF}\rangle$. Same numbers; the kernel form
makes field swaps (root-zone, gradients, land cover) one-liners, makes the
influence region plottable, and is the established source–receptor formalism
(Seibert & Frank 2004; STILT, Lin et al. 2003).

**Note (review round): the kernel file's per-lag normalization is a
convention, not lost physics.** In the stored kernel each hourly lag slice
is normalized to unit sum — a plotting/mapping convention that makes every
lag's source map directly comparable and cleanly NaN-renormalizable. The
physical hour-to-hour energy weight the normalization divides out is carried
alongside as `lag_weight` (one number per lag: the slice's
pre-normalization mass), and the Ψ contraction
(`apply.apply_kernel(lag_weights=...)`) multiplies it back in before summing
over lags, so the unnormalized $w_k$ sums are reproduced exactly. The stored
per-lag maps answer "where, within this hour?", `lag_weight` answers "how
much did this hour count?" — together they close the gap HANDOFF §8.3
flagged: a per-lag-normalized kernel contracted *without* the weights would
silently equalize every contact hour.

### 1.7 Which S goes in? Absolute wetness, in two forms — not the anomaly

The partitioning physics runs on **absolute** wetness: evaporative fraction
$\mathrm{EF} = \mathrm{LE}/(H+\mathrm{LE})$ is a saturating function of soil
water *availability*, roughly
$\mathrm{EF} \approx f\!\big((\mathrm{SM}-\mathrm{SM_{wilt}})/(\mathrm{SM_{fc}}-\mathrm{SM_{wilt}})\big)$
— soil-limited below a critical wetness, energy-limited above. A Georgia
cell at its climatological mean partitions latent-dominant; a Nebraska cell
at *its* mean does not. A difference-from-climatology anomaly erases exactly
that level information, so **an anomaly-only index cannot represent the
partition** (Zach's objection; the 18 Aug draft had this wrong).

Two distinct jobs hide under "normalize the soil moisture":

1. **Texture normalization (physical).** Raw m³ m⁻³ is not comparable across
   cells: 0.25 is near-saturated sand but near-wilting clay. The comparable
   quantity is the wetness index above, which needs wilting/field-capacity
   maps we don't have. The per-cell historical CDF $F(\mathrm{SM})$ is a
   data-driven surrogate — each cell's observed range roughly brackets its
   wilting↔field-capacity span. This makes the feature *more* faithful to
   partitioning.
2. **Removing standing geography (statistical).** The CDF also maps every
   cell's typical state to 0.5, which is what lets it answer *"does today's
   soil state carry signal beyond a static map of CONUS?"* — the interesting
   claim for the skill framework, but a different question from "what is the
   partition."

No single transform does both: raw SM preserves the level but confounds
texture; the per-cell CDF fixes texture but erases the level (it maps every
cell's median to 0.5, so a perennially wet cell and a perennially dry one
look identical at their own medians).

**The core form: the cardinal monthly anomaly `psi_anom`** (Zach's decision,
19 Aug). Convolve

$$S'(\mathbf{x}, t) \;=\; \mathrm{SM}(\mathbf{x}, t)
\;-\; \overline{\mathrm{SM}}\big(\mathrm{month}(t), \mathbf{x}\big)$$

where $\overline{\mathrm{SM}}$ is the per-1°-cell mean of `SMAP_L4_smsfc_av`
pooled by calendar month over the full 2016–2021 record (all days, all 5
analysis hours; built by `scripts/build_smap_l4_baseline.py`; to be extended
as the record grows). Rationale: this keeps the *relative* property (removes
standing geography — the "relearned a map of CONUS" hazard under
region/date-blocked CV) while staying **cardinal rather than ordinal**: an
m³ m⁻³ departure is linear in water, the same currency as the latent flux it
modulates, and preserves the magnitude of departures that the percentile
form flattens into ranks. Because the contraction is linear, convolving $S'$
directly is exactly equivalent to convolving raw SM and subtracting the
convolved baseline — the spatially varying reference is handled for free.

Scale constraint, stated honestly: the EF-curve argument (§1.9) wants a Ψ
scale where equal values mean equal availability everywhere. The cardinal
anomaly does not texture-normalize (a 0.05 m³ m⁻³ departure moves sand and
clay differently through the EF curve) — that job is deferred entirely to
the eventual wetness index
$W = (\mathrm{SM}-\mathrm{SM_{wilt}})/(\mathrm{SM_{fc}}-\mathrm{SM_{wilt}})$
from soil-texture maps (STATSGO/POLARIS), which would preserve cardinality
*and* fix texture. Until then, `psi_pct` (per-cell CDF: texture-corrected
but ordinal and level-erasing) and `psi_raw` (absolute level, geography-
confounded) sit in the ablation tier, with a decision rule for `psi_raw`:
skill beyond `psi_anom` that dies under region-blocked CV is geography
leakage (drop); skill that survives is a real absolute-level effect
(build $W$).

Drop the domain z-score (`standardize()`; a longitude proxy, F4). The
"wet path / dry path" sign language applies to `psi_anom` and `psi_pct`;
`psi_raw` is a level, not a direction.

### 1.8 Step 6 — The second pathway: geometry, no trajectory needed

Soil moisture also acts **geometrically**: dry soil deepens the PBL until
$z_i$ reaches the level of free convection ($z_{\rm LFC}$, m); wet soil
lowers the LFC toward a shallower PBL top. Neither Ψ nor Φ can express this
race. With an assessed PBLH and the forecast LFC both on the grid, the gap

$$\Gamma_{\rm gap} = z_{\rm LFC} - z_i$$

per (day, slot, cell) is one subtraction — and it is the lever the coupling
literature (Ek & Holtslag 2004; Findell & Eltahir 2003) says matters most.

### 1.9 The feature set, and why Ψ and Ω stay separate

Core features (each carries a distinct physical axis):

| Feature | Question it answers | Needs |
|---|---|---|
| `psi_anom` | *which way* was the surface energy split — how anomalously wet/dry (m³ m⁻³, vs the cell's monthly baseline) was the soil the air was coupled to? | trajectories + SMAP_L4 + monthly baseline + solar geometry |
| `omega` = Φ/m* | *how much* surviving surface energy per kg of arriving air? | path + solar geometry + PBLH (assessed — F2) |
| `gamma_gap` | how close is the PBL top to free convection? | PBLH + FCST_LFC, **no trajectory** |
| `psi_grad_2h` | was there a soil-moisture *boundary* nearby recently (mesoscale circulations)? | last-2 h weights, gradient field |
| `s_endpoint_anom` | the naive point predictor — the control Ψ must beat | SMAP_L4 at receptor |

Ablation/honesty columns (not core predictors): `psi_raw` and `psi_pct`
(region-blocked decision rule, §1.7), `phi`, `m_star` (verify importance
≈ 0 given `omega` and `gamma_gap`), `s_endpoint_raw`, `n_parcels`,
`coverage`, `n_front_x`.

**Why Ψ and Ω are two features rather than one combined index.** The physical
statement is: *Ω joules per kilogram were delivered, split between warming
and moistening with evaporative fraction set by the soil wetness Ψ* —
delivered warming ≈ $(1-\mathrm{EF}(\Psi))\,\Omega/c_p$, moistening ≈
$\mathrm{EF}(\Psi)\,\Omega/L_v$. Combining them into one number requires
committing to the EF(SM) curve, which is the least-trusted relation in
land–atmosphere physics (nonlinear, saturating, texture- and
vegetation-dependent), and any product collapses physically opposite cases
(extreme-dry soil × weak delivery vs mild soil × strong delivery) onto one
value. A forest constructs the interaction itself from the two axes. And if
one *did* trust an EF curve, the "combined" version is the pair
$(A_s, A_l) = ((1{-}\mathrm{EF})\Omega,\ \mathrm{EF}\,\Omega)$ — an
invertible remapping of $(\Psi, \Omega)$, so the forest gains nothing from
the transformation. Two axes in, interaction learned, curve left to the
data.

---

## 2. Adversarial evaluation of HANDOFF.md

Findings ranked by severity. Each: the claim, the problem, the fix. Symbols
as defined in §1.

### F1 — IMPORTANT: the plan never discusses the production sweep

The three-day plan (HANDOFF §10) wires the energy weight, runs
`build_predictors`, and evaluates — all on the 2019-06-05 demo day. The
deliverable is a predictor column in `FCST_SMAP_MRMS` (2016–2021, 28×43
cells, 7 afternoon slots ≈ 9 000 rows/year). The full HYSPLIT archive exists
on the compute cluster, so the machinery *can* sweep it — but the HANDOFF is
silent on everything that makes that real: per-day job structure, the fact
that dense kernels are ~0.6 GB/variable/day (≈ 1 TB over the record — the
kernels must not be persisted; the predictors, ~kB/day, are the product),
per-day PBL and SMAP inputs, and at-scale QA that a one-day demo never
exposes (coverage per day, thin-parcel receptors, terrain-limited western
granules). §4 supplies the missing plan.

*(Retraction: the 18 Aug draft claimed trajectories exist only for the demo
day and proposed a cheap back-trajectory on the `FCST_u/v` gridded winds as
the product. Wrong premise — only the D: copy is demo-only. The cheap
version survives only as an optional robustness check, §4.5.)*

### F2 — IMPORTANT: `m_star` as shipped is an information-free feature, and the geometric pathway is no longer blocked

`predictors.m_star` defaults to `ClimatologicalPBL`, a smooth deterministic
function of (local hour, longitude). A forest that already sees location and
time gains nothing from it — worse, it soaks up split importance as a
geography proxy. HANDOFF is honest that the PBL pathway is "absent" (§7.7)
but ships the feature anyway.

It can now be made real: `data/PBL_depth/Guo2024_model/` is an **assessed**
(ML-merged, radiosonde/lidar-constrained) PBLH at 0.25°/3-hourly, 2017–2021 —
it carries the actual day's boundary-layer depth, including its
soil-moisture-driven part. That also unlocks $\Gamma_{\rm gap}$ (§1.8):
`FCST_MU_LFC`/`FCST_MML_LFC` are already in the match-up file, so the
highest-value feature in this whole effort costs one subtraction. (Verify
the LFC datum — AGL vs ASL — first; Guo PBLH is height above ground.)

Note HANDOFF §9.2's "circularity is void" argument inverts here: an assessed
PBLH deliberately re-imports the day's soil signal. That is the *mechanism*,
not contamination — but PBLH-derived and SMAP-derived features share a
cause; expect correlated importances and say so.

### F3 — IMPORTANT: Φ is nearly deterministic, and its error is anti-correlated with the label

Under clear-sky DSWF and a climatological PBL, Φ (§1.5) for a fixed horizon
is almost a closed-form function of (latitude, arrival hour, day of year);
residual variance is mostly coverage geometry (which lags are populated,
terrain-limited parcels). Consequences HANDOFF understates:

1. As a feature, Φ largely duplicates (lat, slot, month). Before trusting
   any importance on it, regress Φ on those three and report the residual.
2. The clear-sky assumption fails **most on pre-convective days** — the days
   with the positive label — so the error is anti-correlated with the
   outcome. "Treat as upper bound, rely on rankings" (HANDOFF §6.6) is weak
   comfort: day-to-day rankings are exactly what cloud cover corrupts. Φ is
   a diagnostic, not a core feature (§1.9); the clear-sky caveat propagates
   into Ω, which inherits Φ's numerator.

### F4 — MODERATE: the anomaly treatment needed rework (now done in §1.7)

Two separate defects in HANDOFF's soil-moisture handling. (a) Its v1
`standardize()` subtracts the *spatial* domain mean, so on any one day
`psi_*_std` is dominated by the standing east-wet/west-dry gradient — a
longitude proxy the forest will read as geography. (b) More fundamentally,
any anomaly-only formulation discards the absolute wetness level that the
EF partition physics actually runs on (Zach's point). Resolution: the two
absolute forms of §1.7 (`psi_raw`, `psi_pct`), both single contractions by
the linearity HANDOFF itself proves in §8.4; no difference anomaly, no
domain z-score. The per-cell CDF is buildable from the full SMAP_L4 record —
HANDOFF's "skip the climatology" time-saving argument was only ever valid
for a one-day demo.

### F5 — MODERATE: the "exact" dilution result quietly swaps a column for a parcel

The boxed result $c_N = \frac{1}{m^*}\int H\,dt$ (HANDOFF §6.4) is exact for
a *single Lagrangian column* growing in place by entrainment. A trajectory
crosses columns of different depth, so variation of $m_k$ along a path is
partly **spatial**, and $m^* = \max_k m_k$ conflates "the PBL grew (dilution
happened)" with "the parcel drifted over the deep-PBL High Plains (no
dilution happened)". Second-order over a 2–7 h afternoon path, and the
practical prescription (divide once by the max, never per step) survives —
but call it a well-motivated approximation, not "exact," or a reviewer will.

### F6 — AGREE with HANDOFF §8.6, plus one addition: NaN renormalization moves the source region

The containment order-statistic defect is real: at $n = 4$ parcels,
`dist[ceil(0.9·n)−1]` is the maximum, so containment silently does nothing,
and at moderate $n$ it is noisy — sample size leaks into Ψ and will look
like signal. Addition: `apply_kernel`'s renormalization over the
*finite-SMAP* retained weight means two receptors with identical kernels but
different SMAP gap patterns average over different effective source regions.
`min_coverage=0.5` truncates the worst cases; also **export the retained
coverage fraction as a column** so the forest (and you) can see it.

### F7 — STRUCTURAL: the confusion is the parcel machinery, not the physics

HANDOFF interleaves the concept (§1 here, six defined quantities) with
machinery that exists only because demo parcels are ~4 sparse columns per
cell: Stohl fuzz, containment circles, arrival membership, per-lag
normalization, NaN-lag conventions. Those are *implementation* answers to
sparse sampling, not part of the idea — and presenting them inside the
derivation is why the document reads as confusing. Separate the two layers
in any write-up: the index (§1) in one section, the sparse-parcel estimator
of it in another.

### F8 — PROCESS: the front flags are named and never wired

HANDOFF §7.6 correctly identifies three roles for fronts (validity flag,
confounder, collinearity with |∇SM|) and then nothing uses them. The repo
now has front products (`data/front_id/`, `src/front_formats/`, dataset-v9
front flags), so `n_front_crossings` along a path and `min_dist_to_front`
at arrival are cheap. At minimum wire the validity flag: a path that crosses
a front is a broken premise (the air mass changed), and frontal days are
precisely the high-CAPE days that dominate the label.

### What survives the attack (worth saying)

- **The kernel identity** (HANDOFF §6.8): index = ⟨kernel, field⟩ — right,
  standard, and the reason field swaps are free.
- **Separate features, never composites** (§7.1–7.2) — right, and matches
  how the convection_skill RF work is already run.
- **Label-leakage guards** (§7.5): fixed forward horizons; split by
  date/region — right, consistent with the cell-day/block-bootstrap
  conventions in the hypothesis battery.
- **The resolution analysis** (§7.8): decorrelation-length reasoning is
  sound; "the method earns its keep on 0–5 cm, not root-zone" is the right
  falsifiable prediction to carry forward.
- **q is a conserved tracer** → geometric footprint only; rain-out discount
  exact. Verified against the data previously; correct.
- **Day-1 priorities**: the energy-weight patch (§8.5) and containment fix
  (§8.6) are the correct first moves for the production sweep too.

---

## 3. What is actually available (verified 18–19 Aug 2026)

| Need | Have | Where | Gap |
|---|---|---|---|
| Trajectories, every day | full HYSPLIT archive (per Zach) | compute cluster | confirm format matches demo `nogrid_*` granules so `trajectories.load_day()` works unmodified |
| Trajectories, demo day | `nogrid_*`/`fullgrid_*` 2019-06-05 | `data/HYSPLIT_demo/` (D:) | — |
| Soil moisture, 3-hourly | `SMAP_L4_smsfc_av` (+sd, gradients) at 5 analysis hours 16:30–28:30 UTC | `FCST_SMAP_MRMS_*.nc` | per-cell CDF $F(\mathrm{SM})$ to be built once from the full record |
| PBLH climatology | 1° monthly-diurnal, built | `PBL_depth/derived/PBL_climatology_1deg_global_lst_2017-2021.nc` | conus/utc variant one command away |
| PBLH assessed | 0.25°, 3-hourly, 2017–2021 | `PBL_depth/Guo2024_model/` | needs a per-day 1° aggregate (cheap; same rule as the climatology script minus the pooling). **2016 has no assessed PBLH** |
| LFC / CAPE / CIN | `FCST_MU_*`, `FCST_MML_*` per (date, slot, cell) | match-up files | confirm AGL vs ASL for LFC |
| Steering winds (fallback only) | `FCST_u/v/w`, `FCST_alt` per (date, slot, cell) | match-up files | box-mean over all release levels (demo-day mean alt ≈ 2.1 km) — not a PBL wind |
| Land mask | `data/lsm.nc` + the PBLH product's own land mask | — | — |
| Fronts | `data/front_id/`, dataset-v9 flags | repo | wire `n_front_crossings` (F8) |

**On PBL "1° gridded vs hyperspecific":** the two axes are resolution (1° vs
0.25°) and **information (climatology vs the day's assessed value)** — and
the second matters far more. Recommended product input: **1° AND assessed**
(aggregate the raw 3-hourly Guo files per timestamp, not pooled into a
climatology). That is barely more work than the climatology join and is the
difference between `m_star`/`gamma_gap` carrying the day's soil signal and
carrying none (F2). "Hyperspecific" then means purely the 0.25°-along-path
variant (§4.4, run P2).

---

## 4. Production implementation (cluster sweep of the kernel machinery)

### 4.1 One-time preparations

```
# (a) footprint.py energy patch (HANDOFF §8.5): energy_fn=None backward-
#     compatible; ClearSkyAvailableEnergy() for production builds
# (b) containment fix (HANDOFF §8.6 / F6): containment_frac=None (or an
#     n-dependent rule) below ~20 parcels; record which rule fired
# (c) pbl.GriddedPBL: a PBLModel reading the per-day 1° assessed PBLH
#     aggregate; falls back to the monthly-diurnal climatology where the
#     assessed product has no coverage (and for all of 2016)
# (d) per-day 1° PBLH aggregate from Guo2024_model (reuse
#     build_pbl_climatology.py's pooling rule, keyed by timestamp):
#     PBLH_1deg[date, hour∈{0,3,…,21}, lat, lon], 2017–2021 CONUS
# (e) per-cell monthly SMAP_L4 baseline over the full record: DONE —
#     scripts/build_smap_l4_baseline.py →
#     data/soil_moisture/SMAP_L4_smsfc_monthly_baseline_2016-2021.nc
#     (S' = SM − baseline is the core anomaly, §1.7; a per-cell CDF lookup
#      is optional, ablation tier only. Dec–Feb are all-NaN: the L4 record
#      only covers the Mar–Nov season)
# (f) verify FCST_MML_LFC / FCST_MU_LFC datum (AGL vs ASL)
```

### 4.2 The per-day job (slurm array, one day per task)

```
day   = trajectories.load_day(date)                  # cluster HYSPLIT files
pbl   = GriddedPBL(date)                             # assessed, 1°, 3-hourly
kern  = footprint.build_all(day, pbl_model=pbl,
                            energy_fn=ClearSkyAvailableEnergy(),
                            containment_frac=n_dependent_rule)
sm_raw  = SMAP_L4 smsfc for the date (5 analysis slots; nearest-in-time)
sm_anom = sm_raw - sm_baseline(month(date))          # cardinal anomaly, §1.7

pred  = predictors.build_predictors(kern, sm_anom)   # psi_anom, phi, m_star,
                                                     # endpoint, meso, n_parcels
pred["omega"]           = pred["phi"] / pred["m_star"]  # the core intensity
                                                        # feature (§1.5, §1.9);
                                                        # phi & m_star kept as
                                                        # diagnostics only
pred["psi_raw"]         = predictors.psi(kern, sm_raw)  # ablation tier
pred["s_endpoint_raw"]  = predictors.endpoint_value(kern, sm_raw)
pred["gamma_gap"]      = FCST_MML_LFC(date) - PBLH_1deg(date)   # no trajectory
pred["coverage"]       = retained-kernel-weight fraction (F6)
pred["n_front_x"]      = front crossings along member paths (F8)

write pred → per-day nc (~kB)          # the product
DISCARD kern                           # ~0.6 GB/var/day; persist sparse COO
                                       # kernels only for sampled case-study days
# NaN, never fabricated: empty receptors, coverage < 0.5, all-water paths
```

Merge step: concatenate per-day predictor files onto the `FCST_SMAP_MRMS`
`(date, time, lat, lon)` axes as new `UPW_*` variables.

Drop from `build_predictors` before the sweep: `psi_surface_std` /
`standardize()` (F4) and the climatological-PBL `m_star` default (F2 — pass
`pbl_model=GriddedPBL` explicitly).

### 4.3 At-scale QA (things a one-day demo never shows)

- Per-day: land-cell fraction with `n_parcels > 0` per arrival slot; SMAP
  `min_coverage` rejection rate; count of receptors where the containment
  fallback rule fired.
- Distributional drift of `n_parcels` across seasons and the terrain-limited
  West (near-surface parcels are sparse over high terrain).
- The HANDOFF day-3 tests, run on a stratified sample of days, not one:
  energy-weight A/B (`UniformEnergy` vs clear-sky, r of the Ψ fields);
  `marginal_value(psi, s_endpoint)` split by layer (the §7.8 prediction);
  uniform-field and land-fraction closure checks (label-free validation).

### 4.3b Cluster deployment

Env vars required at submit time: `JPL_AIRS_REPO` (checkout), `JPL_AIRS_DATA`
(data root), `JPL_AIRS_RESULTS` (outputs), `UPWIND_YEAR` (per-year jobs),
`UPWIND_TRAJ_ROOT` (the HYSPLIT per-day archive root — the directory holding
the `wrf27km_<YYYYMMDD>` day dirs; the features `.sbatch` forwards it as
`--traj-root`, and a CLI `--traj-root` appended at submit time still wins),
and the cluster's `SBATCH_PARTITION` / `SBATCH_ACCOUNT` (never hardcoded in
the `.sbatch` files). Full 2017–2021 production run, in order:

```
export UPWIND_TRAJ_ROOT=/path/to/HYSPLIT/per_day_archive    # once
sbatch slurm/upwind_pblh_3hrly.sbatch                        # once (§4.1d)
for Y in 2017 2018 2019 2020 2021; do
    UPWIND_YEAR=$Y sbatch --array=0-274%40 slurm/upwind_features.sbatch
done                                                         # §4.2, per day
for Y in 2017 2018 2019 2020 2021; do
    UPWIND_YEAR=$Y sbatch slurm/upwind_merge.sbatch \
        --daily-dir "${JPL_AIRS_RESULTS:-results}/upwind_features/daily"
done                                                         # §4.2 merge
```

The merge's `--daily-dir` must be the features array's `--out-dir`
(default `<results>/upwind_features/daily`); it is passed explicitly because
`merge_upwind_features.py` defaults `--daily-dir` to `None`, a
trajectory-free-only build with all kernel-borne features NaN.

Array index → date: task *N* is the *N*-th day (0-based) of Mar 1–Nov 30,
the 275-day Mar–Nov season. All three jobs are idempotent (skip-if-exists /
cached), so a failed year is resubmitted with the same line. 2016 has no
assessed PBLH, so its `gamma_gap` / `pblh_anom` are honestly NaN.

### 4.4 The PBL ablation (the "hyperspecific" experiment)

Freeze everything in §4.2 and vary one input:

| Run | PBLH input | Reads |
|---|---|---|
| P0 | monthly-diurnal climatology (1°) | baseline `m_star`, `gamma_gap`, gate Λ |
| P1 | per-day assessed, 1° | does the day's PBL carry signal? (expect yes, mostly via `gamma_gap`) |
| P2 | per-day assessed, 0.25° sampled along the path | does sub-cell PBL structure add anything at 1° prediction scale? (expect marginal) |

ΔAUC / Δimportance between consecutive rows isolates the information axis
(P0→P1) from the resolution axis (P1→P2). The interesting claim to test:
P0→P1 large, P1→P2 small.

### 4.5 Optional robustness check: the steering-wind surrogate

A 2-D hourly back-trajectory on the match-up file's own `FCST_u/v`
(deep-layer steering wind), with the same $w_k$ weights, costs ≤7 bilinear
lookups per row and needs no cluster. Comparing its Ψ against the kernel Ψ
on a sample of days quantifies how sensitive the index is to the wind
representation — useful for the paper's robustness section and as a cheap
recomputation path — but it is not the product.

---

## 5. Open items

1. Confirm the cluster HYSPLIT files match the demo-day `nogrid_*` granule
   format (`trajectories.load_day()` unmodified?).
2. Confirm `FCST_MML_LFC` / `FCST_MU_LFC` datum (AGL vs ASL) before
   `gamma_gap`.
3. HANDOFF open questions 1–2 (level-ladder semantics; deterministic vs
   ensemble parcels) still stand and now matter at scale.
4. Wire the front-crossing validity flag once `front_id` products are
   finalized (F8).
5. Soil-texture wetness index (STATSGO/POLARIS) as the eventual replacement
   for the CDF surrogate (§1.7).
6. Sync this file to `D:\JPL_AIRS\src\trajectory_kernels\` next to
   HANDOFF.md (D: was unmounted when this revision was written; the 18 Aug
   draft there is superseded).
