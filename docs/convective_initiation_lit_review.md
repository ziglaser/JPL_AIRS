# Mechanisms of Convective Initiation: A Literature Review

**Purpose.** This review catalogs the mechanisms that the literature has theoretically and observationally linked to the initiation of deep moist convection (CI), organized so they can be mapped onto quantities **observable from AIRS soundings and SMAP soil moisture** (with HYSPLIT trajectories for advection). It is scoped to support extending Richardson et al. (2024), who showed that AIRS-derived CAPE — advected forward with forecast winds — skillfully predicts heavy hourly precipitation over the **central-eastern U.S. / Great Plains**, toward an improved extreme-convection signal that also incorporates SMAP.

**Scope constraints (per study design).**
- **Observable from AIRS:** temperature and moisture profiles → CAPE, CIN, lapse rates, mid-level EML/lid strength, column/lower-tropospheric humidity (PWAT), LCL/LFC, and (with advection) their time tendency.
- **Observable from SMAP:** surface (and derived root-zone) soil moisture — point values, spatial gradients/heterogeneity, and antecedent-memory time series.
- **Region:** Eastern U.S. and the Great Plains specifically (not the Sahel/semi-arid tropics where much of the mechanistic literature was developed). Where the strongest evidence comes from other regions, it is retained as a *mechanism analog* and paired with the U.S. counterpart.

**How to read it.** Mechanisms are grouped into categories (A–M). Each entry states the physical mechanism, the sign/nature of its effect on CI, its key references, and — most importantly — an **AIRS/SMAP observability + regional-relevance tag**:
- ✅ **In scope** — directly derivable from AIRS and/or SMAP and relevant to Eastern/Great Plains US.
- 🟡 **Partial** — the *environmental preconditioning* is AIRS/SMAP-observable, but the km-scale trigger itself is not.
- ⛔ **Out of scope** — sub-grid/transient or requires data we don't have; listed for completeness.

**Section N is the operational core:** an observable-keyed catalog of explicit causal chains (observable → mediator → CI) with quantified effect sizes, keyed strictly to AIRS + SMAP + reanalysis over the target region. Categories A–M provide the mechanistic background; Section N is what maps onto predictors.

---

## Conceptual framing: the ingredients-based view

Deep moist convection requires the near-simultaneous presence of three ingredients: **moisture**, **conditional instability**, and a **lifting mechanism** sufficient to raise a parcel to its level of free convection (LFC), i.e., to overcome convective inhibition (CIN). Heavy precipitation additionally requires high **precipitation efficiency** and slow system motion (duration). This "ingredients-based" decomposition is the standard organizing principle and is worth adopting as the review's backbone.

- Doswell, Brooks & Maddox (1996) — canonical ingredients-based methodology for heavy-precipitation/flash-flood forecasting.
- Johns & Doswell (1992) — ingredients framing applied to severe-storm forecasting (parameters: instability, moisture, lift, shear).
- Markowski & Richardson (2010) — graduate textbook synthesizing CI, boundaries, and mesoscale forcing.
- Weckwerth & Parsons (2006) — the definitive observational review of CI mechanisms (IHOP_2002); emphasizes that CI is overwhelmingly boundary-driven and that moisture/lift at the km scale is the hardest and most decisive ingredient to observe.

---

## A. Conditional/thermodynamic instability (CAPE, CIN)

**Mechanism.** Positive buoyancy integrated from the LFC to the equilibrium level (CAPE) sets the potential updraft strength; CIN is the energy barrier a parcel must overcome. This is the thermodynamic core of the Richardson et al. (2024) approach. CAPE/CIN depend sensitively on the low-level temperature/moisture and the mid-level lapse rate, all of which are modified by advection between an AIRS overpass and CI.

- Richardson, Kahn & Kalmus (2024) — AIRS T/q soundings + forecast-wind advection produce a CAPE field that predicts intense hourly precipitation over the central-eastern U.S., near ERA5 skill and far above raw AIRS soundings; demonstrates that mesoscale advection of the thermodynamic environment is a first-order control. **(Target paper.)**
- Moncrieff & Miller (1976) — classic treatment linking CAPE/buoyancy to convective structure.
- **Retrievability:** High. CAPE/CIN from AIRS, ERA5, RTMA; the paper's HYSPLIT/advection step directly addresses the time-evolution problem.

## B. Low-level moisture and moisture (flux) convergence

**Mechanism.** Boundary-layer moisture largely determines the LFC/LCL and the amount of CAPE realized; local moistening lowers the barrier to CI. Horizontal convergence of moisture concentrates water vapor and forces low-level ascent along mesoscale lines. Moisture is the ingredient most poorly observed at the scales relevant to CI (Weckwerth & Parsons 2006).

- Banacos & Schultz (2005) — history and operational use of moisture flux convergence for CI; argues mass convergence is the physically appropriate diagnostic within an ingredients framework.
- Weckwerth (2000) — boundary-layer water-vapor variability strongly modulates whether CI occurs along a given boundary.
- **Retrievability:** Moderate–high. Column/lower-tropospheric moisture from AIRS/SMAP-adjacent products, PWAT and moisture convergence from reanalysis; sub-grid BL moisture heterogeneity remains hard.

## C. Capping inversion, elevated mixed layer (EML), and the "lid"

**Mechanism.** A stable layer (often an EML advected from elevated terrain such as the Mexican Plateau) caps the boundary layer, suppressing premature convection and allowing CAPE to build until strong enough forcing or heating breaches the cap — frequently yielding explosive CI. Too strong a cap gives a "null" case; too weak allows early, non-severe convection.

- Carlson, Benjamin, Forbes & Li (1983) — conceptual model of EMLs in the severe-storm environment; the lid mechanism.
- Ribeiro & Bosart (2018) — climatology of EMLs over CONUS/N. America and their link to severe environments.
- **Retrievability:** High. Lapse-rate/CIN structure and lid strength from AIRS/ERA5 profiles.

## D. Mesoscale boundaries and low-level convergence lines (dominant CI trigger)

**Mechanism.** The great majority of observed CI occurs at boundary-layer convergence lines, where forced ascent lifts parcels to the LFC. Boundaries include fronts, drylines, sea/lake breezes, thunderstorm outflow (gust fronts/cold pools), and their intersections. Colliding boundaries are especially productive.

- Wilson & Schreiber (1986) — foundational: 79% of storms (95% of intense storms) initiated near radar-observed convergence lines; colliding lines triggered CI in 71% of cases.
- **Drylines:** Duda & Gallus (2013)/Peterson-style climatologies; drylines separate moist Gulf air from dry continental air and are a preferred CI locus in the southern Great Plains.
- **Sea/lake breezes:** Wakimoto & Atkins (1994) and CI ahead of the sea-breeze front — thermally forced convergence boundaries.
- **Outflow boundaries / cold pools (secondary CI):** Purdom (1976, and Purdom & Marcus 1982) — satellite "arc clouds"; storm outflow triggers new/organized convection. Torri, Kuang & Tian (2015) — mechanical (gust-front vorticity) vs. thermodynamic (moisture accumulation at the edge) triggering by cold pools.
- **Retrievability:** Mixed. Fronts/drylines/sea-breezes partly resolvable in high-res reanalysis/satellite; individual gust fronts and km-scale collisions are largely sub-grid and a known predictability limit.

## E. Boundary-layer structure and lifting to the LFC (thermals, rolls, PBL depth)

**Mechanism.** Even without a synoptic boundary, the daytime convective boundary layer produces coherent updrafts — thermals and horizontal convective rolls (HCRs) — whose intersections with boundaries locally enhance moisture and lift, controlling exactly where along a boundary CI occurs. PBL growth toward the LCL/LFC and entrainment drying are decisive.

- Weckwerth, Wilson, Wakimoto & Crook (1997) — environmental conditions for HCRs; roll updrafts modulate BL moisture and CI location.
- Weckwerth & Parsons (2006) — synthesis of BL controls on CI location/timing.
- **Retrievability:** Low for individual thermals/rolls (sub-grid); PBL depth, LCL, and entrainment tendencies are obtainable from reanalysis/soundings.

## F. Synoptic-scale forcing and large-scale ascent

**Mechanism.** Upper-level troughs, jet-streak circulations, and quasi-geostrophic ascent provide broad, sustained lift and lapse-rate steepening that precondition the environment (reduce CIN, moisten, cool aloft) — setting the stage for mesoscale triggers. Large-scale ascent also advects EMLs and moisture into place.

- Doswell, Brooks & Maddox (1996); Johns & Doswell (1992) — synoptic ingredients and pattern recognition.
- Maddox (1980) — mesoscale convective complexes and their synoptic settings.
- **Retrievability:** High. QG forcing, jet-level divergence, 500-hPa vorticity advection, omega from reanalysis.

## G. Low-level jet and nocturnal/elevated convection

**Mechanism.** The Great Plains nocturnal low-level jet (LLJ) transports moisture and warm-air advection northward, generating low-level convergence/isentropic ascent at its terminus (nose) and destabilizing elevated layers. It underlies the nocturnal warm-season precipitation maximum, often via elevated CI decoupled from the surface.

- Higgins et al. (1997) — LLJ control on summertime moisture transport and nocturnal precipitation over the central U.S.
- Geerts et al. (2017) — PECAN field campaign overview of nocturnal, elevated convection mechanisms.
- **Retrievability:** High for the LLJ itself (reanalysis winds; also the natural input for HYSPLIT trajectories); elevated LFC/CIN from profiles.

## H. Gravity waves and atmospheric bores

**Mechanism.** In the stable nocturnal boundary layer, undular bores and trapped gravity waves — often generated by MCS cold pools impinging on the stable layer — lift convectively unstable air above the inversion, initiating or maintaining elevated convection.

- Parsons et al. (2019) — potential role of bores/gravity waves in initiating and maintaining nocturnal convection over the southern Great Plains (PECAN).
- Koch et al. — structure/evolution of convectively generated undular bores as CI agents.
- **Retrievability:** Low. Bores/waves are transient and sub-grid; generally out of scope observationally, but worth flagging for nocturnal cases.

## I. Orographic and terrain-driven forcing

**Mechanism.** Terrain forces CI via mechanical uplift (flow forced over/around a barrier) and thermal forcing (daytime upslope/valley winds converging over ridges). Convection initiates where terrain-induced ascent locally overcomes CIN in a conditionally unstable, moist airmass.

- Houze (2012) — comprehensive review of orographic effects on precipitating clouds.
- Kirshbaum et al. (2018) — mechanisms of moist orographic convection and links to surface fluxes.
- **Retrievability:** High for terrain and thermally forced upslope flow (static + reanalysis winds); exact CI cells remain sub-grid.

## J. Land surface and soil moisture (core of the proposed study)

Soil moisture modulates the partition of net radiation into latent vs. sensible heat (Bowen ratio), thereby controlling BL growth, humidity, temperature, and the LCL/LFC — with competing pathways whose net sign depends on regime and scale. This is the richest and most contested category, so it is broken out below.

**J1. Temporal (one-dimensional) feedback — sign is regime-dependent.**
- **Wet-soil / positive pathway:** higher latent heat flux moistens and cools the BL, lowers the LCL, and can raise CAPE / reduce CIN, favoring CI locally.
- **Dry-soil / negative pathway:** higher sensible heat flux deepens the BL, promotes stronger thermals that can break the cap, and (via a drier-but-hotter BL) can still reach the LFC; several observational studies find afternoon rain preferentially over locally dry soils.
- Tuttle & Salvucci (2016) — remotely-sensed causal analysis: **positive** feedback in the western U.S., **negative** in the eastern U.S.; regional aridity governs the sign. **(Uploaded paper; central to the study.)**
- Taylor et al. (2012) — across six continents, afternoon rain falls preferentially over locally drier soils; models tend to show the opposite (spurious positive feedback).
- Findell & Eltahir (2003a,b) — CTP–HIlow framework: early-morning lapse rate (CTP) and low-level humidity index (HIlow) classify soundings as wet-soil-advantage, dry-soil-advantage, or insensitive. Directly relevant to combining SMAP with AIRS soundings.
- Ek & Holtslag (2004) — mechanistic BL model showing how soil moisture, entrainment, and BL growth jointly determine whether cloud/CI is favored over wet vs. dry soils.
- Liu et al. (2022) — recent review of soil moisture's influence on convective activity (useful for structuring this category).

**J2. Spatial heterogeneity — mesoscale circulations from soil-moisture gradients.**
- Soil-moisture (and induced flux) gradients on ~10–50 km scales set up thermally direct mesoscale circulations (sea-breeze-like), whose convergence branch preferentially triggers CI on the dry side / downwind of gradients.
- Taylor et al. (2007) — observational case of soil-moisture-induced mesoscale circulations.
- Taylor et al. (2011) — Sahelian storm-initiation frequency doubled over ~30-km soil-moisture patterns via enhanced convergence.
- Guillod et al. (2015) — reconciles the debate: **spatial** correlation (rain over locally dry patches) and **temporal** correlation (rain on overall-wet, heterogeneous days) have opposite signs and coexist; the distinction matters for how SMAP fields are used.
- Klein & Taylor (2020) — dry soils (patches ≥200 km) can intensify propagating MCS cores downstream.

**J3. Regional-aridity dependence and direct CI observations.**
- AlNasser, Short Gianotti & Entekhabi (2026) — tracks CI directly via geostationary cloud-top temperature (not precipitation proxy) with SMAP soil moisture across North America, Africa, Australia (2015–2020); the wet- vs. dry-soil preference varies systematically with regional aridity. **(Uploaded paper; methodological template for direct CI + SMAP.)**

**J4. Coupling frameworks and metrics (how to quantify L–A coupling).**
- Koster et al. (2004) — GLACE: identifies "hotspots" of strong soil-moisture–precipitation coupling in semi-arid transition zones.
- Santanello et al. (2018) — the LoCo (Local Land–Atmosphere Coupling) process chain and its metrics (CTP–HIlow, mixing diagrams, heated-condensation framework, two-legged metrics) — a menu of diagnostics for linking SMAP to CI.
- Dirmeyer (2011) — the "two-legged" coupling metric (terrestrial + atmospheric legs).
- **Retrievability:** High for the soil-moisture state (SMAP) and derived flux/BL diagnostics from reanalysis; spatial-gradient metrics require careful scale choices.

## K. Vertical wind shear (organization/mode and CI feedback)

**Mechanism.** Shear is not primarily a trigger but governs storm organization, longevity, and secondary CI: it interacts with cold-pool vorticity (RKW-type balance) to produce upright, sustained updrafts at gust fronts, enabling continued initiation. Relevant to whether initiated cells become long-lived, heavy-rain-producing systems.

- Weisman & Klemp (1982) — dependence of simulated storm mode (cells → multicell → supercell) on shear and buoyancy.
- Rotunno, Klemp & Weisman (1988) — RKW theory: optimal cold-pool/shear balance for squall-line maintenance and gust-front CI.
- **Retrievability:** High. Bulk shear (0–6 km), storm-relative helicity from reanalysis winds.

## L. Aerosols and cloud microphysics

**Mechanism.** Aerosols acting as CCN alter droplet spectra, can delay warm rain, and may "invigorate" deep convection through enhanced latent heating aloft (freezing of extra condensate). Effects on precipitation are regime-dependent (can enhance or suppress), and are secondary to dynamics/thermodynamics for CI timing/location.

- Rosenfeld et al. (2008) — "Flood or drought": framework for aerosol effects on precipitation, including invigoration.
- Tao, Chen, Li, Wang & Zhang (2012) — review of aerosol impacts on convective clouds and precipitation.
- **Retrievability:** Moderate (AOD from satellite) but mechanistically noisy; likely out of scope for a first-order CI signal.

## M. Antecedent conditions and land-surface memory

**Mechanism.** Prior precipitation sets soil-moisture memory (days–weeks), which conditions subsequent flux partitioning and thus later CI — a slow, persistent control that is well suited to SMAP's strengths and to conditioning composites.

- Koster et al. (2004); Santanello et al. (2018) — memory as the mechanism translating a slowly varying land state into an atmospheric influence.
- Tuttle & Salvucci (2017) — cautions on confounders (persistence, synoptic co-variability) when inferring causal soil-moisture→precipitation links; directly relevant to designing the study's statistical tests.
- **Retrievability:** High. SMAP time series; antecedent precipitation from IMERG/gauge/reanalysis.

---

## Cross-cutting notes for the study design

1. **CAPE is necessary but not sufficient.** Richardson et al. (2024) show advected CAPE already captures much skill for heavy rain in the central-eastern U.S., but CI *timing and location* are set by lift (category D/E) and CIN release (C) — the parts least visible to a polar-orbiter snapshot. SMAP's value proposition is that it constrains the *surface-flux → BL → CIN/LCL* chain that governs whether CAPE is realized.
2. **Sign of the soil-moisture effect is not universal** (J1–J3): it flips with aridity (Tuttle & Salvucci 2016; AlNasser et al. 2026) and with whether you measure spatial vs. temporal correlations (Guillod et al. 2015). Any SMAP predictor should be regime-stratified.
3. **Spatial gradients may matter more than point values** (Taylor et al. 2007, 2011; Klein & Taylor 2020) — consider SMAP gradient/heterogeneity features, not just local soil moisture.
4. **Confounding is the central inferential risk** (Tuttle & Salvucci 2017): soil moisture, antecedent rain, and synoptic forcing co-vary. HYSPLIT back-trajectories can help attribute the airmass and separate advective from local-surface contributions.
5. **Regional specificity.** Over the **Eastern U.S.**, the temporal soil-moisture–precipitation feedback is observationally **negative** (Tuttle & Salvucci 2016) — wetter-than-normal soils are associated with *reduced* subsequent rain probability — whereas the West is positive. Over the **Great Plains**, CI is dominated by drylines, the nocturnal LLJ, EML/lid release, and nocturnal elevated MCSs; the region is also a GLACE-type coupling "hotspot" and hosts the strongest afternoon soil-moisture signals in CONUS (Findell & Eltahir 2003b). A single SMAP predictor should therefore be **regime/region-stratified**, not applied with one sign.

---

## N. Observable-keyed causal-chain catalog (recommended predictor menu)

Each entry is an explicit chain **observable → mediator → convective initiation/intensification**, with mechanism, quantified effect size where the literature provides one, sign/geometry/regional notes, and the concrete AIRS/SMAP/reanalysis predictor. Chains N1 and N2 both start from the **soil-moisture gradient** but proceed through *different mediators* (wind shear vs. buoyancy/convergence) — they are physically and statistically distinct and should be separate predictors. Because they use different *moments* of the SMAP field (gradient vs. mean) and different mediators, they do not collinearly double-count (see note at end).

**Observable notation.** SM = SMAP surface soil moisture; ∇SM = horizontal gradient with zonal (∂SM/∂x) and meridional (∂SM/∂y) components (compute both; retain vector orientation, not just |∇SM|); MLCAPE / MLCIN = mixed-layer (lowest ~100 hPa) CAPE/CIN from the (advected, revised) AIRS sounding; S06/S03 = 0–6 km / 0–3 km bulk vertical wind shear from reanalysis winds.

---

### N1. ∇SM (mesoscale, ~200–500 km) → vertical wind shear → MCS organization & extreme intensification

**Mechanism.** A persistent soil-moisture gradient (typically set by antecedent rainfall) drives a gradient in the sensible-heat flux: the dry side warms the near-surface air faster than the wet side, sharpening the low-level horizontal potential-temperature gradient. By thermal-wind balance this **strengthens the vertical wind shear** (and low-level convergence) across the gradient zone. Enhanced shear organizes convection — it modifies the MCS-relative inflow of moist unstable air, enhances cloud–cloud interactions, tilts updrafts, and *reduces entrainment dilution* — driving upscale growth, longer lifetimes, and heavier rain. This is an **organization/intensity** pathway acting on the extreme tail, not a trigger for isolated cells.

**Effect size.**
- Barton et al. (2025): on favorable- vs. unfavorable-gradient days, the largest storms show a **10–30% increase in precipitation-feature size and rainfall**; gradients act at ~500 km; seven MCS hotspots including the **US Great Plains**.
- Lu et al. (2025), East China convection-permitting (22 summers): convective cores are **~2.5× more frequent downstream of the steepest 10% of SM gradients** (~200 km) vs. a near-uniform surface; explicitly attributes it to enhanced **zonal** wind shear + low-level convergence from the θ-anomaly gradient.
- Taylor, Klein, Barton et al. (2026, *Nature*): across 2.2 M afternoon events, **68% more "extreme" initiations** under favorable soil conditions; the effect is maximized when the SM-driven circulation **opposes the direction of shear-induced cloud displacement** — i.e., *directional* shear is the decisive mediator.
- Klein & Taylor (2020): dry patches ≥200 km favor intensified cores on the downstream (downshear) side.

**Sign / geometry / region.** Favorability is **directional**: it depends on the orientation of ∇SM relative to the low-level and mid-level shear vectors, not on |∇SM| alone. Established in the Sahel/monsoon regions and East China; the Great Plains is included in Barton et al. and is physically analogous (antecedent-MCS-generated wet/dry patches + strong warm-season shear).

**Observable predictors.** Zonal and meridional SMAP gradient components at ~200–500 km (36 km SMAP product is adequate and low-noise at this scale); ambient S06/S03 from reanalysis; and a **∇SM-projected-onto-shear** interaction term (signed alignment) to capture the directional effect. HYSPLIT identifies the storm inflow sector over which to evaluate ∇SM.

**Key refs.** Barton et al. 2025; Taylor, Klein, Barton et al. 2026; Lu et al. 2025; Klein & Taylor 2020; (organization background: Weisman & Klemp 1982; Rotunno, Klemp & Weisman 1988).

---

### N2. ∇SM (mesoscale, ~10–100 km) → differential heating → mesoscale circulation → low-level convergence + local CAPE/CIN modification → CI

**Mechanism.** The same flux gradient that drives shear (N1) also drives a **thermally direct, sea-breeze-like mesoscale circulation**: air rises over the hot, dry patch and sinks over the cool, moist patch, producing a low-level **convergence/ascent branch** near the dry–wet boundary (preferentially on the dry side / downwind end). This concentrates low-level moisture and provides mesoscale lift that helps parcels overcome **CIN** and reach the LFC, while the dry-side deeper, warmer boundary layer locally raises surface-based/mixed-layer **CAPE**. Net result: CI is favored over/adjacent to the drier patch. This is an **initiation** pathway (distinct from N1's organization pathway).

**Effect size.**
- Taylor et al. (2011): frequency of Sahelian storm initiation **doubled (~2×)** over ~30 km soil-moisture patterns via enhanced convergence.
- Taylor et al. (2012): global observations — afternoon rain falls preferentially over **locally drier** soils (spatial signal strongest in semi-arid regions).
- Taylor et al. (2015): detectable SM control on CI extended to **Europe** (mid-latitude, more Great-Plains-like).
- Rochetin et al. (2017), Froidevaux et al. (2014): LES/convection-resolving experiments confirm heterogeneity-induced breeze circulations trigger deep convection over the drier/warmer patch, and that **background wind speed modulates** the effect (weak-wind days favor it; strong flow advects/erodes the circulation).
- Guillod et al. (2015): reconciles sign — the **spatial** correlation (CI over locally dry patches) coexists with an opposite-signed **temporal** correlation (more CI on overall-wet, heterogeneous days).

**Sign / geometry / region.** Spatial preference is for the **dry/downwind side** of the gradient; strength scales with gradient magnitude, patch length scale (~10 km and up), and *inversely* with background wind. Semi-arid/transition regions and weakly-forced synoptic days show it most; over the wetter Eastern US it is weaker and can be masked by synoptic forcing.

**Observable predictors.** SMAP zonal/meridional gradient at ~10–100 km (needs the **9 km enhanced** product to resolve sharper patches); background low-level wind speed (reanalysis) as a modulator/gate; AIRS MLCAPE and MLCIN as the buoyancy/inhibition state the circulation acts on. In the parcel framework, this pathway is partly captured by letting SMAP modify Δθ,Δq along in-BL trajectory segments (N3), plus an explicit local-∇SM convergence proxy.

**Key refs.** Taylor et al. 2007, 2011, 2012, 2015; Rochetin et al. 2017; Froidevaux et al. 2014; Garcia-Carreras et al. 2011; Guillod et al. 2015.

---

### N3. Mean SM along BL trajectory → evaporative fraction → mixed-layer θ,q → MLCAPE / MLCIN → CI

**Mechanism.** Point-value (not gradient) soil moisture sets the **evaporative fraction** and thus how available energy partitions into latent vs. sensible heat as the parcel traverses the boundary layer. Wet soil → moister, cooler BL → lower LCL, higher low-level moisture (can raise MLCAPE, lower MLCIN); dry soil → warmer, deeper, drier BL with more entrainment (can raise MLCIN and the LCL). This is the **thermodynamic** channel that folds directly into the advected, revised AIRS sounding (see the flux-budget method note).

**Effect size / sign.** Regime-dependent and the crux of the study: **positive** SM–precipitation feedback in the western US, **negative** in the eastern US (Tuttle & Salvucci 2016); afternoon rain over locally drier soils globally (Taylor et al. 2012); the wet/dry-soil CI preference varies systematically with **regional aridity** (AlNasser et al. 2026). Magnitudes are best expressed as shifts in MLCIN/MLCAPE per unit SM anomaly rather than a single number.

**Observable predictors.** SMAP mean along in-BL, daytime, over-land trajectory segments (residence-time & insolation weighted) → EF → Δθ,Δq → recomputed MLCAPE/MLCIN/LCL. **Fold this into one physically-revised CAPE predictor** rather than also entering raw SM separately (avoid double-count).

**Key refs.** Tuttle & Salvucci 2016; Taylor et al. 2012; AlNasser et al. 2026; Ek & Holtslag 2004; Findell & Eltahir 2003a,b.

---

### N4. Antecedent SM (memory, days–weeks) → MLCIN → CI (inhibition gate)

**Mechanism.** Slowly-varying antecedent soil moisture modulates **convective inhibition** more than CAPE: low antecedent SM warms the low levels and raises CIN (and the 700-hPa "lid" temperature), suppressing CI even when CAPE and moisture are ample; the effect persists via soil-moisture memory. This makes CIN a first-order gate on whether otherwise-favorable environments actually initiate.

**Effect size / sign.** Myoung & Nielsen-Gammon (2010, Texas warm season): CIN — controlled by 700-hPa temperature and surface dewpoint — is the dominant convective parameter separating rainy from dry months; **low antecedent SM enhances CIN and suppresses summer convection** (a negative-feedback, drought-reinforcing pathway consistent with the eastern-US sign).

**Observable predictors.** SMAP antecedent time series (memory); AIRS-derived MLCIN and 700-hPa temperature; pair with N3.

**Key refs.** Myoung & Nielsen-Gammon 2010; Tuttle & Salvucci 2017 (confounding cautions); Koster et al. 2004 (memory).

---

### N5. Mid-tropospheric lapse rate / EML "lid" (AIRS) → CIN & CAPE build-up → delayed but explosive CI

**Mechanism.** A capping inversion / elevated mixed layer suppresses premature convection and lets CAPE accumulate until a trigger breaches the lid — favoring the intense, discrete initiations relevant to extremes. AIRS resolves this in the mid-troposphere (a relative instrument strength).

**Effect size / sign.** Non-monotonic: moderate cap → higher-end CAPE and stronger storms; too-strong cap → null (no CI); too-weak → early, weak convection. EML climatologies peak over the Plains in warm season (Ribeiro & Bosart 2018).

**Observable predictors.** AIRS lapse rate over ~1–3 km and 700–500 hPa; MLCIN; lid strength. Also feeds mixed-layer-growth closure in the N3 budget.

**Key refs.** Carlson et al. 1983; Ribeiro & Bosart 2018.

---

### N6. Low-level moisture / PWAT (AIRS) → LCL, LFC, MLCAPE → CI

**Mechanism.** Boundary-layer moisture largely sets the LCL/LFC and how much CAPE is realized; small low-level moisture differences flip whether a boundary initiates. It is also the ingredient most poorly observed at CI scales, so the advected AIRS moisture field (plus SMAP-driven moistening from N3) is where marginal skill likely lives.

**Effect size / sign.** Monotonic and strong: higher low-level q → lower LFC, higher MLCAPE, lower MLCIN → more likely CI. Precipitation-efficiency/duration for *extremes* also rises with column moisture (Doswell ingredients).

**Observable predictors.** AIRS lower-tropospheric q, PWAT, derived LCL/LFC; advect with HYSPLIT.

**Key refs.** Weckwerth 2000; Banacos & Schultz 2005; Doswell et al. 1996.

---

### N7. Ambient deep-layer shear (reanalysis) → storm mode & longevity → extreme rainfall

**Mechanism.** Independent of soil moisture, environmental shear governs whether initiated convection becomes organized, long-lived, and heavy-raining (multicell/supercell/MCS), via the cold-pool/shear (RKW) balance. It is the ambient baseline onto which the SM-gradient shear increment (N1) adds.

**Effect size / sign.** Storm mode transitions with 0–6 km bulk shear (organized/supercell modes favored above ~15–20 m s⁻¹; Weisman & Klemp 1982). Interacts multiplicatively with CAPE (bulk-Richardson / composite-parameter tradition).

**Observable predictors.** Reanalysis S06/S03, storm-relative helicity. **Isolate from N1** by treating N1 as the SM-attributable increment (or the residual after removing synoptic shear) to avoid double-counting.

**Key refs.** Weisman & Klemp 1982; Rotunno, Klemp & Weisman 1988.

---

### N8. Lower-free-tropospheric (LFT, ~1–3 km) humidity → entrainment dilution of the rising plume → shallow-to-deep transition

**Mechanism.** Whether shallow convection deepens depends less on CAPE than on how much the rising plume is diluted by entraining environmental air. A **moist LFT** preserves plume buoyancy above the LFC and lets convection deepen; a **dry LFT** entrains unsaturated air, kills buoyancy, and vetoes deep CI even when boundary-layer CAPE is high. This is arguably AIRS's single most distinctive contribution beyond CAPE/CIN, because AIRS retrieves the free-tropospheric humidity profile directly.

**Effect size.** Precipitation onset rises sharply once column saturation fraction exceeds ~0.7; entraining-plume-buoyancy calculations show that land–ocean, seasonal, and diurnal differences in convective onset are governed mainly by the LFT-humidity contribution, not total column water (Ahmed & Neelin 2018; Schiro et al. 2018). The inhibitor side: LFT (1–3 km) dry-air entrainment measurably suppresses deep convection and raises near-surface moist heat (Duan, Ahmed & Neelin 2024).

**Sign / region.** Enabler when moist, inhibitor when dry. Established primarily in the tropics/subtropics; Duan et al. (2024) show mid-latitude/subtropical relevance. Treat the exact threshold as a hypothesis to recalibrate for the Eastern US / Plains.

**Observable predictors.** AIRS RH/q at 850–600 hPa; an entraining-CAPE (deep-inflow plume-buoyancy) diagnostic rather than undilute CAPE; column saturation fraction. Advect with HYSPLIT.

**Key refs.** Ahmed & Neelin 2018; Schiro et al. 2018; Morrison et al. 2022; Duan, Ahmed & Neelin 2024.

---

### N9. Heated Condensation Framework (HCF): SM + AIRS sounding → energy-to-CI threshold → locally-triggered CI

**Mechanism.** HCF quantifies, from a single sounding, the sensible+latent heat energy still required to reach the buoyant condensation level (BCL) and initiate convection, and diagnoses whether the atmosphere is in a **moistening-advantage** (needs moisture — favors wet-soil trigger) or **PBL-growth-advantage** (needs heat — favors dry-soil trigger) regime. It is a physically-based, AIRS+SMAP-native alternative/complement to CTP–HIlow that directly flags *locally-originated* (surface-triggered) convection.

**Effect size.** Framework/diagnostic (energy deficit in J kg⁻¹; advantage-transition boundary), mapped climatologically over CONUS.

**Observable predictors.** AIRS T,q profile → BCL, energy deficit, advantage regime; SMAP → surface state driving the energy input.

**Key refs.** Tawfik & Dirmeyer 2014; Tawfik, Dirmeyer & Santanello 2015 (Parts I–II).

---

### N10. Frontal overrunning / isentropic upglide → elevated MUCAPE → nocturnal/elevated CI

**Mechanism.** At night the surface decouples under a stable inversion; convection initiates from an **elevated** most-unstable layer lifted along sloping isentropes (warm-air advection over a frontal/stable surface, often coupled to the LLJ). Surface-based CAPE is misleading here — the relevant quantity is **MUCAPE** of the elevated layer, which AIRS (seeing above the PBL) is well suited to detect. A leading nocturnal CI mode over the Great Plains.

**Effect size.** Nocturnal warm-season CI over the central/southern Plains is frequently elevated and frontal/LLJ-forced (climatology in Reif & Bluestein 2017); underlies the nocturnal precipitation maximum.

**Observable predictors.** AIRS MUCAPE / most-unstable-parcel LFC (vs. SBCAPE); 700-hPa warm advection; reanalysis LLJ.

**Key refs.** Reif & Bluestein 2017; Geerts et al. 2017 (PECAN); (background: Higgins et al. 1997).

---

### N11. Precipitation / moisture recycling → local moisture supply → CI (HYSPLIT-native)

**Mechanism.** A fraction of the moisture feeding a storm was locally evapotranspired (set by upwind soil moisture) rather than advected from the Gulf; higher antecedent SM raises the recycled contribution — a **moisture-supply** (thermodynamic) pathway distinct from the dynamic gradient/shear effects. HYSPLIT back-trajectories are the natural tool to attribute Gulf vs. locally-recycled source.

**Effect size.** Continental-interior recycling ratios ~20–25% (Mississippi basin), rising toward ~0.3 in summer with high ET.

**Observable predictors.** HYSPLIT back-trajectory moisture-uptake attribution over SMAP-wet vs. -dry source regions; AIRS PWAT along the path.

**Key refs.** Dominguez et al. 2006; Dirmeyer, Schlosser & Brubaker 2009; (tool: Stein et al. 2015).

---

### N12. SM → competing lapse-rate steepening vs. low-level moisture loss → net MLCAPE/MLCIN (CAPE decomposition)

**Mechanism.** Over drier soil, stronger sensible heating **steepens the near-surface lapse rate** (raises buoyancy) but simultaneously **removes low-level moisture and raises the LCL** (cuts CAPE, raises CIN). The net CAPE response is the competition of these opposing terms — usually moisture-dominated (wetter soil → higher MLCAPE) but regime-dependent. Decomposing AIRS-derived CAPE into lapse-rate vs. moisture contributions makes the sign explicit and testable.

**Effect size / sign.** Sign competition; during dry-downs, rising sensible heat often coincides with *falling* MLCAPE (moisture loss wins). Regime boundary aligns with the CTP–HIlow windows.

**Observable predictors.** AIRS-sounding decomposition of ΔMLCAPE into lapse-rate and low-level-moisture terms; SMAP for the SM state.

**Key refs.** Findell & Eltahir 2003a,b; Ek & Holtslag 2004.

---

**Region- and timing-specific nuances (Eastern US / Great Plains).**

- *Sign-gating by free-tropospheric moisture (SGP):* the ambient lower-tropospheric humidity, not soil moisture alone, sets the *sign* of the local SM–precipitation feedback — a dry free troposphere favors the wet-soil (positive) pathway; a moist one favors the dry-soil (negative) pathway (Wang et al. 2024). Stratify SMAP predictors by AIRS free-tropospheric RH.
- *Antecedent-MCS memory (central US):* MCSs contribute 30–70% of warm-season rainfall; early-warm-season (Apr–Jun) MCS rain creates the coherent mesoscale soil-moisture heterogeneity that then organizes July afternoon (non-MCS) CI — and the feedback sign differs for MCS vs. non-MCS antecedents (Hu, Leung & Feng 2021). Separate antecedent-rain type when using SMAP memory.
- *Diurnal timing:* over drier soils, faster PBL growth breaches the LCL earlier, shifting CI ~1–2 h earlier and toward the afternoon–evening maximum (Klein & Taylor 2020).
- *Land-use heterogeneity:* irrigation/land-cover contrasts (High Plains Aquifer belt) create flux-gradient circulations analogous to N2, raising downwind CAPE/PWAT and CTP while lowering the LCL deficit (GRAINEX). SMAP resolution (9–36 km) only marginally resolves these — a caveat, and a confounder to control for.

---

**Satellite-sounding methodology & caveats (study design).**

- The advection-of-thermodynamics logic your study inherits was first validated in a nature-run simulation experiment (Richardson et al. 2023) before the observational 2024 paper — a template for a skill test and confirmation that air-motion advection, not diabatic processes, dominates pre-convective development on these timescales.
- AIRS (and CrIS/IASI) soundings carry a persistent **boundary-layer cold/dry bias** that depresses surface-based CAPE; blending with surface stations and/or CrIS/ATMS (NUCAPS, later overpass) improves both the BL parcel and temporal sampling in the advected gap.
- Recent SMAP-native evidence supports antecedent SM as a first-order predictor: SMAP L4 antecedent soil moisture (~7-h lag window) enhances cloud reflectivity by up to ~4 dBZ over the central US (Gao et al. 2024); ML predictor-importance studies consistently rank CAPE/CIN + moisture combinations, and low/mid-level shear, above any single index (Ukkonen & Mäkelä 2019).

---

**Recommended architecture (two channels, no double-count).**

| Channel | Observable (moment) | Sampling | Mediator → target | Chains |
|---|---|---|---|---|
| Thermodynamic / initiation | SMAP **mean**; AIRS T,q | Lagrangian (in-BL trajectory) + advected sounding | MLCAPE/MLCIN, convergence → CI | N2–N6 |
| Kinematic / organization | SMAP **gradient** (∂/∂x, ∂/∂y); reanalysis winds | Eulerian (~200–500 km inflow neighborhood) | shear → MCS growth/intensity | N1, N7 |

Mean-SM (thermodynamic) and gradient-SM (kinematic) are different moments of the field, so both can enter without collinearity. Within the thermodynamic channel, fold SM into the **revised CAPE** rather than also adding raw SM. Within the kinematic channel, use the **SM-attributable shear increment**, not total ambient shear twice.

**Deprioritized (sub-grid/transient or not AIRS/SMAP-observable):** individual gust fronts/cold-pool collisions, horizontal convective rolls, thermals, bores/gravity waves, aerosols, and — as *direct* triggers — the km-scale lift of drylines/LLJ/orography (their thermodynamic preconditioning is retained via N5/N6; the lift itself is not resolved).

---

## References (with brief notes)

1. **AlNasser, F., D. J. Short Gianotti, and D. Entekhabi (2026).** *Soil Moisture Impact on Convective Initiation.* Water Resources Research. — Directly tracks CI via geostationary cloud-top temperature (not a precip proxy) with SMAP across three continents; wet/dry-soil CI preference varies with regional aridity. *(Uploaded.)*

2. **Banacos, P. C., and D. M. Schultz (2005).** *The Use of Moisture Flux Convergence in Forecasting Convective Initiation: Historical and Operational Perspectives.* Weather and Forecasting, 20, 351–366. — Critical history of MFC; recommends mass convergence within an ingredients approach.

3. **Carlson, T. N., S. G. Benjamin, G. S. Forbes, and Y.-F. Li (1983).** *Elevated Mixed Layers in the Regional Severe Storm Environment: Conceptual Model and Case Studies.* Monthly Weather Review, 111, 1453–1474. — Origin of the EML/"lid" concept for capping and CAPE build-up.

4. **Dirmeyer, P. A. (2011).** *The terrestrial segment of soil moisture–climate coupling.* Geophysical Research Letters, 38, L16702. — Introduces the two-legged coupling metric (terrestrial + atmospheric legs).

5. **Doswell, C. A., H. E. Brooks, and R. A. Maddox (1996).** *Flash Flood Forecasting: An Ingredients-Based Methodology.* Weather and Forecasting, 11, 560–581. — The canonical ingredients framework (moisture, instability, lift; efficiency, duration).

6. **Ek, M. B., and A. A. M. Holtslag (2004).** *Influence of Soil Moisture on Boundary Layer Cloud Development.* Journal of Hydrometeorology, 5, 86–99. — Mechanistic BL model of how entrainment/BL growth can favor cloud/CI over dry soils.

7. **Findell, K. L., and E. A. B. Eltahir (2003a).** *Atmospheric Controls on Soil Moisture–Boundary Layer Interactions. Part I: Framework Development.* Journal of Hydrometeorology, 4, 552–569. — Defines CTP and HIlow.

8. **Findell, K. L., and E. A. B. Eltahir (2003b).** *…Part II: Feedbacks within the Continental United States.* Journal of Hydrometeorology, 4, 570–583. — Maps wet-/dry-soil-advantage regimes across CONUS.

9. **Geerts, B., et al. (2017).** *The 2015 Plains Elevated Convection at Night Field Project (PECAN).* Bulletin of the American Meteorological Society, 98, 767–786. — Overview of nocturnal/elevated CI mechanisms (LLJ, bores, MCS).

10. **Guillod, B. P., B. Orlowsky, D. G. Miralles, A. J. Teuling, and S. I. Seneviratne (2015).** *Reconciling spatial and temporal soil moisture effects on afternoon rainfall.* Nature Communications, 6, 6443. — Spatial (rain over dry patches) and temporal (rain on wet, heterogeneous days) correlations have opposite signs and coexist.

11. **Higgins, R. W., Y. Yao, E. S. Yarosh, J. E. Janowiak, and K. C. Mo (1997).** *Influence of the Great Plains Low-Level Jet on Summertime Precipitation and Moisture Transport over the Central United States.* Journal of Climate, 10, 481–507. — LLJ control on nocturnal precipitation and moisture budget.

12. **Houze, R. A. (2012).** *Orographic effects on precipitating clouds.* Reviews of Geophysics, 50, RG1001. — Comprehensive review of terrain-forced precipitation/convection.

13. **Johns, R. H., and C. A. Doswell (1992).** *Severe Local Storms Forecasting.* Weather and Forecasting, 7, 588–612. — Ingredients/parameters (instability, moisture, lift, shear) for severe convection.

14. **Kirshbaum, D. J., B. Adler, N. Kalthoff, C. Barthlott, and S. Serafin (2018).** *Moist Orographic Convection: Physical Mechanisms and Links to Surface-Exchange Processes.* Atmosphere, 9(3), 80. — Mechanical vs. thermal orographic CI mechanisms.

15. **Klein, C., and C. M. Taylor (2020).** *Dry soils can intensify mesoscale convective systems.* Proceedings of the National Academy of Sciences, 117, 21132–21137. — Dry patches ≥200 km intensify MCS cores downstream (Sahel).

16. **Koster, R. D., et al. (2004).** *Regions of Strong Coupling Between Soil Moisture and Precipitation.* Science, 305, 1138–1140. — GLACE multi-model "hotspots" in semi-arid transition zones.

17. **Liu, W., Q. Zhang, C. Li, L. Xu, et al. (2022).** *The influence of soil moisture on convective activity: a review.* Theoretical and Applied Climatology, 149, 221–232. — Recent topical review; useful for structuring the soil-moisture category.

18. **Maddox, R. A. (1980).** *Mesoscale Convective Complexes.* Bulletin of the American Meteorological Society, 61, 1374–1387. — Defines MCCs and their synoptic/nocturnal settings.

19. **Markowski, P., and Y. Richardson (2010).** *Mesoscale Meteorology in Midlatitudes.* Wiley-Blackwell. — Textbook synthesis of boundaries, CI, and mesoscale forcing.

20. **Moncrieff, M. W., and M. J. Miller (1976).** *The dynamics and simulation of tropical cumulonimbus and squall lines.* Quarterly Journal of the Royal Meteorological Society, 102, 373–394. — Early buoyancy/CAPE–structure theory.

21. **Parsons, D. B., et al. (2019).** *The Potential Role of Atmospheric Bores and Gravity Waves in the Initiation and Maintenance of Nocturnal Convection over the Southern Great Plains.* Journal of the Atmospheric Sciences, 76, 43–68. — Bores lift unstable air above the nocturnal inversion to trigger elevated CI.

22. **Purdom, J. F. W. (1976).** *Some uses of high-resolution GOES imagery in the mesoscale forecasting of convection and its behavior.* Monthly Weather Review, 104, 1474–1483. — Satellite "arc cloud"/outflow-boundary CI. (See also Purdom & Marcus 1982.)

23. **Ribeiro, B. Z., and L. F. Bosart (2018).** *Elevated Mixed Layers and Associated Severe Thunderstorm Environments in South and North America.* Monthly Weather Review, 146, 3–28. — Climatology of EMLs and the lid.

24. **Richardson, M. T., B. H. Kahn, and P. M. Kalmus (2024).** *Mesoscale air motion and thermodynamics predict heavy hourly U.S. precipitation.* Communications Earth & Environment, 5, 472. — AIRS CAPE + forecast-wind advection predicts intense hourly precip near ERA5 skill; mesoscale advection is a first-order control. *(Uploaded; target paper.)*

25. **Rosenfeld, D., et al. (2008).** *Flood or Drought: How Do Aerosols Affect Precipitation?* Science, 321, 1309–1313. — Aerosol/CCN effects and convective invigoration.

26. **Rotunno, R., J. B. Klemp, and M. L. Weisman (1988).** *A Theory for Strong, Long-Lived Squall Lines.* Journal of the Atmospheric Sciences, 45, 463–485. — RKW cold-pool/shear balance for gust-front CI and squall-line longevity.

27. **Santanello, J. A., et al. (2018).** *Land–Atmosphere Interactions: The LoCo Perspective.* Bulletin of the American Meteorological Society, 99, 1253–1272. — LoCo process chain and its coupling metrics (CTP–HIlow, mixing diagrams, HCF, two-legged).

28. **Tao, W.-K., J.-P. Chen, Z. Li, C. Wang, and C. Zhang (2012).** *Impact of aerosols on convective clouds and precipitation.* Reviews of Geophysics, 50, RG2001. — Review of aerosol–convection microphysical/dynamical pathways.

29. **Taylor, C. M., et al. (2007).** *An observational case study of mesoscale atmospheric circulations induced by soil moisture.* Geophysical Research Letters, 34, L15801. — Observed soil-moisture-driven mesoscale circulations.

30. **Taylor, C. M., et al. (2011).** *Frequency of Sahelian storm initiation enhanced over mesoscale soil-moisture patterns.* Nature Geoscience, 4, 430–433. — CI frequency doubled over ~30-km soil-moisture gradients via convergence.

31. **Taylor, C. M., R. A. M. de Jeu, F. Guichard, P. P. Harris, and W. A. Dorigo (2012).** *Afternoon rain more likely over drier soils.* Nature, 489, 423–426. — Global observational evidence for rain over locally dry soils; models show opposite sign.

32. **Torri, G., Z. Kuang, and Y. Tian (2015).** *Mechanisms for convection triggering by cold pools.* Geophysical Research Letters, 42, 1943–1950. — Mechanical vs. thermodynamic cold-pool CI.

33. **Tuttle, S., and G. Salvucci (2016).** *Empirical evidence of contrasting soil moisture–precipitation feedbacks across the United States.* Science, 352, 825–828. — Causal analysis: positive feedback in the western U.S., negative in the east; aridity governs sign. *(Uploaded.)*

34. **Tuttle, S. E., and G. D. Salvucci (2017).** *Confounding factors in determining causal soil moisture–precipitation feedback.* Water Resources Research, 53, 5531–5544. — Methodological cautions on persistence/synoptic confounders in causal inference.

35. **Wakimoto, R. M., and N. T. Atkins (1994).** *Observations of the sea-breeze front during CaPE.* Monthly Weather Review, 122, 1092–1114. — Sea-breeze boundary structure and CI. (See also CI ahead of the sea-breeze front, MWR 2005.)

36. **Weckwerth, T. M. (2000).** *The Effect of Small-Scale Moisture Variability on Thunderstorm Initiation.* Monthly Weather Review, 128, 4017–4030. — BL water-vapor variability determines whether CI occurs along a boundary.

37. **Weckwerth, T. M., and D. B. Parsons (2006).** *A Review of Convection Initiation and Motivation for IHOP_2002.* Monthly Weather Review, 134, 5–22. — The definitive observational CI review; boundaries dominate, moisture/lift are the decisive under-observed ingredients.

38. **Weckwerth, T. M., J. W. Wilson, R. M. Wakimoto, and N. A. Crook (1997).** *Horizontal Convective Rolls: Determining the Environmental Conditions Supporting their Existence and Characteristics.* Monthly Weather Review, 125, 505–526. — HCR environments and their role in localizing CI.

39. **Weisman, M. L., and J. B. Klemp (1982).** *The Dependence of Numerically Simulated Convective Storms on Vertical Wind Shear and Buoyancy.* Monthly Weather Review, 110, 504–520. — Shear/buoyancy control of storm mode.

40. **Wilson, J. W., and W. E. Schreiber (1986).** *Initiation of Convective Storms at Radar-Observed Boundary-Layer Convergence Lines.* Monthly Weather Review, 114, 2516–2536. — Foundational: most CI occurs at (and especially at collisions of) convergence lines.

### References added in this revision (soil-moisture-gradient → shear/CAPE mechanisms)

41. **Barton, E. J., et al. (2025).** *Soil moisture gradients strengthen mesoscale convective systems by increasing wind shear.* Nature Geoscience, 18, 330–336. — ~500 km SM gradients sharpen temperature gradients and vertical shear; +10–30% size/rainfall for the largest storms on favorable-gradient days; seven hotspots incl. US Great Plains. *(Uploaded; primary source for chain N1.)*

42. **Taylor, C. M., C. Klein, E. J. Barton, S. Hahn, and W. Wagner (2026).** *Wind shear enhances soil moisture influence on rapid thunderstorm growth.* Nature, 651, 116–121. — 2.2 M afternoon events; 68% more extreme initiations under favorable soil conditions; effect maximized when SM-driven circulation opposes shear-induced cloud displacement — *directional* shear is the key mediator (chain N1).

43. **Lu, Y., et al. (2025).** *Role of Soil Moisture Gradients in Favoring Mesoscale Convective Systems in East China.* Geophysical Research Letters, 52, e2025GL117137. — 22-summer convection-permitting sims; cores ~2.5× more frequent downstream of the steepest 10% of ~200 km SM gradients; attributes it to θ-gradient-enhanced zonal shear + low-level convergence; extends Klein & Taylor (2020).

44. **Rochetin, N., F. Couvreux, and F. Guichard (2017).** *Morphology of breeze circulations induced by surface flux heterogeneities and their impact on convection initiation.* Quarterly Journal of the Royal Meteorological Society, 143, 463–478. — LES: heterogeneity-driven breeze circulations trigger deep convection over the drier/warmer patch (chain N2).

45. **Froidevaux, P., L. Schlemmer, J. Schmidli, W. Langhans, and C. Schär (2014).** *Influence of the Background Wind on the Local Soil Moisture–Precipitation Feedback.* Journal of the Atmospheric Sciences, 71, 782–799. — Convection-resolving: small-scale SM variability drives afternoon convection; background wind speed modulates the sign/strength (chain N2 gate).

46. **Myoung, B., and J. W. Nielsen-Gammon (2010).** *The Convective Instability Pathway to Warm Season Drought in Texas (Parts I–II).* Journal of Climate, 23, 4461–4489. — CIN (set by 700-hPa T and surface dewpoint) is the dominant parameter; low antecedent SM raises CIN and suppresses summer convection (chain N4).

47. **Taylor, C. M. (2015).** *Detecting soil moisture impacts on convective initiation in Europe.* Geophysical Research Letters, 42, 4631–4638. — Extends the observed SM-gradient CI control to mid-latitude Europe (more Great-Plains-like than the Sahel).

48. **Garcia-Carreras, L., D. J. Parker, and J. H. Marsham (2011).** *What is the mechanism for the modification of convective cloud distributions by land surface–induced flows?* Journal of the Atmospheric Sciences, 68, 619–634. — Mechanism for surface-flux-gradient-induced mesoscale flows organizing convective cloud (supports chain N2).

### References added in the exhaustive-sweep revision (free-troposphere, coupling frameworks, methodology)

49. **Ahmed, F., and J. D. Neelin (2018).** *Reverse engineering the tropical precipitation–buoyancy relationship.* Journal of the Atmospheric Sciences, 75, 1587–1608. — Entraining-plume buoyancy with deep-inflow mixing predicts convective onset far better than undilute CAPE (chain N8).

50. **Schiro, K. A., F. Ahmed, S. E. Giangrande, and J. D. Neelin (2018).** *GoAmazon2014/5 campaign points to deep-inflow approach to deep convection across scales.* Proceedings of the National Academy of Sciences, 115, 4577–4582. — Lower-free-tropospheric humidity is the dominant control on the shallow-to-deep transition (chain N8).

51. **Morrison, H., et al. (2022).** *Relationships between environmental humidity, the horizontal scale of subcloud ascent, and convective initiation.* Journal of the Atmospheric Sciences, 79. — The subcloud-ascent scale needed for deep CI decreases as free-tropospheric humidity rises (chain N8).

52. **Duan, S. Q., F. Ahmed, and J. D. Neelin (2024).** *Moist heatwaves intensified by entrainment of dry air that limits deep convection.* Nature Geoscience, 17, 837–844. — Dry lower-free-troposphere (1–3 km) entrainment vetoes deep convection even with high boundary-layer CAPE (chain N8, inhibitor side).

53. **Tawfik, A. B., and P. A. Dirmeyer (2014).** *A process-based framework for quantifying the atmospheric preconditioning of surface-triggered convection.* Geophysical Research Letters, 41, 173–178. — Introduces the Heated Condensation Framework (chain N9).

54. **Tawfik, A. B., P. A. Dirmeyer, and J. A. Santanello (2015).** *The Heated Condensation Framework, Parts I–II.* Journal of Hydrometeorology, 16, 1929–1956. — HCF description + CONUS climatology of CI and land–atmosphere coupling (chain N9).

55. **Reif, D. W., and H. B. Bluestein (2017).** *A 20-Year Climatology of Nocturnal Convection Initiation over the Central and Southern Great Plains during the Warm Season.* Monthly Weather Review, 145, 1615–1639. — Documents frontal/LLJ-forced elevated nocturnal CI (chain N10).

56. **Dominguez, F., P. Kumar, X.-Z. Liang, and M. Ting (2006).** *Impact of Atmospheric Moisture Storage on Precipitation Recycling.* Journal of Climate, 19, 1513–1530. — Quantifies continental precipitation recycling (chain N11).

57. **Dirmeyer, P. A., C. A. Schlosser, and K. L. Brubaker (2009).** *Precipitation, Recycling, and Land Memory: An Integrated Analysis.* Journal of Hydrometeorology, 10, 278–288. — Links soil-moisture memory to recycled precipitation (chain N11).

58. **Stein, A. F., et al. (2015).** *NOAA's HYSPLIT Atmospheric Transport and Dispersion Modeling System.* Bulletin of the American Meteorological Society, 96, 2059–2077. — Canonical HYSPLIT reference (trajectory/recycling tool).

59. **Hu, H., L. R. Leung, and Z. Feng (2021).** *Early warm-season mesoscale convective systems dominate soil moisture–precipitation feedback for summer rainfall in central United States.* Proceedings of the National Academy of Sciences, 118, e2105260118. — Early-season MCS rain sets the mesoscale SM heterogeneity that organizes later afternoon CI; feedback sign differs for MCS vs. non-MCS (region/timing nuance).

60. **Wang, et al. (2024).** *Influence of lower-tropospheric moisture on local soil moisture–precipitation feedback over the US Southern Great Plains.* Atmospheric Chemistry and Physics, 24, 3857–. — Free-tropospheric humidity gates the *sign* of the local SM–precipitation feedback (region nuance).

61. **Richardson, M. T., B. H. Kahn, and P. M. Kalmus (2023).** *Trajectory enhancement of low-earth orbiter thermodynamic retrievals to predict convection: a simulation experiment.* Atmospheric Chemistry and Physics, 23, 7699–7724. — Nature-run validation and template for the advected-AIRS-CAPE method (study foundation).

62. **Gao, Y., et al. (2024).** *Soil Moisture–Cloud–Precipitation Feedback in the Lower Atmosphere from Functional Decomposition of Satellite Observations.* Geophysical Research Letters, 51, e2024GL110347. — SMAP L4 antecedent soil moisture (~7-h lag) enhances cloud reflectivity up to ~4 dBZ over the central US (SMAP predictor support).

63. **Ukkonen, P., and A. Mäkelä (2019).** *Evaluation of Machine Learning Classifiers for Predicting Deep Convection.* Journal of Advances in Modeling Earth Systems, 11, 1784–1802. — Multi-predictor ML (CAPE/CIN/moisture/shear) beats any single instability index (predictor-selection support).

---

*Prepared as a scoping review to structure predictor selection for an AIRS-CAPE + SMAP extreme-convection study. Some categories (H, L, and sub-grid parts of D/E) are flagged as likely out of observational scope; the next step is to intersect this list with the specific AIRS/SMAP/HYSPLIT/reanalysis fields actually available.*
