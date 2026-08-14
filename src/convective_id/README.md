# convective_id — finding convective storms the PrecipFlag misses

**Problem.** The MRMS PrecipFlag `Convection` category marks strict convective
cores. In a 1°×1° cell holding a big convective storm, most rain (anvil, MCS
trailing stratiform) is flagged stratiform: P99.9 QPE events average a
convective share of only ~0.2 and just ~2% are majority-convective. Any
flag-share filter therefore throws away organized convection.

**Approach.** Classify precipitating cell-hours as convective from **MRMS data
only** (no AIRS, no SMAP anywhere in the inputs, so the labels can be used
alongside those products without double-dipping), then validate independently
with AIRS-FCST MU CAPE, which no method ever sees.

## The four methods (`methods.py`)

| method | idea | 2019 coverage of precip | CAPE AUC conv-vs-non |
|---|---|---|---|
| `threshold` | sub-pixel peak ≥ 20 mm/h, or ≥ 10 mm/h towering 5× over its wet-area mean (embedded core) | 4% | 0.81 |
| `gmm_cluster` | 4-component GMM on flag-free structure; convective component = largest mean sub-pixel peak | 24% | 0.81 |
| `random_forest` | weak supervision: supported flag cores (+) vs core-free-neighborhood stratiform (−); features are flag-free structure, so core-LIKE anvil cells are recovered | 29% | 0.83 |
| `storm_object` | connected precipitating objects per (date, slot); cells within 2 cells of a core seed along the object are convective — the anvil inherits its storm's label | 29% | 0.84 |

Shared feature vector (`features.FEATURES`, strictly flag-free): wet-area mean
rate, sub-pixel peak, sub-pixel skewness, wet fraction, peak/mean prominence,
3×3-neighborhood peak, precipitating-neighbor count, peak growth since the
previous slot, cell-day max peak. PrecipFlag information appears **only** as
weak labels (forest) and object seeds — never as a feature.

## Validation (`validate.py`) — 2019 headline

CAPE is the physical arbiter: real convection lives in high-CAPE
environments. References: *supported flagged cores* (share ≥ 0.2, ≥ 10 wet
sub-pixels; median CAPE ≈ 1156 J/kg) and *clean stratiform* (zero share, no
core in the 3×3 neighborhood; median CAPE ≈ 0).

The decisive test is the **rescue set** — cells a method calls convective
despite a flag share < 0.1 (the anvil/MCS cases the flags miss). For every
method the rescued cells look like cores, not stratiform: AUC vs clean
stratiform 0.92–0.96 (threshold/forest/object), rescued median CAPE
600–900 J/kg vs stratiform's ~0. The three broad methods (GMM/forest/object)
independently converge on ~25–30% of precipitating cell-hours being
convective — ~5× the flag-level core rate — with pairwise Jaccard 0.4–0.55.

## Run it

```bash
PYTHONPATH=src python -m convective_id.demo [year]   # default 2019
```

Artifacts → `results/convective_id/`: `summary_<year>.csv` (the validation
table), `agreement_<year>.csv`, `labels_<year>.parquet` (per-row labels +
scores for all methods, joinable to the suite table on date/slot/lat/lon),
`figures/cape_validation_<year>.png`.

## Caveats / next steps

- The 1-h QPE sub-pixel peak is rate-based; hail/graupel contamination in
  MRMS gauge-corrected QPE is not screened.
- Thresholds (20 mm/h peak, 5× prominence, 2-cell anvil reach) are physical
  defaults, not fitted; a small CAPE-blind sensitivity sweep is the obvious
  next step.
- `storm_object` works per slot; tracking objects across slots (and inheriting
  labels through time, e.g. a decaying MCS's evening anvil) is the natural
  extension.
- To use in the suite: join `labels_<year>.parquet` onto the analysis table
  and pass any method's `label` as a custom convective filter (the suite's
  `convective_col` accepts any column).
