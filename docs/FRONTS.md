# CODSUS surface-front flags in the analysis dataset

> **See also**: `data/front_id/README.md` for the full front-dataset
> inventory (manifest reorg 2026-08-13),
> including the FrontFinder-format conversions (NOAA XML labels with
> DRYLINES, 0.25° grid; `src/front_formats/`) and cross-format products
> added 2026-08-05.

*Added 2026-08-05 (dataset v9). Code: `src/convection_skill/fronts.py` (loader),
`src/codsus_regen.py` (2019+ regeneration), `src/convection_skill/models.py`
(Mark's RF, verbatim + adapters).*

## What was added

Ten slot-level binary columns in the cached base table, from the NCICS
"Coded Surface Bulletins" gridded front product (Biard & Kunkel,
[zenodo.org/records/2651361](https://zenodo.org/records/2651361)) — human
WPC analyst fronts on the MERRA2 1° grid, 3-hourly:

| columns | meaning |
|---|---|
| `front_{cold,warm,stationary,occluded,any}_1w` | front line itself (1 cell wide) touches the cell |
| `front_{cold,warm,stationary,occluded,any}_3w` | cell within the 3-cell-wide near-front neighborhood |

Alignment decisions (both in `fronts.py`):

- **Grid**: front grid is integer-centered, ours half-degree-centered; a flag
  is ON if any of the four overlapping front cells is on (2×2 max-pool).
- **Time**: each forecast slot takes the most recent bulletin at or before its
  hour — slots 1–3 → same-day 21 UTC analysis, slots 4–6 → next-day 00 UTC.
  These are **concurrent with the 21–02 UTC target window** (Zach's call,
  2026-08-05): fronts are a synoptic-environment covariate like CAPE, not a
  timing-guarded antecedent predictor.

In the cell-day wide table (`to_cell_days`) they widen to
`front_*_h1..h6`, which is how they enter the random forests.

## Hypothesis tests

Registry ids `F1_any`, `F1_any_1w`, `F2_cold`, `F3_stationary`, `F4_warm`,
`F5_occluded` (all secondary tier, `mu_cape` control), plus a `front`
stratifier (frontal vs non-frontal environment by `front_any_3w`) usable on
any hypothesis. NaN rows (years without front data) drop out of the Gini
automatically.

First results (default screens, pooled, iid; `results/fronts/f_specs_results.csv`):

- `F1_any` (3-wide any front): Gini **+0.53** overall, **+0.34** conditional
  on CAPE. 76% of heavy events (P99.9) have a front within the 3-wide
  neighborhood vs a 24% base rate.
- `F3_stationary` +0.33 (strongest single type — training rain);
  `F2_cold` +0.07 (weak marginal, +0.08 beyond CAPE); warm/occluded ~0.
- Mark-style RF (`models.compare_with_fronts`): test R² 0.039 → 0.048 when
  front features are added to CAPE/CIN series + `sm_anom`; front features take
  ~14% of total importance.

## Verification animations

`results/fronts/fronts_compare_2016_{1,3}wide.gif` — daily 21 UTC frames for
2016 with the regenerated product as light full cells and the published one
as dark inner squares (per-type synoptic hues): a match reads as a dark dot
inside a light cell; regenerated-only cells are light-only, published-only
are bare dark dots.

## 2019–2021: regenerated files

The published product ends in 2018 and no updated version exists (checked
Zenodo/NCICS/NOAA 2026-08-05). The raw CODSUS bulletins are issued to the
present and archived by the [IEM AFOS archive](https://mesonet.agron.iastate.edu/wx/afos/);
WPC itself only keeps a rolling 2-week window. `src/codsus_regen.py`
rebuilds the gridded product from those bulletins:

```
python src/codsus_regen.py fetch 2019 2020 2021     # cache bulletins (data/front_id/csb_raw/)
python src/codsus_regen.py build 2019 2020 2021     # -> data/front_id/CODSUS_regen/ (staging)
python src/codsus_regen.py validate 2016            # parity vs the published files
```

The loader prefers the published files and falls back to
`CODSUS_regen/` for missing years, so regenerated years flow into the
dataset on the next cache rebuild.

**Fidelity: the published rasterization was fully reverse-engineered on the
2016 overlap (2026-08-05 forensics; line-level IoU 0.995–0.998 per type,
both widths, base rates matching to 4 decimals).** The published product is,
per valid time:

1. the most complete 1° ("LR", ASUS01) bulletin (the IEM archive holds
   partial retransmissions alongside full analyses — "last wins" dedup was
   silently keeping partials);
2. each front stroked as a **line of width w in plain lat/lon space** — a
   cell is marked when its center lies within w/2 of the polyline, round
   caps; where the line passes exactly midway between two centers BOTH cells
   are marked (their renderer anti-aliases and thresholds coverage at 50%);
3. all fronts painted into **one categorical image — one type per cell** —
   with overlap priority **warm > occluded > stationary > cold** (inferred
   from pairwise win matrices, each >99% one-sided; this also explains the
   apparent "missing endpoint" cells: fronts meet other fronts at triple
   points, where the higher-priority type claims the cell);
4. hard-zeroed outside `codsus_merra2-1deg_mask.nc` (WPC reliable-analysis
   region — ships with the download; only 13 land cells of our domain touch
   the mask edge, all on the Mexican border ≤27.5°N).

The ~0.4% residual is float-tie jitter in their coverage threshold (front
endpoint cells sit exactly at 50%) plus rare dedup ambiguities — not
reproducible deterministically, and negligible after the loader's 2×2
pooling. Rejected hypotheses along the way: point-sampled rounding (any
rounding rule), integer Bresenham, supercover, spline paths, NARR/Lambert
projected geometry, high-res (0.1°) bulletin source, 3×3/plus dilations,
flat or reduced stroke caps.

Regenerated years are therefore interchangeable with the published product
for analysis purposes; the `front` stratifier and F-specs work unchanged.
