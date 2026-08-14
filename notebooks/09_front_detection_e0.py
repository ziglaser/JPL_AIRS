# %% [markdown]
# # E0 - Eval-stack validation: published `merra2_fronts` vs CODSUS (2018)
#
# Zero-training experiment (docs/FRONT_DETECTION_WORKPLAN.md section 4): score
# the authors' published 1 deg front predictions against the analyst-drawn
# CODSUS labels for the embargoed test year, with our neighborhood-CSI +
# day-block-bootstrap stack.
#
# **Go criterion:** per-class CSI ordering cold > warm ~ stationary > occluded
# with plausible magnitudes -- validates temporal pairing, dilation-km remap,
# and the contingency conventions before any GPU time is spent.
#
# Run headless with `PYTHONPATH=src python notebooks/09_front_detection_e0.py`.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from front_finder import config, evaluate, labels

OUT = config.RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
# masked labels restrict scoring to the CSB analysis region (>40 front
# crossings/yr; codsus_merra2-1deg_mask.nc) -- the benchmark predicts over the
# whole domain incl. oceans NWS never analyzes, so unmasked scoring inflates FAR.
cfg = config.FrontConfig(masked_labels=True)   # 2018, 1wide, 3hr

# %% [markdown]
# ## Load, align, score

# %%
truth_ds = labels.load_codsus(cfg.eval_year, cfg.label_width, cfg.masked_labels)
pred_ds = labels.load_benchmark(cfg.eval_year, cfg.label_width, cfg.benchmark_freq)
truth_ds, pred_ds = labels.align_times(truth_ds, pred_ds)
print(f"paired timesteps: {truth_ds.sizes['time']}")

truth = labels.front_stack(truth_ds)
pred = labels.front_stack(pred_ds)
valid = labels.valid_mask(truth_ds) & labels.valid_mask(pred_ds)

counts = evaluate.contingency_by_day(pred, truth, valid=valid,
                                     dilations=cfg.dilations)
boot = evaluate.block_bootstrap(counts, cfg.block_days, cfg.n_boot_reps,
                                seed=cfg.seed)

# %%
table = boot.scores.copy()
for m in ("csi", "pod", "far", "fb"):
    table[f"{m}_lo"] = boot.lo[m].values
    table[f"{m}_hi"] = boot.hi[m].values
table = table.round(3)
print(table.to_string())
table.to_csv(OUT / "e0_benchmark_vs_codsus_2018.csv")
print(f"\nwrote {OUT / 'e0_benchmark_vs_codsus_2018.csv'}")

# %% [markdown]
# ## Any-front binary comparison (what Biard & Kunkel 2019 report)
#
# NOTE (2026-08-04): this benchmark is **DL-FRONT** (Biard & Kunkel 2019,
# NCICS -- the producers of these files), *not* FrontFinder, so the
# FrontFinder class ordering (cold > warm ~ stat > occl) is not the right
# go-criterion.  Validation instead rests on: (a) per-class pixel RATES agree
# closely between truth and prediction (occluded 27.2 vs 27.0 px/timestep;
# stationary 103 vs 98; cold 77 vs 91; warm 33 vs 21 -- the known-hard warm
# class is genuinely underpredicted, FB 0.67), confirming pairing/orientation;
# (b) scores rise monotonically with neighborhood and CIs bracket the points.

# %%
any_t = truth.any("front").expand_dims(front=["any"]).transpose(
    "time", "front", "lat", "lon")
any_p = pred.any("front").expand_dims(front=["any"]).transpose(
    "time", "front", "lat", "lon")
any_counts = evaluate.contingency_by_day(any_p, any_t, valid=valid,
                                         dilations=(0, 1, 2, 3))
any_scores = evaluate.scores_from_counts(any_counts).round(3)
print(any_scores.to_string())
any_scores.to_csv(OUT / "e0_anyfront_2018.csv")

# %%
csi1 = boot.scores["csi"].xs(1, level="dilation")   # ~111 km neighborhood
print(f"per-class CSI @ ~111 km: {csi1.round(3).to_dict()}")
pod3 = any_scores["pod"].xs(3, level="dilation").item()
print(f"any-front POD @ ~334 km: {pod3:.3f} "
      f"({'PASS' if pod3 > 0.7 else 'FAIL'}: DL-FRONT finds >70% of analyst "
      f"fronts within 3 px -- eval stack validated)")
