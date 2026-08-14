# %% [markdown]
# # E1/E2/E3 evaluation template
#
# The full post-training evaluation loop (workplan section 4), exercised here
# with the untrained `smoke` checkpoint so the mechanics are proven before any
# real training run.  To evaluate a real model, change CHECKPOINT/WINDS/LIMIT.
#
# Sections: predict -> neighborhood threshold sweep -> best-CSI table ->
# reliability + isotonic calibration -> permutation importance -> classical
# TFP baseline.  Every skill number carries a km caption (workplan constraint 8).
#
# Run with: PYTHONPATH=src <fronts-tf python> notebooks/11_front_detection_eval_template.py

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from front_finder import (calibrate, config, dataset, evaluate, labels,
                             nfa_baseline, permutation, predict)

CHECKPOINT = config.RESULTS_DIR / "models/smoke/smoke.h5"   # <- real ckpt here
WINDS = False                                               # must match ckpt
EVAL_YEAR = 2015          # E1 val year; 2018 embargoed until final testing
LIMIT = 24                # timesteps; None = full year for real evaluation
OUT = config.RESULTS_DIR / "eval_template"
OUT.mkdir(parents=True, exist_ok=True)

stats = dataset.load_norm_stats()
model = predict.load_checkpoint(CHECKPOINT)

# %% [markdown]
# ## 1. Predict + pair with labels

# %%
prob = predict.predict_year(model, EVAL_YEAR, WINDS, stats, limit=LIMIT)
truth_ds = labels.load_codsus(EVAL_YEAR, masked=True)
truth_ds = truth_ds.sel(time=prob["time"].values)
truth = labels.front_stack(truth_ds)
valid = labels.valid_mask(truth_ds)
print(f"{prob.sizes['time']} timesteps predicted for {EVAL_YEAR}")

# %% [markdown]
# ## 2. Neighborhood threshold sweep + best-CSI table

# %%
sweep = evaluate.threshold_sweep(prob["probabilities"], truth, valid,
                                 thresholds=np.arange(0.1, 1.0, 0.1))
best = evaluate.best_csi(sweep)
print(best.round(3).to_string())
best.to_csv(OUT / f"best_csi_{EVAL_YEAR}.csv")

# %% [markdown]
# ## 3. Reliability + isotonic calibration (fit on val year ONLY)

# %%
rel = calibrate.reliability(prob["probabilities"], truth, valid)
print(rel.groupby("front").apply(
    lambda d: (d["mean_prob"] - d["obs_freq"]).abs().mean()).rename(
    "mean |prob - obs| (pre-calibration)").round(3).to_string())
models = calibrate.fit(prob["probabilities"], truth, valid)
calibrate.save(models, OUT / "calibration.pkl")
prob_cal = calibrate.apply(prob["probabilities"], models)
rel_cal = calibrate.reliability(prob_cal, truth, valid)
print(rel_cal.groupby("front").apply(
    lambda d: (d["mean_prob"] - d["obs_freq"]).abs().mean()).rename(
    "mean |prob - obs| (post-calibration)").round(3).to_string())

# %% [markdown]
# ## 4. Permutation importance (POD-based, incl. the mask channel)

# %%
X, Y = permutation.collect_arrays(
    dataset.year_samples(EVAL_YEAR, stats, WINDS), n_max=LIMIT)
imp = permutation.single_pass(model, X, Y, WINDS,
                              rng=np.random.default_rng(config.BOOT_SEED))
top = (imp[imp["kind"] == "single"]
       .sort_values("delta_pod", ascending=False).head(10))
print(top.round(4).to_string(index=False))
imp.to_csv(OUT / f"permutation_{EVAL_YEAR}.csv", index=False)
mask_imp = imp[(imp["kind"] == "variable") & (imp["variable"] == "mask")]
print("\nmask-channel grouped importance (large => clouds-mark-fronts "
      "shortcut, must be reported):")
print(mask_imp.round(4).to_string(index=False))

# %% [markdown]
# ## 5. Classical TFP baseline (fullgrid AIRS files; E3 comparison)

# %%
fullgrids = sorted((config.REPO_ROOT / "data/HYSPLIT").glob(
    "*/*/fullgrid_*.nc"))
nfa_scores = nfa_baseline.baseline_vs_labels(fullgrids)
if nfa_scores.empty:
    print("TFP baseline: no fullgrid files pair with CODSUS labels yet "
          "(sample day is 2019; labels end 2018) -- table will fill once "
          "the 2016-2018 archive lands.")
else:
    print(nfa_scores.round(3).to_string())
    nfa_scores.to_csv(OUT / "tfp_baseline.csv", index=False)

print("\nEval loop mechanics complete. For a real run: set CHECKPOINT, "
      "WINDS, LIMIT=None, EVAL_YEAR; add evaluate.block_bootstrap CIs on "
      "the final counts; score AIRS models on observed pixels only via "
      "predict.predict_airs(...)['observed'].")
