# %% [markdown]
# # Stage-C fine-tune pipeline demo (single sample day, 2019-06-05)
#
# Validates the real-AIRS fine-tune path end-to-end with the ONE fullgrid
# AIRS-FCST file on disk.  2019 has no CODSUS labels (they end 2018), so the
# gradient-step demo pairs the 2019-06-05 inputs with the 2018-06-05 CODSUS
# bulletin as an explicitly marked PLACEHOLDER -- it validates mechanics
# (shapes, masking, loss, backprop), not skill.  Actual training happens once
# the 2016-2021 fullgrid archive lands (Zach).
#
# Run with: PYTHONPATH=src <fronts-tf python> notebooks/10_front_detection_finetune_demo.py

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from front_finder import (config, dataset, ingest_hysplit as ih, labels,
                             mask_bank, model as model_mod)

FULLGRID = (config.REPO_ROOT / "data/HYSPLIT/wrf27km_20190605/wrf27km_20190605"
            / "fullgrid_wrf27km_GOOD_1p00deg_20190605_1700-2059.nc")
OUT = config.RESULTS_DIR
stats = dataset.load_norm_stats()

# %% [markdown]
# ## 1. Ingest: fullgrid -> label-grid channels -> normalized input tensor

# %%
ch = ih.to_label_grid(ih.load_fullgrid(FULLGRID), slot=0, winds=True)
x, observed, bulletin = dataset.airs_x(FULLGRID, stats, winds=True)
print(f"x {x.shape}, finite: {np.isfinite(x).all()}")
print(f"observed cells: {int(observed.sum())} / {observed.size} "
      f"({observed.mean():.1%} of label grid; swath covers CONUS subset)")
print(f"overpass mid-time -> paired bulletin: {bulletin}")
vf = x[..., -1]
print(f"mask channel: min {vf.min():.2f} max {vf.max():.2f}, "
      f"mean over observed region {vf[2:70, 1:142][observed].mean():.2f}")
in_range = ((x[..., :-1] >= -0.25) & (x[..., :-1] <= 1.25)).mean()
print(f"normalized channels within [-0.25, 1.25]: {in_range:.1%} "
      f"(frozen MERRA-2 min-max stats applied to AIRS values)")

# %% [markdown]
# ## 2. Mask bank: harvest this day's real gap field

# %%
bank_path = mask_bank.harvest([FULLGRID])
bank_vf, bank_dates = mask_bank.load_bank(bank_path)
print(f"bank: {bank_vf.shape} from dates {list(bank_dates)} -> {bank_path}")
rng = np.random.default_rng(config.BOOT_SEED)
m = mask_bank.sample_mask(bank_vf, rng, month=6, dates=bank_dates)
print(f"sampled mask valid fraction (any level > 0): {(m.max(-1) > 0).mean():.1%}")

# %% [markdown]
# ## 3. PLACEHOLDER label pairing (2018-06-05) + one fine-tune step on GPU

# %%
placeholder_t = pd.Timestamp("2018-06-05") + (bulletin - bulletin.normalize())
truth_ds = labels.load_codsus(2018, masked=True)
fronts = labels.front_stack(truth_ds).values
valid = labels.valid_mask(truth_ds).values
times = pd.DatetimeIndex(truth_ds["time"].values)
i = int(np.flatnonzero(times == placeholder_t)[0])
fr = labels.dilate(fronts[i], 1)
y = dataset.make_y(fr, valid[i] & observed)
print(f"PLACEHOLDER y {y.shape}; loss-weighted pixels: "
      f"{int(y[..., -1].sum())} (label-valid AND on-swath only)")

# %%
import tensorflow as tf

model = model_mod.build(winds=True, learning_rate=2e-5)   # stage-C LR
xb = tf.constant(x[None]); yb = tf.constant(y[None])
loss0 = model.evaluate(xb, [yb] * len(model.outputs), verbose=0)
h = model.fit(xb, [yb] * len(model.outputs), epochs=3, verbose=0)
print(f"fine-tune step check -- loss before: {np.sum(loss0[1:]):.4f}, "
      f"after 3 steps: {h.history['loss'][-1]:.4f} "
      f"({'DECREASING - backprop through masked FSS works' if h.history['loss'][-1] < np.sum(loss0[1:]) else 'NOT DECREASING - investigate'})")
print("\nStage-C mechanics validated. Real fine-tuning: "
      "train --name E3 --airs-glob 'data/HYSPLIT/*/*/fullgrid_*.nc' "
      "--retrain <E1 ckpt> --winds")
