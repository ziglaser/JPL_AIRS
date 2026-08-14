"""Aggregate CODSUS vs NOAA-XML front labels (1 deg, 1wide) for comparison plots.

# Run from the repo root: python notebooks/13_codsus_vs_noaa_crunch.py (once), then _plots.py

One pass over the 2006-2018 overlap years; everything the figures need lands in
a single .npz so plotting can iterate cheaply.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from front_finder import config, labels  # noqa: E402

YEARS = range(2006, 2019)
TYPES = ("cold", "warm", "stationary", "occluded")
OUT = Path("results/fronts/codsus_vs_noaa")
OUT.mkdir(parents=True, exist_ok=True)

GRID = config.GRID_SHAPE

# accumulators
freq_c = {t: np.zeros(GRID) for t in TYPES}          # front-hit counts per pixel
freq_n = {t: np.zeros(GRID) for t in TYPES}
freq_dl = np.zeros(GRID)                             # NOAA dryline
freq_dl_season = {s: np.zeros(GRID) for s in ("MAM", "JJA")}
n_valid = np.zeros(GRID)                             # joint-valid time count
n_valid_season = {s: np.zeros(GRID) for s in ("MAM", "JJA")}
SEASON = {3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}

iou = {t: {} for t in (*TYPES, "any")}               # [type][year] = (I, U)
monthly = []                                         # rows for a tidy frame
best_dl = (0, None)                                  # case-study candidate

for year in YEARS:
    with labels.load_codsus(year) as dc, labels.load_noaa(year) as dn:
        tc = pd.DatetimeIndex(dc["time"].values)
        tn = pd.DatetimeIndex(dn["time"].values)
        common = tc.intersection(tn)
        ic = tc.get_indexer(common)
        i_n = tn.get_indexer(common)

        ftc = [str(s) for s in dc["front_type"].values]
        ftn = [str(s) for s in dn["front_type"].values]
        frc = dc["fronts"].values[ic]                # (T, front, lat, lon) ubyte
        frn = dn["fronts"].values[i_n]               # uint8 (fill already 2)

    valid = ((frc != config.LABEL_FILL).all(1)
             & (frn != config.LABEL_FILL).all(1))    # (T, lat, lon)
    months = common.month.values

    n_valid += valid.sum(0)
    for s in ("MAM", "JJA"):
        sel = np.isin(months, [m for m, ss in SEASON.items() if ss == s])
        n_valid_season[s] += valid[sel].sum(0)

    any_c = np.zeros(valid.shape, dtype=bool)
    any_n = np.zeros(valid.shape, dtype=bool)
    for t in TYPES:
        c = (frc[:, ftc.index(t)] == 1) & valid
        n = (frn[:, ftn.index(t)] == 1) & valid
        any_c |= c
        any_n |= n
        freq_c[t] += c.sum(0)
        freq_n[t] += n.sum(0)
        iou[t][year] = (int((c & n).sum()), int((c | n).sum()))
        for m in range(1, 13):
            sel = months == m
            monthly.append({"year": year, "month": m, "type": t,
                            "codsus": int(c[sel].sum()),
                            "noaa": int(n[sel].sum()),
                            "valid": int(valid[sel].sum())})
    iou["any"][year] = (int((any_c & any_n).sum()), int((any_c | any_n).sum()))

    dl = (frn[:, ftn.index("dryline")] == 1) & valid
    freq_dl += dl.sum(0)
    for s in ("MAM", "JJA"):
        sel = np.isin(months, [m for m, ss in SEASON.items() if ss == s])
        freq_dl_season[s] += dl[sel].sum(0)
    for m in range(1, 13):
        sel = months == m
        monthly.append({"year": year, "month": m, "type": "dryline",
                        "codsus": 0, "noaa": int(dl[sel].sum()),
                        "valid": int(valid[sel].sum())})
    per_time = dl.sum((1, 2))
    k = int(per_time.argmax())
    if per_time[k] > best_dl[0]:
        best_dl = (int(per_time[k]), str(common[k]))
    print(f"{year}: {len(common)} joint times, any-front IoU "
          f"{iou['any'][year][0] / iou['any'][year][1]:.3f}, "
          f"max dryline px/time {per_time[k]}", flush=True)

np.savez_compressed(
    OUT / "aggregates.npz",
    n_valid=n_valid,
    **{f"freq_codsus_{t}": freq_c[t] for t in TYPES},
    **{f"freq_noaa_{t}": freq_n[t] for t in TYPES},
    freq_dryline=freq_dl,
    **{f"freq_dryline_{s}": freq_dl_season[s] for s in ("MAM", "JJA")},
    **{f"n_valid_{s}": n_valid_season[s] for s in ("MAM", "JJA")},
    iou_years=np.array(sorted(iou["any"])),
    **{f"iou_{t}": np.array([iou[t][y] for y in sorted(iou[t])])
       for t in iou},
)
pd.DataFrame(monthly).to_csv(OUT / "monthly_counts.csv", index=False)
print("best dryline case:", best_dl)
