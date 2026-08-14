"""End-to-end demo: run all four methods on one year, validate with CAPE.

    PYTHONPATH=src python -m convective_id.demo [year]     (default 2019)

Methods run on the FULL base grid (spatial contiguity needs the ocean edge);
validation and the summary are restricted to land rows (the suite's domain).
Artifacts -> results/convective_id/:

    summary_<year>.csv        per-method CAPE validation + reference rows
    agreement_<year>.csv      pairwise Jaccard overlap of the labels
    labels_<year>.parquet     per-row labels/scores for every method
    figures/cape_validation_<year>.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from convection_skill import config as cs_config
from convection_skill.config import AnalysisConfig
from convection_skill.dataset import build_base_table

from .features import add_structure_features
from .methods import classify_all
from .validate import agreement_matrix, plot_cape_validation, validate_with_cape

OUT = cs_config.RESULTS_DIR / "convective_id"


def main(year: int = 2019) -> pd.DataFrame:
    t0 = time.time()
    base = build_base_table(AnalysisConfig(years=(year,)))
    df = add_structure_features(base)
    land = (df["land_frac"] >= cs_config.LAND_FRACTION_MIN).to_numpy()
    n_precip = int((df["is_precip"].to_numpy() & land).sum())
    print(f"[{year}] features on {len(df):,} rows; "
          f"{n_precip:,} precipitating land cell-hours ({time.time()-t0:.0f}s)",
          flush=True)

    t0 = time.time()
    results = classify_all(df)
    print(f"[{year}] 4 methods classified ({time.time()-t0:.0f}s)", flush=True)
    if "feature_importances" in results["random_forest"].attrs:
        imps = results["random_forest"].attrs["feature_importances"]
        ranked = sorted(imps.items(), key=lambda kv: -kv[1])
        print("      forest importances:",
              ", ".join(f"{k}={v:g}" for k, v in ranked[:5]), flush=True)

    summary = validate_with_cape(df, results, row_mask=land)
    agreement = agreement_matrix(results, df["is_precip"].to_numpy() & land)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / f"summary_{year}.csv", index=False)
    agreement.to_csv(OUT / f"agreement_{year}.csv")
    labels = pd.concat(
        {name: res[["label", "score"]] for name, res in results.items()}, axis=1)
    labels.columns = [f"{m}_{c}" for m, c in labels.columns]
    pd.concat([df[["date", "slot", "lat", "lon"]], labels], axis=1) \
        .to_parquet(OUT / f"labels_{year}.parquet")
    plot_cape_validation(df, results, OUT / "figures" / f"cape_validation_{year}.png",
                         row_mask=land)

    pd.set_option("display.width", 200)
    print("\n== CAPE validation (land precip rows; AUC 0.5 = none, 1 = perfect) ==")
    print(summary.round(3).to_string(index=False))
    print("\n== label agreement (Jaccard) ==")
    print(agreement.round(2).to_string())
    print(f"\nartifacts -> {OUT}", flush=True)
    return summary


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2019)
