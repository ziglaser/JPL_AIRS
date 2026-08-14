from pathlib import Path

import convection_skill.plotting as pl
import convective_id.plotting as cid_plotting

from convection_skill import config, dataset
from convection_skill.config import AnalysisConfig

OUT = Path("results/replication/figures")

def SM_plots():
    # unified builder; SM is the daily pre-window surface value (sm_raw),
        # constant across the evening hours by construction (timing guard)
        table = dataset.build_dataset(AnalysisConfig.paper())
    
        OUT.mkdir(parents=True, exist_ok=True)
        date = "2020-06-01"
    
        # Per-field colour scales (qpe, CAPE and soil moisture live on different ranges).
        cmaps = {"mu_cape": "turbo", "qpe": "Blues", "sm_raw": "YlGnBu"}
        vmaxs = {"mu_cape": 4000, "qpe": 8, "sm_raw": 0.6}
    
        # 1. Multi-field snapshot: three fields as panels of one CONUS figure.
        fig = pl.plot_field_map(None, table, ["mu_cape", "qpe", "sm_raw"],
                                date=date, hour_utc=21, cmap=cmaps, vmax=vmaxs)
        fig.savefig(OUT / "panels_21UTC.png", dpi=150, bbox_inches="tight")
    
        # 2. Animate the evening forecast (21->02 UTC) and save a GIF.
        pl.animate_field_map(table, ["mu_cape", "qpe"], date=date,
                             hours=config.FORECAST_HOURS_UTC,
                             cmap=cmaps, vmax=vmaxs,
                             save_path=OUT / "evening_cape_qpe.gif", fps=2)

def convective_id_plots():
    """The convective-flag threshold-sweep animation, exactly as generated
    2026-07-23: paper-years base superset, land rows only (the suite's
    domain), defaults for the threshold sweep [1..25]% and the 4000+ J/kg
    CAPE clip bin. ~1 min from the cached base; a few minutes on first build.
    """

    cfg = AnalysisConfig.from_files("configs/data_table.yaml",
                                    "configs/hypothesis_tests.yaml")[0]
    base = dataset.build_dataset(cfg)

    cid_plotting.animate_cape_qpe_convective_threshold(
        df=base,
        save_path=config.RESULTS_DIR / "convective_id" / "figures" / "cape_qpe_convective_threshold.gif",
        thresholds_pct=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0),
        min_count=10,
        cape_max=4000.0,
        qpe_ticks=(80.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.95),
        fps=0.5,
    )

    # the same sweep as one readable line plot: % of high-CAPE cells flagged
    # convective vs threshold, one line per QPE-percentile subset. The QPE
    # cuts come from the UNSCREENED base sample (the replication ladder /
    # suite convention) -- percentiles of the screened frame itself would
    # collapse to 0 mm/h at P90/P95 because the rain screens delete wet rows.
    # Sample for THIS plot: the normal data filter (AIRS/product validity +
    # complete days) but WITHOUT the post-rain screens, which would otherwise
    # empty the extreme-QPE subsets (they exist to delete wet rows). Cheap:
    # every variant shares the one cached base table.
    from dataclasses import replace

    no_rain_screen_cfg = replace(cfg, screen_overpass_rain=False,
                                 screen_forecast_rain=False)
    ladder = dataset.qpe_percentile_thresholds(
        dataset.build_base_table(cfg), cfg,
        percentiles=(90.0, 95.0, 99.0, 99.5, 99.9, 99.95))
    cid_plotting.plot_high_cape_convective_fraction(
        df=dataset.build_dataset(no_rain_screen_cfg),
        save_path=config.RESULTS_DIR / "convective_id" / "figures" / "high_cape_convective_fraction.png",
        thresholds_pct=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0),
        qpe_thresholds=ladder,
    )


if __name__ == "__main__":
    convective_id_plots()
