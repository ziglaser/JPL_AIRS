"""All constants for the front-detection work, each with its source.

Mirrors ``src/convection_skill/config.py``: constants live here and only here;
the YAML file ``configs/front_finder.yaml`` overrides the run-level knobs
via :class:`FrontConfig` (flat mapping, unknown keys raise).

Paper = Justin, McGovern & Allen (2025): "FrontFinder AI", AIES-D-24-0043.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Data/results roots are overridable for cluster deployments where data
#: lives on scratch rather than inside the repo checkout (SLURM, 2026-08-09).
DATA_ROOT = Path(os.environ.get("JPL_AIRS_DATA", REPO_ROOT / "data"))
# Manifest reorg 2026-08-13: the data root now mirrors the cluster tree
# (gattaca2:/gpfs/scratch/smap-convection/AIRS_SMAP_Front_data): labels live
# under front_id/met_drawn_fronts/, reanalysis under front_id/reanalysis/,
# published predictions under front_id/predicted_fronts/, shared masks under
# masks/.  (Supersedes the 2026-08-11 flat front_id/ layout.)
#: Analyst-drawn CSB fronts on the 1 deg MERRA-2 grid (NCICS; audit 2026-08-04).
#: Year files live in ``{width}wide/`` subdirectories (manifest reorg
#: 2026-08-13: the old ``1deg_`` prefix is gone); only the ``codsus_masked_*``
#: variant survived the 2026-08-11 reorg (labels.load_codsus).
CODSUS_DIR = DATA_ROOT / "front_id/met_drawn_fronts/WPC_CODSUS/WPC_1deg_gridded"
#: The authors' published model predictions on the same grid (hourly 1980-2018),
#: in ``1deg_{width}wide/{1hr,3hr}/`` subdirectories (inner layout unchanged
#: by the manifest reorg 2026-08-13).
BENCHMARK_DIR = DATA_ROOT / "front_id/predicted_fronts/bk19"
#: Vendored FrontFinder codebase, consumed as a library.
FRONTS_REPO = REPO_ROOT / "fronts"
RESULTS_DIR = Path(os.environ.get("JPL_AIRS_RESULTS",
                                  REPO_ROOT / "results")) / "front_finder"

# --------------------------------------------------------------------------- #
# Label grid & classes (audit 2026-08-04: ncdump of codsus_*_2018.nc)
# --------------------------------------------------------------------------- #
#: Training-label dilation in 8-connected iterations, threaded through every
#: stage (shards, degraded stage B, AIRS stage C, calibration).  0 as of
#: 2026-08-10: at 1 deg an UNDILATED pixel is already ~111 km wide -- thicker
#: than the paper's dilated 3-px/75-km target at 0.25 deg -- and the FSS
#: (3, 3) pooling supplies +-111 km of positional tolerance on its own.
#: Training at dilation 1 produced ~333 km swaths (quicklook FB 2.4-3.5),
#: which dilutes the per-cell Gini discrimination of the downstream
#: convective-initiation risk features.  Changing this invalidates the
#: materialized y_*.npy shards (materialize --labels-only rebuilds them).
LABEL_DILATION = 0
#: Front-label source, applied by ``labels.load_fronts`` to EVERY label load
#: (training, materialize, evaluation, NFA baseline):
#:   "codsus" -- analyst CSB fronts, 4 types, 2003-2018 (replication default);
#:   "noaa"   -- NOAA XML analyses re-rasterized to the same schema
#:               (src/front_formats/xml_to_codsus.py), 2006-2022, and the only
#:               source with DRYLINES.
#: Chosen via environment variable rather than a CLI flag because the class
#: count shapes the model head, the shard layout and every module's tensor
#: specs at import time -- one process, one label source:
#:   JPL_FRONT_LABELS=noaa python -m front_finder.train --name D1 ...
LABEL_SOURCE = os.environ.get("JPL_FRONT_LABELS", "codsus")
if LABEL_SOURCE not in ("codsus", "noaa"):
    raise ValueError(f"JPL_FRONT_LABELS must be 'codsus' or 'noaa', "
                     f"got {LABEL_SOURCE!r}")
#: NOAA-XML label files in the CODSUS schema, with the extra dryline channel,
#: in ``{width}wide/`` subdirectories (2007-2022; manifest reorg 2026-08-13).
NOAA_LABELS_DIR = (DATA_ROOT
                   / "front_id/met_drawn_fronts/NOAA_CODSUS/NOAA_1deg_gridded")
#: The >40-crossings/yr analysis envelope (Biard & Kunkel Fig. 2); ships with
#: the CODSUS download and defines what ``masked`` labels mean for BOTH sources.
REGION_MASK_PATH = DATA_ROOT / "masks/codsus_merra2-1deg_mask.nc"
#: Order of the ``front`` dimension restricted to the classes we train on.
#: The dryline class exists only in the NOAA XML analyses.
FRONT_TYPES = (("cold", "warm", "stationary", "occluded", "dryline")
               if LABEL_SOURCE == "noaa" else
               ("cold", "warm", "stationary", "occluded"))
#: fronts(time, front, lat, lon) ubyte: 0 = no front, 1 = front, 2 = fill
#: (outside mask in the ``_masked`` variants).
LABEL_FILL = 2
#: Label grid: 1 deg, lat 10..77 N (68), lon -171..-31 E (141).
GRID_SHAPE = (68, 141)
#: Zero-padded model grid.  levels=3 (model.py, 2026-08-10) needs dims % 4;
#: 72x144 (% 8) is kept anyway -- the materialized x shards use it, and the
#: extra padding is loss-weight-0.
PADDED_SHAPE = (72, 144)

# --------------------------------------------------------------------------- #
# Predictors (workplan section 3.2/3.3)
# --------------------------------------------------------------------------- #
#: Level Set A -- shifted up vs the paper's surface..850 because AIRS boundary
#: layer retrievals are weakly constrained (Susskind et al. 2014); 1000/925/850
#: keep the low-level frontal contrast, 700/500 add AIRS's strong layers.
TARGET_LEVELS_HPA = (1000, 925, 850, 700, 500)
#: Thermodynamic channels derivable from AIRS t/q/pres (fronts/utils/variables.py).
THERMO_VARS = ("T", "q", "r", "Td", "theta_e", "Tv", "RH")
WIND_VARS = ("u", "v")

# --------------------------------------------------------------------------- #
# MERRA-2 pretraining corpus (M2I3NPASM: inst3_3d_asm_Np, 0.5 x 0.625 deg,
# 3-hourly, 42 pressure levels; GES DISC OPeNDAP)
# --------------------------------------------------------------------------- #
MERRA2_DIR = DATA_ROOT / "front_id/reanalysis/MERRA2"   # manifest reorg 2026-08-13
#: Materialized training shards (materialize.py): per-year memmap .npy pairs
#: holding fully-derived, normalized samples so training never re-runs the
#: netCDF -> derive -> regrid path (post-mortem 2026-08-09: TF's file-backed
#: dataset.cache() writer leaks ~0.4 MB/sample until finalization).
#: Shards bake the labels in, so each label source gets its own directory
#: (the CODSUS shards predate the source switch and keep the bare name).
SHARD_DIR = MERRA2_DIR / ("shards" if LABEL_SOURCE == "codsus"
                          else f"shards_{LABEL_SOURCE}")
#: lev indices of TARGET_LEVELS_HPA in the 42-level M2I3NPASM grid
#: (lev[0]=1000, spacing 25 hPa to 700 then 50 hPa; verified via DDS 2026-08-04).
MERRA2_LEV_INDEX = {1000: 0, 925: 3, 850: 6, 700: 12, 500: 16}
#: lat[200:334:2] = 10..77 N at exactly 1 deg -- the label-grid latitudes.
MERRA2_LAT_SLICE = (200, 334, 2)
#: lon[14:239] INCLUSIVE (OPeNDAP convention) = -171.25..-30.625 at native
#: 0.625 deg -- must extend past -31.0 so interpolation to the integer-degree
#: label longitudes never extrapolates.
MERRA2_LON_SLICE = (14, 239)
MERRA2_VARS_3D = ("T", "QV", "U", "V")
#: Pretraining years (workplan 3.6): train 2003-2014, val 2015.  2016-2018
#: never enter pretraining (AIRS fine-tune split integrity).  NOAA labels
#: only start 2006, so the dryline runs train on 2006-2014 (same val year,
#: same embargo).
PRETRAIN_TRAIN_YEARS = tuple(range(2006 if LABEL_SOURCE == "noaa" else 2003,
                                   2015))
PRETRAIN_VAL_YEAR = 2015

# --------------------------------------------------------------------------- #
# Evaluation (paper section 2d; fronts/generate_performance_stats.py)
# --------------------------------------------------------------------------- #
#: Neighborhood dilation iterations evaluated on the 1 deg grid. The paper's
#: eval dilates truth by 8-connected iterations at 0.25 deg (2 iters = 50 km);
#: at 1 deg one iteration is ~111 km meridionally (and 111*cos(lat) km
#: zonally), so every reported number carries an explicit km caption.
EVAL_DILATIONS = (0, 1, 2)
#: Nominal km per dilation iteration at 1 deg (meridional; zonal shrinks with
#: cos(lat)).  111.2 km/deg: WGS-84 mean meridional degree.
KM_PER_ITERATION = 111.2
#: Truth-dilation convention copied from generate_performance_stats.py:
#: TP/FP are scored against the DILATED truth, FN against the UNDILATED truth
#: (deliberate upstream choice, kept for comparability).

#: Day-block bootstrap (house norm: convection_skill audit 2026-07-23 found
#: iid resampling understates CIs 2-3x on daily synoptic data).  Block length
#: ~1 synoptic period; sensitivity 3/14 days.
BLOCK_DAYS = 7
N_BOOT_REPS = 1000          # paper section 2d uses 1000 bootstrap iterations
CONFIDENCE_LEVEL = 95.0
BOOT_SEED = 20260804

# --------------------------------------------------------------------------- #
# Run-level knobs (YAML-overridable)
# --------------------------------------------------------------------------- #


@dataclass
class FrontConfig:
    """One evaluation/training run of the front-detection work.

    Only run-level choices live here; physical constants stay module-level.
    """

    # ---- evaluation scope ---------------------------------------------------
    eval_year: int = 2018                 # E0/E3 test year (embargoed for C)
    label_width: int = 1                  # "1wide" | "3wide" label variant
    masked_labels: bool = False           # use codsus_masked_* (>40 fronts/yr)
    #: Benchmark time subset: "3hr" files match the CODSUS 3-hourly axis.
    benchmark_freq: str = "3hr"
    dilations: tuple = EVAL_DILATIONS
    #: Restrict scoring to synoptic hours (0/6/12/18 UTC) as for USAD, or all.
    synoptic_only: bool = False

    # ---- inference ----------------------------------------------------------
    block_days: int = BLOCK_DAYS
    n_boot_reps: int = N_BOOT_REPS
    seed: int = BOOT_SEED

    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "FrontConfig":
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in dict(d).items()}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown FrontConfig keys {sorted(unknown)}; "
                             f"choose from {sorted(cls.__dataclass_fields__)}")
        return cls(**d)

    @classmethod
    def from_file(cls, path) -> "FrontConfig":
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(path.read_text())
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            raise ValueError(f"config must be .yaml/.yml/.json, got {path.name}")
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must contain a mapping at top level")
        return cls.from_dict(data)
