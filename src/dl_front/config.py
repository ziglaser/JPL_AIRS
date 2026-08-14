"""All constants for the DL-FRONT replication, each with its source.

Paper = Biard & Kunkel (2019): "Automated detection of weather fronts using
a deep learning neural network", Adv. Stat. Clim. Meteorol. Oceanogr. 5,
147-160 (docs/papers/Biard_Kunkel_2019.pdf).  Section references below are to
that paper.  Grid/label constants are shared with ``front_finder.config``.

Two kinds of constants live here:

* **Structural** (defined in this file): grid geometry, variable/class
  names, file paths, label-overlap priority -- changing these requires
  matching data or code changes.
* **Tunable** (loaded from ``configs/dl_front.yaml`` at import): every
  hyperparameter and analysis knob, each documented with its paper source in
  the YAML.  Point the ``JPL_DLFRONT_CONFIG`` env var at an alternate YAML
  to run an experiment without editing the tracked defaults.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import yaml

from front_finder import config as fd

REPO_ROOT = fd.REPO_ROOT
#: Overridable like front_finder.config.RESULTS_DIR (cluster scratch).
RESULTS_DIR = Path(os.environ.get("JPL_AIRS_RESULTS",
                                  str(REPO_ROOT / "results"))) / "dl_front"

# --------------------------------------------------------------------------- #
# Input fields (paper "Data availability": M2I1NXASM = MERRA-2 2d 1-hourly
# instantaneous single-level assimilation diagnostics V5.12.4, variables
# QV2M, SLP, T2M, U10M, V10M "remapped by bicubic interpolation to a
# 1x1 deg latitude-longitude grid"; sampled 3-hourly per section 3.1)
# --------------------------------------------------------------------------- #
#: Channel order fixed as in the paper's Fig. 1 caption (temperature,
#: humidity, pressure, u-wind, v-wind).
SFC_VARS = ("T2M", "QV2M", "SLP", "U10M", "V10M")
SFC_DIR = fd.MERRA2_DIR / "sfc_daily"
NORM_STATS_PATH = fd.MERRA2_DIR / "sfc_norm_stats.json"
#: 3-hourly sampling of the hourly collection (section 3.1: "3-hourly
#: instantaneous values"), matching the CSB bulletin hours 00/03/../21 UTC.
SFC_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

#: Label grid (identical to the CODSUS rasters): 1 deg, lat 10..77 N,
#: lon -171..-31 E (section 3.1 "domain of 10-77 N and 171-31 W").
GRID_SHAPE = fd.GRID_SHAPE
LABEL_LATS = tuple(float(v) for v in range(10, 78))
LABEL_LONS = tuple(float(v) for v in range(-171, -30))

# Native-resolution subset fetched for local bicubic remapping: MERRA-2 is
# 0.5 x 0.625 deg (lat = -90 + 0.5j, lon = -180 + 0.625k); indices chosen
# with >=2 native cells of margin so the spline never extrapolates at the
# label-domain edges.
MERRA2_SFC_LAT_SLICE = (196, 338)      # 8.0 .. 79.0 N inclusive (143 pts)
MERRA2_SFC_LON_SLICE = (10, 242)       # -173.75 .. -28.75 E inclusive (233 pts)

# --------------------------------------------------------------------------- #
# Classes & labels
# --------------------------------------------------------------------------- #
#: Paper section 3.1: cold, warm, stationary, occluded, none (one-hot).
CLASS_NAMES_5 = ("cold", "warm", "stationary", "occluded", "none")
#: Dryline extension: NOAA XML analyses re-rasterized to the CODSUS schema
#: (src/front_formats/xml_to_codsus.py) carry an extra dryline channel; the
#: file's front_type order is used verbatim.
CLASS_NAMES_6 = ("cold", "warm", "stationary", "occluded", "dryline", "none")
#: NOAA-XML label files (2007-2022), CODSUS schema + dryline channel.
NOAA_LABELS_DIR = fd.NOAA_LABELS_DIR
#: Section 3.3 / Fig. 2 region mask (>40 front crossings/yr envelope),
#: applied as the per-pixel loss & metric weight for training AND evaluation.
REGION_MASK_PATH = fd.REGION_MASK_PATH
#: Published-raster overlap resolution: the per-type CODSUS channels can
#: overlap where fronts cross; their renderer resolves ties with this
#: painter's priority (reverse-engineered 2026-08-05, line IoU 0.995+;
#: docs/FRONTS.md).  The NOAA-schema files are already exclusive.
TYPE_PRIORITY = ("warm", "occluded", "stationary", "cold", "dryline")

# --------------------------------------------------------------------------- #
# AIRS-FCST fullgrid inputs & kriged gap-filled caches (stage B'/C data)
# --------------------------------------------------------------------------- #
def _resolve_airs_fcst_root() -> Path:
    """Root of the HYSPLIT AIRS-FCST fullgrid archive (manifest reorg
    2026-08-13), layout ``<root>/YYYY/wrf27km_YYYYMMDD/fullgrid_wrf27km_
    GOOD_1p00deg_*`` (2003-2022).  Resolution order:

    1. ``JPL_AIRS_FCST`` env var, when set (explicit override always wins,
       but a value pointing at a nonexistent directory triggers a loud
       warning: every ``find_fullgrid`` lookup against such a root returns
       None, which silently degrades gap-bank harvests and, worse, lets
       krige_fill write empty-but-valid done-marker year caches -- a typo'd
       export must not fail silently; review 2026-08-13);
    2. the canonical cluster location: the archive is a SIBLING of the data
       root (``/gpfs/scratch/smap-convection/AIRS_FCST_1deg`` next to
       ``.../AIRS_SMAP_Front_data``), used whenever it exists;
    3. the single local demo day under ``data/HYSPLIT_demo``.
    """
    env = os.environ.get("JPL_AIRS_FCST", "")
    if env:
        root = Path(env)
        if not root.is_dir():
            warnings.warn(
                f"JPL_AIRS_FCST={env} does not exist (or is not a "
                f"directory): every fullgrid lookup will come back empty. "
                f"Check the export for typos (canonical cluster path: "
                f"/gpfs/scratch/smap-convection/AIRS_FCST_1deg).",
                stacklevel=2)
        return root
    sibling = fd.DATA_ROOT.parent / "AIRS_FCST_1deg"
    if sibling.exists():
        return sibling
    return fd.DATA_ROOT / "HYSPLIT_demo"


AIRS_FCST_ROOT = _resolve_airs_fcst_root()
#: Kriged gap-filled surface caches (dl_front.krige_fill), one
#: ``kriged_sfc_{year}.nc`` per year in PHYSICAL units, keyed by the
#: training/eval ``--source`` value (manifest reorg 2026-08-13, replacing the
#: retired ``KRIGED_DIR = front_id/kriged`` root):
#:
#: * ``kriged-degraded`` -> ``front_id/degraded_reanalysis`` (reanalysis with
#:   AIRS-shaped gaps re-filled) -- present in the cluster manifest;
#: * ``kriged-airs``     -> ``front_id/kriged_airs_fcst`` (real AIRS-FCST
#:   fields) -- OUR naming choice: the manifest carries no placeholder for
#:   this cache, so it is defined as the symmetric sibling of
#:   ``degraded_reanalysis`` and created on demand by the builders.
KRIGED_SOURCE_DIRS = {
    "kriged-degraded": fd.DATA_ROOT / "front_id/degraded_reanalysis",
    "kriged-airs": fd.DATA_ROOT / "front_id/kriged_airs_fcst",
}
#: Published DL-FRONT prediction rasters (Biard & Kunkel 2019), the third
#: leg of the three-way test evaluation (user decision 2026-08-13): yearly
#: ``1deg_{w}wide/{freq}/merra2_merra2-1deg_{w}wide_{freq}_{year}.nc`` files,
#: 1980-2018, binary fronts(time, front, lat, lon) on the exact label grid.
#: (Their ``title`` attr says "coded surface bulletins" -- stale inherited
#: metadata; they ARE the model predictions.)  Manifest reorg 2026-08-13:
#: the cluster data root now carries the same
#: ``front_id/predicted_fronts/bk19`` tree as the local checkout, so the
#: front_finder default resolves everywhere; the JPL_BK19_DIR override is
#: retained for out-of-tree archives but is likely unnecessary now.
BK19_DIR = Path(os.environ.get("JPL_BK19_DIR", str(fd.BENCHMARK_DIR)))
#: Global 1-deg land fraction ('lsm' on half-degree cell centers
#: -89.5..89.5 / -179.5..179.5), bilinearly interpolated onto the integer
#: LABEL_LATS/LONS grid and thresholded at LAND_FRACTION_MIN to build the
#: 6-class analysis domain (dl_front.dataset.analysis_domain; user decision
#: 2026-08-13).
LAND_MASK_PATH = fd.DATA_ROOT / "masks/land_surface_mask.nc"
#: 16-day-repeat swath climatology (dl_front.swath.build_swath_bank): per
#: (cycle day, period hour) frequency of AIRS surface-level coverage,
#: thresholded into the "expected swath" footprint that splits missing
#: pixels into swath-boundary vs retrieval (cloud) gaps.
SWATH_BANK_PATH = fd.DATA_ROOT / "masks/swath_bank.npz"

#: gap_type flag values stored in the kriged caches (int8).  Chosen as one
#: compact ordinal variable (not separate masks) so the CNN can consume it
#: later as a single extra channel or a pair of one-hot channels without a
#: schema change.
GAP_OUT_OF_DOMAIN = -1   # outside the crop domain (analysis box + halo;
                         # schema v3, user decision 2026-08-13)
GAP_OBSERVED = 0         # AIRS retrieval present
GAP_CLOUD = 1            # inside the expected swath but not retrieved
GAP_OUT_OF_SWATH = 2     # outside the expected swath (orbit geometry)

# --------------------------------------------------------------------------- #
# Tunables (configs/dl_front.yaml)
# --------------------------------------------------------------------------- #
#: The tracked defaults; override with JPL_DLFRONT_CONFIG=<path to yaml>.
CONFIG_YAML = Path(os.environ.get("JPL_DLFRONT_CONFIG",
                                  REPO_ROOT / "configs/dl_front.yaml"))

#: The full tunable inventory: (yaml section, yaml key) -> module constant.
#: Sources and rationale for every value are documented in the YAML itself.
TUNABLES = {
    ("model", "n_conv_layers"): "N_CONV_LAYERS",
    ("model", "n_filters"): "N_FILTERS",
    ("model", "kernel_size"): "KERNEL_SIZE",
    ("model", "dropout"): "DROPOUT",
    ("loss", "none_weight"): "NONE_WEIGHT",
    ("training", "learning_rate"): "LEARNING_RATE",
    ("training", "batch_size"): "BATCH_SIZE",
    ("training", "max_epochs"): "MAX_EPOCHS",
    ("training", "patience"): "PAPER_PATIENCE",
    ("training", "n_folds"): "N_FOLDS",
    ("training", "fold_seed"): "FOLD_SEED",
    ("stages", "degraded_lr_factor"): "DEGRADED_LR_FACTOR",
    ("stages", "finetune_lr_factor"): "FINETUNE_LR_FACTOR",
    ("stages", "finetune_patience"): "FINETUNE_PATIENCE",
    ("stages", "severity_ramp_epochs"): "SEVERITY_RAMP_EPOCHS",
    ("labels", "label_width"): "LABEL_WIDTH",
    ("splits", "train_years_5class"): "TRAIN_YEARS_5",
    ("splits", "eval_years_5class"): "EVAL_YEARS_5",
    ("splits", "train_years_6class"): "TRAIN_YEARS_6",
    ("splits", "eval_years_6class"): "EVAL_YEARS_6",
    ("degradation", "t2m_noise_sigma_k"): "T2M_NOISE_SIGMA_K",
    ("degradation", "q2m_noise_frac_sigma"): "Q2M_NOISE_FRAC_SIGMA",
    ("degradation", "observed_min_fraction"): "OBSERVED_MIN_FRACTION",
    ("evaluation", "roc_factors"): "ROC_FACTORS",
    ("airs", "hours"): "AIRS_HOURS",
    ("airs", "surface_level_hpa"): "AIRS_SURFACE_LEVEL_HPA",
    ("airs", "kriged_channels"): "KRIGED_CHANNELS",
    ("airs", "swath_min_fraction"): "SWATH_MIN_FRACTION",
    ("domain", "lat_range"): "ANALYSIS_LAT_RANGE",
    ("domain", "lon_range"): "ANALYSIS_LON_RANGE",
    ("domain", "land_fraction_min"): "LAND_FRACTION_MIN",
    ("kriging", "variogram_model"): "KRIGE_VARIOGRAM",
    ("kriging", "max_obs_points"): "KRIGE_MAX_OBS",
    ("kriging", "seed"): "KRIGE_SEED",
}


def load_tunables(path: Path | str = None) -> dict:
    """Read a tunables YAML -> {module constant: value}.

    Strict by design: every (section, key) in :data:`TUNABLES` must be
    present, and nothing else may be -- an unknown section or key (a typo,
    or a knob that no longer exists) raises immediately rather than being
    silently ignored.  Lists become tuples so the values are hashable and
    immutable like the constants they replace.
    """
    path = Path(path) if path is not None else CONFIG_YAML
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at top level")

    sections = {s for s, _ in TUNABLES}
    unknown_sections = set(raw) - sections
    if unknown_sections:
        raise ValueError(f"{path.name}: unknown sections {sorted(unknown_sections)}; "
                         f"expected {sorted(sections)}")

    out = {}
    problems = []
    for (section, key), const in TUNABLES.items():
        try:
            value = raw[section][key]
        except (KeyError, TypeError):
            problems.append(f"missing {section}.{key} (-> {const})")
            continue
        out[const] = tuple(value) if isinstance(value, list) else value
    for section in sections:
        extra = set(raw.get(section, {})) - {k for s, k in TUNABLES if s == section}
        if extra:
            problems.append(f"unknown keys in '{section}': {sorted(extra)}")
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return out


globals().update(load_tunables())

#: Per-variable normalization is not described in the paper (raw SLP ~1e5 Pa
#: cannot train against T ~3e2 K); standardization to zero mean / unit
#: variance with constants frozen from the training years is used, mirroring
#: the frozen-constants rule of front_finder.dataset (decision 2026-08-09).
