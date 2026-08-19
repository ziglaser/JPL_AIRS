"""Central configuration for the trajectory-kernel tool: every constant in one
place, each annotated with the data fact or paper it comes from.

Same interpretability contract as :mod:`convection_skill.config` -- to check the
code implements the intended physics, a reader compares this one file against the
data audit (``docs/TRAJECTORY_KERNEL_WORKPLAN.md`` sections 0-1) and the cited
literature; to re-target it, they edit this file.

Literature keys used below:
- Lin+2003 / Fasoli+2018 : STILT footprint = near-surface residence-time
  sensitivity; contact layer ~0.5*PBLH.
- Sodemann+2008          : Lagrangian moisture-source diagnostic; PBL gate at
  1.5*PBLH; rain-out discounting.
- Stohl 1998            : single-trajectory position error ~20% of distance
  travelled, ~linear in age (the fuzz rule).
- McGrath-Spangler & Denning 2012 ; Seidel+2012 : N. American summer PBL depth
  climatology and diurnal cycle (deep afternoon, ~200 m nocturnal collapse).
- Guillod+2015          : soil-moisture -> afternoon-rain coupling scales
  (~140 km, 3-15 h).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
#: Data root: the JPL_AIRS_DATA env var when set (dev: /mnt/d/JPL_AIRS/data,
#: cluster: the AIRS_SMAP_Front_data root), else the repo-local data/ tree --
#: the same convention as convection_skill.config and the slurm scripts.
DATA_DIR: Path = Path(os.environ.get("JPL_AIRS_DATA", REPO_ROOT / "data"))
RESULTS_DIR: Path = Path(os.environ.get("JPL_AIRS_RESULTS", REPO_ROOT / "results"))
#: Global 1-deg fractional land-sea mask (variable ``lsm``). The historical
#: location was <data root>/lsm.nc; current data trees carry it at
#: masks/land_surface_mask.nc -- prefer whichever exists (checked in order).
_LSM_CANDIDATES = (DATA_DIR / "lsm.nc", DATA_DIR / "masks" / "land_surface_mask.nc")
LSM_PATH: Path = next((p for p in _LSM_CANDIDATES if p.exists()), _LSM_CANDIDATES[0])

#: Derived inputs for the upwind feature pipeline (scripts/build_upwind_features.py).
#: Each is overridable by CLI; these are the canonical locations under DATA_DIR.
SMAP_BASELINE_PATH: Path = (
    DATA_DIR / "soil_moisture" / "SMAP_L4_smsfc_monthly_baseline_2016-2021.nc")
PBLH_3HRLY_PATH: Path = (
    DATA_DIR / "PBL_depth" / "derived" / "PBLH_1deg_3hrly_2017-2021.nc")
PBLH_CLIM_PATH: Path = (
    DATA_DIR / "PBL_depth" / "derived" / "PBL_climatology_1deg_conus_utc_2017-2021.nc")
FCST_TABLE_DIR: Path = DATA_DIR / "FCST_SMAP_MRMS"

#: The HYSPLIT trajectory day used for method development.
TRAJ_DIR: Path = DATA_DIR / "wrf27km_20190605" / "wrf27km_20190605"
NOGRID_TEMPLATE: str = "nogrid_wrf27km_GOOD_20190605_{granule}.nc"
FULLGRID_NAME: str = "fullgrid_wrf27km_GOOD_1p00deg_20190605_1700-2059.nc"

# --------------------------------------------------------------------------- #
# Granules and overpass swaths  (audit: WORKPLAN section 1)
# --------------------------------------------------------------------------- #
#: The five AIRS granules present for 2019-06-05, grouped by overpass swath.
#: Early swath ~18:50-19:02 UTC (Aqua early-afternoon pass); late swath
#: ~20:32-20:38 UTC (next orbit, ~1.7 h later). Kept as a mapping so a parcel can
#: be tagged with its swath (the swaths have different available look-back time).
SWATHS: dict[str, tuple[int, ...]] = {
    "early": (188, 189, 190),
    "late": (205, 206),
}
ALL_GRANULES: tuple[int, ...] = tuple(g for gs in SWATHS.values() for g in gs)

# --------------------------------------------------------------------------- #
# Trajectory vertical axis  (audit)
# --------------------------------------------------------------------------- #
#: ``level`` is a fixed pressure ladder shared by every release column: index 0
#: is the top (~102 hPa / ~16 km), increasing index = lower/denser air, down to
#: the near-surface. Indices 53-56 are unused (always NaN).
N_LEVELS_USED: int = 53  # levels 0..52 carry data
NEAR_SURFACE_LEVEL: int = 52  # ~987 hPa / ~256 m at release

# --------------------------------------------------------------------------- #
# Output grid  (from the fullgrid twin; == the CAPE analysis grid + lsm.nc)
# --------------------------------------------------------------------------- #
GRID_LAT: tuple[float, float, float] = (25.5, 52.5, 1.0)  # (min, max, step) inclusive
GRID_LON: tuple[float, float, float] = (-106.5, -64.5, 1.0)

#: Source-cell size (degrees) of the kernel window around each receptor
#: (2026-07-23, Zach: resolve the actual parcel-cloud shape instead of 1-deg
#: blocks). The TARGET grid stays the 1-deg CAPE/lsm grid above; only the
#: relative source window is gridded finer. Source cells NEST inside the 1-deg
#: target cells (offsets at odd multiples of step/2), so 16 quarter-degree
#: cells tile each target cell and staircase outlines align exactly with the
#: receptor-cell edges.
SOURCE_STEP_DEG: float = 0.25

# --------------------------------------------------------------------------- #
# Unit sanity ranges  (asserted at ingest; audit "Q6 data quality" bounds,
# padded slightly). A file outside these signals a units/version mismatch.
# --------------------------------------------------------------------------- #
RANGE_LAT: tuple[float, float] = (15.0, 60.0)
RANGE_LON: tuple[float, float] = (-115.0, -55.0)
RANGE_ALT_M: tuple[float, float] = (-50.0, 20000.0)
RANGE_PRES_HPA: tuple[float, float] = (80.0, 1050.0)
RANGE_T_K: tuple[float, float] = (180.0, 330.0)
RANGE_Q_GKG: tuple[float, float] = (0.0, 30.0)  # g/kg (audit Q4)

# --------------------------------------------------------------------------- #
# Boundary layer  (pbl.py)
# --------------------------------------------------------------------------- #
#: Climatological daytime well-mixed PBL depth over summer CONUS, and the shallow
#: nocturnal stable-layer depth it collapses to (McGrath-Spangler & Denning 2012;
#: Seidel+2012). The nocturnal value is decisive here: the trajectory window runs
#: to 02 UTC (~evening local), so late-lag surface coupling should nearly shut
#: off. West->east the daytime depth runs ~3 km (Rockies) -> ~1 km (East);
#: PBL_DAYTIME_M is a domain-mean default, PBL_DEPTH_GRADIENT the west-east slope.
PBL_DAYTIME_M: float = 1800.0
PBL_NOCTURNAL_M: float = 200.0
#: Local hours (solar-ish, derived from longitude) bracketing the daytime plateau
#: and the night. A smooth cosine ramps between PBL_NOCTURNAL_M and PBL_DAYTIME_M.
PBL_PEAK_HOUR_LOCAL: float = 15.0   # mid-afternoon maximum
PBL_SUNRISE_HOUR_LOCAL: float = 7.0
PBL_SUNSET_HOUR_LOCAL: float = 19.0
#: Extra daytime depth per degree longitude west of the domain's east edge, m/deg.
#: (~ (3000-1000) m over ~40 deg of longitude.)
PBL_WEST_DEEPENING_M_PER_DEG: float = 50.0
PBL_REFERENCE_LON: float = -65.0  # east edge; deepening measured west of here
PBL_FIXED_DEFAULT_M: float = PBL_DAYTIME_M  # ConstantPBL convenience default
#: Nearest-time tolerance (hours) for the assessed 3-hourly PBLH lookup
#: (pbl.GriddedPBL): half the Guo et al. (2024) product's 3-h sampling interval,
#: so inside temporal coverage the nearest sample always qualifies, while queries
#: beyond the record's ends (2016, the Oct-2021 gap) fall through to climatology.
PBLH_TIME_TOLERANCE_H: float = 1.5

# --------------------------------------------------------------------------- #
# Surface contact  (contact.py)
# --------------------------------------------------------------------------- #
#: A parcel is surface-coupled when its altitude is below CONTACT_FRACTION*PBLH.
#: Default 1.0 (compromise); STILT uses 0.5 (strong coupling), Sodemann 1.5
#: (inclusive of the entrainment zone). The weight tapers smoothly to 0 across
#: the top TAPER_FRACTION of that layer rather than a hard cut.
CONTACT_FRACTION: float = 1.0
CONTACT_FRACTION_STILT: float = 0.5
CONTACT_FRACTION_SODEMANN: float = 1.5
CONTACT_TAPER_FRACTION: float = 0.25  # top 25% of the contact layer ramps 1->0

# --------------------------------------------------------------------------- #
# Available surface energy  (insolation.py)
# --------------------------------------------------------------------------- #
#: The hour-to-hour weight that footprint.py leaves to a downstream step. Contact
#: time is multiplied by the surface AVAILABLE ENERGY Rn - G ~= a * DSWF, the
#: reservoir the Bowen ratio partitions between sensible and latent heating.
#: Without it, a contact hour at 01 UTC (dark) counts as much as one at 20 UTC.
SOLAR_CONSTANT_WM2: float = 1361.0
#: Bulk clear-sky transmissivity, DSWF = S0*cosZ*(B0 + B1*cosZ) (Liu & Jordan
#: 1960 form): transmissivity rises with sun elevation as air mass falls.
CLEARSKY_B0: float = 0.60
CLEARSKY_B1: float = 0.20
#: a = (Rn - G)/DSWF. Rn/DSWF ~= 0.60-0.70 for vegetated land (albedo 0.15-0.25,
#: net longwave -60 to -100 W/m2) times (1 - G/Rn), G/Rn ~= 0.05-0.15 vegetated
#: and 0.2-0.4 bare. Cross-check: 0.55*750 = 413 W/m2 vs H + LE = 400 W/m2 for a
#: high-plains June afternoon (3%).
AVAILABLE_ENERGY_COEF: float = 0.55
AVAILABLE_ENERGY_COEF_BARE: float = 0.42
AVAILABLE_ENERGY_COEF_VEG: float = 0.58

#: Mean mixed-layer air density used for the column mass m* = RHO_ML * PBLH
#: (kg/m2), which converts the extensive influence Phi (J/m2) to the intensive
#: Omega (J/kg). 1.1 kg/m3 is the mixed-layer mean over the high plains
#: (1.05-1.20 across the domain); good to about 10%.
RHO_ML_KG_M3: float = 1.10

# --------------------------------------------------------------------------- #
# Land mask
# --------------------------------------------------------------------------- #
#: Land-fraction cutoff for *masking and plotting* (a cell is "land" above this).
#: NOTE: the footprint itself weights each source point by the *continuous* land
#: fraction (0-1) from ``land.make_land_lookup`` -- a 60%-land cell contributes
#: 0.6, which is smoother and more honest than a hard 0.5 cut -- so this constant
#: is not applied inside the footprint integral, only in coverage masks/plots.
#: Matches convection_skill.config.LAND_FRACTION_MIN.
LAND_FRACTION_MIN: float = 0.50

# --------------------------------------------------------------------------- #
# Sub-hourly resampling  (resample.py)
# --------------------------------------------------------------------------- #
#: The trajectories are stored hourly; the footprint integral is sampled on a
#: finer step so a fast parcel does not skip source cells. Minutes.
RESAMPLE_STEP_MIN: float = 10.0

# --------------------------------------------------------------------------- #
# Fuzzing  (fuzz.py) -- Stohl 1998
# --------------------------------------------------------------------------- #
#: Trajectory-position uncertainty sigma(tau) = FUZZ_SIGMA0_KM + FUZZ_ALPHA * D,
#: where D is the along-track distance travelled since the receptor (km). Stohl's
#: review gives error ~20% of distance travelled, growing ~linearly with age.
#: Default sigma0 = 0 (Zach, 2026-07-23): the fuzz is a pure fraction of the
#: backward travel distance, so at lag 0 the deposit collapses to the receptor
#: cell (we conditioned on the parcel being there) and the kernel inflates only
#: as the air is traced upstream. Set sigma0 > 0 to add a constant within-cell /
#: release-geometry blur on top (50 km ~ half a 1-deg cell).
FUZZ_SIGMA0_KM: float = 0.0
FUZZ_ALPHA: float = 0.20         # 20%-of-distance rule (Stohl 1998)
#: User-facing knob multiplying the *growth* term to make kernels sharper/blurrier.
FUZZINESS: float = 1.0

# --------------------------------------------------------------------------- #
# Receptor and kernel extents  (footprint.py)
# --------------------------------------------------------------------------- #
#: RECEPTOR_BAND (user decision): which arriving parcels count as "what the
#: surface cell sees", as an altitude band in metres. Default near-surface, for
#: the clean surface-coupling story; widen toward a full low-troposphere column
#: for broader inflow. (Tracking property changes at all levels -- e.g. to adjust
#: advected CAPE/CIN -- is a separate feature, not built here.)
RECEPTOR_BAND_M: tuple[float, float] = (0.0, 1000.0)

#: Horizontal half-width (degrees) of the receptor catch box at arrival time: a
#: parcel counts as "arriving" if within this of the target cell centre. Default
#: = half the 1-deg grid step, i.e. the parcel is literally IN the target cell at
#: the arrival hour (matching ``build_all``'s nearest-cell assignment); widen it
#: to borrow parcels from neighboring cells when counts are thin.
RECEPTOR_CATCH_HALFWIDTH_DEG: float = 0.5  # == the grid cell itself

#: Relative source window half-width (degrees) stored per receptor. Guillod+2015
#: coupling reaches ~140 km (~1.3 deg); the window must also hold the parcels
#: themselves: measured on 2019-06-05, the farthest near-surface parcel sits
#: 4.6 deg from its arrival cell over the full <=7 h look-back (p99 = 2.6 deg),
#: so 6 deg covers every parcel plus fuzz margin. (Was 10 deg on the 1-deg
#: source grid; at SOURCE_STEP_DEG = 0.25 that would balloon build_all's dense
#: array ~5x for empty far field.) Source axis is stored as offsets from the
#: receptor.
SOURCE_WINDOW_HALFWIDTH_DEG: float = 6.0

#: Kernel support truncation (2026-07-23, Zach): at each lag the kernel keeps
#: only the cells inside the smallest circle, grown outward from the member
#: parcels' center of mass, that contains this fraction of the parcels in PBL
#: contact at that lag (the deposit's own gate). Mass outside -- mostly the
#: far-field Gaussian fuzz tails -- is dropped and the lag renormalized, so
#: the kernel follows the actual parcel cloud. The physical ``footprint`` is
#: never truncated. ``None`` disables truncation.
KERNEL_CONTAINMENT_FRAC: float = 0.90

#: Below this member count the containment order statistic dist[ceil(0.9*n)-1]
#: degenerates (at n=4 it is the maximum, so containment silently does nothing)
#: and sample size leaks into Psi; receptors under this threshold skip
#: containment entirely and are flagged (HANDOFF.md 8.6 /
#: UPWIND_INDEX_REVIEW.md F6).
CONTAINMENT_MIN_PARCELS: int = 20

#: Maximum look-back lag (hours). Bounded in general by soil-moisture memory
#: (~24 h; Guillod+2015) but capped by this dataset at t_arrival - t_overpass
#: (<= ~7 h here). The builder never exceeds available trajectory length.
MAX_LAG_HOURS: float = 12.0

# --------------------------------------------------------------------------- #
# Reproducibility / geodesy
# --------------------------------------------------------------------------- #
EARTH_RADIUS_KM: float = 6371.0
RANDOM_SEED: int = 20240813
