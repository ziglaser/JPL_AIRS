"""Tests for scripts/upwind_qa.py -- the physical-realism QA battery -- on a
synthetic world only (no JPL_AIRS_DATA, no real files).

The battery is exercised strictly through its pinned public contract:

* ``main(argv)`` on a yearly companion (``UPWIND_FEATURES_<YYYY>.nc``), the
  match-up file (``FCST_SMAP_MRMS_<YYYY>.nc``), and the daily kernel dir
  (``UPW_<YYYYMMDD>.nc``, which is where ``lag_weight`` lives -- it is never
  merged into the companion because its dims are wrong for the (date, time,
  lat, lon) axes);
* ``<out-dir>/qa_report.json``: a LIST of verdicts, each carrying
  ``name`` / ``status`` / ``metrics`` / ``detail`` / ``figures``, with
  ``status`` in {PASS, WARN, FAIL, SKIP}.

The synthetic world is CONSTRUCTED so each check's physical expectation holds
by design, and each adversarial variant breaks exactly one expectation:

* check 2 (upwind direction): the stored (upwind_dlat, upwind_dlon)
  displacement is exactly anti-parallel, in local-km space
  (dx = dlon * 111 km * cos(lat), dy = dlat * 111 km), to the FCST_u/v wind
  we write -- back-trajectories must run UPWIND, so the median cosine between
  displacement and wind is -1.  Flipping the displacement sign makes the
  parcels appear to have come from DOWNWIND (cosine +1), which must FAIL.
* check 3 (lag structure): later arrival slots integrate longer histories, so
  the lag_weight centroid must grow with arrival_step and the last slot must
  put ~no mass at lag 0 (a slot-6 arrival cannot be dominated by its own
  arrival hour).  Moving 30 % of slot-6 mass to lag 0 must be flagged.
* check 4 (kernel features carry information): omega sits near its physical
  scale (~2000, clear-sky insolation-hours) with in-band medians, and
  psi_anom = s_endpoint_anom + slot-GROWING noise, so r(psi, s_endpoint)
  decreases with slot -- the kernel average genuinely departs from the
  endpoint value as the footprint lengthens.  psi_anom == s_endpoint_anom
  exactly (r = 1 everywhere) means the convolution adds nothing and must FAIL.
* check 6 (sampling-density independence): n_parcels varies widely while every
  feature is constructed independent of it (equal psi variance across
  n_parcels bins).  Wiring omega to n_parcels breaks the |r| < 0.1 gate.
* check 7 (gamma_gap sanity vs rain): MRMS rain is wired to DECREASE with
  gamma_gap (a smaller LFC-minus-PBLH gap means convection fires more easily),
  so the rank correlation is negative.

SKIP paths: a missing daily dir skips check 3; a match-up file without MRMS
variables skips checks 5 and 7; an all-NaN kernel-feature year (the 2016
situation -- no assessed PBLH, no kernel features) skips checks 2-6.

The implementation is written in parallel from the same spec; where an
internal name must be guessed, verdicts are located first by "check <n>" in
the name, then by physics keywords, never by import.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before the script module (which draws figures) loads

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_SCRIPT = REPO_ROOT / "scripts" / "upwind_qa.py"

VALID_STATUSES = {"PASS", "WARN", "FAIL", "SKIP"}

# --------------------------------------------------------------------------- #
# the synthetic world's fixed geometry and clock
# --------------------------------------------------------------------------- #
_DATES = np.arange("2019-06-01", "2019-06-21",
                   dtype="datetime64[D]").astype("datetime64[ns]")   # 20 dates
#: The merged companion's time axis is 7 slots: slot 0 = the overpass itself
#: (kernel features honestly NaN there -- no arrival window), slots 1..6 = the
#: 21..02 UTC arrival ladder.  This mirrors merge_upwind_features exactly.
_TIME = np.arange(7)
_SLOTS = np.arange(1, 7)                                             # kernel slots
_LAT = 30.5 + np.arange(6)                                           # 6 rows
_LON = -100.5 + np.arange(8)                                         # 8 cols
_LAGS = np.arange(8.0)                                               # hours 0..7
_SHAPE = (_DATES.size, _TIME.size, _LAT.size, _LON.size)
_PBLH_M = 800.0
#: lag centroid per slot: later arrivals integrate longer upwind histories
#: (kernel extent <= ~7 h, Stohl-style growth); slot 6 centres at 4.6 h, so
#: its Gaussian (sigma 0.7 h) mass at lag 0 is ~4e-10 -- "approximately zero".
_LAG_MU = 1.0 + 0.6 * _SLOTS.astype(float)
_LAG_SIGMA = 0.7


def _lag_profile(step_index: int, slot6_lag0_mass: float = 0.0) -> np.ndarray:
    """The normalized lag_weight profile for one arrival step."""
    w = np.exp(-0.5 * ((_LAGS - _LAG_MU[step_index]) / _LAG_SIGMA) ** 2)
    w /= w.sum()
    if slot6_lag0_mass > 0.0 and step_index == _SLOTS.size - 1:
        w[0] = 0.0
        w *= (1.0 - slot6_lag0_mass) / w.sum()
        w[0] = slot6_lag0_mass                 # the adversarial early-mass spike
    return w


def _build_world(root: Path, *, flip_wind: bool = False,
                 psi_equals_endpoint: bool = False,
                 omega_tracks_nparcels: bool = False,
                 all_nan_kernel: bool = False,
                 drop_mrms: bool = False,
                 slot6_lag0_mass: float = 0.0,
                 with_daily: bool = True) -> dict:
    """Fabricate one complete QA input world under ``root``.

    Returns {"features_dir", "fcst_dir", "daily_dir", "out_dir"}; the keyword
    toggles produce the adversarial variants described in the module
    docstring, each breaking exactly one check's constructed expectation.
    """
    rng = np.random.default_rng(7)
    features_dir = root / "features"
    fcst_dir = root / "fcst"
    daily_dir = root / "daily"
    out_dir = root / "qa_out"
    features_dir.mkdir(parents=True)
    fcst_dir.mkdir()

    slot_b = _TIME.astype(float)[None, :, None, None]      # 0..6; slot 0 NaN'd below
    coslat_b = np.cos(np.radians(_LAT))[None, None, :, None]

    # --- winds and the exactly-anti-parallel upwind displacement (check 2)
    u = 5.0 + rng.normal(0.0, 0.5, _SHAPE)                 # m/s, eastward
    v = 2.0 + rng.normal(0.0, 0.3, _SHAPE)                 # m/s, northward
    mean_lag = np.broadcast_to(1.0 + 0.6 * slot_b, _SHAPE).copy()  # h; == lag centroid
    sign = 1.0 if flip_wind else -1.0                      # -1: truly upwind
    upwind_dlat = sign * (v * 3.6 * mean_lag) / 111.0      # deg; 3.6: m/s -> km/h
    upwind_dlon = sign * (u * 3.6 * mean_lag) / (111.0 * coslat_b)
    upwind_km = np.hypot(u, v) * 3.6 * mean_lag

    # --- kernel soil-moisture features (checks 4b, 6)
    s_endpoint_anom = rng.normal(0.0, 0.05, _SHAPE)        # m3/m3 anomaly
    kernel_noise = rng.normal(0.0, 1.0, _SHAPE) * 0.012 * slot_b
    psi_anom = (s_endpoint_anom if psi_equals_endpoint
                else s_endpoint_anom + kernel_noise)
    s_endpoint_raw = 0.20 + s_endpoint_anom
    psi_raw = 0.20 + psi_anom

    # --- sampling density, deliberately independent of every feature (check 6);
    # the 1..220 span populates every containment bin (1-4 .. 50+), so the
    # variance-step-across-the-seam gate actually has both sides to compare
    n_parcels = rng.integers(1, 220, _SHAPE).astype(float)
    omega = (1200.0 + 8.0 * n_parcels + rng.normal(0.0, 20.0, _SHAPE)
             if omega_tracks_nparcels
             else 2000.0 + rng.normal(0.0, 150.0, _SHAPE))  # check 4a band

    # --- the trajectory-free gate and the rain it should anticipate (check 7).
    # A small NEGATIVE tail (~3%: PBL top occasionally above the LFC) keeps the
    # sign census alive -- an identically-zero negative fraction is a dead
    # field.  Event rates decay smoothly with the gap so decile monotonicity
    # is strict rather than saturated.
    gamma_gap = rng.uniform(-100.0, 3000.0, _SHAPE)         # m; LFC - PBLH
    rain_max = (6.0 * np.exp(-gamma_gap / 800.0)
                * rng.uniform(0.0, 2.0, _SHAPE))            # mm/h; smaller gap -> more rain
    rain_av = rain_max / 4.0
    # the Taylor et al. (2012) sign: rain falls over LOCALLY DRY soil, so the
    # mesoscale anomaly is slightly depressed where the cell's own slot rains
    # (precip-minus-no-precip mean comes out negative by construction)
    psi_meso_anom = (psi_anom + rng.normal(0.0, 0.01, _SHAPE)
                     - 0.02 * (rain_max > 1.0))

    phi = rng.uniform(0.2, 0.8, _SHAPE)
    m_star = rng.uniform(0.3, 0.7, _SHAPE)
    coverage = rng.uniform(0.7, 1.0, _SHAPE)
    containment_applied = (rng.uniform(size=_SHAPE) < 0.05).astype(float)
    # land closure: 1.0 everywhere the kernel exists; the border ring is NaN
    # (never-populated coastal cells), so "interior" is well defined
    psi_land = np.ones(_SHAPE)
    psi_land[:, :, [0, -1], :] = np.nan
    psi_land[:, :, :, [0, -1]] = np.nan

    kernel_fields = {
        "psi_anom": psi_anom, "omega": omega, "phi": phi, "m_star": m_star,
        "psi_raw": psi_raw, "coverage": coverage, "n_parcels": n_parcels,
        "containment_applied": containment_applied,
        "s_endpoint_anom": s_endpoint_anom, "s_endpoint_raw": s_endpoint_raw,
        "psi_meso_anom": psi_meso_anom,
        "upwind_dlat": upwind_dlat, "upwind_dlon": upwind_dlon,
        "upwind_km": upwind_km, "mean_lag_hours": mean_lag,
        "psi_land": psi_land,
    }
    for arr in kernel_fields.values():                      # slot 0 = overpass:
        arr[:, 0] = np.nan                                  # no arrival window
    if all_nan_kernel:                                      # the 2016 situation
        kernel_fields = {k: np.full(_SHAPE, np.nan) for k in kernel_fields}

    coords = {"date": _DATES, "time": _TIME, "lat": _LAT, "lon": _LON}
    dims = ("date", "time", "lat", "lon")

    companion_vars = {f"UPW_{k}": (dims, arr.astype(np.float32))
                      for k, arr in kernel_fields.items()}
    # the trajectory-free merge stream rides along (check 7 reads gamma_gap);
    # it stays finite even in the all-NaN-KERNEL world -- only the kernel
    # features are 2016-honest-NaN, and only checks 2-6 are asserted there
    companion_vars["UPW_gamma_gap_mml"] = (dims, gamma_gap.astype(np.float32))
    companion_vars["UPW_gamma_gap_mu"] = (
        dims, (gamma_gap + 1000.0).astype(np.float32))
    companion_vars["UPW_pblh"] = (dims, np.full(_SHAPE, _PBLH_M, np.float32))
    companion_vars["UPW_pblh_anom"] = (
        dims, rng.normal(0.0, 100.0, _SHAPE).astype(np.float32))
    companion = xr.Dataset(companion_vars, coords=coords)
    companion.attrs.update({
        "nan_policy": "NaN, never fabricated",
        "daily_files_found": 3 if with_daily else 0,
        "n_dates_with_daily_file": 3 if with_daily else 0,
    })
    companion.to_netcdf(features_dir / "UPWIND_FEATURES_2019.nc")

    # --- the match-up file the QA cross-references
    fcst_vars = {
        "FCST_u": (dims, u.astype(np.float32)),
        "FCST_v": (dims, v.astype(np.float32)),
        "FCST_MML_LFC": (dims, (gamma_gap + _PBLH_M).astype(np.float32)),
    }
    if not drop_mrms:
        # real FCST_SMAP_MRMS files carry the MRMS variables on a dim named
        # `nhours` (slot-aligned with `time`, degenerate all-zero coord) --
        # mirror that exactly; a `time`-dimmed MRMS fixture masked the
        # 2026-08-21 production crash in _slot_field
        mrms_dims = ("date", "nhours", "lat", "lon")
        fcst_vars["MRMS_GaugeCorrQPE01H_av"] = (mrms_dims, rain_av.astype(np.float32))
        fcst_vars["MRMS_GaugeCorrQPE01H_max"] = (mrms_dims, rain_max.astype(np.float32))
        coords = {**coords, "nhours": ("nhours", np.zeros(len(coords["time"][1])
                  if isinstance(coords["time"], tuple) else len(coords["time"])))}
    xr.Dataset(fcst_vars, coords=coords).to_netcdf(
        fcst_dir / "FCST_SMAP_MRMS_2019.nc")

    # --- three daily files carrying lag_weight (never merged: wrong dims)
    if with_daily:
        daily_dir.mkdir()
        lag_w = np.empty((_SLOTS.size, _LAT.size, _LON.size, _LAGS.size))
        for si in range(_SLOTS.size):
            lag_w[si] = _lag_profile(si, slot6_lag0_mass)[None, None, :]
        for di in range(3):
            stamp = str(_DATES[di].astype("datetime64[D]")).replace("-", "")
            day_vars = {
                k: (("arrival_step", "target_lat", "target_lon"),
                    arr[di, 1:].astype(np.float32))         # kernel slots 1..6
                for k, arr in kernel_fields.items()}
            day_vars["lag_weight"] = (
                ("arrival_step", "target_lat", "target_lon", "lag"),
                lag_w.astype(np.float32))
            xr.Dataset(day_vars, coords={
                "arrival_step": _SLOTS, "target_lat": _LAT,
                "target_lon": _LON, "lag": _LAGS,
            }).to_netcdf(daily_dir / f"UPW_{stamp}.nc")

    return {"features_dir": features_dir, "fcst_dir": fcst_dir,
            "daily_dir": daily_dir, "out_dir": out_dir}


# --------------------------------------------------------------------------- #
# driving the battery and reading its report
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qa_mod():
    if not QA_SCRIPT.exists():
        pytest.skip("scripts/upwind_qa.py not present yet (written in parallel)")
    spec = importlib.util.spec_from_file_location("upwind_qa", QA_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["upwind_qa"] = module        # dataclasses resolve via sys.modules
    spec.loader.exec_module(module)
    return module


def _run_qa(qa_mod, world: dict, daily_dir=None) -> list:
    """main(argv) on a fabricated world; return the parsed qa_report.json.

    A nonzero SystemExit is tolerated (a FAIL-verdict exit-code convention is
    legitimate) -- the report file itself is the contract under test.
    """
    daily = world["daily_dir"] if daily_dir is None else daily_dir
    argv = ["--years", "2019",
            "--companion-dir", str(world["features_dir"]),
            "--fcst-dir", str(world["fcst_dir"]),
            "--daily-dir", str(daily),
            "--out-dir", str(world["out_dir"])]
    try:
        qa_mod.main(argv)
    except SystemExit:
        pass
    report_path = world["out_dir"] / "qa_report.json"
    assert report_path.exists(), "main() must always write qa_report.json"
    report = json.loads(report_path.read_text())
    # the pinned contract is the LIST of verdicts; a provenance wrapper dict
    # holding that list under "checks" is accepted (it strictly adds metadata)
    if isinstance(report, dict):
        report = report["checks"]
    assert isinstance(report, list)
    return report


#: fallback physics keywords per check, used only when no verdict name carries
#: an identifiable "check <n>" tag (the implementation is written in parallel)
_KEYWORDS = {
    1: ("land", "closure", "presence", "schema", "inventory"),
    2: ("align", "direction", "wind", "anti"),
    3: ("lag",),
    4: ("magnitude", "realism", "endpoint", "insolation"),
    5: ("event", "case", "mrms", "qpe"),
    6: ("leakage", "sample", "parcel", "density"),
    7: ("label", "monoton", "gamma", "lfc"),
}


def _verdicts_for(report: list, num: int) -> list:
    """Every verdict belonging to check ``num`` (4 may split into 4a/4b)."""
    tagged = re.compile(rf"check[\s_\-]*0?{num}(?!\d)", re.IGNORECASE)
    hits = [v for v in report if tagged.search(str(v.get("name", "")))]
    if not hits:
        kws = _KEYWORDS[num]
        hits = [v for v in report
                if any(k in str(v.get("name", "")).lower() for k in kws)]
    if not hits:
        bare = re.compile(rf"(?<!\d)0?{num}(?!\d)")
        hits = [v for v in report if bare.search(str(v.get("name", "")))]
    assert hits, (f"no verdict identifiable as check {num}; names: "
                  f"{[v.get('name') for v in report]}")
    return hits


def _statuses(report: list, num: int) -> set:
    return {v["status"] for v in _verdicts_for(report, num)}


@pytest.fixture(scope="module")
def base_world(tmp_path_factory):
    return _build_world(tmp_path_factory.mktemp("qa_base"))


@pytest.fixture(scope="module")
def base_report(qa_mod, base_world):
    return _run_qa(qa_mod, base_world)


# =========================================================================== #
# 1. end to end: report contract + the constructed world's PASS verdicts
# =========================================================================== #
def test_report_contract_and_all_checks_present(base_report):
    """Every verdict carries the five pinned fields with a legal status, and
    all seven checks are represented (check 4 may appear as 4a/4b)."""
    assert len(base_report) >= 7
    for v in base_report:
        assert set(v) >= {"name", "status", "metrics", "detail", "figures"}, v
        assert v["status"] in VALID_STATUSES, v
        assert isinstance(v["name"], str) and v["name"]
        assert isinstance(v["metrics"], dict)
        assert isinstance(v["detail"], str)
        assert isinstance(v["figures"], list)
    for num in range(1, 8):
        _verdicts_for(base_report, num)          # asserts internally


def test_constructed_world_passes(base_report):
    """The world was BUILT to satisfy checks 2, 3, 4 (a and b), 6 and 7:
    anti-parallel displacement, growing lag centroid with ~zero slot-6 lag-0
    mass, in-band omega + slot-decaying r(psi, endpoint), n_parcels-independent
    features, and rain that increases as gamma_gap shrinks."""
    for num in (2, 3, 4, 6, 7):
        assert _statuses(base_report, num) == {"PASS"}, (
            num, _verdicts_for(base_report, num))


def test_check2_reports_the_anti_parallel_cosine(base_report):
    """If check 2 exposes a cosine metric, it must sit at ~-1: the displacement
    was written exactly anti-parallel to FCST_u/v in local-km space."""
    metrics = {}
    for v in _verdicts_for(base_report, 2):
        metrics.update(v["metrics"])
    cosines = [val for key, val in metrics.items()
               if "cos" in key.lower() and isinstance(val, (int, float))]
    if cosines:                                   # metric names are theirs to pick
        assert min(cosines) < -0.9


# =========================================================================== #
# 2. adversarial constructions: each breaks exactly one check
# =========================================================================== #
def test_flipped_wind_fails_check2(qa_mod, tmp_path):
    """Negating the displacement makes parcels appear to come from DOWNWIND
    (median cosine +1) -- physically impossible for a back-trajectory."""
    report = _run_qa(qa_mod, _build_world(tmp_path, flip_wind=True))
    assert "FAIL" in _statuses(report, 2)


def test_psi_equals_endpoint_fails_check4(qa_mod, tmp_path):
    """psi_anom == s_endpoint_anom exactly (r = 1 in every slot): the kernel
    convolution adds nothing beyond the endpoint value, and the verdict must
    say so."""
    report = _run_qa(qa_mod, _build_world(tmp_path, psi_equals_endpoint=True))
    failed = [v for v in _verdicts_for(report, 4) if v["status"] == "FAIL"]
    assert failed
    assert any("nothing" in v["detail"].lower() for v in failed), failed


def test_slot6_lag0_mass_flags_check3(qa_mod, tmp_path):
    """30 % of slot-6 lag mass at lag 0 contradicts the growing-history
    physics (a 6-hours-later arrival dominated by its own arrival hour)."""
    report = _run_qa(qa_mod, _build_world(tmp_path, slot6_lag0_mass=0.3))
    assert _statuses(report, 3) & {"WARN", "FAIL"}


def test_omega_tracking_nparcels_fails_check6(qa_mod, tmp_path):
    """omega = 1200 + 8 * n_parcels is a sampling-density artefact in feature
    clothing; it must break the |r| < 0.1 independence gate."""
    report = _run_qa(qa_mod, _build_world(tmp_path, omega_tracks_nparcels=True))
    assert "FAIL" in _statuses(report, 6)


# =========================================================================== #
# 3. SKIP paths: absent inputs are named, never silently passed
# =========================================================================== #
def test_missing_daily_dir_skips_check3(qa_mod, tmp_path):
    """lag_weight lives ONLY in the daily files (wrong dims for the merge);
    no daily dir means check 3 has nothing to test -> SKIP, not PASS/FAIL."""
    world = _build_world(tmp_path, with_daily=False)
    report = _run_qa(qa_mod, world, daily_dir=tmp_path / "no_such_daily_dir")
    assert _statuses(report, 3) == {"SKIP"}


def test_missing_mrms_skips_checks_5_and_7(qa_mod, tmp_path):
    """Without MRMS_GaugeCorrQPE01H_{av,max} the rain-referenced checks (5, 7)
    must SKIP; the trajectory checks are untouched (2 still PASSes)."""
    report = _run_qa(qa_mod, _build_world(tmp_path, drop_mrms=True))
    assert _statuses(report, 5) == {"SKIP"}
    assert _statuses(report, 7) == {"SKIP"}
    assert _statuses(report, 2) == {"PASS"}


def test_all_nan_kernel_year_skips_checks_2_to_6(qa_mod, tmp_path):
    """An all-NaN kernel-feature year (2016: no assessed PBLH, no kernel
    features -- honest NaN by the pipeline's nan_policy) must SKIP every
    kernel-referenced check rather than fail it, and say why."""
    world = _build_world(tmp_path, all_nan_kernel=True, with_daily=False)
    report = _run_qa(qa_mod, world, daily_dir=tmp_path / "no_such_daily_dir")
    for num in (2, 3, 4, 5, 6):
        verdicts = _verdicts_for(report, num)
        assert {v["status"] for v in verdicts} == {"SKIP"}, (num, verdicts)
        assert all(v["detail"] for v in verdicts)           # a stated reason


# =========================================================================== #
# 4. figures: real, non-empty PNGs in the out dir
# =========================================================================== #
def test_figures_are_nonempty_pngs_in_out_dir(base_report, base_world):
    out_dir = base_world["out_dir"]
    listed = [f for v in base_report for f in v["figures"]]
    assert listed, "the battery must produce at least one figure"
    for name in listed:
        path = Path(name)
        if not path.is_absolute():
            path = out_dir / path
        assert path.suffix == ".png", name
        assert path.exists() and path.stat().st_size > 0, name
