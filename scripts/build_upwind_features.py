"""Per-day cluster job: HYSPLIT trajectories -> upwind soil-moisture features.

One day per invocation (slurm array task), implementing the production plan of
src/trajectory_kernels/UPWIND_INDEX_REVIEW.md section 4.2:

    day     = trajectories.load_day_dir(<traj-root>/.../<date>)
    pbl     = GriddedPBL(date)              assessed 1-deg 3-hourly PBLH, with
                                            climatology + analytic fallback
    kernels = footprint.build_all(day, pbl_model=pbl,
                                  energy_fn=ClearSkyAvailableEnergy())
    sm_anom = SMAP_L4 surface SM (nearest analysis slot) - monthly baseline
    feats   = predictors.build_features(kernels, sm_anom, sm_raw, pbl)
    write UPW_<YYYYMMDD>.nc (~kB-MB); DISCARD the kernels

The dense kernels are ~0.6 GB per variable per day (~1 TB over the record) and
are never persisted -- the per-day feature table is the product (review F1).
A later merge step concatenates the daily files onto the FCST_SMAP_MRMS
(date, time, lat, lon) axes; days with no trajectory archive are recorded
there as NaN, so this script treats an absent day directory as an expected
gap (exit 0, no output), never an error.

SMAP slot choice: SMAP_L4_smsfc_av carries 5 analysis slots per day (16.5,
19.5, 22.5, 25.5, 28.5 hours after the date's 00 UTC). Surface soil moisture
decorrelates over days, not hours, so the single slot nearest the day's mean
kernel arrival time is used rather than interpolating between slots -- the
sub-daily difference is noise next to the day-to-day signal the anomaly
carries (HANDOFF 7.8 / review 1.7).

Energy A/B (--uniform-energy): rebuilds the kernels with UniformEnergy so the
Psi fields can be compared against the clear-sky-weighted production run
(review section 4.3). Under a uniform weight the footprint sums are contact
hours, not J/m2, so phi/omega/m_star are deliberately omitted from the output
(predictors.phi would refuse them anyway) and the file is tagged
UPW_<YYYYMMDD>_uniform_energy.nc so it can never shadow a production file.

Usage (dev smoke test):
  JPL_AIRS_DATA=/mnt/d/JPL_AIRS/data PYTHONPATH=src \\
    python scripts/build_upwind_features.py --date 2019-06-05 \\
    --traj-root /mnt/d/JPL_AIRS/data/HYSPLIT_demo --out-dir /tmp/upw
Cluster: one date per slurm array task; all paths env-injected via
JPL_AIRS_DATA / JPL_AIRS_RESULTS or the explicit flags below.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date as Date
from pathlib import Path

import numpy as np
import xarray as xr

from trajectory_kernels import config, footprint, land, predictors, trajectories
from trajectory_kernels.insolation import ClearSkyAvailableEnergy, UniformEnergy
from trajectory_kernels.pbl import GriddedPBL, PBLModel


def default_lsm_path() -> Path:
    """The land-sea mask: ``config.LSM_PATH`` when present, else the dev copy.

    The canonical location is ``<data root>/lsm.nc``; the dev data root on D:
    carries the identical global 1-deg field as
    ``masks/land_surface_mask.nc`` (variable ``lsm`` either way).
    """
    if config.LSM_PATH.exists():
        return config.LSM_PATH
    return config.DATA_DIR / "masks" / "land_surface_mask.nc"


class TalliedPBL(PBLModel):
    """Wrap a GriddedPBL to accumulate its per-call source fractions.

    ``GriddedPBL.last_source_fractions`` reports only the most recent
    ``depth()`` call; the kernel builder makes thousands of calls of varying
    size, so the honest per-day QA number (review 4.3) is the point-weighted
    tally over all of them, which this wrapper keeps.
    """

    def __init__(self, inner: GriddedPBL):
        self.inner = inner
        self._points = {"assessed": 0.0, "climatology": 0.0, "analytic": 0.0}
        self._n = 0

    def depth(self, lat, lon, time_utc) -> np.ndarray:
        out = self.inner.depth(lat, lon, time_utc)
        n = int(np.size(out))
        for layer, frac in self.inner.last_source_fractions.items():
            self._points[layer] += frac * n
        self._n += n
        return out

    @property
    def source_fractions(self) -> dict[str, float]:
        n = max(self._n, 1)
        return {layer: pts / n for layer, pts in self._points.items()}

    def __repr__(self) -> str:
        return f"GriddedPBL(available={self.inner.available})"


def resolve_day_dir(traj_root: Path, yyyymmdd: str) -> Path | None:
    """First existing per-day directory under the archive root, else None.

    ONLY the confirmed cluster archive layout is accepted (Zach, 2026-08-19,
    closing review open item 1): year-nested
    ``<root>/YYYY/wrf27km_<YYYYMMDD>``, with the doubled-day variant
    ``.../wrf27km_<YYYYMMDD>/wrf27km_<YYYYMMDD>`` (as the dev D: copy has)
    preferred when present. Anything else is a miss -- guessing at alternative
    layouts risks silently loading the wrong day. To run against a
    differently-arranged tree (e.g. the dev HYSPLIT_demo copy), symlink it
    into shape: ``<root>/2019/wrf27km_20190605 -> .../wrf27km_20190605``.
    """
    year = yyyymmdd[:4]
    candidates = (
        traj_root / year / f"wrf27km_{yyyymmdd}" / f"wrf27km_{yyyymmdd}",
        traj_root / year / f"wrf27km_{yyyymmdd}",
    )
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def select_smap_slot(fcst_dir: Path, day: Date, arrival_times: np.ndarray,
                     baseline_path: Path) -> tuple[xr.DataArray, xr.DataArray, float]:
    """The day's SMAP_L4 surface SM at the analysis slot nearest mean arrival.

    Returns ``(sm_raw, sm_anom, slot_nhours)`` with both fields on (lat, lon)
    in m3/m3. ``slot_nhours`` is the chosen L4 analysis slot, in hours after
    the date's 00 UTC (the file's ``L4_nhours`` convention; arrivals past
    midnight land at 24+). Nearest slot, never interpolated -- see module
    docstring. The anomaly reference is the per-cell monthly baseline
    (review 1.7); a month the L4 record does not cover (Dec-Feb) yields an
    all-NaN anomaly, which downstream code reports as honest gaps.
    """
    path = fcst_dir / f"FCST_SMAP_MRMS_{day.year}.nc"
    if not path.exists():
        raise SystemExit(f"SMAP input missing: {path} (is --fcst-dir/JPL_AIRS_DATA right?)")

    midnight = np.datetime64(day.isoformat())
    arrival_nhours = (arrival_times - midnight) / np.timedelta64(1, "h")
    mean_nhours = float(arrival_nhours.mean())

    with xr.open_dataset(path) as ds:
        slots = ds["L4_nhours"].values.astype(float)
        slot = float(slots[np.argmin(np.abs(slots - mean_nhours))])
        sm_raw = (ds["SMAP_L4_smsfc_av"]
                  .sel(date=midnight, L4_nhours=slot)
                  .load()
                  .drop_vars(("date", "L4_nhours"), errors="ignore"))

    with xr.open_dataset(baseline_path) as base:
        reference = (base["sm_baseline"].sel(month=day.month).load()
                     .drop_vars("month", errors="ignore"))

    sm_anom = (sm_raw - reference).rename("sm_anom")
    sm_anom.attrs.update({"units": "m3 m-3",
                          "long_name": "SMAP L4 surface SM minus per-cell monthly baseline"})
    return sm_raw.rename("sm_raw"), sm_anom, slot


def atomic_to_netcdf(ds: xr.Dataset, out_path: Path, encoding: dict | None = None) -> None:
    """Write ``ds`` to ``out_path`` atomically: tmp file in the same dir, then rename.

    A slurm preemption (or node death) mid-``to_netcdf`` leaves a truncated
    file at the final path, which the skip-if-exists resume check would then
    treat as complete forever. Writing to ``<name>.nc.tmp`` and renaming with
    ``os.replace`` (atomic on POSIX within one filesystem, which "same
    directory" guarantees) makes the final path appear only once the file is
    whole; a leftover ``.tmp`` from a killed job is inert and cleaned up on
    the next attempt.
    """
    tmp = out_path.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)  # survives only if the write/rename failed


def git_describe() -> str:
    """Repo version stamp for provenance; 'unknown' if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=config.REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True, help="day to process, YYYY-MM-DD")
    p.add_argument("--traj-root", type=Path, required=True,
                   help="directory containing per-day HYSPLIT subdirectories")
    p.add_argument("--out-dir", type=Path,
                   default=config.RESULTS_DIR / "upwind_features" / "daily",
                   help="destination for the per-day UPW_<YYYYMMDD>.nc files")
    p.add_argument("--baseline", type=Path, default=config.SMAP_BASELINE_PATH,
                   help="per-cell monthly SMAP_L4 baseline (build_smap_l4_baseline.py)")
    p.add_argument("--pblh-3hrly", type=Path, default=config.PBLH_3HRLY_PATH,
                   help="assessed 1-deg 3-hourly PBLH (build_pblh_3hrly_1deg.py)")
    p.add_argument("--pblh-clim", type=Path, default=config.PBLH_CLIM_PATH,
                   help="monthly-diurnal PBLH climatology fallback")
    p.add_argument("--fcst-dir", type=Path, default=config.FCST_TABLE_DIR,
                   help="directory of FCST_SMAP_MRMS_<year>.nc match-up files")
    p.add_argument("--lsm", type=Path, default=default_lsm_path(),
                   help="global 1-deg land-sea mask (variable 'lsm')")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if the output file already exists")
    p.add_argument("--allow-pblh-fallback", action="store_true",
                   help="proceed even when the assessed 3-hourly PBLH file is "
                        "absent (climatology/analytic fallback only) -- the "
                        "resulting m_star/omega are information-free (review "
                        "F2), so this is for diagnostics, never production; "
                        "the output is stamped pblh_fallback='allowed'")
    p.add_argument("--rainout-discount", action="store_true",
                   help="discount kernel weights upstream of condensation events")
    p.add_argument("--uniform-energy", action="store_true",
                   help="A/B diagnostic: uniform hour weights instead of clear-sky "
                        "available energy; output tagged _uniform_energy, no phi/omega")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    day = Date.fromisoformat(args.date)
    yyyymmdd = day.strftime("%Y%m%d")
    tag = "_uniform_energy" if args.uniform_energy else ""
    out_path = args.out_dir / f"UPW_{yyyymmdd}{tag}.nc"

    if out_path.exists() and not args.force:
        print(f"exists, skipping (use --force to rebuild): {out_path}")
        return 0

    if not args.traj_root.is_dir():
        raise SystemExit(f"--traj-root does not exist: {args.traj_root} "
                         "(a missing root would silently skip every day)")
    day_dir = resolve_day_dir(args.traj_root, yyyymmdd)
    if day_dir is None:
        print(f"no trajectories for {args.date} under {args.traj_root}; "
              "skipping (merge step records this day as NaN)")
        return 0

    # Preflight: fail on bad inputs NOW, before the expensive kernel build.
    # (select_smap_slot would catch the SMAP file too, but only after the
    # ~hour-long footprint sweep has already been paid for.)
    fcst_path = args.fcst_dir / f"FCST_SMAP_MRMS_{day.year}.nc"
    if not fcst_path.exists():
        raise SystemExit(f"SMAP match-up file missing: {fcst_path} "
                         "(is --fcst-dir/JPL_AIRS_DATA right?); refusing to "
                         "start the kernel build only to fail at the SMAP step")
    if not args.baseline.exists():
        raise SystemExit(f"SMAP monthly baseline missing: {args.baseline} "
                         "(build it with scripts/build_smap_l4_baseline.py, or "
                         "point --baseline at it); refusing to start the "
                         "kernel build only to fail at the anomaly step")

    pbl = TalliedPBL(GriddedPBL(three_hourly_path=args.pblh_3hrly,
                                clim_path=args.pblh_clim, date=args.date))
    if not pbl.inner.available["assessed"]:
        # Without the assessed layer the PBL model degrades to the analytic
        # climatology, producing exactly the information-free m_star/omega
        # that review F2 forbids -- while the output would otherwise look
        # healthy. Refuse unless the caller explicitly opts in.
        msg = (f"assessed 3-hourly PBLH unavailable at {args.pblh_3hrly}: "
               "the kernel build would silently degrade to the analytic "
               "climatological PBL (information-free m_star/omega, review F2). "
               "Build it with scripts/build_pblh_3hrly_1deg.py (cluster: "
               "slurm/upwind_pblh_3hrly.sbatch) or fix --pblh-3hrly; pass "
               "--allow-pblh-fallback only for diagnostics.")
        if not args.allow_pblh_fallback:
            raise SystemExit(msg)
        print(f"WARNING: {msg} Proceeding under --allow-pblh-fallback.")

    day_ds = trajectories.load_day_dir(day_dir)
    energy_fn = UniformEnergy() if args.uniform_energy else ClearSkyAvailableEnergy()
    land_fn = land.make_land_lookup(args.lsm)

    kernels = footprint.build_all(day_ds, pbl_model=pbl, energy_fn=energy_fn,
                                  land_fn=land_fn,
                                  rainout_discount=args.rainout_discount)

    arrival_times = np.asarray(kernels.attrs["arrival_times_utc"], dtype="datetime64[s]")
    sm_raw, sm_anom, slot = select_smap_slot(
        args.fcst_dir, day, arrival_times, args.baseline)

    # Under a uniform weight the footprint sums are contact hours, so the
    # energy-denominated phi/omega/m_star tier is withheld (pbl_model=None);
    # the contact gate above still used the full assessed PBL model.
    feats = predictors.build_features(
        kernels, sm_anom, sm_raw=sm_raw,
        pbl_model=None if args.uniform_energy else pbl)
    del kernels  # ~0.6 GB/variable; the feature table is the product

    fractions = pbl.source_fractions
    feats.attrs.update({
        "date": day.isoformat(),
        "traj_day_dir": str(day_dir),
        "n_parcels_loaded": int(day_ds.sizes["parcel"]),
        "smap_slot_nhours": slot,
        "smap_slot_choice": ("analysis slot nearest the mean kernel arrival "
                             "time; nearest not interpolated, SM decorrelates "
                             "over days (HANDOFF 7.8)"),
        "pblh_source_fractions": "; ".join(
            f"{layer}={frac:.3f}" for layer, frac in fractions.items()),
        "rainout_discount": int(args.rainout_discount),
        "git_describe": git_describe(),
        "command_line": " ".join(sys.argv),
    })
    if args.allow_pblh_fallback:
        feats.attrs["pblh_fallback"] = "allowed"

    # The F2 guard above and this provenance record must never disagree: the
    # tallied assessed fraction is the QA number a reader checks to know the
    # guard actually held, so its absence is a bug, not a soft gap.
    assert feats.attrs["pblh_source_fractions"].startswith("assessed="), (
        "pblh_source_fractions attr must record the tallied assessed fraction "
        f"first; got {feats.attrs['pblh_source_fractions']!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in feats.data_vars}
    atomic_to_netcdf(feats, out_path, encoding=encoding)

    populated = int((feats["n_parcels"] > 0).sum())
    finite = int(np.isfinite(feats["psi_anom"].values).sum())
    cov = feats["coverage"].values
    cov = cov[np.isfinite(cov)]
    cov_note = (f"coverage median {np.median(cov):.2f} min {cov.min():.2f}"
                if cov.size else "coverage all-NaN")
    print(f"wrote {out_path}: {populated} populated receptors, "
          f"{finite} finite psi_anom, {cov_note}, "
          f"pblh {feats.attrs['pblh_source_fractions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
