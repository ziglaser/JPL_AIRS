"""Build the kriged gap-filled surface caches (the separate kriged dataset).

Two cache flavors, both fully gap-filled fields in PHYSICAL units so they
z-score with the SAME frozen stats as the reanalysis (dataset.kriged_year_arrays):

* ``degraded_reanalysis`` (stage B'): clean MERRA-2 reanalysis steps with
  AIRS-shaped gaps punched into ``config.KRIGED_CHANNELS`` and re-filled by
  ordinary kriging of the surviving pixels; SLP/U10M/V10M stay clean.  The
  gap shape for a (date, hour) comes from the real fullgrid file when one
  exists, else from a season-matched draw of the harvested real-gap bank
  (front_finder.mask_bank).
* ``airs_fcst`` (stage C): real AIRS-FCST surface fields
  (dl_front.airs_fcst.period_fields), ALL four AIRS-derived channels
  kriged over the full label grid; SLP -- which AIRS does not retrieve --
  is copied clean from the reanalysis step at the period timestamp.

Cache schema v3 (user decision 2026-08-13): ``kriged_sfc_{year}.nc`` under
the per-flavor ``config.KRIGED_SOURCE_DIRS`` directory (manifest reorg
2026-08-13: ``front_id/degraded_reanalysis`` and ``front_id/
kriged_airs_fcst``), with dims (time, lat, lon) on the label grid, data vars:

* config.SFC_VARS float32, defined (no NaN) ONLY inside the CROP domain
  -- ``dl_front.dataset.crop_domain()``, the analysis box (lat 32..53 N,
  lon -107..-64 E) expanded by the derived receptive-field halo
  (``dataset.halo_px()``, 8 px at default config) -- and NaN everywhere
  outside it.  The halo is filled from ALL real observations that fall in
  it (the fullgrid archive reaches 25.5 N, so the southern halo has real
  obs); reanalysis is NOT substituted into the halo -- it is kriged
  AIRS-informed data, accepted drift and all, and the halo is minimal
  (exactly the receptive-field radius) to limit that drift.  Beyond
  box + halo nothing can influence an in-box prediction, so nothing is
  filled.  This applies to the NON-kriged clean channels too (SLP/U10M/
  V10M in the degraded flavor, SLP in airs): schema v3 defines nothing
  out-of-crop.  (Schema v2 used the codsus region mask as the fill
  domain; the user judged it 'far too large'.)
* ``valid_frac`` float32 in [0, 1], the PRE-kriging AIRS availability
  on the FULL grid (semantics unchanged from v1).
* ``gap_type`` int8, per config.GAP_*: -1 out-of-crop, 0 observed,
  1 cloud/retrieval gap inside the expected swath, 2 out-of-swath
  (dl_front.swath.gap_type_for; the climatological swath bank when
  available, else the per-day morphological envelope).  EXCEPTION
  (review 2026-08-13): steps whose availability mask is a gap-BANK draw
  (no readable fullgrid file) are classified against the drawn mask's
  OWN morphological envelope -- the donor date sits at a different
  16-day cycle position, so the requested date's climatological
  footprint would scramble cloud vs out-of-swath.  The per-step log
  note records every 'bank mask used' step.

Global attrs: source/variogram_model/max_obs_points/created plus
``schema_version=3``, ``domain_lat_range``/``domain_lon_range``/
``land_fraction_min``/``halo_px`` (the resolved domain decision), and
``swath_bank`` ('<bank path>' or 'per-day-envelope').  Steps belong to
the file of their own calendar year (a 00 UTC step on Jan 1 goes in the
new year).

Determinism: every random choice (obs subsampling, gap-bank draws) uses a
generator seeded from (config.KRIGE_SEED, date, hour[, channel]), so
reruns and worker counts cannot change the output.

Incremental & resumable: years build one at a time and each year's cache is
written the moment the year completes (even a ZERO-step year gets an empty
file, so a sparse archive cannot leave phantom missing caches).  A rerun
skips years whose cache file exists; ``--force`` rebuilds them.

CLI::

    python -m dl_front.krige_fill build-degraded --years 2007-2015 \
        [--hours 18,21,0] [--workers N] [--out-dir DIR] [--force] \
        [--allow-small-bank]
    python -m dl_front.krige_fill build-airs --years 2007-2021 ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import airs_fcst, config, swath
from .acquire_merra2_sfc import day_path

#: build flavor name -> ``--source`` value (the config.KRIGED_SOURCE_DIRS key)
FLAVORS = {"degraded_reanalysis": "kriged-degraded", "airs_fcst": "kriged-airs"}


# --------------------------------------------------------------------------- #
# Kriging one field
# --------------------------------------------------------------------------- #

def krige_fill(field: np.ndarray, lats=None, lons=None,
               rng: np.random.Generator | None = None,
               target: np.ndarray | None = None,
               variogram: str | None = None) -> np.ndarray:
    """Fill the NaNs of one (68, 141) field by ordinary kriging.

    Observed (finite) pixels are returned bit-identical; only ``target``
    pixels (default: every NaN) are predicted -- NaNs outside the target
    stay NaN, which is how the builders leave everything outside the
    analysis domain unfilled.  PyKrige OrdinaryKriging with its default
    ('standard') settings, variogram model from config unless ``variogram``
    overrides it (the validation study sweeps models); when more than
    ``config.KRIGE_MAX_OBS`` pixels are observed the fit uses a seeded
    random subsample (an exact solve is O(n^3) in the obs count).
    Degenerate inputs: < 10 observed points -> fill target with the
    observed mean; 0 observed points -> field returned untouched (caller
    handles).
    """
    from pykrige.ok import OrdinaryKriging

    lats = np.asarray(config.LABEL_LATS if lats is None else lats, float)
    lons = np.asarray(config.LABEL_LONS if lons is None else lons, float)
    out = np.array(field, dtype=np.float64)
    observed = np.isfinite(out)
    target = (~observed if target is None
              else np.asarray(target, bool) & ~observed)
    n_obs = int(observed.sum())
    if not target.any() or n_obs == 0:
        return out
    if n_obs < 10:
        out[target] = out[observed].mean()
        return out

    lon2d, lat2d = np.meshgrid(lons, lats)
    ox, oy, oz = lon2d[observed], lat2d[observed], out[observed]
    if n_obs > config.KRIGE_MAX_OBS:
        rng = np.random.default_rng(config.KRIGE_SEED) if rng is None else rng
        pick = rng.choice(n_obs, size=config.KRIGE_MAX_OBS, replace=False)
        ox, oy, oz = ox[pick], oy[pick], oz[pick]
    ok = OrdinaryKriging(ox, oy, oz,
                         variogram_model=variogram or config.KRIGE_VARIOGRAM)
    z, _ = ok.execute("points", lon2d[target], lat2d[target])
    out[target] = np.ma.getdata(z)
    return out


def _step_rng(date: pd.Timestamp, hour: int,
              channel: str | None = None) -> np.random.Generator:
    """The deterministic stream for one (date, hour[, channel]) fill."""
    key = [config.KRIGE_SEED, int(f"{date:%Y%m%d}"), hour]
    if channel is not None:
        key.append(config.SFC_VARS.index(channel))
    return np.random.default_rng(key)


# --------------------------------------------------------------------------- #
# Per-day builders (module-level so multiprocessing can pickle them)
# --------------------------------------------------------------------------- #

def _step_timestamp(date: pd.Timestamp, hour: int) -> pd.Timestamp:
    """Label timestamp of one AIRS period: hour 0 means next-day 00 UTC."""
    days = 1 if hour == 0 else 0
    return date.normalize() + pd.Timedelta(days=days, hours=hour)


def _reanalysis_step(when: pd.Timestamp) -> xr.Dataset | None:
    """One 3-hourly sfc_daily step (physical units) or None if absent."""
    path = day_path(when)
    if not path.exists():
        return None
    with xr.open_dataset(path) as day:
        day = day.load()
    times = pd.DatetimeIndex(day["time"].values)
    if when not in times:
        return None
    return day.sel(time=when)


def _crop() -> np.ndarray:
    """The crop domain (analysis box + halo) as (68, 141) bool.

    Schema v3 (user decision 2026-08-13) defines nothing outside this
    mask -- it is the kriging target boundary AND the NaN boundary of
    every stored channel.  dataset.crop_domain() is module-cached, so
    forked workers pay the (cheap) construction at most once each.
    """
    from .dataset import crop_domain
    return crop_domain()


_BANK = None    # (vf, dates) gap-bank cache, loaded once per process

#: A malformed fullgrid file (truncated time axis, all-fill parceltime,
#: missing variables, HDF read errors, ...) must skip THAT day with a logged
#: note, never abort a multi-year build (spec: only-absent-file handling is
#: not enough on a 15-year archive).
_FULLGRID_ERRORS = (ValueError, KeyError, IndexError, OSError)


def _gap_valid_frac(date: pd.Timestamp, hour: int, allow_small_bank: bool
                    ) -> tuple[np.ndarray, str | None, bool]:
    """Surface-level AIRS availability (68, 141) for one degraded step.

    Prefers the real fullgrid file for the date; otherwise (missing OR
    unreadable file) draws a season-matched surface field (bank lev index 0)
    from the harvested real-gap bank.  Returns (valid_frac, note or None,
    used_bank).  ``used_bank`` tells the caller the mask is a gap-BANK draw
    from a DONOR date, so gap classification must not use the requested
    date's climatological swath (review 2026-08-13; see
    :func:`_classify_step_gaps`), and BOTH bank branches -- missing file
    and unreadable file -- carry a 'bank mask used' note so the builder log
    and krige_validate can quantify how many steps ran on donor geometry.
    """
    fullgrid = airs_fcst.find_fullgrid(date)
    if fullgrid is not None:
        try:
            period = airs_fcst.period_fields(fullgrid, hour)
            return period["valid_frac"].values, None, False
        except _FULLGRID_ERRORS as err:
            note = (f"{hour:02d}Z: unreadable fullgrid "
                    f"{Path(fullgrid).name} ({err}); bank mask used")
    else:
        note = f"{hour:02d}Z: no fullgrid file; bank mask used"
    global _BANK
    from front_finder import mask_bank

    from . import degrade_sfc
    if _BANK is None:
        _BANK = mask_bank.load_bank()
    vf, dates = _BANK
    if len(vf) < mask_bank.MIN_REAL_BANK and not allow_small_bank:
        raise RuntimeError(
            f"gap bank {mask_bank.BANK_PATH} holds only {len(vf)} field(s) "
            f"(< MIN_REAL_BANK={mask_bank.MIN_REAL_BANK}): every degraded "
            f"step would reuse near-identical gap geometry.  Re-harvest it "
            f"from the JPL fullgrid archive (front_finder.mask_bank.harvest) "
            f"or pass --allow-small-bank for a smoke test.")
    # surface_gap_field also corrects the harvested lev-0 halving (see its
    # docstring), so the degraded masks match airs_fcst.period_fields'
    mask = degrade_sfc.surface_gap_field(vf, _step_rng(date, hour),
                                         month=date.month, dates=dates)
    return mask, note, True


def _classify_step_gaps(date, hour: int, valid_frac: np.ndarray,
                        domain: np.ndarray, used_bank: bool) -> np.ndarray:
    """gap_type for one step, honoring the mask's provenance.

    A real fullgrid mask belongs to ``date``, so swath.gap_type_for may
    split its gaps with the date's climatological footprint.  A gap-bank
    draw is a DONOR date's swath at a different 16-day cycle position:
    classifying it against the requested date's climatology labels the
    drawn swath's cloud holes out-of-swath and the empty climatological
    footprint cloud -- systematically scrambled strata (review
    2026-08-13).  Bank draws are therefore classified against the drawn
    mask's own morphological envelope.
    """
    if used_bank:
        observed = swath.observed_mask(valid_frac)
        return swath.classify_gaps(observed, swath.swath_envelope(observed),
                                   domain=domain)
    return swath.gap_type_for(date, hour, valid_frac, domain=domain)


def _build_degraded_day(args) -> tuple[str, list, list]:
    """One overpass day -> degraded-reanalysis steps.

    Returns (date iso, [(timestamp, {var: (68,141) float32}, valid_frac,
    gap_type)], [skip notes]).  Schema v3 (user decision 2026-08-13):
    only the crop domain (analysis box + halo) is filled -- observed
    pixels in the HALO are used (real obs constrain the box edges),
    observed pixels outside the crop are dropped, kriging targets
    crop & ~observed, and every channel (kriged AND clean) is NaN
    out-of-crop.
    """
    date_iso, hours, allow_small_bank = args
    date = pd.Timestamp(date_iso)
    crop = _crop()
    steps, notes = [], []
    for hour in hours:
        when = _step_timestamp(date, hour)
        rea = _reanalysis_step(when)
        if rea is None:
            notes.append(f"{hour:02d}Z: no sfc_daily reanalysis step")
            continue
        valid_frac, note, used_bank = _gap_valid_frac(date, hour,
                                                      allow_small_bank)
        valid_frac = valid_frac.astype(np.float32)
        if note:
            notes.append(note)
        observed = (valid_frac >= airs_fcst.OBSERVED_MIN_FRACTION) & crop
        if not observed.any():
            # krige_fill would return the all-NaN in-crop field untouched,
            # breaking the no-NaN-inside-crop schema -- skip the step.
            notes.append(f"{hour:02d}Z: zero observed pixels inside crop, "
                         f"step skipped")
            continue
        target = crop & ~observed
        fields = {}
        for var in config.SFC_VARS:
            grid = rea[var].values.astype(np.float64)
            if var in config.KRIGED_CHANNELS:
                grid = np.where(observed, grid, np.nan)
                grid = krige_fill(grid, rng=_step_rng(date, hour, var),
                                  target=target)
            else:                       # clean channel: still crop-only
                grid = np.where(crop, grid, np.nan)
            fields[var] = grid.astype(np.float32)
        gap_type = _classify_step_gaps(date, hour, valid_frac, crop,
                                       used_bank)
        steps.append((when, fields, valid_frac, gap_type))
    return date_iso, steps, notes


def _build_airs_day(args) -> tuple[str, list, list]:
    """One overpass day -> real AIRS-FCST steps (same return as degraded)."""
    date_iso, hours, _allow_small_bank = args
    date = pd.Timestamp(date_iso)
    crop = _crop()
    fullgrid = airs_fcst.find_fullgrid(date)
    if fullgrid is None:
        return date_iso, [], ["no fullgrid file"]
    steps, notes = [], []
    for hour in hours:
        try:
            period = airs_fcst.period_fields(fullgrid, hour)
        except _FULLGRID_ERRORS as err:
            notes.append(f"{hour:02d}Z: unreadable fullgrid "
                         f"{Path(fullgrid).name} ({err}), skipped")
            continue
        when = pd.Timestamp(period["time"].values)
        valid_frac = period["valid_frac"].values.astype(np.float32)
        # EVERY kriged channel needs >= 1 observed pixel INSIDE the crop
        # (schema v3, user decision 2026-08-13): krige_fill returns an
        # all-NaN field untouched, which would violate the
        # no-NaN-inside-crop schema (and silently NaN a training loss).
        empty = [var for var in airs_fcst.CHANNEL_MAP.values()
                 if not np.isfinite(period[var].values[crop]).any()]
        if empty:
            notes.append(f"{hour:02d}Z: zero observed pixels inside crop "
                         f"({','.join(empty)}), step skipped")
            continue
        rea = _reanalysis_step(when)     # AIRS retrieves no SLP: copy clean
        if rea is None:
            raise FileNotFoundError(
                f"no sfc_daily reanalysis step at {when} (needed for the "
                f"clean SLP channel); fetch it with "
                f"'python -m dl_front.acquire_merra2_sfc {when.year}'")
        # clean SLP is crop-only too: schema v3 defines nothing outside
        fields = {"SLP": np.where(crop, rea["SLP"].values,
                                  np.nan).astype(np.float32)}
        for var in airs_fcst.CHANNEL_MAP.values():
            # drop observations outside the crop (HALO obs are kept: real
            # AIRS data constrains the box edges, user decision 2026-08-13),
            # then fill only inside it (krige_fill intersects target with
            # the remaining NaNs)
            grid = np.where(crop, period[var].values, np.nan)
            fields[var] = krige_fill(
                grid, rng=_step_rng(date, hour, var),
                target=crop).astype(np.float32)
        gap_type = swath.gap_type_for(date, hour, valid_frac, domain=crop)
        steps.append((when, fields, valid_frac, gap_type))
    return date_iso, steps, notes


# --------------------------------------------------------------------------- #
# Drivers & cache writing
# --------------------------------------------------------------------------- #

def _year_dates(year: int, hours) -> pd.DatetimeIndex:
    """Every overpass date that can contribute a step to ``year``'s file.

    A Dec 31 overpass feeds the NEXT year's file through its 00 UTC step,
    so the previous Dec 31 is included whenever hour 0 is requested (its
    18/21 UTC steps land in the previous year and are filtered out by the
    caller's ``ts.year == year`` cut).
    """
    dates = list(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
    if 0 in hours:
        dates = [pd.Timestamp(f"{year - 1}-12-31")] + dates
    return pd.DatetimeIndex(dates)


def _run_days(worker, dates, hours, workers: int,
              allow_small_bank: bool = False) -> list:
    """Map a per-day builder over dates, log one line per day, collect steps."""
    args = [(d.isoformat(), tuple(hours), allow_small_bank) for d in dates]
    if workers > 1:
        pool = Pool(workers)
        results = pool.imap(worker, args)
    else:
        pool, results = None, map(worker, args)
    steps = []
    try:
        for date_iso, day_steps, notes in results:
            built = ",".join(f"{s[0]:%H}Z" for s in day_steps)
            obs = np.mean([(s[2] >= airs_fcst.OBSERVED_MIN_FRACTION).mean()
                           for s in day_steps]) if day_steps else 0.0
            note = ("; " + "; ".join(notes)) if notes else ""
            print(f"{date_iso[:10]}: hours=[{built}] obs_frac={obs:.3f}{note}",
                  flush=True)
            steps.extend(day_steps)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return steps


def _write_year_cache(steps: list, source: str, year: int, path: Path,
                      bank_used: str) -> Path:
    """One year of steps -> ``kriged_sfc_{year}.nc`` (written even if empty).

    A zero-step year still gets a (time=0) file: it is the phase's
    done-marker, and downstream readers handle an empty year gracefully
    instead of crashing on a missing file.  ``bank_used`` is the swath_bank
    provenance string resolved by :func:`_build` BEFORE the workers were
    forked (see the comment there) -- it must not be re-resolved here.
    """
    group = sorted(steps, key=lambda s: s[0])
    n, shape = len(group), config.GRID_SHAPE
    times = pd.DatetimeIndex([s[0] for s in group]).astype("datetime64[ns]")

    def stack(pick, dtype=np.float32):
        return (np.stack([pick(s) for s in group]) if group
                else np.empty((0, *shape), dtype))

    data = {var: (("time", "lat", "lon"),
                  stack(lambda s, v=var: s[1][v])) for var in config.SFC_VARS}
    data["valid_frac"] = (("time", "lat", "lon"), stack(lambda s: s[2]))
    data["gap_type"] = (("time", "lat", "lon"),
                        stack(lambda s: s[3], np.int8))
    ds = xr.Dataset(data, coords={"time": times,
                                  "lat": np.asarray(config.LABEL_LATS),
                                  "lon": np.asarray(config.LABEL_LONS)})
    from .dataset import halo_px
    ds.attrs = dict(source=source,
                    variogram_model=config.KRIGE_VARIOGRAM,
                    max_obs_points=config.KRIGE_MAX_OBS,
                    created=datetime.now(timezone.utc).isoformat(),
                    schema_version=3,
                    # the resolved domain decision (user 2026-08-13), so a
                    # cache is self-describing even after the YAML changes
                    domain_lat_range=list(config.ANALYSIS_LAT_RANGE),
                    domain_lon_range=list(config.ANALYSIS_LON_RANGE),
                    land_fraction_min=config.LAND_FRACTION_MIN,
                    halo_px=halo_px(),
                    swath_bank=bank_used)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    tag = "" if n else "  (WARNING: empty year)"
    print(f"{year}: wrote {n} steps -> {path}{tag}", flush=True)
    return path


def _build(worker, source: str, years, hours, workers: int, out_dir,
           max_days: int | None, force: bool,
           allow_small_bank: bool = False) -> list[Path]:
    """Shared driver: one year at a time, cache written as soon as the year
    completes (a mid-run crash or walltime kill loses at most the current
    year), existing year caches skipped on rerun unless ``force``."""
    hours = config.AIRS_HOURS if hours is None else tuple(hours)
    # Aggregate zero-fullgrids guard (review 2026-08-13): with a mistyped
    # fullgrid root (e.g. `export JPL_AIRS_FCST=/gpfs/.../AIRS_FCST_ldeg`)
    # find_fullgrid returns None for EVERY date.  The airs flavor would then
    # write empty-but-schema-valid year caches that act as done-markers
    # (skip_krige treats them as complete forever); the degraded flavor
    # would silently run 100% on donor gap-bank masks.  A sparse archive is
    # expected (per-day fallbacks handle that) -- an archive with ZERO
    # fullgrid files is a misconfiguration, so fail before writing anything.
    if not airs_fcst._archive_index(config.AIRS_FCST_ROOT):
        raise RuntimeError(
            f"no fullgrid_* files found anywhere under AIRS_FCST_ROOT="
            f"{config.AIRS_FCST_ROOT}"
            f"{'' if config.AIRS_FCST_ROOT.is_dir() else ' (directory does not exist)'}"
            f": refusing to build '{source}' caches from an empty archive. "
            f"Check the JPL_AIRS_FCST export (canonical cluster path: "
            f"/gpfs/scratch/smap-convection/AIRS_FCST_1deg).")
    # manifest reorg 2026-08-13: the canonical per-flavor cache dir comes
    # from config.KRIGED_SOURCE_DIRS; an --out-dir override still gets the
    # canonical directory NAME appended so both flavors can share one root.
    canonical = config.KRIGED_SOURCE_DIRS[FLAVORS[source]]
    out_root = (Path(out_dir) / canonical.name if out_dir is not None
                else canonical)
    # Resolve the swath-bank provenance ONCE, before any pool fork: this
    # load populates swath._BANK_CACHE, which forked workers inherit, so
    # every worker classifies with exactly the bank (or absence) recorded
    # in the attr -- a bank npz appearing mid-build can no longer produce
    # mixed classification with a mislabeling attr (review 2026-08-13).
    bank_used = (str(config.SWATH_BANK_PATH)
                 if swath.load_swath_bank() is not None
                 else "per-day-envelope")
    written = []
    for year in sorted(years):
        path = out_root / f"kriged_sfc_{year}.nc"
        if path.exists() and not force:
            print(f"{year}: {path} exists, skipped (--force to rebuild)",
                  flush=True)
            written.append(path)
            continue
        dates = _year_dates(year, hours)[:max_days]
        steps = _run_days(worker, dates, hours, workers, allow_small_bank)
        steps = [s for s in steps if s[0].year == year]
        written.append(_write_year_cache(steps, source, year, path,
                                         bank_used))
    return written


def build_degraded(years, hours=None, workers: int = 1,
                   out_dir=None, max_days: int | None = None,
                   force: bool = False,
                   allow_small_bank: bool = False) -> list[Path]:
    """Reanalysis with AIRS-shaped gaps kriged back: degraded_reanalysis."""
    return _build(_build_degraded_day, "degraded_reanalysis", years, hours,
                  workers, out_dir, max_days, force, allow_small_bank)


def build_airs(years, hours=None, workers: int = 1,
               out_dir=None, max_days: int | None = None,
               force: bool = False,
               allow_small_bank: bool = False) -> list[Path]:
    """Real AIRS-FCST surface fields, kriged gap-free: airs_fcst."""
    return _build(_build_airs_day, "airs_fcst", years, hours,
                  workers, out_dir, max_days, force, allow_small_bank)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_years(text: str) -> list[int]:
    """'2007-2015' -> [2007..2015]; '2007,2010' -> [2007, 2010]."""
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in text.split(",")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn in (("build-degraded", build_degraded),
                     ("build-airs", build_airs)):
        p = sub.add_parser(name, help=fn.__doc__)
        p.set_defaults(fn=fn)
        p.add_argument("--years", required=True,
                       help="'2007-2015' or '2007,2010,2012'")
        p.add_argument("--hours", default=None,
                       help=f"comma UTC hours (default {config.AIRS_HOURS})")
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--out-dir", default=None,
                       help="cache root override (default: the flavor's "
                            "config.KRIGED_SOURCE_DIRS directory; the "
                            "flavor's directory name is still appended)")
        p.add_argument("--max-days", type=int, default=None,
                       help="only build the first N overpass days per year "
                            "(bounded smoke-test path)")
        p.add_argument("--force", action="store_true",
                       help="rebuild year caches that already exist "
                            "(default: skip them and resume)")
        p.add_argument("--allow-small-bank", action="store_true",
                       help="build-degraded only: accept a gap bank smaller "
                            "than mask_bank.MIN_REAL_BANK (smoke tests)")
    args = ap.parse_args(argv)
    hours = (tuple(int(h) for h in args.hours.split(","))
             if args.hours else None)
    args.fn(parse_years(args.years), hours=hours, workers=args.workers,
            out_dir=args.out_dir, max_days=args.max_days, force=args.force,
            allow_small_bank=args.allow_small_bank)


if __name__ == "__main__":
    main()
