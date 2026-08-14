"""AIRS swath geometry: expected-coverage climatology and gap decomposition.

A missing pixel in an AIRS-FCST period is missing for one of two very
different reasons, and the CNN (and the kriging validation) should be able
to tell them apart:

* **out-of-swath** -- the satellite simply did not fly over the cell at that
  time.  Orbit geometry repeats every :data:`CYCLE_DAYS` = 16 days, so this
  part is deterministic and predictable from a climatology.
* **retrieval (cloud) gap** -- the cell was inside the swath but the
  retrieval failed (overwhelmingly cloud contamination).  This part is
  weather, not geometry.

The decomposition is: composite the *observed* masks of many days that share
the same position in the 16-day cycle (per period hour), keep cells covered
in at least ``config.SWATH_MIN_FRACTION`` of those days as the **expected
swath** (the composite frequency is well below 1 even at swath centers
because of clouds -- hence the deliberately low 5-10 % threshold), and flag
every missing in-swath cell as a cloud gap:

    gap_type (int8): config.GAP_OUT_OF_DOMAIN (-1) outside the caller's
    domain (the cache builders pass dataset.crop_domain(), the analysis
    box + halo -- user decision 2026-08-13), GAP_OBSERVED (0),
    GAP_CLOUD (1), GAP_OUT_OF_SWATH (2).

One ordinal variable (rather than separate masks) so a later experiment can
feed it to the CNN as a single extra channel -- or split it into one-hot
channels -- without another cache-schema change.

Forward projection (the 21Z / next-day-00Z periods): the fullgrid forecast
slots advect the overpass airmass, so the covered region drifts downstream.
Three footprint predictors of increasing physics (and cost) are provided and
raced against each other by ``compare_projections`` (IoU vs the actual
envelope on days with real files; used by dl_front.krige_validate):

* ``project_composite`` -- no motion model at all: the per-(cycle-day, hour)
  climatology already *is* the projected swath.  The bank stores it.
* ``project_shift``     -- rigid translation of the overpass envelope by the
  swath-mean wind (one displacement vector, O(1) rolls).
* ``project_hull``      -- convex hull of the overpass envelope, each vertex
  advected by the locally interpolated wind, hull re-rasterized (the
  "move the edges, not every cell" minimalist approach).

Bank CLI (run on the JPL archive; one pass over the fullgrid files)::

    JPL_AIRS_FCST=... PYTHONPATH=src \
        python -m dl_front.swath build-bank --years 2016-2021
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import airs_fcst, config

#: AIRS/Aqua ground-track repeat cycle, days.
CYCLE_DAYS = 16
#: Cycle days with fewer composited overpasses than this are too thin to
#: threshold into a footprint (expected_swath returns None for them).
MIN_DAYS_PER_CYCLE_DAY = 5
#: Structuring-element iterations for the morphological envelope closing:
#: 2 one-degree dilate/erode rounds bridge typical cloud holes without
#: welding separate swath passes together (demo-day audit 2026-08-13).
ENVELOPE_CLOSING_ITERS = 2


def cycle_day(date) -> int:
    """Position of ``date`` in the 16-day repeat cycle.

    The epoch (proleptic ordinal 0) is arbitrary -- only consistency between
    bank building and lookup matters, and both go through this function.
    """
    return pd.Timestamp(date).toordinal() % CYCLE_DAYS


def observed_mask(valid_frac: np.ndarray) -> np.ndarray:
    """(68, 141) bool: pixels with an actual AIRS surface retrieval."""
    return np.asarray(valid_frac) >= airs_fcst.OBSERVED_MIN_FRACTION


def swath_envelope(observed: np.ndarray,
                   iters: int = ENVELOPE_CLOSING_ITERS) -> np.ndarray:
    """Close the cloud holes of one day's observed mask -> swath envelope.

    Morphological closing (dilate then erode, ``iters`` rounds of the
    3x3 structuring element) bridges gaps up to ~2*iters degrees wide, then
    interior holes are filled outright.  The result is the estimated area
    the swath *covered*, clouds included -- the per-day fallback truth when
    no climatological bank footprint is available.
    """
    observed = np.asarray(observed, bool)
    if not observed.any():
        return observed.copy()
    closed = ndimage.binary_closing(
        observed, structure=np.ones((3, 3), bool), iterations=iters)
    return ndimage.binary_fill_holes(closed)


def classify_gaps(observed: np.ndarray, envelope: np.ndarray,
                  domain: np.ndarray | None = None) -> np.ndarray:
    """(68, 141) int8 gap_type from an observed mask and a swath envelope.

    ``domain`` (bool, default all-True) marks the analysis region; outside
    it every pixel is GAP_OUT_OF_DOMAIN regardless of coverage.
    """
    observed = np.asarray(observed, bool)
    # an observed pixel is in-swath by definition
    envelope = np.asarray(envelope, bool) | observed
    out = np.full(observed.shape, config.GAP_OUT_OF_SWATH, np.int8)
    out[envelope] = config.GAP_CLOUD
    out[observed] = config.GAP_OBSERVED
    if domain is not None:
        out[~np.asarray(domain, bool)] = config.GAP_OUT_OF_DOMAIN
    return out


# --------------------------------------------------------------------------- #
# The swath bank: per-(cycle day, hour) coverage climatology
# --------------------------------------------------------------------------- #

def build_swath_bank(years, hours=None, root=None,
                     path: Path = None) -> Path:
    """Composite observed masks by (cycle day, period hour) -> swath_bank.npz.

    For every date in ``years`` with a fullgrid file, the surface observed
    mask of each period is accumulated into its (cycle_day, hour) bin; the
    bank stores the per-bin coverage *frequency* plus the day count, and
    the footprint threshold (config.SWATH_MIN_FRACTION) is applied at READ
    time (expected_swath), so re-thresholding needs no rebuild.
    """
    hours = tuple(config.AIRS_HOURS if hours is None else hours)
    path = Path(config.SWATH_BANK_PATH if path is None else path)
    shape = (CYCLE_DAYS, len(hours), *config.GRID_SHAPE)
    counts = np.zeros(shape, np.int32)
    n_days = np.zeros((CYCLE_DAYS, len(hours)), np.int32)

    dates = pd.DatetimeIndex([])
    for year in years:
        dates = dates.append(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
    for date in dates:
        fullgrid = airs_fcst.find_fullgrid(date, root=root)
        if fullgrid is None:
            continue
        # one load per archive day, shared across the three period hours --
        # period_fields would otherwise re-open and fully re-load the same
        # (NFS) file per hour, tripling the I/O of a mask-only pass
        # (review 2026-08-13)
        try:
            ds = airs_fcst.load_fullgrid(fullgrid)
        except (ValueError, KeyError, IndexError, OSError) as err:
            print(f"{date:%Y-%m-%d}: skipped ({err})", flush=True)
            continue
        cyc = cycle_day(date)
        for h_idx, hour in enumerate(hours):
            try:
                period = airs_fcst.period_fields(fullgrid, hour, ds=ds)
            except (ValueError, KeyError, IndexError, OSError) as err:
                print(f"{date:%Y-%m-%d} {hour:02d}Z: skipped ({err})",
                      flush=True)
                continue
            counts[cyc, h_idx] += observed_mask(period["valid_frac"].values)
            n_days[cyc, h_idx] += 1
        print(f"{date:%Y-%m-%d}: composited into cycle day {cyc}", flush=True)

    freq = counts / np.maximum(n_days[..., None, None], 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, freq=freq.astype(np.float32), n_days=n_days,
        hours=np.asarray(hours),
        years=np.asarray(sorted(set(int(y) for y in years))))
    print(f"wrote {path}: {int(n_days.sum())} (day, hour) composites, "
          f"cycle-day day counts {n_days.min(1).tolist()}", flush=True)
    return path


_BANK_CACHE: dict[Path, dict | None] = {}


def load_swath_bank(path: Path = None) -> dict | None:
    """The bank as {freq, n_days, hours, years} arrays, or None if absent."""
    path = Path(config.SWATH_BANK_PATH if path is None else path)
    if path not in _BANK_CACHE:
        if not path.exists():
            _BANK_CACHE[path] = None
        else:
            with np.load(path) as z:
                _BANK_CACHE[path] = {k: z[k] for k in z.files}
    return _BANK_CACHE[path]


def expected_swath(date, hour: int, bank: dict | None = None,
                   path: Path = None) -> np.ndarray | None:
    """Climatological footprint for (date, hour), or None if unavailable.

    None means: no bank on disk, the hour was not composited, or the cycle
    day has fewer than MIN_DAYS_PER_CYCLE_DAY contributing overpasses --
    callers fall back to the per-day morphological envelope.
    """
    bank = load_swath_bank(path) if bank is None else bank
    if bank is None:
        return None
    hours = bank["hours"].tolist()
    if hour not in hours:
        return None
    cyc, h_idx = cycle_day(date), hours.index(hour)
    if bank["n_days"][cyc, h_idx] < MIN_DAYS_PER_CYCLE_DAY:
        return None
    return bank["freq"][cyc, h_idx] >= config.SWATH_MIN_FRACTION


def gap_type_for(date, hour: int, valid_frac: np.ndarray,
                 domain: np.ndarray | None = None) -> np.ndarray:
    """The (68, 141) int8 gap_type of one period.

    Envelope preference: the climatological expected swath (bank) when it
    covers this (cycle day, hour); else the day's own morphological
    envelope.  The bank is the better cloud/swath splitter -- a fully
    overcast swath segment leaves no observed pixels for closing to work
    with, but the climatology still knows the satellite was there.
    """
    observed = observed_mask(valid_frac)
    envelope = expected_swath(date, hour)
    if envelope is None:
        envelope = swath_envelope(observed)
    return classify_gaps(observed, envelope, domain=domain)


# --------------------------------------------------------------------------- #
# Forward-projection experiments (overpass envelope -> later-hour footprint)
# --------------------------------------------------------------------------- #

def _mean_wind(u: np.ndarray, v: np.ndarray,
               where: np.ndarray) -> tuple[float, float]:
    """Swath-mean (u, v) m/s over ``where`` pixels (NaN-safe)."""
    pick_u, pick_v = u[where], v[where]
    return (float(np.nanmean(pick_u)), float(np.nanmean(pick_v)))


def _displacement_deg(u_ms: float, v_ms: float, dt_hours: float,
                      lat_deg: float) -> tuple[float, float]:
    """Wind (m/s) x time -> (dlat, dlon) in degrees at latitude lat_deg."""
    meters = 111_000.0
    dlat = v_ms * dt_hours * 3600 / meters
    dlon = u_ms * dt_hours * 3600 / (meters * np.cos(np.radians(lat_deg)))
    return dlat, dlon


def project_shift(envelope: np.ndarray, u: np.ndarray, v: np.ndarray,
                  dt_hours: float) -> np.ndarray:
    """Rigid translation of the envelope by the swath-mean wind.

    The 1-degree grid makes the roll a whole-degree shift; sub-degree
    displacement (common at these 2-6 h leads) rounds to zero, which is
    exactly the null hypothesis this cheap method represents.
    """
    envelope = np.asarray(envelope, bool)
    if not envelope.any():
        return envelope.copy()
    lat0 = float(np.asarray(config.LABEL_LATS)[envelope.any(1)].mean())
    u0, v0 = _mean_wind(u, v, envelope)
    dlat, dlon = _displacement_deg(u0, v0, dt_hours, lat0)
    out = np.roll(envelope, (round(dlat), round(dlon)), axis=(0, 1))
    # roll wraps around the grid edge; wrapped rows/cols are nonsense
    if round(dlat):
        edge = slice(round(dlat)) if dlat > 0 else slice(round(dlat), None)
        out[edge, :] = False
    if round(dlon):
        edge = slice(round(dlon)) if dlon > 0 else slice(round(dlon), None)
        out[:, edge] = False
    return out


def project_hull(envelope: np.ndarray, u: np.ndarray, v: np.ndarray,
                 dt_hours: float) -> np.ndarray:
    """Advect the convex hull's vertices by the local wind, re-rasterize.

    Moves O(10) boundary points instead of every cell; the local wind is
    the mean over the 3x3 neighborhood of each vertex (NaN-safe, falling
    back to the swath mean where the neighborhood is unobserved).
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull, QhullError

    envelope = np.asarray(envelope, bool)
    lats = np.asarray(config.LABEL_LATS, float)
    lons = np.asarray(config.LABEL_LONS, float)
    iy, ix = np.nonzero(envelope)
    if len(iy) < 3:
        return envelope.copy()
    pts = np.column_stack([lons[ix], lats[iy]])
    try:
        hull = ConvexHull(pts)
    except QhullError:                       # degenerate (collinear) swath
        return project_shift(envelope, u, v, dt_hours)

    u0, v0 = _mean_wind(u, v, envelope)
    moved = []
    for k in hull.vertices:
        vy, vx = iy[k], ix[k]
        ny = slice(max(vy - 1, 0), vy + 2)
        nx = slice(max(vx - 1, 0), vx + 2)
        uu = np.nanmean(u[ny, nx]) if np.isfinite(u[ny, nx]).any() else u0
        vv = np.nanmean(v[ny, nx]) if np.isfinite(v[ny, nx]).any() else v0
        dlat, dlon = _displacement_deg(uu, vv, dt_hours, lats[vy])
        moved.append([lons[vx] + dlon, lats[vy] + dlat])

    lon2d, lat2d = np.meshgrid(lons, lats)
    inside = MplPath(moved).contains_points(
        np.column_stack([lon2d.ravel(), lat2d.ravel()]))
    return inside.reshape(envelope.shape)


def compare_projections(dates, hours=(21, 0), root=None) -> pd.DataFrame:
    """Race the three footprint predictors on days with real fullgrid files.

    Truth for (date, hour) = the morphological envelope of the ACTUAL
    observed mask at that hour.  Every method predicts it from overpass-time
    (18Z) information only: the 18Z envelope and the 18Z surface winds,
    advected by the PHYSICAL lead time from the mid-overpass instant
    (``airs_fcst.overpass_midpoint``, e.g. ~18:59) to the forecast slot.
    Using the rounded 18Z slot-0 label instead would over-advect shift/hull
    by up to 50 % at the 21Z lead (~2 h true vs 3 h nominal) and bias the
    method race against the motionless composite (review 2026-08-13).
    Hour 0 keeps its next-day-00-UTC semantics: the slot timestamp from
    period_fields already lands on the following day, so the lead is ~5 h.
    Returns one tidy row per (date, hour, method) with IoU and wall
    milliseconds.
    """
    bank = load_swath_bank()
    rows = []
    for date in pd.DatetimeIndex(dates):
        fullgrid = airs_fcst.find_fullgrid(date, root=root)
        if fullgrid is None:
            continue
        try:
            ds = airs_fcst.load_fullgrid(fullgrid)
            over = airs_fcst.period_fields(fullgrid, 18, ds=ds)
        except (ValueError, KeyError, IndexError, OSError):
            continue
        env18 = swath_envelope(observed_mask(over["valid_frac"].values))
        u18 = over["U10M"].values
        v18 = over["V10M"].values
        t0 = airs_fcst.overpass_midpoint(fullgrid)   # physical, unrounded
        for hour in hours:
            try:
                actual = airs_fcst.period_fields(fullgrid, hour, ds=ds)
            except (ValueError, KeyError, IndexError, OSError):
                continue
            truth = swath_envelope(
                observed_mask(actual["valid_frac"].values))
            if not truth.any():
                continue
            dt = (pd.Timestamp(actual["time"].values) - t0
                  ).total_seconds() / 3600
            methods = {
                "composite": lambda: expected_swath(date, hour, bank=bank),
                "shift": lambda: project_shift(env18, u18, v18, dt),
                "hull": lambda: project_hull(env18, u18, v18, dt),
            }
            for name, fn in methods.items():
                t0 = time.perf_counter()
                pred = fn()
                ms = (time.perf_counter() - t0) * 1e3
                if pred is None:              # composite without a bank
                    continue
                inter = (pred & truth).sum()
                union = (pred | truth).sum()
                rows.append({"date": date, "hour": hour, "method": name,
                             "iou": inter / union if union else np.nan,
                             "ms": ms})
    # Always emit the tidy schema, even when no sampled date has a real
    # fullgrid file (local runs) -- an empty but headed CSV keeps downstream
    # pd.read_csv from choking on a zero-byte file.
    return pd.DataFrame(rows, columns=["date", "hour", "method", "iou", "ms"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _year_list(spec: str) -> list[int]:
    """'2016-2021' or '2016,2018' -> [ints]."""
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bank = sub.add_parser("build-bank",
                          help="composite the 16-day swath climatology")
    bank.add_argument("--years", required=True, type=_year_list)
    bank.add_argument("--hours", default=None,
                      help="comma list, default config airs.hours")
    bank.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    hours = (tuple(int(h) for h in args.hours.split(","))
             if args.hours else None)
    build_swath_bank(args.years, hours=hours,
                     path=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
