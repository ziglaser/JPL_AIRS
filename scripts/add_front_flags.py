"""Write surface-front flag variables into the FCST_SMAP_MRMS year files.

Until now the front flags existed only in memory: ``convection_skill.dataset``
called ``fronts.year_front_flags`` while building the base table and nothing
was ever written back (and, from the 2026-08-13 manifest reorg until the path
fix on 2026-08-18, that call silently produced all-NaN columns).  This script
materializes them as ordinary netCDF variables on the dataset's own
``(date, time, lat, lon)`` grid so that any consumer -- the base table, Mark's
RF, a notebook -- reads the same numbers without re-deriving the alignment.

Two families of flags, deliberately named so they can never be confused:

* ``front_{cold,warm,stationary,occluded,dryline,any}_{1,3}w`` -- MET-DRAWN
  analyst fronts.  The names are exactly ``fronts.front_columns()`` (plus
  ``dryline`` when the source has it) so the base table can read them straight
  out of the file instead of recomputing; the source (NOAA-XML or WPC/CODSUS)
  is recorded in each variable's ``source`` attribute rather than in its name,
  which is why ``--label-source`` takes ONE source per output file.
* ``pred_front_{type}_3w`` -- OUR MODEL's predictions, read from a bk19-schema
  prediction tree (``--pred-dir``).  The distinct ``pred_`` prefix is load
  bearing: it keeps model output out of ``fronts.front_columns()``, so
  ``convection_skill.dataset``/``hypotheses``/``models`` cannot silently
  substitute predicted fronts for analyst fronts in the pre-registered F1-F5
  tests, and a reader scanning variable names can tell truth from model at a
  glance.  3wide only -- every model we trained saw 3wide labels
  (``dl_front.config.LABEL_WIDTH = 3``) and a 3wide prediction cannot be
  re-thinned to a 1-cell line.

Alignment is NOT reimplemented here: :func:`convection_skill.fronts.file_front_flags`
does the 2x2 grid max-pool and the concurrent slot->bulletin mapping, so these
variables are identical to the in-memory columns by construction.  NaN (never
0) wherever the governing analysis is unavailable: slot 0 (the overpass slot
has no forecast hour), dates whose 3-hourly bulletin is missing from the
source, Dec 31 slots 4-6 (their 00 UTC bulletin lives in the next year's file),
and -- for predictions -- cells whose four overlapping 1-degree cells are all
outside the model's trained analysis domain (bk19-schema fill byte 2).

The primary dataset is never modified by default: the year file is COPIED
(so every existing variable, attribute, dtype, _FillValue and the ``date``
encoding are preserved bit-for-bit -- nothing is re-encoded) and the flags are
appended to the copy.  Each run starts from the pristine primary, so the output
is a pure function of (primary file, front source, checkpoint tag, git rev) and
a rerun replaces the flags rather than patching them.  ``--in-place`` appends
to the primary itself and is an explicit opt-in.

Usage::

    PYTHONPATH=src python scripts/add_front_flags.py --years 2016-2021 \\
        --label-source noaa --out-dir data/FCST_SMAP_MRMS_fronts
    PYTHONPATH=src python scripts/add_front_flags.py --years 2016-2018 \\
        --pred-dir $JPL_AIRS_DATA/front_id/predicted_fronts/dlfront_D6C-ens3_kriged-airs \\
        --pred-tag D6C-ens3_kriged-airs
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import netCDF4
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convection_skill import config as cs_config          # noqa: E402
from convection_skill import fronts as fr                  # noqa: E402
from front_finder import config as ff_config               # noqa: E402

#: The flag variables live on the dataset's forecast axis (named ``time``,
#: values 0..6), matching every FCST_* variable.  The separate ``nhours``
#: dimension and the zero-filled ``nhours(time)`` variable are MRMS
#: bookkeeping and are left untouched (data_loading.make_uniform works around
#: them, and it passes through any var whose dims are exactly these four).
FLAG_DIMS: tuple[str, ...] = ("date", "time", "lat", "lon")
#: Prefixes that identify variables this script owns (for --force detection).
FLAG_PREFIXES: tuple[str, ...] = ("front_", "pred_front_")
#: float32 + NaN fill matches every other variable in these files AND
#: fronts.py's "NaN where unavailable" policy.  Unlike the primary variables
#: these are compressed: a flag field is mostly constant, and zlib=4 took a
#: real 2018 NOAA any-front field from 12.31 MB to 0.39 MB (measured
#: 2026-08-18).  Compression is invisible to readers.
FLAG_DTYPE: str = "f4"
ZLIB_COMPLEVEL: int = 4
#: date chunk of 30 keeps the chunk under the 366-day years (a chunksize may
#: not exceed its dimension) and gives cheap month-at-a-time access.
DATE_CHUNK: int = 30
#: One-line statement of the concurrent slot->bulletin rule, stamped on every
#: variable so the file is self-documenting (Zach 2026-08-05: concurrent only).
TIME_MAPPING: str = ("concurrent: slots 1-3 use the same-day 21 UTC analysis, "
                     "slots 4-6 the next-day 00 UTC analysis; slot 0 (overpass) "
                     "has no forecast hour and is NaN")
#: Where copies go by default (a parallel tree, same basenames, so a consumer
#: switches trees by changing one directory).
DEFAULT_OUT_DIR = cs_config.DATA_DIR / "FCST_SMAP_MRMS_fronts"


def _git_rev() -> str | None:
    """Current commit, or None outside a checkout (evaluate_test's helper)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(Path(__file__).parent),
                                       text=True).strip()
    except Exception:
        return None


def parse_years(spec: str) -> tuple[int, ...]:
    """``2016-2021`` or ``2016,2018`` -> a tuple of years."""
    out: list[int] = []
    for piece in spec.split(","):
        if "-" in piece:
            lo, hi = (int(v) for v in piece.split("-"))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(piece))
    return tuple(out)


def primary_path(primary_dir: Path, year: int) -> Path:
    """The read-only source year file (basename fixed by the dataset)."""
    return primary_dir / f"FCST_SMAP_MRMS_{year}.nc"


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(dates, slots, lats, lons) from the year file's own coordinates.

    ``decode_cf`` must stay on: ``nflags`` is a ``char(nflags, string22)``
    label array whose name collides with its dimension, so ``decode_cf=False``
    raises MissingDimensionsError.  ``FCST_parceltime`` is dropped only to
    silence its pre-existing "nanoseconds since ..." cast warning.
    """
    with xr.open_dataset(path, drop_variables=["FCST_parceltime"]) as ds:
        return (ds["date"].values, ds["time"].values,
                ds["lat"].values, ds["lon"].values)


def any_over_common_types(flags: dict) -> np.ndarray:
    """``any`` = max over the FOUR types every source has.

    Recomputed here rather than taken from the file's own channel list so that
    ``front_any_*``/``pred_front_any_*`` mean the SAME thing in every file --
    and exactly what the base table's ``front_any_*`` column means, which is
    built from ``fronts.FRONT_TYPES`` (no dryline: only the NOAA source and our
    6-class model have that class).  The dryline flag is written as its own
    variable, so ``np.fmax`` of the two gives a dryline-inclusive "any".
    """
    return np.fmax.reduce([flags[t] for t in fr.FRONT_TYPES if t in flags])


def _long_name_what(ftype: str, types: tuple[str, ...]) -> str:
    """Human phrase for one channel ("dryline" is not a "... front")."""
    if ftype == "any":
        return "any of " + "/".join(t for t in fr.FRONT_TYPES if t in types)
    return ftype if ftype == fr.DRYLINE_TYPE else f"{ftype} front"


def met_drawn_flags(source: str, year: int, dates, slots, lats, lons) -> dict:
    """``{name: (values, attrs)}`` for the met-drawn flags, both widths.

    A missing width/year yields all-NaN variables with the same names, exactly
    as ``fronts.year_front_flags`` does, so the file schema is stable across
    years and sources.
    """
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for width in fr.FRONT_WIDTHS:
        path = fr.label_path(source, width, year)
        if path is None:
            expected = fr.label_search_dirs(source, width)
            print(f"  MISSING {source} {width}wide {year}: nothing in "
                  f"{[str(d) for d in expected]} -> all-NaN flags", flush=True)
            # Same type set file_front_types would report for this source
            # (NOAA files carry drylines, WPC files do not), so a missing
            # year still writes the full schema the docstring promises.
            types = fr.FRONT_TYPES + ((fr.DRYLINE_TYPE,)
                                      if source == "noaa" else ())
            flags = {t: np.full((len(dates), len(slots), len(lats), len(lons)),
                                np.nan, dtype=np.float32)
                     for t in types + ("any",)}
            origin = "MISSING"
        else:
            types = fr.file_front_types(path)
            flags = fr.file_front_flags(path, types, dates, tuple(slots),
                                        lats, lons)
            origin = path.name
        flags["any"] = any_over_common_types(flags)
        for ftype, values in flags.items():
            what = _long_name_what(ftype, types)
            out[f"front_{ftype}_{width}w"] = (values, {
                "long_name": f"met-drawn {what} within the cell "
                             f"({width}-cell-wide line, {source.upper()})",
                "source": origin,
                "front_source": source,
                "width": np.int32(width),
                "time_mapping": TIME_MAPPING,
                "units": "1",
            })
    return out


def prediction_flags(pred_dir: Path, tag: str, year: int,
                     dates, slots, lats, lons) -> dict:
    """``{name: (values, attrs)}`` for the model flags (3wide only).

    ``pred_dir`` is any bk19-schema tree; the channel set is taken from the
    first year present so a missing year still gets the same variable names.
    """
    width = 3
    types = None
    for probe_year in (year, *range(2003, 2023)):
        probe = fr.prediction_path(pred_dir, width, probe_year)
        if probe is not None:
            types = fr.file_front_types(probe)
            break
    if types is None:
        raise SystemExit(f"no bk19-schema year file under {pred_dir} "
                         f"(expected .../1deg_{width}wide/3hr/"
                         f"{fr.PRED_FILE_TEMPLATE.format(width=width, year=year)})")

    path = fr.prediction_path(pred_dir, width, year)
    if path is None:
        print(f"  MISSING predictions {tag} {year} under {pred_dir} "
              f"-> all-NaN flags", flush=True)
        flags = {t: np.full((len(dates), len(slots), len(lats), len(lons)),
                            np.nan, dtype=np.float32)
                 for t in types + ("any",)}
        origin = "MISSING"
    else:
        flags = fr.file_front_flags(path, types, dates, tuple(slots), lats, lons)
        origin = path.name

    flags["any"] = any_over_common_types(flags)
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for ftype, values in flags.items():
        what = _long_name_what(ftype, types)
        out[f"pred_front_{ftype}_{width}w"] = (values, {
            "long_name": f"MODEL-PREDICTED {what} within the cell "
                         f"({width}-cell-wide equivalent, tag {tag})",
            "source": origin,
            "model_tag": tag,
            "prediction_tree": str(pred_dir),
            "width": np.int32(width),
            "time_mapping": TIME_MAPPING,
            "units": "1",
            "comment": "NaN where all four overlapping 1-degree cells are "
                       "outside the model's trained analysis domain "
                       "(bk19-schema fill value 2); see pred_front_valid_frac",
        })
    if path is not None:
        out["pred_front_valid_frac"] = (valid_fraction(path, lats, lons), {
            "long_name": "fraction of the four overlapping 1-degree prediction "
                         "cells that lie inside the model's trained domain",
            "source": path.name,
            "model_tag": tag,
            "units": "1",
        }, ("lat", "lon"))
    return out


def valid_fraction(path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """How many of each cell's four 1-degree neighbours are non-fill, /4.

    Static in time (the fill mask is the time-invariant analysis domain), so
    one time step suffices.  Companion to the ANY-overlap pooling rule: 1.0
    means all four corners were predicted, 0.25-0.75 a domain-edge cell,
    0.0 a cell the model says nothing about.
    """
    with xr.open_dataset(path) as f:
        step = f["fronts"].isel(time=0, front=0)
        corners = [step.sel(lat=lats + dlat, lon=lons + dlon).notnull().values
                   for dlat in (-0.5, 0.5) for dlon in (-0.5, 0.5)]
    return (np.sum(corners, axis=0) / 4.0).astype(np.float32)


def existing_flags(path: Path) -> list[str]:
    """Flag variables already in the file (idempotency / --force check)."""
    with netCDF4.Dataset(path) as nc:
        return [n for n in nc.variables
                if n.startswith(FLAG_PREFIXES)]


def append_flags(path: Path, variables: dict) -> None:
    """Append (or overwrite) the flag variables in place, netCDF4 mode 'a'.

    netCDF4 cannot re-create an existing variable ("String match to name in
    use"), so an existing variable is assigned into instead.  Everything else
    in the file is left byte-for-byte alone -- no decode/re-encode pass.
    """
    with netCDF4.Dataset(path, "a") as nc:
        for name, spec in variables.items():
            values, attrs = spec[0], spec[1]
            dims = spec[2] if len(spec) > 2 else FLAG_DIMS
            if name in nc.variables:
                nc[name][:] = values
                var = nc[name]
            else:
                chunks = tuple(min(DATE_CHUNK, len(nc.dimensions[d]))
                               if d == "date" else len(nc.dimensions[d])
                               for d in dims)
                var = nc.createVariable(name, FLAG_DTYPE, dims,
                                        zlib=True, complevel=ZLIB_COMPLEVEL,
                                        chunksizes=chunks,
                                        fill_value=np.float32(np.nan))
                var[:] = values
            for key, value in attrs.items():
                var.setncattr(key, value)


def summarize(variables: dict, lats: np.ndarray) -> dict:
    """Per-variable flag rate / NaN count / covered lat band (for the log)."""
    rows = {}
    for name, spec in variables.items():
        values = np.asarray(spec[0], dtype=np.float32)
        finite = np.isfinite(values)
        if values.ndim == 4:
            rows_with_data = np.isfinite(values).any(axis=(0, 1, 3))
            band = ([float(lats[rows_with_data].min()),
                     float(lats[rows_with_data].max())]
                    if rows_with_data.any() else None)
        else:
            band = None
        rows[name] = {
            "flag_rate": (float(np.nanmean(values)) if finite.any()
                          else None),
            "n_valid": int(finite.sum()),
            "n_nan": int((~finite).sum()),
            "lat_band": band,
        }
    return rows


def process_year(year: int, args) -> dict:
    """Build and write one year; returns its provenance record."""
    src = primary_path(Path(args.primary_dir), year)
    if not src.exists():
        print(f"{year}: MISSING primary file {src} -> skipped", flush=True)
        return {"year": year, "skipped": f"no primary file {src}"}

    dates, slots, lats, lons = read_grid(src)
    print(f"{year}: date={len(dates)} time={len(slots)} "
          f"lat={len(lats)} lon={len(lons)}", flush=True)

    variables: dict = {}
    if args.label_source:
        variables.update(met_drawn_flags(args.label_source, year,
                                        dates, slots, lats, lons))
    if args.pred_dir:
        variables.update(prediction_flags(Path(args.pred_dir), args.pred_tag,
                                          year, dates, slots, lats, lons))
    if not variables:
        raise SystemExit("nothing to write: pass --label-source and/or --pred-dir")

    if args.in_place:
        target = src
    else:
        target = Path(args.out_dir) / src.name
        target.parent.mkdir(parents=True, exist_ok=True)

    present = existing_flags(target) if target.exists() else []
    if present and not args.force:
        raise SystemExit(f"{target} already carries flag variables "
                         f"{present}; rerun with --force to replace them")
    if not args.in_place:
        # Copy first: the output is then a pure function of the pristine
        # primary plus this run, and every original byte is preserved.
        shutil.copy2(src, target)
    print(f"  -> {target}", flush=True)

    append_flags(target, variables)
    stats = summarize(variables, lats)
    for name, row in stats.items():
        print(f"  {name:28s} rate={row['flag_rate']!s:>8.8} "
              f"valid={row['n_valid']:>9d} nan={row['n_nan']:>9d} "
              f"lat={row['lat_band']}", flush=True)
    return {
        "year": year,
        "primary": str(src),
        "output": str(target),
        "in_place": bool(args.in_place),
        "label_source": args.label_source,
        "label_dirs": {"wpc": str(ff_config.CODSUS_DIR),
                       "noaa": str(ff_config.NOAA_LABELS_DIR)},
        "pred_dir": args.pred_dir,
        "pred_tag": args.pred_tag,
        "variables": stats,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--years", default="2016-2021",
                   help="e.g. 2016-2021 or 2016,2018 (default: all six)")
    p.add_argument("--label-source", choices=sorted(fr.LABEL_FILE_TEMPLATES) + [""],
                   default="noaa",
                   help="met-drawn source; NOAA is the dateline-fixed 6-class "
                        "product with drylines, WPC the 4-class product the "
                        "pre-registered F1-F5 tests used. Pass '' to skip.")
    p.add_argument("--pred-dir", default=None,
                   help="a bk19-schema prediction tree "
                        "(.../<tag>/1deg_3wide/3hr/); omit to skip model flags")
    p.add_argument("--pred-tag", default=None,
                   help="name recorded in the model flags' attrs "
                        "(default: --pred-dir's basename)")
    p.add_argument("--primary-dir",
                   default=str(cs_config.DATA_DIR / "FCST_SMAP_MRMS"),
                   help="directory holding the read-only FCST_SMAP_MRMS_*.nc")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="where the flagged copies go")
    p.add_argument("--in-place", action="store_true",
                   help="append into the PRIMARY files instead of copies")
    p.add_argument("--force", action="store_true",
                   help="replace flag variables that are already present")
    args = p.parse_args(argv)
    if args.label_source == "":
        args.label_source = None
    if args.pred_dir and not args.pred_tag:
        args.pred_tag = Path(args.pred_dir).name

    # The two packages resolve different data roots (front_finder honours
    # $JPL_AIRS_DATA, convection_skill is hard-wired to the checkout), so echo
    # both -- on the cluster they are NOT the same tree.
    print(f"labels:  wpc={ff_config.CODSUS_DIR}", flush=True)
    print(f"         noaa={ff_config.NOAA_LABELS_DIR}", flush=True)
    print(f"primary: {args.primary_dir}", flush=True)

    records = [process_year(y, args) for y in parse_years(args.years)]
    sidecar = Path(args.out_dir if not args.in_place else args.primary_dir)
    sidecar.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar / "add_front_flags_run.json"
    sidecar.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "time_mapping": TIME_MAPPING,
        "years": records,
    }, indent=2))
    print(f"provenance -> {sidecar}", flush=True)


if __name__ == "__main__":
    main()
