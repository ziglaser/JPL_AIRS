"""Bank of REAL AIRS gap/swath masks for stage-B degraded pretraining.

Random synthetic masking would miss the decisive property of AIRS gaps:
retrieval failures cluster under cloud, which is anticorrelated with clear
sky exactly where fronts live (workplan 3.6-B).  So stage B applies masks
HARVESTED from real fullgrid files: per-level valid-fraction fields on the
label grid, stored as one compressed npz.

With the single sample day currently on disk the bank has size 1 -- the
machinery is validated now and the bank simply grows as Zach's 2016-2021
fullgrid pull lands (rebuild with ``harvest``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, ingest_hysplit as ih

BANK_PATH = config.DATA_ROOT / "masks" / "gap_bank.npz"
#: Minimum bank size before stage B trusts the real bank over the synthetic
#: swath+cloud generator (synth_gaps): fewer fields than this cannot span
#: seasons or swath geometries, and the model would memorize them
#: (dataset.make_vf_sampler, 2026-08-10).
MIN_REAL_BANK = 30


def harvest(fullgrid_paths, out_path=BANK_PATH, slot: int = 0) -> Path:
    """fullgrid files -> npz bank of per-level valid-fraction fields.

    Arrays: ``vf`` (n, lat 68, lon 141, lev 5) float16 and ``date``
    (n,) str -- dates kept so stage B can sample season-conditionally.
    """
    vfs, dates = [], []
    for p in fullgrid_paths:
        ch = ih.to_label_grid(ih.load_fullgrid(p), slot=slot)
        vf = ch["valid_fraction"].transpose("lat", "lon", "lev").values
        vfs.append(np.nan_to_num(vf).astype(np.float16))
        dates.append(str(ih.overpass_time(p).date()))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, vf=np.stack(vfs), date=np.array(dates))
    return out_path


def load_bank(path=BANK_PATH) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["vf"].astype(np.float32), z["date"]


def sample_mask(bank_vf: np.ndarray, rng: np.random.Generator,
                month: int | None = None,
                dates: np.ndarray | None = None) -> np.ndarray:
    """Draw one real (lat, lon, lev) valid-fraction field from the bank.

    If ``month`` and ``dates`` are given, prefer masks within +-1 month
    (seasonal swath/cloud statistics); falls back to the whole bank when the
    season has no entries (inevitable while the bank is small).
    """
    idx = np.arange(len(bank_vf))
    if month is not None and dates is not None:
        months = np.array([int(d[5:7]) for d in dates])
        near = idx[np.minimum(np.abs(months - month),
                              12 - np.abs(months - month)) <= 1]
        if len(near):
            idx = near
    return bank_vf[rng.choice(idx)]


def apply_mask(x: np.ndarray, vf: np.ndarray,
               impute_value: float) -> np.ndarray:
    """Impose a real gap field on a clean pretraining input (72,144,5,C).

    Channels where the (unpadded) valid fraction < OBSERVED_MIN_FRACTION are
    imputed; the trailing mask channel becomes the graded valid fraction --
    exactly the semantics of dataset.airs_x for real AIRS inputs.
    """
    from .dataset import _pad

    vf_p = _pad(vf)                                   # (72, 144, 5)
    out = x.copy()
    invalid = vf_p < ih.OBSERVED_MIN_FRACTION
    out[..., :-1] = np.where(invalid[..., None], impute_value, out[..., :-1])
    out[..., -1] = vf_p
    return out
