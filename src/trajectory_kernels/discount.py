"""Optional Sodemann-style rain-out discount.

Sodemann, Schwierz & Wernli (2008, JGR, doi:10.1029/2007JD008503) discount every
prior uptake proportionally at each en-route precipitation event. Here ``q`` is
a conserved tracer reduced *only* by condensation (removal logged in
``q_excess``), so the proportional discounting telescopes exactly:

    w_disc(t) = q(t_arrival) / q(t)     in (0, 1]; == 1 where nothing condensed.
"""

from __future__ import annotations

import numpy as np


def condensation_discount(q: np.ndarray) -> np.ndarray:
    """Retained-humidity weight ``q[-1] / q(t)`` per along-track point.

    ``q`` is one parcel's specific humidity in time order, arrival last. Weights
    are in [0, 1]: < 1 upstream of condensation. Non-finite or non-positive ``q``
    (including at arrival) yields weight 1, so the discount degrades to a no-op
    rather than zeroing contributions; clipping at 1 keeps numerical noise (or
    genuine moistening) from ever amplifying a contribution.
    """
    q = np.asarray(q, dtype=float)
    if q.size == 0:
        return np.ones(0)
    q_arrival = q[-1]
    ok = np.isfinite(q) & (q > 0.0) & np.isfinite(q_arrival) & (q_arrival > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(ok, q_arrival / q, 1.0)
    return np.clip(w, 0.0, 1.0)
