"""Trajectory-uncertainty fuzzing: deposit each along-track point as a 2-D
Gaussian whose width grows with distance travelled.

Stohl (1998) puts trajectory position error at ~20% of the distance travelled,
growing ~linearly with age, so kernels are sharp near the receptor and broad far
upstream. ``FuzzKernel`` is the pluggable interface
(``sigma_km(distance_km) -> km``); ``config.FUZZINESS`` scales the growth.
"""

from __future__ import annotations

import numpy as np

from . import config, geo


class FuzzKernel:
    """Base class: ``sigma_km(distance_km) -> spread in km``."""

    def sigma_km(self, distance_km) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class StohlFuzz(FuzzKernel):
    """``sigma = sigma0 + fuzziness * alpha * distance`` (Stohl 1998 20%-rule)."""

    def __init__(
        self,
        sigma0_km: float = config.FUZZ_SIGMA0_KM,
        alpha: float = config.FUZZ_ALPHA,
        fuzziness: float = config.FUZZINESS,
    ):
        self.sigma0_km = float(sigma0_km)
        self.alpha = float(alpha)
        self.fuzziness = float(fuzziness)

    def sigma_km(self, distance_km) -> np.ndarray:
        d = np.asarray(distance_km, dtype=float)
        return self.sigma0_km + self.fuzziness * self.alpha * d


class EmpiricalFuzz(StohlFuzz):
    """Same linear growth law, but ``alpha`` measured from this day's data.

    The fullgrid file stores, per 1-deg box, the spread (``u_std``, ``v_std``) of
    the parcel winds it aggregates. Over well-populated low-level boxes,

        alpha = <sqrt(u_std^2 + v_std^2)> / <sqrt(u^2 + v^2)>

    is the (dimensionless) velocity dispersion per unit travel speed, i.e. a
    position-error growth rate per unit distance -- the data-driven analog of
    Stohl (1998)'s 20%-of-distance rule. Construct with :meth:`from_fullgrid`.
    """

    @classmethod
    def from_fullgrid(
        cls,
        path=None,
        min_pres_hpa: float = 850.0,
        min_parcels: int = 5,
        sigma0_km: float = config.FUZZ_SIGMA0_KM,
        fuzziness: float = config.FUZZINESS,
    ) -> "EmpiricalFuzz":
        """Measure ``alpha`` from the fullgrid's low-level wind spread.

        Uses boxes at pressures >= ``min_pres_hpa`` (the layer the near-surface
        parcels fly in) with at least ``min_parcels`` aggregated parcels.
        """
        import xarray as xr  # local import: fuzz stays light for synthetic use

        if path is None:
            path = config.TRAJ_DIR / config.FULLGRID_NAME
        with xr.open_dataset(path) as ds:
            low = ds.sel(level=slice(min_pres_hpa, None))
            ok = (low["N"].values >= min_parcels)
            u, v = low["u"].values, low["v"].values
            us, vs = low["u_std"].values, low["v_std"].values
            ok &= np.isfinite(u) & np.isfinite(v) & np.isfinite(us) & np.isfinite(vs)
            if not ok.any():
                raise ValueError(f"no populated low-level boxes in {path}")
            speed = np.sqrt(u[ok] ** 2 + v[ok] ** 2)
            spread = np.sqrt(us[ok] ** 2 + vs[ok] ** 2)
            alpha = float(spread.mean() / speed.mean())
        return cls(sigma0_km=sigma0_km, alpha=alpha, fuzziness=fuzziness)


def deposit_gaussian(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    sigma_km: float,
    weight: float,
    accumulator: np.ndarray,
    n_sigma: float = 4.0,
) -> np.ndarray:
    """Add a mass-conserving 2-D Gaussian of total mass ``weight`` to ``accumulator``.

    ``grid_lat`` / ``grid_lon`` are 1-D ascending cell-centre axes; ``accumulator``
    is ``(len(grid_lat), len(grid_lon))``. The Gaussian is isotropic in km (so
    anisotropic in degrees: meridians converge with latitude) and is normalized on
    the *discrete grid* so the deposited mass equals ``weight`` exactly regardless
    of ``sigma`` -- the property the footprint's mass-conservation test relies on.
    A ``sigma`` at/below zero (or below half a cell) collapses to the nearest cell.
    Cells beyond ``n_sigma`` are skipped for speed.
    """
    if not (np.isfinite(center_lat) and np.isfinite(center_lon) and np.isfinite(weight)):
        return accumulator
    if weight == 0.0:
        return accumulator

    dlat_km = geo.km_per_deg_lat()
    dlon_km = geo.km_per_deg_lon(center_lat)

    # nearest-cell fallback for a (near-)delta
    if grid_lat.size > 1:
        cell_lat_km = abs(grid_lat[1] - grid_lat[0]) * dlat_km
    else:
        cell_lat_km = np.inf
    if grid_lon.size > 1:
        cell_lon_km = abs(grid_lon[1] - grid_lon[0]) * dlon_km
    else:
        cell_lon_km = np.inf
    cell_km = min(cell_lat_km, cell_lon_km)
    if sigma_km <= 0.0 or sigma_km < 0.5 * cell_km:
        i = int(np.argmin(np.abs(grid_lat - center_lat)))
        j = int(np.argmin(np.abs(grid_lon - center_lon)))
        accumulator[i, j] += weight
        return accumulator

    sig_lat_deg = sigma_km / dlat_km
    sig_lon_deg = sigma_km / dlon_km

    lat_lo = center_lat - n_sigma * sig_lat_deg
    lat_hi = center_lat + n_sigma * sig_lat_deg
    lon_lo = center_lon - n_sigma * sig_lon_deg
    lon_hi = center_lon + n_sigma * sig_lon_deg
    ii = np.where((grid_lat >= lat_lo) & (grid_lat <= lat_hi))[0]
    jj = np.where((grid_lon >= lon_lo) & (grid_lon <= lon_hi))[0]
    if ii.size == 0 or jj.size == 0:
        return accumulator

    gauss_lat = np.exp(-0.5 * ((grid_lat[ii] - center_lat) / sig_lat_deg) ** 2)
    gauss_lon = np.exp(-0.5 * ((grid_lon[jj] - center_lon) / sig_lon_deg) ** 2)
    block = np.outer(gauss_lat, gauss_lon)
    total = block.sum()
    if total <= 0.0:
        return accumulator
    accumulator[np.ix_(ii, jj)] += weight * block / total  # discrete normalization
    return accumulator
