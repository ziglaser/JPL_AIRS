"""Kernels + surface fields -> the upwind feature table.

Thin layer on top of :mod:`apply`, implementing UPWIND_INDEX_REVIEW.md
sections 1.5-1.9. The two core features per (arrival_step, target cell):

    Psi[S] [units of S]  DIRECTION. The energy- and contact-weighted mean of a
                         soil field S along the air's coupled path,
                         sum(w_k S_k) / sum(w_k) with w_k the per-hour energy
                         weight (review 1.6). The kernel is per-lag
                         normalized, so ``psi`` multiplies the hour-to-hour
                         mass back in from the kernel dataset's ``lag_weight``
                         variable; kernels predating that variable fall back
                         (with a warning) to the equal-hour mean. Fed the
                         cardinal monthly anomaly S' = SM - monthly baseline
                         (m3/m3), it answers "was the soil the air touched
                         anomalously wet or dry?"
    Omega  [J/kg]        INTENSITY. Phi / m*: available surface energy
                         delivered along the coupled path, per kilogram of
                         the arriving mixed-layer column.

with the extensive building blocks retained as ablation diagnostics only:

    Phi    [J/m2]        parcel-ensemble mean of sum of the per-step weights
                         w_k = a*DSWF*contact*land*dt -- the energy the
                         surface COULD deliver to one arriving column,
                         carrying no soil information. The mean over the
                         arriving parcels (never the raw sum, which scales
                         with parcel count -- sample size, not physics).
    m*     [kg/m2]       rho * max-over-path PBLH: the mass the surface input
                         is stirred through by arrival (divide once by the
                         maximum, never per step -- entrainment dilution and
                         the instantaneous 1/m_k weighting cancel; a
                         well-motivated approximation, not exact, since a
                         moving parcel also crosses columns of differing
                         depth).

Why Omega rather than the pair (Phi, m*) (review 1.5, superseding the D:
draft's argument): m* has exactly two predictor roles -- a geometric one
(deep PBL top nearer the LFC), which Gamma_gap = z_LFC - z_i expresses
better because it references the PBL top to the threshold that matters, and
a dilution one, which is only meaningful as the denominator of Phi. With
Gamma_gap in the feature set (computed trajectory-free downstream), Omega's
1/PBLH dependence is physics -- the surface imprint genuinely is more dilute
in a deeper layer -- not cancelled signal. Phi's independent variance is
mostly (latitude, slot, month) plus coverage-geometry artifacts, so it is a
diagnostic, not a feature.

Why Psi and Omega stay two features rather than one index (review 1.9): the
physical statement is "Omega J/kg were delivered, split between warming and
moistening with evaporative fraction set by the soil wetness Psi". Combining
them commits to an EF(SM) curve -- the least-trusted relation in
land-atmosphere physics -- and any product collapses opposite cases
(extreme-dry x weak delivery vs mild x strong delivery) onto one value. The
forest constructs the interaction itself from the two axes.

Why the soil input is the CARDINAL MONTHLY ANOMALY (review 1.7): the
partition physics runs on absolute wetness, but raw m3/m3 confounds the
standing east-wet/west-dry geography (under region-blocked CV, a map of
CONUS relearned). S' = SM - per-cell monthly-mean baseline removes the
standing map while staying cardinal -- an m3/m3 departure is linear in
water, the same currency as the latent flux it modulates, unlike the
percentile form which flattens departures into ranks. It does NOT
texture-normalize; that job is deferred to a soil-texture wetness index.
Because the kernel contraction is linear, convolving S' equals convolving
raw SM and subtracting the convolved baseline -- the spatially varying
reference is handled for free.

The anomaly-after-convolution identity, for scalar references:

    <K, S - c> / <K, 1> = Psi[S] - c

so any SCALAR reference commutes with the convolution; a spatially varying
reference must be convolved too (see ``anomalize``).
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from . import config, geo
from .apply import apply_kernel
from .pbl import PBLModel

_SEC_PER_HOUR = 3600.0


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #
def psi(kernel_ds: xr.Dataset, surface, min_coverage: float = 0.5,
        return_coverage: bool = False, **kw):
    """Contact- and energy-weighted mean of ``surface`` over the source region.

    Units are those of ``surface`` (m3/m3 for SMAP fields). Requires that the
    footprint was built with an energy weight (see ``insolation.py``); with a
    uniform weight this is the contact-hour-weighted average instead.

    The kernel is normalized PER LAG HOUR, so the hour-to-hour energy weight
    must be multiplied back in: ``psi`` passes the kernel dataset's
    ``lag_weight`` variable (the physical mass of each lag hour) to
    ``apply_kernel``, giving exactly Psi = sum(w_k S_k) / sum(w_k) (review
    1.6). Kernels built before ``lag_weight`` existed fall back to the
    equal-hour mean of populated lags, with a warning -- rebuild them to get
    the weighted form.

    ``min_coverage=0.5``: SMAP is roughly half NaN on this domain, so receptors
    whose retained kernel weight falls below half are set to NaN rather than
    reported from a small unrepresentative corner of the source region.
    ``return_coverage=True`` also returns that retained-weight fraction, which
    ``build_features`` exports as an honesty column: two receptors with
    identical kernels but different SMAP gap patterns average over different
    effective source regions, and the forest should be able to see that.
    """
    if "lag_weight" in kernel_ds:
        kw.setdefault("lag_weights", kernel_ds["lag_weight"])
    else:
        warnings.warn(
            "kernel dataset has no 'lag_weight' variable (built before the "
            "hour-to-hour weighting fix): Psi falls back to the equal-hour "
            "mean over populated lags instead of sum(w S)/sum(w). Rebuild "
            "the kernels to restore the energy weighting.", stacklevel=2)
    return apply_kernel(kernel_ds, surface, which="kernel",
                        min_coverage=min_coverage,
                        return_coverage=return_coverage, **kw)


def anomalize(psi_field: xr.DataArray, reference) -> xr.DataArray:
    """Turn a mean into a signed anomaly: positive = anomalously wet path.

    ``reference`` is either a scalar or a field already convolved through the
    same kernels -- a spatially varying reference does NOT commute with the
    convolution unless it is convolved too. For a scalar this is exact by
    linearity. In production the cardinal monthly anomaly is formed BEFORE the
    convolution (S' = SM - monthly baseline, review 1.7), which by the same
    linearity is equivalent and handles the varying reference for free.
    """
    if np.isscalar(reference):
        return psi_field - float(reference)
    return psi_field - reference


def standardize(psi_field: xr.DataArray) -> xr.DataArray:
    """ABLATION/DIAGNOSTIC ONLY - on any one day this is dominated by the
    standing east-west soil-moisture gradient (a longitude proxy); superseded
    by the cardinal monthly anomaly (UPWIND_INDEX_REVIEW.md 1.7).

    Domain z-score of a Psi field, roughly in [-2, +2], sign as in
    ``anomalize``. Kept only so single-day case studies without a monthly
    baseline can still be normalized for plotting.
    """
    mean = float(psi_field.mean(skipna=True))
    std = float(psi_field.std(skipna=True))
    if not np.isfinite(std) or std == 0.0:
        return psi_field * np.nan
    out = (psi_field - mean) / std
    out.name = "psi_standardized"
    out.attrs["long_name"] = "domain-standardised soil-moisture influence direction"
    return out


# --------------------------------------------------------------------------- #
# Intensity: Phi, m*, and their ratio Omega
# --------------------------------------------------------------------------- #
def phi(kernel_ds: xr.Dataset, check_energy: bool = True) -> xr.DataArray:
    """Available surface energy delivered along the coupled path, J/m2.

    Sums the physical (untruncated) footprint over lag and source cells and
    DIVIDES BY ``n_parcels``: the multi-parcel footprint is the superposition
    of every arriving parcel's path integral, so the raw sum scales linearly
    with how many parcels HYSPLIT happened to release into the receptor --
    sample size, not physics (a 16-parcel receptor would get 16x the phi of an
    identical 1-parcel one). The parcel-ensemble MEAN is the Monte Carlo
    estimate of the single-column path integral of review 1.5, invariant under
    parcel count. Receptors with ``n_parcels == 0`` are NaN.

    ASSUMES the footprint was built with ``energy_fn=ClearSkyAvailableEnergy``,
    so its per-lag values are W/m2 x hours and the factor 3600 gives J/m2.
    ``kernel_ds.attrs["energy_model"]`` is checked and a ValueError raised if
    the kernels were built with ``UniformEnergy`` (or carry no energy
    provenance at all): under a uniform weight this sum is land-contact hours,
    and a silent unit change is worse than an error. With
    ``check_energy=False`` (only to inspect contact deliberately) the return
    is 3600 x contact-hours per parcel, i.e. contact-seconds per parcel --
    not hours.

    Reference value: 4.5 MJ/m2 for a parcel with ~3 h of clear-sky afternoon
    contact (0.55 x 750 W/m2 x 10800 s) -- per parcel, so the same whether 1
    or 16 such parcels arrive. Clear-sky is an upper bound that fails most
    on pre-convective days, so Phi (and Omega's numerator) are trustworthy in
    rank more than in level.
    """
    if check_energy:
        model = kernel_ds.attrs.get("energy_model")
        if model is None or "UniformEnergy" in str(model):
            raise ValueError(
                f"phi() needs kernels built with energy_fn=ClearSkyAvailableEnergy "
                f"(got energy_model={model!r}); the footprint sum would be contact "
                f"hours, not J/m2. Pass check_energy=False to get hours deliberately.")
    fp = kernel_ds["footprint"].sum(dim=("lag", "dlat", "dlon"), skipna=True)
    populated = kernel_ds["n_parcels"] > 0
    # parcel-ensemble mean: divide the superposed footprint by parcel count
    # so phi measures the path, not the sample size (masked to n_parcels > 0)
    out = (fp * _SEC_PER_HOUR) / kernel_ds["n_parcels"].where(populated)
    out.name = "phi"
    out.attrs.update({
        "units": "J m-2",
        "long_name": "available surface energy delivered along the coupled path",
    })
    return out


def m_star(kernel_ds: xr.Dataset, pbl_model: PBLModel,
           rho: float = config.RHO_ML_KG_M3) -> xr.DataArray:
    """Dilution column mass m* = rho * max PBLH over the look-back window, kg/m2.

    The MAXIMUM over the window, never the arrival-time value: the tracer
    dilution result (review 1.5) says surface-delivered energy ends up spread
    through the DEEPEST layer the air ever occupied. The evening PBL collapse
    detrains mass without changing the surviving concentration, so sampling
    PBLH at a 00-02 UTC arrival (nocturnal ~200 m) understates the reservoir
    and inflates Omega ~10x -- observed on the 2019-06-05 demo before this
    was corrected. The window is the kernel's own lag axis, and the parcel
    path is approximated by the receptor-cell column (paths displace only a
    cell or two over the horizon, broader than afternoon-PBLH gradients).

    ``pbl_model`` is REQUIRED: a climatological default is an information-free
    feature (a smooth function of local hour and longitude that a forest
    already seeing location and time reads as geography, review F2). Pass the
    assessed per-day PBL model (``pbl.GriddedPBL``) explicitly; that
    deliberately re-imports the day's soil-driven boundary-layer signal, which
    is the mechanism, not contamination -- but expect correlated importances
    with the SMAP-derived features and say so.

    Good to about 10% from ``rho`` (config.RHO_ML_KG_M3 = 1.10 kg/m3, mean
    mixed-layer density); the PBLH model is the larger uncertainty. Reference
    value: 1.10 x 1800 = 1980 kg/m2.
    """
    if pbl_model is None:
        raise TypeError(
            "m_star requires an explicit pbl_model: a climatological default is "
            "an information-free geography proxy (UPWIND_INDEX_REVIEW.md F2).")
    tlat = kernel_ds["target_lat"].values
    tlon = kernel_ds["target_lon"].values
    steps = kernel_ds["arrival_step"].values

    arrival_times = _arrival_times(kernel_ds)
    lag_hours = kernel_ds["lag"].values.astype(float)
    grid_lat, grid_lon = np.meshgrid(tlat, tlon, indexing="ij")
    out = np.full((steps.size, tlat.size, tlon.size), np.nan)
    for s in range(steps.size):
        deepest = np.full(grid_lat.shape, -np.inf)
        for h in lag_hours:
            when = np.full(grid_lat.shape,
                           arrival_times[s] - np.timedelta64(int(round(h * 3600)), "s"))
            depth = np.asarray(pbl_model(grid_lat, grid_lon, when), dtype=float)
            deepest = np.fmax(deepest, depth)  # fmax: NaN slots never win
        out[s] = rho * np.where(np.isfinite(deepest), deepest, np.nan)

    da = xr.DataArray(
        out, dims=("arrival_step", "target_lat", "target_lon"),
        coords={"arrival_step": kernel_ds["arrival_step"],
                "target_lat": tlat, "target_lon": tlon},
        name="m_star",
        attrs={"units": "kg m-2",
               "long_name": ("dilution column mass: rho * max PBLH over the "
                             "look-back window (deepest layer occupied)")},
    )
    return da.where(kernel_ds["n_parcels"] > 0)


def omega(kernel_ds: xr.Dataset, pbl_model: PBLModel) -> xr.DataArray:
    """Omega = Phi / m*: surviving surface energy per kg of arriving air, J/kg.

    THE core intensity feature (review 1.5). Worked example (per arriving
    parcel -- phi is the parcel-ensemble mean, so parcel count cancels): 3 h
    of full contact at path-mean clear-sky DSWF 750 W/m2 gives
    Phi = 0.55 x 750 x 10800 = 4.5 MJ/m2; stirred through a
    max-1800 m mixed layer, m* = 1.10 x 1800 = 1980 kg/m2, so
    Omega = 2250 J/kg -- about 2.2 K of warming or 0.9 g/kg of moistening if
    delivered purely one way, with the split set by the soil wetness that
    ``psi`` reads.

    The 1/PBLH dependence is physics, not cancelled signal, PROVIDED the
    geometric pathway is carried separately by Gamma_gap = z_LFC - z_i
    (computed trajectory-free in scripts/merge_upwind_features.py). Inherits
    phi's clear-sky upper-bound caveat and m_star's requirement for an
    assessed ``pbl_model``.
    """
    out = phi(kernel_ds) / m_star(kernel_ds, pbl_model)
    out.name = "omega"
    out.attrs.update({
        "units": "J kg-1",
        "long_name": ("available surface energy per unit arriving mixed-layer "
                      "mass, phi / m_star"),
    })
    return out


def _arrival_times(kernel_ds: xr.Dataset) -> np.ndarray:
    """Arrival datetime per arrival_step, from attrs when present.

    Falls back to the 2019-06-05 demo release clock (hourly from 19 UTC) only
    with an explicit warning: at production scale the wrong clock silently
    corrupts every diurnal quantity downstream.
    """
    stamp = kernel_ds.attrs.get("arrival_times_utc")
    if stamp is not None:
        return np.asarray(stamp, dtype="datetime64[s]")
    warnings.warn(
        "kernel_ds carries no attrs['arrival_times_utc']; assuming the "
        "2019-06-05 demo clock (hourly from 19 UTC). Set the attr for any "
        "other day.", stacklevel=3)
    base = np.datetime64("2019-06-05T19:00:00")
    return base + kernel_ds["arrival_step"].values.astype("timedelta64[h]")


# --------------------------------------------------------------------------- #
# Diagnostics that decide whether any of this was worth doing
# --------------------------------------------------------------------------- #
def endpoint_value(kernel_ds: xr.Dataset, surface) -> xr.DataArray:
    """The naive point predictor: ``surface`` sampled at the receptor cell.

    Include this in the feature table deliberately. Its importance relative to
    Psi answers, directly and without argument, whether the Lagrangian
    accumulation beats reading the soil moisture under the cell.
    """
    from .apply import lookup_from_dataarray
    fn = surface if callable(surface) else lookup_from_dataarray(surface)
    tlat = kernel_ds["target_lat"].values
    tlon = kernel_ds["target_lon"].values
    grid_lat, grid_lon = np.meshgrid(tlat, tlon, indexing="ij")
    vals = fn(grid_lat, grid_lon)
    da = xr.DataArray(vals, dims=("target_lat", "target_lon"),
                      coords={"target_lat": tlat, "target_lon": tlon},
                      name="s_endpoint")
    return da.broadcast_like(kernel_ds["n_parcels"]).where(kernel_ds["n_parcels"] > 0)


def marginal_value(psi_field: xr.DataArray, endpoint: xr.DataArray) -> float:
    """Pearson r between the accumulated index and the naive point value.

    The negative control. If r > 0.95 the Lagrangian accumulation is adding
    nothing over a point predictor, and that is a result worth reporting rather
    than discovering in review. Expected: lower for surface (0-5 cm) soil
    moisture, which decorrelates over 20-50 km, than for root-zone, which
    decorrelates over 100-200 km and so is nearly path-invariant.
    """
    a = np.asarray(psi_field.values, dtype=float).ravel()
    b = np.asarray(endpoint.values, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def psi_mesoscale(kernel_ds: xr.Dataset, surface, smooth_deg: float = 3.0,
                  **kw) -> xr.DataArray:
    """Isolate the heterogeneity pathway: ``Psi[S] - Psi[<S>_smooth]``.

    The smoothed term carries the mean-state (thermodynamic) pathway; the
    residual carries the mesoscale gradient that drives solenoidal circulations.
    Costs one extra convolution because ``apply`` is field-agnostic --
    essentially free given the kernels already exist. ``smooth_deg=3`` puts the
    cut near the synoptic scale and above the ~140 km soil-moisture/rain
    coupling length of Guillod+2015.
    """
    if callable(surface):
        raise TypeError("psi_mesoscale needs a gridded DataArray, not a callable")
    n = max(int(round(smooth_deg / float(abs(surface["lat"][1] - surface["lat"][0])))), 1)
    smooth = (surface.rolling(lat=n, center=True, min_periods=1).mean()
                     .rolling(lon=n, center=True, min_periods=1).mean())
    out = psi(kernel_ds, surface, **kw) - psi(kernel_ds, smooth, **kw)
    out.name = "psi_mesoscale"
    out.attrs["long_name"] = (
        f"mesoscale component of the direction index (S minus {smooth_deg} deg mean)")
    return out


# --------------------------------------------------------------------------- #
# Kernel geometry: where the influence sits and how old it is (QA diagnostics)
# --------------------------------------------------------------------------- #
def kernel_shape(kernel_ds: xr.Dataset) -> xr.Dataset:
    """Geometry of the influence kernel, on (arrival_step, target_lat, target_lon).

    Purely kernel-derived (no surface field enters), so these must be exported
    by the per-day builder BEFORE the kernels are discarded -- they cannot be
    recovered from the feature table afterwards. Variables:

        upwind_dlat, upwind_dlon [deg]  the influence-centroid offset from the
                                        receptor: the weighted mean of the
                                        relative ``dlat``/``dlon`` coordinates
                                        under w = lag_weight * kernel over
                                        (lag, dlat, dlon). The kernel is
                                        normalized PER LAG, so ``lag_weight``
                                        restores the physical hour-to-hour
                                        mass -- exactly the weighting ``psi``
                                        applies to the soil field (review 1.6).
        upwind_km                [km]   haversine length of that centroid
                                        offset evaluated at the receptor
                                        latitude (``geo.haversine_km``): the
                                        effective fetch of the source region.
        mean_lag_hours           [h]    the lag_weight-weighted mean lag: the
                                        age of the influence reaching the
                                        receptor (HANDOFF 7.7 -- nearly free
                                        once the kernels exist; here it is).

    Physics the QA battery checks against (UPWIND_INDEX_REVIEW.md QA check 2):
    the air arrives FROM upwind, so the centroid must sit on the upstream side
    of the receptor -- dotting (upwind_dlon, upwind_dlat) against the day's
    low-level wind vector should give a NEGATIVE (upstream) projection over
    almost all receptors; a positive projection means the trajectory clock or
    the offset sign convention is broken. Consistency scales: with ~10 m/s
    low-level flow and mean_lag_hours ~ 3 h, upwind_km should sit near
    100 km -- upwind_km >> wind x lag flags a kernel/geometry bug.

    NaN policy: every variable is NaN where ``n_parcels == 0`` or the total
    kernel weight is zero (nothing arrived -- there is no geometry to report).
    Kernels predating ``lag_weight`` fall back (with the same warning as
    ``psi``) to equal weight per populated lag.
    """
    kernel = np.nan_to_num(
        kernel_ds["kernel"].transpose("arrival_step", "target_lat", "target_lon",
                                      "lag", "dlat", "dlon").values.astype(float),
        nan=0.0)  # (step, tlat, tlon, lag, dlat, dlon)
    if "lag_weight" in kernel_ds:
        lw = np.nan_to_num(
            kernel_ds["lag_weight"].transpose(
                "arrival_step", "target_lat", "target_lon", "lag"
            ).values.astype(float), nan=0.0)
    else:
        warnings.warn(
            "kernel dataset has no 'lag_weight' variable (built before the "
            "hour-to-hour weighting fix): kernel_shape falls back to equal "
            "weight per populated lag. Rebuild the kernels to restore the "
            "energy weighting.", stacklevel=2)
        lw = (kernel.sum(axis=(-2, -1)) > 0).astype(float)

    dlat = kernel_ds["dlat"].values.astype(float)
    dlon = kernel_ds["dlon"].values.astype(float)
    lag = kernel_ds["lag"].values.astype(float)

    w = kernel * lw[..., None, None]           # physical mass per source cell
    total = w.sum(axis=(3, 4, 5))              # (step, tlat, tlon)
    lag_total = lw.sum(axis=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen_dlat = (w * dlat[None, None, None, None, :, None]).sum(axis=(3, 4, 5)) / total
        cen_dlon = (w * dlon[None, None, None, None, None, :]).sum(axis=(3, 4, 5)) / total
        mean_lag = (lw * lag[None, None, None, :]).sum(axis=3) / lag_total

    tlat = kernel_ds["target_lat"].values.astype(float)
    tlon = kernel_ds["target_lon"].values.astype(float)
    grid_lat, grid_lon = np.meshgrid(tlat, tlon, indexing="ij")
    km = geo.haversine_km(grid_lat, grid_lon,
                          grid_lat + cen_dlat, grid_lon + cen_dlon)

    valid = (kernel_ds["n_parcels"].values > 0) & (total > 0) & (lag_total > 0)
    coords = {"arrival_step": kernel_ds["arrival_step"],
              "target_lat": tlat, "target_lon": tlon}
    dims = ("arrival_step", "target_lat", "target_lon")

    def _da(vals, name, units, long_name):
        return xr.DataArray(np.where(valid, vals, np.nan), dims=dims,
                            coords=coords, name=name,
                            attrs={"units": units, "long_name": long_name})

    return xr.Dataset({
        "upwind_dlat": _da(cen_dlat, "upwind_dlat", "degrees",
                           "influence-centroid latitude offset from the "
                           "receptor (lag_weight x kernel weighted)"),
        "upwind_dlon": _da(cen_dlon, "upwind_dlon", "degrees",
                           "influence-centroid longitude offset from the "
                           "receptor (lag_weight x kernel weighted)"),
        "upwind_km": _da(km, "upwind_km", "km",
                         "haversine length of the influence-centroid offset "
                         "at the receptor latitude (effective fetch)"),
        "mean_lag_hours": _da(mean_lag, "mean_lag_hours", "h",
                              "lag_weight-weighted mean lag: age of the "
                              "influence reaching the receptor"),
    })


# --------------------------------------------------------------------------- #
# One call for the whole feature table
# --------------------------------------------------------------------------- #
def build_features(kernel_ds: xr.Dataset, sm_anom, sm_raw=None,
                   pbl_model: PBLModel | None = None) -> xr.Dataset:
    """Assemble the per-day feature table on (arrival_step, target_lat, target_lon).

    ``sm_anom`` is the cardinal monthly anomaly S' = SM - per-cell monthly
    baseline, m3/m3 (review 1.7) -- formed BEFORE the call; the convolution's
    linearity makes that equivalent to anomalizing after. ``sm_raw`` (raw SMAP
    m3/m3), if given, adds the ablation-tier absolute-level columns whose
    region-blocked decision rule is stated in review 1.7. ``pbl_model`` should
    be the assessed per-day model; if None, ``omega``/``m_star``/``phi`` are
    omitted (never silently computed from a climatology, review F2) and the
    reason recorded in attrs.

    Tiers (recorded per variable as attrs["feature_tier"] and in the dataset
    attrs): core = predictors carrying a distinct physical axis; ablation =
    columns kept to verify their importance is ~0 or to run the psi_raw
    decision rule; honesty = sampling/validity columns the forest (and the
    reader) should see. Gamma_gap and pblh_anom are trajectory-free and are
    computed in scripts/merge_upwind_features.py, not here.
    """
    psi_anom, coverage = psi(kernel_ds, sm_anom, return_coverage=True)
    psi_anom.name = "psi_anom"
    psi_anom.attrs["units"] = "m3 m-3"
    psi_anom.attrs["long_name"] = (
        "energy-weighted path-mean soil-moisture anomaly vs per-cell monthly baseline")

    tiers = {"psi_anom": "core", "coverage": "honesty", "n_parcels": "honesty",
             "s_endpoint_anom": "honesty", "psi_meso_anom": "honesty"}
    out = {
        "psi_anom": psi_anom,
        "coverage": coverage,
        "n_parcels": kernel_ds["n_parcels"],
        "s_endpoint_anom": endpoint_value(kernel_ds, sm_anom).rename("s_endpoint_anom"),
        "psi_meso_anom": psi_mesoscale(kernel_ds, sm_anom).rename("psi_meso_anom"),
    }
    if "containment_applied" in kernel_ds:
        out["containment_applied"] = kernel_ds["containment_applied"]
        tiers["containment_applied"] = "honesty"

    pbl_note = None
    if pbl_model is not None:
        out["omega"] = omega(kernel_ds, pbl_model)
        out["phi"] = phi(kernel_ds)
        out["m_star"] = m_star(kernel_ds, pbl_model)
        tiers.update({"omega": "core", "phi": "ablation", "m_star": "ablation"})
    else:
        pbl_note = ("omega/m_star/phi omitted: no pbl_model was passed, and a "
                    "climatological default would be an information-free "
                    "geography proxy (UPWIND_INDEX_REVIEW.md F2).")

    if sm_raw is not None:
        out["psi_raw"] = psi(kernel_ds, sm_raw).rename("psi_raw")
        out["s_endpoint_raw"] = endpoint_value(kernel_ds, sm_raw).rename("s_endpoint_raw")
        tiers.update({"psi_raw": "ablation", "s_endpoint_raw": "ablation"})

    ds = xr.Dataset(out)
    for name, tier in tiers.items():
        ds[name].attrs["feature_tier"] = tier
    ds.attrs.update({
        "energy_model": str(kernel_ds.attrs.get("energy_model", "unknown")),
        "pbl_model": repr(pbl_model) if pbl_model is not None else "none",
        "feature_tiers": "; ".join(f"{k}={v}" for k, v in tiers.items()),
        "note": ("gamma_gap and pblh_anom are computed trajectory-free in "
                 "scripts/merge_upwind_features.py, not here."),
    })
    if pbl_note is not None:
        ds.attrs["pbl_note"] = pbl_note
    return ds
