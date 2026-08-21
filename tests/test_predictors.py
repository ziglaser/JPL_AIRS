"""Tests for the predictor layer (psi / phi / m* / omega / build_features).

Everything runs on a hand-built 2-receptor x 2-arrival-step kernel dataset with
known weights, so each feature's arithmetic can be checked against a value
computed by hand. No real data or JPL_AIRS_DATA needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from trajectory_kernels import config
from trajectory_kernels import predictors as P
from trajectory_kernels.pbl import PBLModel, _utc_fractional_hour

# ---------------------------------------------------------------------------
# Fixture geometry: 1 target lat x 2 target lons = 2 receptors, 2 arrival
# steps, 2 lag hours, 3x3 relative source window at 1-degree offsets.
# ---------------------------------------------------------------------------
TLAT = np.array([40.5])
TLON = np.array([-90.5, -89.5])
LAG = np.array([0.0, 1.0])
DREL = np.array([-1.0, 0.0, 1.0])
ARRIVALS = ["2019-06-05T21:00:00", "2019-06-05T23:00:00"]
FP_FILL = 0.5  # W/m2-weighted hours per source cell (uniform, easy to sum)
N_PARCELS = 25  # arriving parcels per receptor (phi divides by this)


@pytest.fixture
def kernel_ds() -> xr.Dataset:
    """Minimal multi-receptor kernel dataset in the collect_kernels layout.

    Each populated (step, receptor, lag) kernel slice is uniform 1/9 over the
    3x3 window (sums to 1, as build_footprint guarantees); the footprint is a
    uniform FP_FILL per cell so phi is FP_FILL * n_cells * 3600 / n_parcels
    by hand (phi is the parcel-ensemble MEAN). lag_weight is uniform (equal
    hours), so psi's weighted mean reduces to the equal-hour form here; the
    dedicated 3:1 test below exercises unequal weights.
    """
    shape = (2, TLAT.size, TLON.size, LAG.size, DREL.size, DREL.size)
    kernel = np.full(shape, 1.0 / 9.0)
    footprint = np.full(shape, FP_FILL)
    lag_weight = np.full(shape[:4], 9.0 * FP_FILL, dtype="float32")
    n_parcels = np.full((2, TLAT.size, TLON.size), N_PARCELS, dtype=int)
    containment = np.ones((2, TLAT.size, TLON.size), dtype=int)
    return xr.Dataset(
        {
            "kernel": (("arrival_step", "target_lat", "target_lon",
                        "lag", "dlat", "dlon"), kernel),
            "footprint": (("arrival_step", "target_lat", "target_lon",
                           "lag", "dlat", "dlon"), footprint),
            "lag_weight": (("arrival_step", "target_lat", "target_lon", "lag"),
                           lag_weight),
            "n_parcels": (("arrival_step", "target_lat", "target_lon"), n_parcels),
            "containment_applied": (("arrival_step", "target_lat", "target_lon"),
                                    containment),
        },
        coords={"arrival_step": np.arange(2), "target_lat": TLAT,
                "target_lon": TLON, "lag": LAG, "dlat": DREL, "dlon": DREL},
        attrs={"energy_model": "ClearSkyAvailableEnergy",
               "arrival_times_utc": ARRIVALS,
               "kernel_containment_frac": 0.90,
               "containment_min_parcels": config.CONTAINMENT_MIN_PARCELS},
    )


def _surface(values_fn, lat_step=0.5):
    """A (lat, lon) surface generously covering every source cell."""
    lat = np.arange(35.0, 46.0 + 1e-9, lat_step)
    lon = np.arange(-96.0, -84.0 + 1e-9, lat_step)
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    return xr.DataArray(values_fn(glat, glon), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})


class RampPBL(PBLModel):
    """PBLH falling 100 m per UTC hour: max over any look-back window sits at
    the LARGEST lag, never at arrival -- so a test can tell max from arrival."""

    def depth(self, lat, lon, time_utc):
        hour = _utc_fractional_hour(time_utc)
        shape = np.broadcast(np.asarray(lat), np.asarray(lon), hour).shape
        return np.broadcast_to(3000.0 - 100.0 * hour, shape).astype(float)


# ---------------------------------------------------------------------------
# (1) psi renormalization guarantee
# ---------------------------------------------------------------------------
def test_psi_of_uniform_field_is_the_constant_despite_nan_gaps(kernel_ds):
    """A weighted MEAN of a constant field must return that constant even when
    NaN gaps remove kernel weight -- the renormalization guarantee."""
    def field(glat, glon):
        vals = np.full(glat.shape, 0.27)
        vals[(glat > 40.6) & (glon > -90.6)] = np.nan  # a NaN quadrant
        return vals

    out = P.psi(kernel_ds, _surface(field), min_coverage=0.0)
    assert out.dims == ("arrival_step", "target_lat", "target_lon")
    assert np.allclose(out.values, 0.27, atol=1e-12)


# ---------------------------------------------------------------------------
# (1b) psi hour-to-hour weighting via lag_weight (review 1.6)
# ---------------------------------------------------------------------------
def _lag_split_kernel_ds() -> xr.Dataset:
    """One receptor whose two lag hours sample DIFFERENT cells with DIFFERENT
    physical mass: lag 0 is a delta on the receptor's own cell with
    lag_weight 3, lag 1 a delta one cell west with lag_weight 1. A field
    differing between those cells separates the 3:1 weighted mean from the
    equal-hour (1:1) mean."""
    shape = (1, 1, 1, LAG.size, DREL.size, DREL.size)
    kernel = np.full(shape, np.nan)
    kernel[0, 0, 0, 0, 1, 1] = 1.0  # lag 0: own cell (dlat=0, dlon=0)
    kernel[0, 0, 0, 1, 1, 0] = 1.0  # lag 1: one cell west (dlon=-1)
    footprint = np.nan_to_num(kernel) * np.array([3.0, 1.0])[:, None, None]
    lag_weight = np.array([[[[3.0, 1.0]]]])
    return xr.Dataset(
        {
            "kernel": (("arrival_step", "target_lat", "target_lon",
                        "lag", "dlat", "dlon"), kernel),
            "footprint": (("arrival_step", "target_lat", "target_lon",
                           "lag", "dlat", "dlon"), footprint),
            "lag_weight": (("arrival_step", "target_lat", "target_lon", "lag"),
                           lag_weight),
            "n_parcels": (("arrival_step", "target_lat", "target_lon"),
                          np.array([[[5]]])),
        },
        coords={"arrival_step": [0], "target_lat": TLAT, "target_lon": [-90.5],
                "lag": LAG, "dlat": DREL, "dlon": DREL},
        attrs={"energy_model": "ClearSkyAvailableEnergy",
               "arrival_times_utc": ARRIVALS[:1]},
    )


def test_psi_weights_hours_by_lag_weight_not_equally():
    """REGRESSION (review critical): the per-lag kernel normalization must NOT
    equal-weight the hours -- psi restores the physical hour mass from
    lag_weight, giving sum(w S)/sum(w). Field: 0.4 on the lag-0 cell, 0.2 on
    the lag-1 cell; weights 3:1 -> (3*0.4 + 1*0.2)/4 = 0.35, not the
    equal-hour 0.30."""
    ds = _lag_split_kernel_ds()
    surf = _surface(lambda glat, glon: np.where(glon > -91.0, 0.4, 0.2))
    out = P.psi(ds, surf, min_coverage=0.0)
    assert np.allclose(out.values, 0.35, atol=1e-12)
    # sanity: dropping lag_weight recovers (with a warning) the 1:1 mean
    with pytest.warns(UserWarning, match="lag_weight"):
        legacy = P.psi(ds.drop_vars("lag_weight"), surf, min_coverage=0.0)
    assert np.allclose(legacy.values, 0.30, atol=1e-12)


def test_psi_with_lag_weight_keeps_uniform_field_invariant():
    """A weighted MEAN of a constant field must still return the constant with
    unequal lag weights -- the NaN-gap renormalization stays intact."""
    ds = _lag_split_kernel_ds()
    out = P.psi(ds, _surface(lambda glat, glon: np.full(glat.shape, 0.27)),
                min_coverage=0.0)
    assert np.allclose(out.values, 0.27, atol=1e-12)


# ---------------------------------------------------------------------------
# (2) psi min_coverage and the coverage honesty column
# ---------------------------------------------------------------------------
def test_psi_min_coverage_blanks_gappy_receptor_and_coverage_in_unit_interval(kernel_ds):
    """Receptor 2 (-89.5): all nine source cells NaN -> psi NaN regardless of
    threshold. Receptor 1 (-90.5): 3 of 9 uniform-weight cells valid ->
    coverage 1/3, below the 0.5 default -> NaN; passes with min_coverage=0.2."""
    def field(glat, glon):
        vals = np.full(glat.shape, 0.30)
        vals[glon > -91.2] = np.nan  # only the dlon=-1 column of receptor 1 survives
        return vals

    surf = _surface(field)
    out, cov = P.psi(kernel_ds, surf, min_coverage=0.5, return_coverage=True)
    assert np.isnan(out.sel(target_lon=-89.5)).all()
    assert np.isnan(out.sel(target_lon=-90.5)).all()
    assert np.allclose(cov.sel(target_lon=-90.5).values, 1.0 / 3.0, atol=1e-12)
    assert np.allclose(cov.sel(target_lon=-89.5).values, 0.0, atol=1e-12)
    finite = cov.values[np.isfinite(cov.values)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()

    relaxed = P.psi(kernel_ds, surf, min_coverage=0.2)
    assert np.allclose(relaxed.sel(target_lon=-90.5).values, 0.30, atol=1e-12)


# ---------------------------------------------------------------------------
# (3) phi arithmetic and energy-provenance guard
# ---------------------------------------------------------------------------
def test_phi_equals_footprint_sum_times_3600_per_parcel(kernel_ds):
    """phi is the parcel-ensemble MEAN: footprint sum x 3600 / n_parcels
    (the raw sum would scale with sample size, not physics)."""
    n_cells = LAG.size * DREL.size * DREL.size  # 2 lags x 3x3 window
    expected = FP_FILL * n_cells * 3600.0 / N_PARCELS
    out = P.phi(kernel_ds)
    assert out.attrs["units"] == "J m-2"
    assert np.allclose(out.values, expected)


def test_phi_invariant_under_parcel_count(kernel_ds):
    """REGRESSION (review critical): two receptors identical except that one
    was sampled with 16x the parcels -- so its superposed footprint is 16x
    larger -- must get EQUAL phi. Receptor 1 (-90.5): 1 parcel, footprint
    FP_FILL; receptor 2 (-89.5): 16 parcels, footprint 16*FP_FILL."""
    ds = kernel_ds.copy(deep=True)
    ds["n_parcels"].values[:] = [[[1, 16]]] * 2
    ds["footprint"].values[:, :, 1] *= 16.0
    out = P.phi(ds)
    one = out.sel(target_lon=-90.5).values
    sixteen = out.sel(target_lon=-89.5).values
    assert np.allclose(one, sixteen)
    n_cells = LAG.size * DREL.size * DREL.size
    assert np.allclose(one, FP_FILL * n_cells * 3600.0)  # the 1-parcel integral


def test_phi_rejects_uniform_energy_kernels_unless_told_otherwise(kernel_ds):
    """Under a uniform energy weight the footprint sum is contact HOURS, not
    J/m2 -- a silent unit change, so phi must refuse it by default."""
    bad = kernel_ds.copy()
    bad.attrs["energy_model"] = "UniformEnergy"
    with pytest.raises(ValueError, match="UniformEnergy"):
        P.phi(bad)
    # no energy provenance at all is equally refused
    anon = kernel_ds.copy()
    del anon.attrs["energy_model"]
    with pytest.raises(ValueError):
        P.phi(anon)
    # the deliberate escape hatch still computes
    out = P.phi(bad, check_energy=False)
    assert np.isfinite(out.values).all()


# ---------------------------------------------------------------------------
# (4) m* takes the MAX PBLH over the lag window (review 1.5)
# ---------------------------------------------------------------------------
def test_m_star_uses_window_max_not_arrival_pblh(kernel_ds):
    """Guards the 19-Aug dilution fix (UPWIND_INDEX_REVIEW.md 1.5): the evening
    PBL collapse detrains mass without re-concentrating the surface signal, so
    m* must use the DEEPEST layer over the look-back window. RampPBL loses
    100 m/h, so the window max sits at lag 1 h, above the arrival-time value."""
    out = P.m_star(kernel_ds, RampPBL())
    # step 0 arrives 21 UTC: depths {lag0: 900, lag1: 1000} -> max 1000
    # step 1 arrives 23 UTC: depths {lag0: 700, lag1: 800} -> max 800
    assert np.allclose(out.isel(arrival_step=0).values,
                       config.RHO_ML_KG_M3 * 1000.0)
    assert np.allclose(out.isel(arrival_step=1).values,
                       config.RHO_ML_KG_M3 * 800.0)
    # explicitly NOT the arrival-instant value
    assert not np.allclose(out.isel(arrival_step=0).values,
                           config.RHO_ML_KG_M3 * 900.0)


def test_m_star_requires_an_explicit_pbl_model(kernel_ds):
    with pytest.raises(TypeError, match="pbl_model"):
        P.m_star(kernel_ds, None)


# ---------------------------------------------------------------------------
# (5) omega = phi / m*
# ---------------------------------------------------------------------------
def test_omega_is_phi_over_m_star_elementwise(kernel_ds):
    pbl = RampPBL()
    out = P.omega(kernel_ds, pbl)
    expected = P.phi(kernel_ds) / P.m_star(kernel_ds, pbl)
    assert out.attrs["units"] == "J kg-1"
    assert np.allclose(out.values, expected.values)


# ---------------------------------------------------------------------------
# (6) endpoint_value samples the receptor cell
# ---------------------------------------------------------------------------
def test_endpoint_value_reads_the_receptor_cell(kernel_ds):
    surf = _surface(lambda glat, glon: glon)  # value == the cell's longitude
    out = P.endpoint_value(kernel_ds, surf)
    assert set(out.dims) == {"arrival_step", "target_lat", "target_lon"}
    assert np.allclose(out.sel(target_lon=-90.5).values, -90.5)
    assert np.allclose(out.sel(target_lon=-89.5).values, -89.5)


# ---------------------------------------------------------------------------
# (6b) kernel_shape: centroid offset, fetch distance, influence age
# ---------------------------------------------------------------------------
def test_kernel_shape_centroid_at_known_offset_cell():
    """Mass concentrated at known cells: lag 0 is a delta on the receptor's own
    cell (offset 0,0) with lag_weight 3, lag 1 a delta one cell west (dlon=-1)
    with lag_weight 1. The lag_weight x kernel centroid is therefore
    (0, (3*0 + 1*(-1))/4) = (0, -0.25) deg, the fetch is the haversine length
    of that offset at the receptor latitude, and the influence age is the
    3:1-weighted mean lag (3*0 + 1*1)/4 = 0.25 h."""
    from trajectory_kernels import geo

    ds = _lag_split_kernel_ds()
    out = P.kernel_shape(ds)
    assert set(out.data_vars) == {"upwind_dlat", "upwind_dlon", "upwind_km",
                                  "mean_lag_hours"}
    for name in out.data_vars:
        assert out[name].dims == ("arrival_step", "target_lat", "target_lon")
    assert np.allclose(out["upwind_dlat"].values, 0.0, atol=1e-12)
    assert np.allclose(out["upwind_dlon"].values, -0.25, atol=1e-12)
    assert np.allclose(out["mean_lag_hours"].values, 0.25, atol=1e-12)
    expected_km = geo.haversine_km(40.5, -90.5, 40.5, -90.75)
    assert np.allclose(out["upwind_km"].values, expected_km, atol=1e-9)
    assert expected_km == pytest.approx(21.1, abs=0.5)  # 0.25 deg lon at 40.5 N


def test_kernel_shape_symmetric_kernel_centroid_is_the_receptor(kernel_ds):
    """A uniform kernel over a symmetric window has its centroid ON the
    receptor: zero offset, zero fetch; equal lag weights -> mean lag 0.5 h."""
    out = P.kernel_shape(kernel_ds)
    assert np.allclose(out["upwind_dlat"].values, 0.0, atol=1e-12)
    assert np.allclose(out["upwind_dlon"].values, 0.0, atol=1e-12)
    assert np.allclose(out["upwind_km"].values, 0.0, atol=1e-6)
    assert np.allclose(out["mean_lag_hours"].values, 0.5, atol=1e-12)


def test_kernel_shape_nan_where_nothing_arrived(kernel_ds):
    """Receptors with n_parcels == 0 (or zero total weight) have no geometry
    to report: every kernel_shape variable is NaN there, finite elsewhere."""
    ds = kernel_ds.copy(deep=True)
    ds["n_parcels"].values[:, 0, 1] = 0
    out = P.kernel_shape(ds)
    for name in out.data_vars:
        assert np.isnan(out[name].sel(target_lon=-89.5).values).all(), name
        assert np.isfinite(out[name].sel(target_lon=-90.5).values).all(), name


def test_kernel_shape_falls_back_to_equal_lag_weights_with_warning():
    """Kernels predating lag_weight get the equal-per-populated-lag fallback
    (same contract as psi): the 3:1 centroid becomes the 1:1 one."""
    ds = _lag_split_kernel_ds().drop_vars("lag_weight")
    with pytest.warns(UserWarning, match="lag_weight"):
        out = P.kernel_shape(ds)
    assert np.allclose(out["upwind_dlon"].values, -0.5, atol=1e-12)
    assert np.allclose(out["mean_lag_hours"].values, 0.5, atol=1e-12)


# ---------------------------------------------------------------------------
# (7) build_features assembly and tier bookkeeping
# ---------------------------------------------------------------------------
def test_build_features_tiers_dims_and_optional_columns(kernel_ds):
    sm_anom = _surface(lambda glat, glon: np.full(glat.shape, 0.02))
    sm_raw = _surface(lambda glat, glon: np.full(glat.shape, 0.25))

    full = P.build_features(kernel_ds, sm_anom, sm_raw=sm_raw, pbl_model=RampPBL())
    expected_dims = ("arrival_step", "target_lat", "target_lon")
    core = {"psi_anom", "omega"}
    ablation = {"phi", "m_star", "psi_raw", "s_endpoint_raw"}
    honesty = {"coverage", "n_parcels", "s_endpoint_anom", "psi_meso_anom",
               "containment_applied"}
    assert set(full.data_vars) == core | ablation | honesty
    for name in full.data_vars:
        assert full[name].dims == expected_dims, name
        tier = full[name].attrs["feature_tier"]
        assert tier == ("core" if name in core else
                        "ablation" if name in ablation else "honesty")
    assert full.attrs["energy_model"] == "ClearSkyAvailableEnergy"
    assert "feature_tiers" in full.attrs

    no_raw = P.build_features(kernel_ds, sm_anom, pbl_model=RampPBL())
    assert "psi_raw" not in no_raw and "s_endpoint_raw" not in no_raw

    no_pbl = P.build_features(kernel_ds, sm_anom, sm_raw=sm_raw)
    for name in ("omega", "m_star", "phi"):
        assert name not in no_pbl
    assert "no pbl_model" in no_pbl.attrs["pbl_note"]  # the omission is explained
    assert no_pbl.attrs["pbl_model"] == "none"


# ---------------------------------------------------------------------------
# (8) psi_mesoscale vanishes for a large-scale-only field
# ---------------------------------------------------------------------------
def test_psi_mesoscale_near_zero_for_smooth_field(kernel_ds):
    """A broad linear gradient carries no mesoscale structure: subtracting the
    3-degree-smoothed convolution leaves ~0 (only the half-cell asymmetry of
    the even rolling window survives, an order of magnitude below the field's
    variation across one source window)."""
    smooth_field = _surface(lambda glat, glon: 0.30 + 0.01 * glon)
    out = P.psi_mesoscale(kernel_ds, smooth_field, min_coverage=0.0)
    assert np.all(np.abs(out.values) < 5e-3)  # vs 0.02 m3/m3 across the window
    # and a callable is refused: smoothing needs a grid to roll over
    with pytest.raises(TypeError, match="DataArray"):
        P.psi_mesoscale(kernel_ds, lambda glat, glon: glon)
