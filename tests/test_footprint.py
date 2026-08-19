"""Tests for the footprint builder.

The core is exercised with synthetic straight-line parcels of known answer
(no real data needed); a final smoke test runs one real receptor.
"""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kernels import config
from trajectory_kernels.fuzz import StohlFuzz
from trajectory_kernels.pbl import ConstantPBL
from trajectory_kernels import footprint as F


class _DeltaFuzz(StohlFuzz):
    """sigma == 0 everywhere -> each point deposits on its nearest cell (a delta)."""

    def sigma_km(self, distance_km):
        return np.zeros_like(np.asarray(distance_km, dtype=float))


def _all_land(lat, lon):
    return np.ones_like(np.asarray(lat, dtype=float))


def _eastward_parcel(alt_m=200.0):
    """A parcel moving due east onto the receptor at (40.5, -90.5), hourly steps,
    over grid-cell centres and in the daytime (deep PBL, full contact)."""
    lons = np.array([-93.5, -92.5, -91.5, -90.5])  # west -> east (history -> arrival)
    lats = np.full(4, 40.5)
    times = np.array(
        ["2019-06-05T18:00", "2019-06-05T19:00", "2019-06-05T20:00", "2019-06-05T21:00"],
        dtype="datetime64[ns]",
    )
    return {
        "t_hours": np.array([0.0, 1.0, 2.0, 3.0]),
        "time_utc": times,
        "lat": lats, "lon": lons, "alt": np.full(4, alt_m),
    }


def _run(trajs, contact_fraction=config.CONTACT_FRACTION, step_min=60.0, alt=200.0):
    sl, so = F.source_window(40.5, -90.5)
    lag_hours = np.arange(0, 4, dtype=float)
    acc = F.footprint_from_trajectories(
        trajs, sl, so, lag_hours, ConstantPBL(2000.0), _DeltaFuzz(), _all_land,
        contact_fraction=contact_fraction, resample_step_min=step_min,
    )
    return acc, sl, so, lag_hours


def test_total_footprint_equals_contact_duration():
    """All-land, full-contact parcel: total footprint == its in-PBL time (3 h)."""
    acc, *_ = _run([_eastward_parcel()])
    assert acc.sum() == pytest.approx(3.0, abs=1e-6)


def test_lag0_mass_sits_on_receptor_cell():
    acc, sl, so, lag = _run([_eastward_parcel()])
    lag0 = acc[0]
    i, j = np.unravel_index(np.argmax(lag0), lag0.shape)
    half = config.SOURCE_STEP_DEG / 2.0  # arrival point sits on a cell corner
    assert abs(sl[i] - 40.5) <= half + 1e-9
    assert abs(so[j] - (-90.5)) <= half + 1e-9  # a cell touching the arrival point


def test_footprint_is_upwind_only():
    """A parcel arriving from the west leaves all footprint mass to the west."""
    acc, sl, so, lag = _run([_eastward_parcel()])
    total_by_cell = acc.sum(axis=0)
    half = config.SOURCE_STEP_DEG / 2.0
    nz_lat = sl[np.any(total_by_cell > 0, axis=1)]
    nz_lon = so[np.any(total_by_cell > 0, axis=0)]
    # the track runs along source-cell corners, so nearest-cell deposits can
    # land half a cell off the line but never further
    assert np.all(np.abs(nz_lat - 40.5) <= half + 1e-9)   # stays on the flow line
    assert np.all(nz_lon <= -90.5 + half + 1e-9)          # never east (downwind)


def test_source_window_nests_in_target_cells():
    """Quarter-degree source cells tile the 1-deg target cells exactly: the
    receptor cell's edges (target +/- 0.5) fall on source-cell edges."""
    sl, so = F.source_window(40.5, -90.5)
    step = config.SOURCE_STEP_DEG
    assert np.allclose(np.diff(sl), step)
    assert sl.size == so.size == int(2 * config.SOURCE_WINDOW_HALFWIDTH_DEG / step)
    edges = sl - step / 2.0
    assert np.allclose((edges - 40.0) % 1.0 % step, 0.0)   # edges hit x.0/x.25/...
    assert np.any(np.isclose(edges, 40.0))                 # receptor west edge exact
    assert np.allclose(sl - 40.5, so - (-90.5))            # same relative offsets


def test_lag_increases_westward():
    """Mass at larger lag sits further upwind (west)."""
    acc, sl, so, lag = _run([_eastward_parcel()])
    # centroid longitude per lag bin
    lon_centroid = []
    for k in range(acc.shape[0]):
        m = acc[k]
        if m.sum() == 0:
            lon_centroid.append(np.nan)
            continue
        lon_centroid.append((m.sum(axis=0) * so).sum() / m.sum())
    lon_centroid = np.array(lon_centroid)
    finite = np.isfinite(lon_centroid)
    assert np.all(np.diff(lon_centroid[finite]) < 0)  # west with increasing lag


def test_kernel_sums_to_one_within_each_lag_hour():
    """Per-lag normalization: every populated lag slice sums to 1, empty -> NaN."""
    acc, sl, so, lag = _run([_eastward_parcel()])
    lag_totals = acc.sum(axis=(1, 2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.where(lag_totals > 0, acc / lag_totals, np.nan)
    for k, tot in zip(kernel, lag_totals.ravel()):
        if tot > 0:
            assert k.sum() == pytest.approx(1.0, abs=1e-9)
        else:
            assert np.isnan(k).all()


def test_above_pbl_parcel_contributes_nothing():
    acc, *_ = _run([_eastward_parcel(alt_m=5000.0)])  # well above the 2000 m PBL
    assert acc.sum() == pytest.approx(0.0)


def test_subhourly_preserves_total():
    """Finer resampling changes the spatial spread but not the total residence time."""
    coarse, *_ = _run([_eastward_parcel()], step_min=60.0)
    fine, *_ = _run([_eastward_parcel()], step_min=10.0)
    assert fine.sum() == pytest.approx(coarse.sum(), rel=1e-6)


def test_contact_fraction_preset_changes_result():
    """A parcel at 0.7*PBL is included by Sodemann (1.5) but excluded by STILT (0.5)."""
    p = _eastward_parcel(alt_m=0.7 * 2000.0)
    stilt, *_ = _run([p], contact_fraction=config.CONTACT_FRACTION_STILT)
    sod, *_ = _run([p], contact_fraction=config.CONTACT_FRACTION_SODEMANN)
    assert stilt.sum() == pytest.approx(0.0)
    assert sod.sum() > 0.0


def _synthetic_day(n_parcels=2):
    """A minimal load_day-shaped dataset: parcels arriving at (40.5, -90.5)."""
    import xarray as xr
    times = np.array(["2019-06-05T18:00", "2019-06-05T21:00",
                      "2019-06-05T22:00", "2019-06-05T23:00"],
                     dtype="datetime64[ns]")
    lons = np.linspace([-93.5, -92.5, -91.5, -90.5],
                       [-94.5, -93.2, -91.8, -90.5], n_parcels)
    return xr.Dataset(
        {"time_utc": (("parcel", "step"), np.tile(times, (n_parcels, 1))),
         "lat": (("parcel", "step"), np.full((n_parcels, 4), 40.5)),
         "lon": (("parcel", "step"), lons),
         "alt": (("parcel", "step"), np.full((n_parcels, 4), 200.0)),
         "q": (("parcel", "step"), np.full((n_parcels, 4), 8.0)),
         "swath": ("parcel", np.array(["early"] * n_parcels))},
        coords={"parcel": np.arange(n_parcels), "step": np.arange(4)},
    )


def test_build_footprint_per_lag_normalization_and_members():
    day = _synthetic_day()
    ds = F.build_footprint(day, 40.5, -90.5, arrival_step=3,
                           pbl_model=ConstantPBL(2000.0),
                           fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    assert list(ds["member_parcel"].values) == [0, 1]
    lag_sums = ds["kernel"].sum(dim=("source_lat", "source_lon")).values
    fp_sums = ds["footprint"].sum(dim=("source_lat", "source_lon")).values
    assert np.allclose(lag_sums[fp_sums > 0], 1.0, atol=1e-9)
    assert np.isnan(ds["kernel"].values[fp_sums == 0]).all()
    assert (fp_sums > 0).any()


def test_empty_receptor_has_empty_member_list():
    day = _synthetic_day()
    ds = F.build_footprint(day, 30.5, -100.5, arrival_step=3,
                           pbl_model=ConstantPBL(2000.0),
                           fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    assert ds.attrs["n_parcels"] == 0
    assert ds.sizes["member"] == 0
    assert np.isnan(ds["kernel"].values).all()


def _cluster_and_outlier_trajs(n_cluster=9):
    """``n_cluster`` parcels whose lag-2 position is a tight cluster near
    (40.5, -92.5) plus ONE outlier at (40.5, -95.5); all arrive at the
    receptor (40.5, -90.5), all at 200 m (full contact in a 2000 m PBL)."""
    times = np.array(["2019-06-05T19:00", "2019-06-05T20:00", "2019-06-05T21:00"],
                     dtype="datetime64[ns]")
    trajs = []
    for i in range(n_cluster):
        lon2 = -92.5 + 0.02 * i  # tight cluster (~a few km spread)
        trajs.append({
            "t_hours": np.array([0.0, 1.0, 2.0]), "time_utc": times,
            "lat": np.full(3, 40.5), "lon": np.array([lon2, -91.5, -90.5]),
            "alt": np.full(3, 200.0),
        })
    trajs.append({
        "t_hours": np.array([0.0, 1.0, 2.0]), "time_utc": times,
        "lat": np.full(3, 40.5), "lon": np.array([-95.5, -91.5, -90.5]),
        "alt": np.full(3, 200.0),
    })
    return trajs


def test_containment_mask_drops_the_outlier():
    """90% of 10 parcels = 9: the smallest COM-centred circle holding 9 parcels
    covers the cluster but not the ~230-km outlier, whose cells fall out."""
    sl, so = F.source_window(40.5, -90.5)
    lag_hours = np.arange(0, 3, dtype=float)
    mask = F.containment_mask(sl, so, _cluster_and_outlier_trajs(), lag_hours,
                              ConstantPBL(2000.0), frac=0.90)
    lag2 = mask[2]
    near = np.abs(so - (-92.5)) <= 0.25   # cluster cells stay in
    far = np.abs(so - (-95.5)) <= 0.25    # outlier cells fall out
    on_line = np.abs(sl - 40.5) <= 0.25
    assert lag2[np.ix_(on_line, near)].any()
    assert not lag2[np.ix_(on_line, far)].any()
    # lag 0: every parcel is inside the receptor cell -> support stays local
    assert not mask[0][:, np.abs(so - (-93.0)) <= 0.5].any()


def test_kernel_truncated_but_footprint_physical():
    """End-to-end: the outlier's deposit survives in the physical footprint but
    is cut from the kernel, whose populated lags still sum to exactly 1.
    (19 cluster parcels + 1 outlier keeps n at the CONTAINMENT_MIN_PARCELS
    threshold so the mask is actually applied.)"""
    import xarray as xr
    trajs = _cluster_and_outlier_trajs(n_cluster=19)
    n = len(trajs)
    day = xr.Dataset(
        {"time_utc": (("parcel", "step"), np.stack([t["time_utc"] for t in trajs])),
         "lat": (("parcel", "step"), np.stack([t["lat"] for t in trajs])),
         "lon": (("parcel", "step"), np.stack([t["lon"] for t in trajs])),
         "alt": (("parcel", "step"), np.stack([t["alt"] for t in trajs])),
         "q": (("parcel", "step"), np.full((n, 3), 8.0)),
         "swath": ("parcel", np.array(["early"] * n))},
        coords={"parcel": np.arange(n), "step": np.arange(3)},
    )
    ds = F.build_footprint(day, 40.5, -90.5, arrival_step=2,
                           pbl_model=ConstantPBL(2000.0),
                           fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    far = np.abs(ds["source_lon"].values - (-95.5)) <= 0.25
    fp2 = ds["footprint"].sel(lag=2.0).values
    k2 = ds["kernel"].sel(lag=2.0).values
    assert fp2[:, far].sum() > 0                      # physics kept
    assert np.nansum(k2[:, far]) == pytest.approx(0)  # kernel truncated
    assert np.nansum(k2) == pytest.approx(1.0, abs=1e-6)
    assert ds.attrs["kernel_containment_frac"] == pytest.approx(0.90)
    assert ds.attrs["containment_applied"] == 1
    # containment off -> the outlier's kernel weight comes back
    ds_off = F.build_footprint(day, 40.5, -90.5, arrival_step=2,
                               pbl_model=ConstantPBL(2000.0),
                               fuzz_kernel=_DeltaFuzz(), land_fn=_all_land,
                               containment_frac=None)
    assert np.nansum(ds_off["kernel"].sel(lag=2.0).values[:, far]) > 0


# --------------------------------------------------------------------------- #
# Energy weighting (insolation.py plug-in) and the small-n containment guard
# --------------------------------------------------------------------------- #
def _run_energy(**kwargs):
    """The standard eastward-parcel run, forwarding ``energy_fn`` (or nothing,
    to exercise the pre-energy call signature)."""
    sl, so = F.source_window(40.5, -90.5)
    lag_hours = np.arange(0, 4, dtype=float)
    return F.footprint_from_trajectories(
        [_eastward_parcel()], sl, so, lag_hours,
        ConstantPBL(2000.0), _DeltaFuzz(), _all_land, **kwargs,
    )


def test_energy_fn_none_bit_identical_to_omitted():
    """Passing ``energy_fn=None`` explicitly is the documented no-op: the
    footprint is bit-identical to omitting the kwarg altogether."""
    assert np.array_equal(_run_energy(energy_fn=None), _run_energy())


def test_constant_energy_fn_scales_footprint_exactly():
    """A constant weight c multiplies every deposit by exactly c (c = 2 is a
    power of two, so even the floating-point scaling is exact)."""
    c = 2.0
    baseline = _run_energy()
    scaled = _run_energy(energy_fn=lambda lat, lon, t: np.full(np.shape(lat), c))
    assert np.array_equal(scaled, c * baseline)


def test_uniform_energy_matches_no_energy_fn():
    """UniformEnergy is the A/B control: weight 1.0 everywhere reproduces the
    unweighted footprint exactly."""
    from trajectory_kernels.insolation import UniformEnergy
    assert np.array_equal(_run_energy(energy_fn=UniformEnergy()), _run_energy())


def test_energy_attrs_flip_units_and_model_name():
    """build_footprint records the energy model and switches the footprint
    units between contact-hours and energy-weighted W m-2 hours."""
    from trajectory_kernels.insolation import ClearSkyAvailableEnergy
    day = _synthetic_day()
    kw = dict(arrival_step=3, pbl_model=ConstantPBL(2000.0),
              fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    plain = F.build_footprint(day, 40.5, -90.5, **kw)
    assert plain.attrs["energy_model"] == "UniformEnergy"
    assert plain["footprint"].attrs["units"] == "hours of land-surface contact"
    weighted = F.build_footprint(day, 40.5, -90.5,
                                 energy_fn=ClearSkyAvailableEnergy(), **kw)
    assert weighted.attrs["energy_model"] == "ClearSkyAvailableEnergy"
    assert weighted["footprint"].attrs["units"] == (
        "W m-2 hours (energy-weighted land contact)")


def _day_from_trajs(trajs):
    """Stack cluster-and-outlier traj dicts into a load_day-shaped dataset."""
    import xarray as xr
    n, n_step = len(trajs), trajs[0]["lat"].size
    return xr.Dataset(
        {"time_utc": (("parcel", "step"), np.stack([t["time_utc"] for t in trajs])),
         "lat": (("parcel", "step"), np.stack([t["lat"] for t in trajs])),
         "lon": (("parcel", "step"), np.stack([t["lon"] for t in trajs])),
         "alt": (("parcel", "step"), np.stack([t["alt"] for t in trajs])),
         "q": (("parcel", "step"), np.full((n, n_step), 8.0)),
         "swath": ("parcel", np.array(["early"] * n))},
        coords={"parcel": np.arange(n), "step": np.arange(n_step)},
    )


def test_small_n_guard_skips_containment_below_threshold():
    """9 cluster parcels + 1 outlier = 10 < CONTAINMENT_MIN_PARCELS: the
    containment radius would be a degenerate order statistic, so the guard
    skips it (containment_applied == 2) and the kernel is UNtruncated --
    identical to running with containment disabled outright."""
    assert config.CONTAINMENT_MIN_PARCELS == 20  # threshold the fixtures assume
    day = _day_from_trajs(_cluster_and_outlier_trajs(n_cluster=9))
    kw = dict(arrival_step=2, pbl_model=ConstantPBL(2000.0),
              fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    ds = F.build_footprint(day, 40.5, -90.5, **kw)
    assert ds.attrs["n_parcels"] == 10
    assert ds.attrs["containment_applied"] == 2
    ds_off = F.build_footprint(day, 40.5, -90.5, containment_frac=None, **kw)
    assert ds_off.attrs["containment_applied"] == 0
    assert np.array_equal(ds["kernel"].values, ds_off["kernel"].values,
                          equal_nan=True)
    # the outlier's kernel weight survives the skipped mask
    far = np.abs(ds["source_lon"].values - (-95.5)) <= 0.25
    assert np.nansum(ds["kernel"].sel(lag=2.0).values[:, far]) > 0


def test_small_n_guard_applies_containment_at_threshold():
    """19 cluster parcels + 1 outlier = 20 parcels meets the threshold: the
    mask IS applied (containment_applied == 1) and the outlier is truncated."""
    day = _day_from_trajs(_cluster_and_outlier_trajs(n_cluster=19))
    ds = F.build_footprint(day, 40.5, -90.5, arrival_step=2,
                           pbl_model=ConstantPBL(2000.0),
                           fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    assert ds.attrs["n_parcels"] == 20
    assert ds.attrs["containment_applied"] == 1
    far = np.abs(ds["source_lon"].values - (-95.5)) <= 0.25
    assert np.nansum(ds["kernel"].sel(lag=2.0).values[:, far]) == pytest.approx(0)


def test_lag_weight_is_the_per_lag_mass_the_normalization_divided_out():
    """REGRESSION (review critical): the per-lag kernel normalization cancels
    the hour-to-hour energy weighting unless the divided-out per-lag totals
    are kept. build_footprint and build_all must both store them as
    ``lag_weight`` -- equal to the containment-masked footprint's per-lag sum
    (== the raw footprint's here, no containment at small n), 0 for empty
    lags, never NaN."""
    day = _synthetic_day()
    kw = dict(pbl_model=ConstantPBL(2000.0), fuzz_kernel=_DeltaFuzz(),
              land_fn=_all_land)
    single = F.build_footprint(day, 40.5, -90.5, arrival_step=3, **kw)
    assert single["lag_weight"].dims == ("lag",)
    fp_sums = single["footprint"].sum(dim=("source_lat", "source_lon")).values
    assert np.allclose(single["lag_weight"].values, fp_sums, rtol=1e-6)
    assert np.isfinite(single["lag_weight"].values).all()

    ds = F.build_all(day, arrival_steps=(1, 2, 3), **kw)
    assert ds["lag_weight"].dims == ("arrival_step", "target_lat",
                                     "target_lon", "lag")
    assert ds["lag_weight"].dtype == np.float32
    fp_sums = ds["footprint"].sum(dim=("dlat", "dlon")).values
    assert np.allclose(ds["lag_weight"].values, fp_sums, rtol=1e-6)
    assert np.isfinite(ds["lag_weight"].values).all()  # 0 for empty, never NaN
    # empty receptors/lags carry exactly zero weight
    empty = ds["n_parcels"].values == 0
    assert (ds["lag_weight"].values[empty] == 0).all()
    # kernel semantics unchanged: populated lag slices still sum to 1
    ksum = np.nansum(ds["kernel"].values, axis=(-2, -1))
    populated = ds["lag_weight"].values > 0
    assert np.allclose(ksum[populated], 1.0, atol=1e-6)


def test_lag_weight_units_follow_footprint_units():
    from trajectory_kernels.insolation import ClearSkyAvailableEnergy
    day = _synthetic_day()
    kw = dict(arrival_steps=(1, 2, 3), pbl_model=ConstantPBL(2000.0),
              fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    plain = F.build_all(day, **kw)
    assert plain["lag_weight"].attrs["units"] == plain["footprint"].attrs["units"]
    weighted = F.build_all(day, energy_fn=ClearSkyAvailableEnergy(), **kw)
    assert weighted["lag_weight"].attrs["units"] == (
        "W m-2 hours (energy-weighted land contact)")


def test_build_all_uniform_energy_instance_keeps_plain_units():
    """REGRESSION (review 1C): build_all used to key the units on
    ``energy_fn is None``, so an explicit UniformEnergy() instance -- the A/B
    control, weight 1.0 everywhere -- mislabelled plain contact hours as
    energy-weighted W m-2 hours."""
    from trajectory_kernels.insolation import UniformEnergy
    day = _synthetic_day()
    kw = dict(arrival_steps=(1, 2, 3), pbl_model=ConstantPBL(2000.0),
              fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    ds = F.build_all(day, energy_fn=UniformEnergy(), **kw)
    assert ds.attrs["energy_model"] == "UniformEnergy"
    assert ds["footprint"].attrs["units"] == "hours of land-surface contact"


def test_build_all_zero_parcel_day_yields_empty_arrival_times():
    """REGRESSION (review 1D): a parcel-free day used to raise IndexError at
    arrays['time_utc'][0, s]; now the arrival_times_utc attr is an empty list
    (the parcel-free-day marker) and every receptor is honestly empty."""
    day = _synthetic_day(n_parcels=0)
    ds = F.build_all(day, arrival_steps=(1, 2, 3), pbl_model=ConstantPBL(2000.0),
                     fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    assert list(ds.attrs["arrival_times_utc"]) == []
    assert (ds["n_parcels"].values == 0).all()
    assert (ds["lag_weight"].values == 0).all()


def test_build_all_arrival_times_attr_matches_day():
    """build_all records the shared hourly arrival slots (parcel 0 is
    representative) so predictors.m_star can window the PBL history."""
    day = _synthetic_day()
    steps = (1, 2, 3)
    ds = F.build_all(day, arrival_steps=steps, pbl_model=ConstantPBL(2000.0),
                     fuzz_kernel=_DeltaFuzz(), land_fn=_all_land)
    expected = [str(day["time_utc"].values[0, s]) for s in steps]
    assert list(ds.attrs["arrival_times_utc"]) == expected


@pytest.mark.skipif(
    not (config.TRAJ_DIR / config.NOGRID_TEMPLATE.format(granule=189)).exists(),
    reason="trajectory data not present",
)
def test_real_receptor_smoke():
    from trajectory_kernels import trajectories as T
    day = T.load_day()
    # pick a plains cell with near-surface arrivals at 23 UTC (step 3)
    ds = F.build_footprint(day, 40.5, -95.5, arrival_step=3)
    assert set(ds["footprint"].dims) == {"lag", "source_lat", "source_lon"}
    assert "n_parcels" in ds.attrs
    assert ds.sizes["member"] == ds.attrs["n_parcels"]
    if ds.attrs["n_parcels"] > 0 and float(ds["footprint"].sum()) > 0:
        lag_sums = ds["kernel"].sum(dim=("source_lat", "source_lon")).values
        fp_sums = ds["footprint"].sum(dim=("source_lat", "source_lon")).values
        populated = fp_sums > 0
        assert np.allclose(lag_sums[populated], 1.0, atol=1e-5)
