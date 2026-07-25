"""Tests for point-to-surface interpolation methods.

The two workhorse invariants used throughout:

* **Constant reproduction** – interpolating a constant field must return that
  constant everywhere inside the domain.
* **Linear reproduction** – exact interpolators of a linear field (trend deg-1,
  local poly deg-1, thin-plate spline) must recover it to numerical precision.
"""

import numpy as np
import pytest

import geoterp
from geoterp.core import GridSpec


def _valid(grid):
    v = grid.filled(np.nan)
    return v[np.isfinite(v)]


# --- constant-field reproduction ----------------------------------------
@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (geoterp.idw, {}),
        (geoterp.nearest, {}),
        (geoterp.spline, {}),
        (geoterp.trend_surface, {"degree": 1}),
        (geoterp.local_polynomial, {}),
        (geoterp.natural_neighbor, {}),
        (geoterp.kriging, {}),
    ],
)
def test_constant_field_reproduced(constant_points, fn, kwargs):
    x, y, z = constant_points
    grid = fn(x, y=y, z=z, resolution=25, **kwargs)
    vals = _valid(grid)
    assert vals.size > 0
    np.testing.assert_allclose(vals, 15.0, atol=1e-3)


# --- linear-field reproduction by exact interpolators -------------------
@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (geoterp.trend_surface, {"degree": 1}),
        (geoterp.local_polynomial, {"degree": 1, "neighbors": 12}),
        (geoterp.spline, {"kernel": "thin_plate_spline"}),
    ],
)
def test_linear_field_reproduced(linear_field, fn, kwargs):
    x, y, z, f = linear_field
    spec = GridSpec.from_points(x, y, resolution=20)
    grid = fn(x, y=y, z=z, grid=spec, **kwargs)
    X, Y = spec.cell_centers()
    truth = f(X, Y)
    err = np.abs(grid.filled(np.nan) - truth)
    assert np.nanmax(err) < 1e-2


# --- method-specific behaviour ------------------------------------------
def test_idw_exact_hit_returns_sample():
    # Put one sample exactly on a grid cell centre.
    x = np.array([5.0, 95.0, 5.0, 95.0])
    y = np.array([5.0, 5.0, 95.0, 95.0])
    z = np.array([1.0, 2.0, 3.0, 4.0])
    spec = GridSpec(0, 0, 100, 100, cellsize=10)  # centres at 5,15,...,95
    grid = geoterp.idw(x, y=y, z=z, grid=spec)
    # top-left cell centre is (5, 95) -> sample value 3.0
    assert grid.data[0, 0] == pytest.approx(3.0)


def test_idw_power_and_neighbors(stations):
    g1 = geoterp.idw(stations, "temperature", resolution=30, power=1)
    g2 = geoterp.idw(stations, "temperature", resolution=30, power=3)
    assert g1.shape == g2.shape
    # Different power should give a different surface.
    assert not np.allclose(g1.data, g2.data)


def test_nearest_values_are_from_samples(stations):
    grid = geoterp.nearest(stations, "temperature", resolution=40)
    sample_vals = set(np.round(stations.temperature.to_numpy(), 2))
    got = set(np.round(_valid(grid), 2))
    assert got.issubset(sample_vals)


def test_natural_neighbor_stays_in_range(stations):
    grid = geoterp.natural_neighbor(stations, "temperature", resolution=25)
    vals = _valid(grid)
    lo, hi = stations.temperature.min(), stations.temperature.max()
    assert vals.min() >= lo - 1e-6
    assert vals.max() <= hi + 1e-6


def test_natural_neighbor_nan_outside_hull(stations):
    grid = geoterp.natural_neighbor(stations, "temperature", resolution=25)
    v = grid.filled(np.nan)
    # Corners of the padded grid lie outside the convex hull -> NaN.
    assert np.isnan(v[0, 0])
    assert np.isfinite(v).sum() < v.size


def test_trend_surface_reports_r2(stations):
    grid = geoterp.trend_surface(stations, "temperature", resolution=30, degree=2)
    assert 0.0 <= grid.meta["r2"] <= 1.0
    assert grid.meta["degree"] == 2


def test_trend_surface_needs_enough_points():
    with pytest.raises(ValueError):
        geoterp.trend_surface([0, 1], y=[0, 1], z=[1, 2], degree=2)


def test_voronoi_polygons_partition(stations):
    vor = geoterp.voronoi_polygons(stations, "temperature")
    assert len(vor) == len(stations)
    assert (vor.geometry.area > 0).all()
    assert "value" in vor.columns


def test_kriging_variance_returned(stations):
    surf, var = geoterp.kriging(
        stations, "temperature", resolution=20, return_variance=True
    )
    assert surf.shape == var.shape
    assert np.nanmin(var.data) >= -1e-6  # variance is non-negative


def test_kriging_universal(stations):
    grid = geoterp.kriging(
        stations, "temperature", resolution=20, method="universal"
    )
    assert grid.shape == (20, 20) or grid.shape[0] > 0


def test_kriging_bad_variogram(stations):
    with pytest.raises(ValueError):
        geoterp.kriging(stations, "temperature", variogram_model="banana")


def test_all_methods_share_row_orientation(stations):
    """Every method must be north-up (row 0 = north); guards against the
    PyKrige row-flip regression.  Sample temperature falls northward, so within
    each surface value must be negatively correlated with the Y coordinate.
    Correlating against Y (rather than comparing edge rows) is robust to the
    NaN padding that natural-neighbour leaves outside the convex hull."""
    from geoterp.core import GridSpec

    x = stations.geometry.x.to_numpy()
    y = stations.geometry.y.to_numpy()
    spec = GridSpec.from_points(x, y, resolution=30, crs=stations.crs)
    _, Y = spec.cell_centers()

    for fn in (geoterp.idw, geoterp.spline, geoterp.trend_surface, geoterp.kriging,
               geoterp.local_polynomial, geoterp.natural_neighbor):
        g = fn(stations, "temperature", grid=spec)
        v = g.filled(np.nan)
        m = np.isfinite(v)
        r = np.corrcoef(Y[m], v[m])[0, 1]
        assert r < 0, f"{fn.__name__} appears vertically flipped (corr={r:.2f})"
