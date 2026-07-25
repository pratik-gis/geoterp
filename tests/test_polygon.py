"""Tests for polygon-to-polygon interpolation.

The defining property of all three methods is **mass preservation**: when the
target zoning fully covers the source, the interpolated totals must sum back to
the source total.
"""

import numpy as np
import pytest

import geoterp


def test_areal_weighting_preserves_mass(zones):
    src, tgt, _ = zones
    out = geoterp.areal_weighting(src, tgt, "population")
    assert out["population_est"].sum() == pytest.approx(src["population"].sum(), rel=1e-6)


def test_areal_weighting_multiple_columns(zones):
    src, tgt, _ = zones
    src = src.copy()
    src["households"] = src["population"] / 3
    out = geoterp.areal_weighting(src, tgt, ["population", "households"])
    assert "population_est" in out.columns
    assert "households_est" in out.columns
    assert out["households_est"].sum() == pytest.approx(src["households"].sum(), rel=1e-6)


def test_areal_weighting_intensive_bounds(zones):
    src, tgt, _ = zones
    src = src.copy()
    src["density"] = src["population"] / src.geometry.area
    out = geoterp.areal_weighting(src, tgt, "density", extensive=False)
    # An area-weighted average must stay within the source value range.
    assert out["density_est"].min() >= src["density"].min() - 1e-9
    assert out["density_est"].max() <= src["density"].max() + 1e-9


def test_dasymetric_preserves_mass(zones):
    src, tgt, lu = zones
    out = geoterp.dasymetric(src, tgt, "population", lu, "weight")
    assert out["population_est"].sum() == pytest.approx(src["population"].sum(), rel=1e-6)


def test_dasymetric_differs_from_areal(zones):
    src, tgt, lu = zones
    a = geoterp.areal_weighting(src, tgt, "population")["population_est"].to_numpy()
    d = geoterp.dasymetric(src, tgt, "population", lu, "weight")["population_est"].to_numpy()
    # Ancillary weighting should redistribute differently than plain area.
    assert not np.allclose(a, d)


def test_dasymetric_requires_crs(zones):
    src, tgt, lu = zones
    src2 = src.copy()
    src2.crs = None
    with pytest.raises(ValueError):
        geoterp.dasymetric(src2, tgt, "population", lu, "weight")


def test_pycnophylactic_preserves_zone_mass(zones):
    src, _, _ = zones
    grid = geoterp.pycnophylactic(src, "population", resolution=60, max_iter=100)
    total = np.nansum(grid.data)
    assert total == pytest.approx(src["population"].sum(), rel=1e-3)


def test_pycnophylactic_non_negative(zones):
    src, _, _ = zones
    grid = geoterp.pycnophylactic(src, "population", resolution=50, max_iter=80)
    v = grid.filled(np.nan)
    assert np.nanmin(v) >= -1e-9


def test_pycnophylactic_is_smoother_than_uniform(zones):
    """The smoothed surface should have a smaller cell-to-cell gradient than the
    blocky uniform-density seed."""
    src, _, _ = zones
    grid = geoterp.pycnophylactic(src, "population", resolution=60, max_iter=150)
    v = grid.filled(np.nan)
    gy, gx = np.gradient(np.nan_to_num(v))
    grad = np.hypot(gx, gy)
    # Uniform seed jumps sharply at borders; smoothed surface should be gentler.
    assert np.nanmean(grad) < np.nanmax(v)


def test_pycnophylactic_to_polygons(zones):
    src, tgt, _ = zones
    out = geoterp.pycnophylactic_to_polygons(src, tgt, "population", resolution=60)
    assert "population_est" in out.columns
    assert out["population_est"].sum() == pytest.approx(src["population"].sum(), rel=1e-2)
