"""Tests for raster resampling and aggregation."""

import numpy as np
import pytest

import geoterp
from geoterp.core import GridSpec, RasterGrid


@pytest.fixture
def ramp_raster():
    spec = GridSpec(0, 0, 20, 20, cellsize=1, crs="EPSG:32643")
    X, Y = spec.cell_centers()
    return RasterGrid(X + Y, spec)  # smooth linear ramp


def test_resample_scale_up(ramp_raster):
    out = geoterp.resample(ramp_raster, scale=2, method="bilinear")
    assert out.shape == (40, 40)
    assert out.spec.cellsize == pytest.approx(0.5)


def test_resample_scale_down(ramp_raster):
    out = geoterp.resample(ramp_raster, scale=0.5, method="nearest")
    assert out.shape == (10, 10)


def test_resample_to_shape(ramp_raster):
    out = geoterp.resample(ramp_raster, shape=(5, 5))
    assert out.shape == (5, 5)


def test_resample_bilinear_preserves_ramp(ramp_raster):
    # A linear ramp is reproduced (almost) exactly by bilinear resampling.
    out = geoterp.resample(ramp_raster, scale=2, method="bilinear")
    X, Y = out.spec.cell_centers()
    truth = X + Y
    interior = np.abs(out.data - truth)[2:-2, 2:-2]
    assert np.nanmax(interior) < 0.6


def test_resample_requires_one_target(ramp_raster):
    with pytest.raises(ValueError):
        geoterp.resample(ramp_raster, scale=2, cellsize=1)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "lanczos"])
def test_resample_all_kernels_run(ramp_raster, method):
    out = geoterp.resample(ramp_raster, scale=2, method=method)
    assert out.shape == (40, 40)
    assert np.isfinite(out.data).all()


def test_aggregate_average_preserves_mean_of_constant():
    spec = GridSpec(0, 0, 8, 8, cellsize=1)
    r = RasterGrid(np.full((8, 8), 7.0), spec)
    out = geoterp.aggregate(r, factor=2, method="average")
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out.data, 7.0)


def test_aggregate_sum_conserves_total():
    spec = GridSpec(0, 0, 8, 8, cellsize=1)
    data = np.arange(64, dtype=float).reshape(8, 8)
    r = RasterGrid(data, spec)
    out = geoterp.aggregate(r, factor=4, method="sum")
    assert np.nansum(out.data) == pytest.approx(data.sum())


def test_aggregate_min_max():
    spec = GridSpec(0, 0, 4, 4, cellsize=1)
    data = np.arange(16, dtype=float).reshape(4, 4)
    r = RasterGrid(data, spec)
    mn = geoterp.aggregate(r, factor=2, method="min")
    mx = geoterp.aggregate(r, factor=2, method="max")
    assert mn.data[0, 0] == 0
    assert mx.data[0, 0] == 5  # block {0,1,4,5}


def test_aggregate_mode_categorical():
    spec = GridSpec(0, 0, 4, 4, cellsize=1)
    data = np.array(
        [[1, 1, 2, 2],
         [1, 3, 2, 5],
         [4, 4, 6, 6],
         [4, 7, 6, 6]], dtype=float,
    )
    r = RasterGrid(data, spec)
    out = geoterp.aggregate(r, factor=2, method="mode")
    assert out.data[0, 0] == 1  # {1,1,1,3} -> 1
    assert out.data[1, 1] == 6  # {6,6,6,6} -> 6


def test_aggregate_ignores_nan():
    spec = GridSpec(0, 0, 2, 2, cellsize=1)
    data = np.array([[np.nan, 4.0], [2.0, 6.0]])
    r = RasterGrid(data, spec)
    out = geoterp.aggregate(r, factor=2, method="average")
    assert out.data[0, 0] == pytest.approx(4.0)  # mean of {4,2,6}


# --- void / NoData filling ----------------------------------------------
@pytest.fixture
def holed_dem():
    from geoterp import datasets

    dem = datasets.load_dem(resolution=60)
    truth = dem.data.copy()
    rng = np.random.default_rng(0)
    holed = truth.copy()
    holed[20:32, 22:38] = np.nan  # rectangular void
    idx = rng.choice(holed.size, size=150, replace=False)
    holed.flat[idx] = np.nan
    mask = ~np.isfinite(holed)
    return RasterGrid(holed, dem.spec), truth, mask


@pytest.mark.parametrize("method", ["idw", "nearest", "linear", "cubic", "rbf", "laplace"])
def test_fill_closes_all_holes(holed_dem, method):
    holed, truth, mask = holed_dem
    filled = geoterp.fill_nodata(holed, method=method)
    assert np.isfinite(filled.data).all(), "every hole must be filled"
    assert filled.meta["filled"] == int(mask.sum())


@pytest.mark.parametrize("method", ["idw", "linear", "cubic", "rbf", "laplace"])
def test_fill_leaves_valid_cells_untouched(holed_dem, method):
    holed, truth, mask = holed_dem
    filled = geoterp.fill_nodata(holed, method=method)
    np.testing.assert_allclose(filled.data[~mask], truth[~mask])


@pytest.mark.parametrize("method", ["idw", "cubic", "rbf", "laplace"])
def test_fill_reconstructs_smooth_surface(holed_dem, method):
    # On a smooth DEM the filled voids should be close to the true values.
    holed, truth, mask = holed_dem
    filled = geoterp.fill_nodata(holed, method=method)
    mae = np.abs(filled.data[mask] - truth[mask]).mean()
    assert mae < 8.0  # metres; the DEM spans ~180-650 m


def test_fill_no_holes_is_identity():
    spec = GridSpec(0, 0, 4, 4, cellsize=1)
    data = np.arange(16, dtype=float).reshape(4, 4)
    filled = geoterp.fill_nodata(RasterGrid(data, spec), method="idw")
    np.testing.assert_array_equal(filled.data, data)
    assert filled.meta["filled"] == 0


def test_fill_all_nodata_raises():
    spec = GridSpec(0, 0, 4, 4, cellsize=1)
    data = np.full((4, 4), np.nan)
    with pytest.raises(ValueError):
        geoterp.fill_nodata(RasterGrid(data, spec))


def test_fill_bad_method_raises(holed_dem):
    holed, _, _ = holed_dem
    with pytest.raises(ValueError):
        geoterp.fill_nodata(holed, method="banana")
