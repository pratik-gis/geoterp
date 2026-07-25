"""Tests for the core grid/raster structures and input coercion."""

import numpy as np
import pytest

from geoterp.core import GridSpec, RasterGrid
from geoterp.io import extract_xyz


def test_gridspec_from_bounds_cellsize():
    g = GridSpec.from_bounds((0, 0, 100, 50), cellsize=10)
    assert g.ncols == 10
    assert g.nrows == 5
    assert g.shape == (5, 10)


def test_gridspec_resolution_keeps_square_cells():
    g = GridSpec.from_bounds((0, 0, 200, 100), resolution=20)
    # longer side is 200 -> cellsize 10 -> 20 x 10 cells
    assert g.cellsize == pytest.approx(10)
    assert g.shape == (10, 20)


def test_gridspec_rejects_double_spec():
    with pytest.raises(ValueError):
        GridSpec.from_bounds((0, 0, 10, 10), cellsize=1, resolution=10)


def test_cell_centers_orientation():
    g = GridSpec(0, 0, 10, 10, cellsize=5)
    X, Y = g.cell_centers()
    # Row 0 is the northern-most row -> larger y.
    assert Y[0, 0] > Y[-1, 0]
    assert X[0, 0] == pytest.approx(2.5)
    assert Y[0, 0] == pytest.approx(7.5)


def test_transform_is_north_up():
    g = GridSpec(0, 0, 10, 10, cellsize=5)
    t = g.transform
    assert t.a == pytest.approx(5)
    assert t.e == pytest.approx(-5)  # north-up: negative y pixel size
    assert (t.c, t.f) == (0, 10)


def test_raster_filled_replaces_nodata():
    data = np.array([[1.0, np.nan], [3.0, 4.0]])
    r = RasterGrid(data, GridSpec(0, 0, 2, 2, 1))
    filled = r.filled(-1)
    assert filled[0, 1] == -1
    assert filled[0, 0] == 1


def test_geotiff_roundtrip(tmp_path):
    from geoterp.io import read_raster

    spec = GridSpec(0, 0, 10, 10, cellsize=1, crs="EPSG:32643")
    data = np.arange(100, dtype=float).reshape(10, 10)
    r = RasterGrid(data, spec)
    path = str(tmp_path / "r.tif")
    r.to_geotiff(path)
    back = read_raster(path)
    assert back.shape == (10, 10)
    np.testing.assert_allclose(back.data, data, atol=1e-4)
    assert back.spec.cellsize == pytest.approx(1)


def test_extract_xyz_from_arrays():
    x, y, z, crs = extract_xyz([0, 1, 2], y=[0, 1, 2], z=[10, 11, 12])
    assert len(x) == 3
    assert crs is None
    np.testing.assert_array_equal(z, [10, 11, 12])


def test_extract_xyz_from_coord_array():
    coords = np.array([[0, 0], [1, 1]])
    x, y, z, _ = extract_xyz(coords, z=[5, 6])
    np.testing.assert_array_equal(z, [5, 6])


def test_extract_xyz_drops_nonfinite():
    x, y, z, _ = extract_xyz([0, 1, 2], y=[0, np.nan, 2], z=[1, 2, np.inf])
    assert len(x) == 1  # only the first row is fully finite


def test_extract_xyz_from_geodataframe(stations):
    x, y, z, crs = extract_xyz(stations, value="temperature")
    assert len(x) == len(stations)
    assert crs is not None


def test_extract_xyz_missing_value_raises(stations):
    with pytest.raises(KeyError):
        extract_xyz(stations, value="nope")
