"""End-to-end tests of the command-line interface."""

import os

import numpy as np
import pytest

from geoterp.cli import main
from geoterp.io import read_raster


@pytest.fixture
def sample_dir(tmp_path):
    d = tmp_path / "data"
    assert main(["sample", "--outdir", str(d)]) == 0
    return d


def test_sample_writes_all_files(sample_dir):
    for name in ("stations.geojson", "source_zones.geojson", "target_zones.geojson",
                 "landuse.geojson", "dem.tif", "stations.csv"):
        assert (sample_dir / name).exists()


def test_cli_point_idw(sample_dir, tmp_path):
    out = str(tmp_path / "idw.tif")
    rc = main(["point", "idw", str(sample_dir / "stations.geojson"),
               "--value", "temperature", "--resolution", "30", "-o", out])
    assert rc == 0
    g = read_raster(out)
    assert g.shape[0] > 0


def test_cli_point_voronoi(sample_dir, tmp_path):
    out = str(tmp_path / "vor.geojson")
    rc = main(["point", "voronoi", str(sample_dir / "stations.geojson"),
               "--value", "temperature", "-o", out])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_point_kriging(sample_dir, tmp_path):
    out = str(tmp_path / "k.tif")
    rc = main(["point", "kriging", str(sample_dir / "stations.geojson"),
               "--value", "temperature", "--resolution", "20",
               "--variogram", "exponential", "-o", out])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_polygon_areal(sample_dir, tmp_path):
    out = str(tmp_path / "areal.geojson")
    rc = main(["polygon", "areal", str(sample_dir / "source_zones.geojson"),
               str(sample_dir / "target_zones.geojson"),
               "--value", "population", "-o", out])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_polygon_pycno_raster(sample_dir, tmp_path):
    out = str(tmp_path / "pycno.tif")
    rc = main(["polygon", "pycno", str(sample_dir / "source_zones.geojson"),
               "--value", "population", "--resolution", "40", "-o", out])
    assert rc == 0
    g = read_raster(out)
    total = np.nansum(g.data)
    assert total > 0


def test_cli_raster_resample(sample_dir, tmp_path):
    out = str(tmp_path / "up.tif")
    rc = main(["raster", "resample", str(sample_dir / "dem.tif"),
               "--scale", "2", "--kernel", "cubic", "-o", out])
    assert rc == 0
    g = read_raster(out)
    assert g.shape[0] > 0


def test_cli_raster_aggregate(sample_dir, tmp_path):
    out = str(tmp_path / "agg.tif")
    rc = main(["raster", "aggregate", str(sample_dir / "dem.tif"),
               "--factor", "4", "--stat", "average", "-o", out])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_bad_value_column_returns_error(sample_dir, tmp_path):
    rc = main(["point", "idw", str(sample_dir / "stations.geojson"),
               "--value", "does_not_exist", "-o", str(tmp_path / "x.tif")])
    assert rc == 2  # handled error, not a crash
