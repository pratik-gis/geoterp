"""Built-in sample datasets for trying out :mod:`geoterp`.

Everything here is generated deterministically (fixed random seed) so results
are reproducible and no files need to ship with the package.  The scenario is a
fictional 10 km × 10 km city, "Terraville", projected in UTM zone 43N
(``EPSG:32643``) so distances and areas are in metres:

* :func:`load_stations` – 80 weather stations with temperature and elevation.
* :func:`load_source_zones` – census-style tracts with population counts.
* :func:`load_target_zones` – an incompatible set of delivery districts.
* :func:`load_landuse` – ancillary land-use classes with density weights.
* :func:`load_dem` – a smooth elevation raster (:class:`RasterGrid`).
* :func:`make_sample_dataset` – write all of the above to a directory.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..core import GridSpec, RasterGrid

CRS = "EPSG:32643"
XMIN, YMIN = 500_000.0, 1_000_000.0
EXTENT = 10_000.0  # metres
XMAX, YMAX = XMIN + EXTENT, YMIN + EXTENT


def _elevation_field(x, y):
    """A smooth synthetic elevation surface (metres) over the study area."""
    u = (np.asarray(x) - XMIN) / EXTENT
    v = (np.asarray(y) - YMIN) / EXTENT
    # A ridge running NE plus a couple of hills.
    z = (
        180.0
        + 260.0 * np.exp(-(((u - 0.7) ** 2 + (v - 0.65) ** 2) / 0.06))
        + 150.0 * np.exp(-(((u - 0.25) ** 2 + (v - 0.3) ** 2) / 0.05))
        + 60.0 * (u + v)
    )
    return z


def load_dem(resolution: int = 120) -> RasterGrid:
    """Return the study-area digital elevation model as a :class:`RasterGrid`."""
    spec = GridSpec.from_bounds(
        (XMIN, YMIN, XMAX, YMAX), resolution=resolution, crs=CRS
    )
    X, Y = spec.cell_centers()
    return RasterGrid(_elevation_field(X, Y), spec, meta={"name": "elevation"})


def load_stations(n: int = 80, seed: int = 42):
    """Weather stations with ``temperature`` (°C) and ``elevation`` (m)."""
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)
    x = rng.uniform(XMIN, XMAX, n)
    y = rng.uniform(YMIN, YMAX, n)
    elev = _elevation_field(x, y)
    # Temperature falls with elevation (environmental lapse rate) and northwards.
    v = (y - YMIN) / EXTENT
    temp = 31.0 - 0.0065 * (elev - 180.0) - 2.5 * v + rng.normal(0, 0.4, n)
    gdf = gpd.GeoDataFrame(
        {
            "station_id": [f"ST{i:03d}" for i in range(n)],
            "temperature": np.round(temp, 2),
            "elevation": np.round(elev, 1),
        },
        geometry=[Point(px, py) for px, py in zip(x, y)],
        crs=CRS,
    )
    return gdf


def _grid_polygons(nx: int, ny: int, dx0: float = 0.0, dy0: float = 0.0):
    from shapely.geometry import box

    cw = EXTENT / nx
    ch = EXTENT / ny
    geoms = []
    for j in range(ny):
        for i in range(nx):
            x0 = XMIN + i * cw + dx0
            y0 = YMIN + j * ch + dy0
            geoms.append(box(x0, y0, x0 + cw, y0 + ch))
    return geoms


def load_source_zones(seed: int = 7):
    """Census-style source tracts (4×4 grid) with ``population`` counts."""
    import geopandas as gpd

    rng = np.random.default_rng(seed)
    geoms = _grid_polygons(4, 4)
    # Population loosely higher toward the south-west (lower elevation).
    cents = np.array([(g.centroid.x, g.centroid.y) for g in geoms])
    u = (cents[:, 0] - XMIN) / EXTENT
    v = (cents[:, 1] - YMIN) / EXTENT
    base = 8000 * np.exp(-((u - 0.2) ** 2 + (v - 0.25) ** 2) / 0.2)
    pop = np.maximum(50, (base + rng.normal(0, 500, len(geoms)))).round().astype(int)
    return gpd.GeoDataFrame(
        {"tract_id": [f"T{i:02d}" for i in range(len(geoms))], "population": pop},
        geometry=geoms,
        crs=CRS,
    )


def load_target_zones():
    """Incompatible target districts (3×3 grid) for areal interpolation demos."""
    import geopandas as gpd

    geoms = _grid_polygons(3, 3)
    return gpd.GeoDataFrame(
        {"district_id": [f"D{i}" for i in range(len(geoms))]},
        geometry=geoms,
        crs=CRS,
    )


def load_landuse(seed: int = 11):
    """Ancillary land-use zones (6×6 grid) with relative density ``weight``."""
    import geopandas as gpd

    rng = np.random.default_rng(seed)
    geoms = _grid_polygons(6, 6)
    classes = np.array(["water", "rural", "suburban", "urban"])
    class_weight = {"water": 0.0, "rural": 1.0, "suburban": 4.0, "urban": 10.0}
    # Bias toward urban in the south-west, water in a small patch.
    cents = np.array([(g.centroid.x, g.centroid.y) for g in geoms])
    u = (cents[:, 0] - XMIN) / EXTENT
    v = (cents[:, 1] - YMIN) / EXTENT
    urban_p = np.clip(1.2 - (u + v), 0.05, 0.95)
    lu = []
    for k in range(len(geoms)):
        p = np.array([0.08, 0.35, 0.32, 0.25])
        p[3] = urban_p[k]
        p = p / p.sum()
        lu.append(rng.choice(classes, p=p))
    lu = np.array(lu)
    return gpd.GeoDataFrame(
        {
            "landuse": lu,
            "weight": [class_weight[c] for c in lu],
        },
        geometry=geoms,
        crs=CRS,
    )


def make_sample_dataset(outdir: str) -> dict:
    """Write every sample dataset to ``outdir``.  Returns a ``{name: path}`` map."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    stations = load_stations()
    p = os.path.join(outdir, "stations.geojson")
    stations.to_file(p, driver="GeoJSON")
    paths["stations"] = p
    # A CSV variant for the "plain arrays" workflow.
    csv = os.path.join(outdir, "stations.csv")
    stations.assign(x=stations.geometry.x, y=stations.geometry.y).drop(
        columns="geometry"
    ).to_csv(csv, index=False)
    paths["stations_csv"] = csv

    for name, gdf in (
        ("source_zones", load_source_zones()),
        ("target_zones", load_target_zones()),
        ("landuse", load_landuse()),
    ):
        p = os.path.join(outdir, f"{name}.geojson")
        gdf.to_file(p, driver="GeoJSON")
        paths[name] = p

    dem = load_dem()
    p = os.path.join(outdir, "dem.tif")
    dem.to_geotiff(p)
    paths["dem"] = p

    return paths


__all__ = [
    "CRS",
    "load_stations",
    "load_source_zones",
    "load_target_zones",
    "load_landuse",
    "load_dem",
    "make_sample_dataset",
]
