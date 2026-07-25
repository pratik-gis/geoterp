"""geoterp — spatial interpolation for vector and raster data.

A single, consistent toolbox for turning sparse measurements into continuous
surfaces or re-expressing them on a different geometry, covering three data
families:

Points → surface
    :func:`idw`, :func:`nearest`, :func:`natural_neighbor`, :func:`spline`,
    :func:`trend_surface`, :func:`local_polynomial`, :func:`kriging`
    (plus :func:`voronoi_polygons` for vector Thiessen output).
Polygons → polygons
    :func:`areal_weighting`, :func:`dasymetric`, :func:`pycnophylactic`.
Raster → raster
    :func:`resample`, :func:`aggregate`.

Every point/raster method returns a :class:`RasterGrid`; polygon methods return
GeoDataFrames.  See the :mod:`geoterp.datasets` module for ready-made sample
data and the README for a tour.
"""

from __future__ import annotations

from .core import GridSpec, RasterGrid
from .point import (
    idw,
    kriging,
    local_polynomial,
    natural_neighbor,
    nearest,
    spline,
    trend_surface,
    voronoi_polygons,
)
from .polygon import (
    areal_weighting,
    dasymetric,
    pycnophylactic,
    pycnophylactic_to_polygons,
)
from .raster import aggregate, resample

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GridSpec",
    "RasterGrid",
    # points
    "idw",
    "nearest",
    "voronoi_polygons",
    "natural_neighbor",
    "spline",
    "trend_surface",
    "local_polynomial",
    "kriging",
    # polygons
    "areal_weighting",
    "dasymetric",
    "pycnophylactic",
    "pycnophylactic_to_polygons",
    # raster
    "resample",
    "aggregate",
]
