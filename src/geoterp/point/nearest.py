"""Nearest-neighbour interpolation and Thiessen (Voronoi) polygons.

Two related things live here:

* :func:`nearest` rasterises a point cloud by assigning every cell the value of
  its closest sample — the raster form of a Thiessen tessellation.
* :func:`voronoi_polygons` returns the tessellation itself as polygon
  geometries, clipped to a bounding region, which is the vector-native form of
  the same idea.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz


def nearest(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    radius: Optional[float] = None,
) -> RasterGrid:
    """Nearest-neighbour (Thiessen) rasterisation of scattered points."""
    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )
    tree = cKDTree(np.column_stack([x, yv]))
    X, Y = grid.cell_centers()
    q = np.column_stack([X.ravel(), Y.ravel()])
    dist, idx = tree.query(q, k=1, workers=-1)
    out = zv[idx].astype(float)
    if radius is not None:
        out[dist > radius] = np.nan
    return RasterGrid(out.reshape(grid.shape), grid, meta={"method": "nearest"})


def voronoi_polygons(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    bounds=None,
    buffer: float = 0.0,
):
    """Return Thiessen/Voronoi polygons as a GeoDataFrame.

    Each output polygon is the region closer to its generating sample than to
    any other, clipped to ``bounds`` (defaults to the samples' extent, padded by
    ``buffer``).  The generating sample's value is carried through in a
    ``value`` column.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPoint, box
    from shapely.ops import voronoi_diagram

    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    pts = MultiPoint(list(zip(x, yv)))

    if bounds is None:
        xmin, ymin = x.min(), yv.min()
        xmax, ymax = x.max(), yv.max()
        span = max(xmax - xmin, ymax - ymin)
        pad = buffer or 0.05 * (span if span > 0 else 1.0)
        envelope = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    else:
        envelope = box(*bounds)

    diagram = voronoi_diagram(pts, envelope=envelope)
    cells = [g.intersection(envelope) for g in diagram.geoms]

    # voronoi_diagram does not preserve input order, so match each cell back to
    # the sample it contains.
    sample_pts = np.column_stack([x, yv])
    tree = cKDTree(sample_pts)
    records = []
    for cell in cells:
        if cell.is_empty:
            continue
        rep = cell.representative_point()
        _, i = tree.query([rep.x, rep.y])
        records.append({"value": float(zv[i]), "geometry": cell})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    return gdf
