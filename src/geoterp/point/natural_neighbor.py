"""Natural-neighbour interpolation (Sibson's method), from scratch.

Sibson interpolation inserts each query point into the Voronoi tessellation of
the samples and weights every "natural neighbour" by the area it *loses* to the
newcomer.  It produces a smooth, C¹ surface (except at the samples) that stays
within the data range and needs no tuning parameters — a very desirable
interpolator that, frustratingly, has no maintained pure-Python implementation.
So we build one here:

1.  Compute the samples' Voronoi cells once (via shapely).
2.  For each output cell, clip a bounding box by the perpendicular bisectors
    between the query and its nearest samples (Sutherland–Hodgman) to obtain the
    query's *would-be* Voronoi cell.
3.  Intersect that cell with each neighbour's original cell; the normalised
    intersection areas are the Sibson weights.

Points outside the samples' convex hull are left as NaN, where natural-neighbour
interpolation is undefined.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz


def _clip_halfplane(poly: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Clip convex polygon ``poly`` to the half-plane ``(X - m) . n >= 0``.

    Standard Sutherland–Hodgman clip against a single line.  ``poly`` is an
    ``(K, 2)`` array of vertices in order; returns the clipped vertices.
    """
    if len(poly) == 0:
        return poly
    out = []
    npts = len(poly)
    for i in range(npts):
        cur = poly[i]
        nxt = poly[(i + 1) % npts]
        dc = np.dot(cur - m, n)
        dn = np.dot(nxt - m, n)
        cur_in = dc >= 0
        nxt_in = dn >= 0
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            t = dc / (dc - dn)
            out.append(cur + t * (nxt - cur))
    return np.asarray(out)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def natural_neighbor(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    max_neighbors: int = 32,
) -> RasterGrid:
    """Sibson natural-neighbour interpolation onto a grid.

    ``max_neighbors`` caps how many nearby samples are considered when building
    each query cell; the true natural neighbours are always among the closest
    samples, so the default of 32 is safe for typical point distributions while
    keeping the per-cell cost bounded.
    """
    from shapely.geometry import MultiPoint, Polygon, box
    from shapely.ops import voronoi_diagram

    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if len(x) < 3:
        raise ValueError("Natural-neighbour interpolation needs at least 3 samples.")
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )

    obs = np.column_stack([x, yv])
    xmin, ymin = obs.min(axis=0)
    xmax, ymax = obs.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.5 * (span if span > 0 else 1.0)
    envelope = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    bbox_poly = np.array(
        [
            [xmin - pad, ymin - pad],
            [xmax + pad, ymin - pad],
            [xmax + pad, ymax + pad],
            [xmin - pad, ymax + pad],
        ]
    )

    # Original Voronoi cells, matched back to sample indices.
    diagram = voronoi_diagram(MultiPoint(list(map(tuple, obs))), envelope=envelope)
    tree = cKDTree(obs)
    orig_cells = [None] * len(obs)
    for geom in diagram.geoms:
        cell = geom.intersection(envelope)
        if cell.is_empty:
            continue
        rep = cell.representative_point()
        _, i = tree.query([rep.x, rep.y])
        orig_cells[i] = cell

    tri = Delaunay(obs)
    X, Y = grid.cell_centers()
    q = np.column_stack([X.ravel(), Y.ravel()])
    inside = tri.find_simplex(q) >= 0

    k = int(min(max_neighbors, len(obs)))
    dist, idx = tree.query(q, k=k, workers=-1)
    if k == 1:  # pragma: no cover - guarded by the >=3 check above
        dist = dist[:, None]
        idx = idx[:, None]

    out = np.full(q.shape[0], np.nan)
    for c in range(q.shape[0]):
        if not inside[c]:
            continue
        if dist[c, 0] <= 1e-12:
            out[c] = zv[idx[c, 0]]
            continue

        qp = q[c]
        cell = bbox_poly.copy()
        for i in idx[c]:
            p = obs[i]
            m = 0.5 * (qp + p)
            n = qp - p  # half-plane on the query's side of the bisector
            cell = _clip_halfplane(cell, m, n)
            if len(cell) < 3:
                break
        if len(cell) < 3:
            continue

        qpoly = Polygon(cell)
        if not qpoly.is_valid or qpoly.area <= 0:
            continue

        num = 0.0
        denom = 0.0
        for i in idx[c]:
            oc = orig_cells[i]
            if oc is None:
                continue
            inter = qpoly.intersection(oc).area
            if inter > 0:
                num += inter * zv[i]
                denom += inter
        if denom > 0:
            out[c] = num / denom

    return RasterGrid(
        out.reshape(grid.shape), grid, meta={"method": "natural_neighbor"}
    )
