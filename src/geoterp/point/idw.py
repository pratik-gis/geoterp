"""Inverse Distance Weighting (IDW).

Built from scratch on a :class:`scipy.spatial.cKDTree` for speed.  Each output
cell is a weighted average of its ``k`` nearest samples, with weights
``1 / d**power``.  This is the classic Shepard interpolator; there is no
mainstream single-purpose library for it, and the KD-tree version here is both
fast and exact, so we implement it directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz


def idw(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    power: float = 2.0,
    neighbors: int = 12,
    radius: Optional[float] = None,
    smoothing: float = 0.0,
) -> RasterGrid:
    """Interpolate scattered points to a grid with inverse distance weighting.

    Parameters
    ----------
    points, value, y, z:
        The sample data — see :func:`geoterp.io.extract_xyz` for accepted forms.
    grid, resolution, cellsize:
        Output grid.  Pass a ready-made :class:`GridSpec` as ``grid``, or let one
        be derived from the sample extent using ``resolution`` or ``cellsize``.
    power:
        Distance-decay exponent (``2`` is the common default).  Larger values
        make the surface more "peaky" around samples.
    neighbors:
        Number of nearest samples that contribute to each cell.
    radius:
        Optional search radius.  Cells with no sample within ``radius`` are set
        to NaN.  ``None`` means unlimited.
    smoothing:
        Added to squared distance (``d**2 + smoothing``) to avoid singularities
        and gently smooth the surface.  Also removes the exact-hit spikes.
    """
    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )

    tree = cKDTree(np.column_stack([x, yv]))
    k = int(min(neighbors, len(x)))

    X, Y = grid.cell_centers()
    q = np.column_stack([X.ravel(), Y.ravel()])

    dist, idx = tree.query(q, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    out = np.full(q.shape[0], np.nan, dtype=float)

    # Exact hits (distance 0) short-circuit to the sample value.
    exact = dist[:, 0] <= 1e-12
    out[exact] = zv[idx[exact, 0]]

    calc = ~exact
    d = dist[calc]
    if radius is not None:
        within = d <= radius
    else:
        within = np.ones_like(d, dtype=bool)

    w = 1.0 / np.power(d * d + smoothing, power / 2.0)
    w = np.where(within, w, 0.0)
    wsum = w.sum(axis=1)
    vals = (w * zv[idx[calc]]).sum(axis=1)
    good = wsum > 0
    res = np.full(d.shape[0], np.nan)
    res[good] = vals[good] / wsum[good]
    out[calc] = res

    return RasterGrid(out.reshape(grid.shape), grid, meta={"method": "idw", "power": power})
