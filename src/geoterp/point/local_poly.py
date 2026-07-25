"""Local polynomial interpolation (moving-window / LOESS-style regression).

At every output cell a low-order polynomial is fit to the nearby samples,
weighted by a distance kernel, and evaluated at the cell centre.  It behaves
like a locally adaptive trend surface: smoother than IDW, more flexible than a
single global polynomial.  No mainstream GIS library exposes this directly, so
it is implemented from scratch here on a KD-tree.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz
from .trend import _design_matrix


def _tricube(u: np.ndarray) -> np.ndarray:
    w = (1.0 - np.clip(u, 0.0, 1.0) ** 3) ** 3
    return w


def local_polynomial(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    degree: int = 1,
    neighbors: int = 16,
    radius: Optional[float] = None,
) -> RasterGrid:
    """Locally weighted polynomial regression onto a grid.

    Parameters
    ----------
    degree:
        Order of the local polynomial (``1`` = local plane, ``2`` = quadratic).
    neighbors:
        Number of nearest samples used in each local fit.  The kernel bandwidth
        is the distance to the furthest of these, giving spatially adaptive
        smoothing.
    radius:
        Optional hard cap on the search distance; cells whose local window is
        empty become NaN.
    """
    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )

    n_terms = (degree + 1) * (degree + 2) // 2
    k = int(min(max(neighbors, n_terms), len(x)))
    tree = cKDTree(np.column_stack([x, yv]))

    X, Y = grid.cell_centers()
    q = np.column_stack([X.ravel(), Y.ravel()])
    dist, idx = tree.query(q, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    out = np.full(q.shape[0], np.nan)
    x0 = x[idx]  # (Ncells, k)
    y0 = yv[idx]
    z0 = zv[idx]

    for c in range(q.shape[0]):
        d = dist[c]
        bw = d[-1]
        if radius is not None and d[0] > radius:
            continue
        if bw <= 0:
            out[c] = z0[c, 0]
            continue
        w = _tricube(d / bw)
        if radius is not None:
            w = np.where(d <= radius, w, 0.0)
        if not np.any(w > 0):
            continue
        # Local coordinates relative to the query cell for conditioning.
        xl = x0[c] - q[c, 0]
        yl = y0[c] - q[c, 1]
        A = _design_matrix(xl, yl, degree)
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        bw_vec = z0[c] * sw
        coef, *_ = np.linalg.lstsq(Aw, bw_vec, rcond=None)
        # At the query point local coords are 0, so prediction is the intercept.
        out[c] = coef[0]

    return RasterGrid(
        out.reshape(grid.shape),
        grid,
        meta={"method": "local_polynomial", "degree": degree},
    )
