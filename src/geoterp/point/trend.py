"""Global trend surfaces (polynomial regression).

A trend surface fits a single low-order polynomial in ``(x, y)`` to *all* the
samples by ordinary least squares.  It is a global, smooth model that captures
broad spatial drift rather than local detail.  There is no dedicated library for
this — it is a few lines of ``numpy.linalg.lstsq`` — so we build it from
scratch, which also lets us return useful regression diagnostics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz


def _design_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial design matrix with all terms ``x**i * y**j`` where i+j<=degree."""
    cols = []
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            cols.append((x**i) * (y**j))
    return np.column_stack(cols)


def trend_surface(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    degree: int = 2,
) -> RasterGrid:
    """Fit a global polynomial trend surface of the given ``degree``.

    ``degree=1`` is a tilted plane, ``degree=2`` a quadratic, and so on.  The
    fitted R² and coefficients are stored in the returned grid's ``meta``.
    Coordinates are centred and scaled internally for numerical stability.
    """
    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if degree < 1:
        raise ValueError("degree must be >= 1")
    n_terms = (degree + 1) * (degree + 2) // 2
    if len(x) < n_terms:
        raise ValueError(
            f"Need at least {n_terms} samples for a degree-{degree} surface, "
            f"got {len(x)}."
        )
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )

    # Centre/scale for conditioning.
    x0, y0 = x.mean(), yv.mean()
    scale = max(x.std(), yv.std(), 1e-9)
    xs = (x - x0) / scale
    ys = (yv - y0) / scale

    A = _design_matrix(xs, ys, degree)
    coef, *_ = np.linalg.lstsq(A, zv, rcond=None)

    pred = A @ coef
    ss_res = float(np.sum((zv - pred) ** 2))
    ss_tot = float(np.sum((zv - zv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    X, Y = grid.cell_centers()
    Xs = (X.ravel() - x0) / scale
    Ys = (Y.ravel() - y0) / scale
    Ag = _design_matrix(Xs, Ys, degree)
    out = (Ag @ coef).reshape(grid.shape)

    return RasterGrid(
        out,
        grid,
        meta={"method": "trend_surface", "degree": degree, "r2": r2, "coef": coef},
    )
