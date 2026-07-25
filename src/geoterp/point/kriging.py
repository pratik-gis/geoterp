"""Geostatistical interpolation (kriging) via PyKrige.

PyKrige is the reference implementation for ordinary and universal kriging in
Python, so we wrap it rather than reimplement variogram fitting.  Both the
predicted surface and, optionally, the kriging variance are returned.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz

_VARIOGRAMS = {
    "linear",
    "power",
    "gaussian",
    "spherical",
    "exponential",
    "hole-effect",
}


def kriging(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    method: str = "ordinary",
    variogram_model: str = "spherical",
    nlags: int = 6,
    drift_degree: int = 1,
    return_variance: bool = False,
):
    """Ordinary or universal kriging onto a grid.

    Parameters
    ----------
    method:
        ``"ordinary"`` (default) or ``"universal"``.  Universal kriging models a
        polynomial trend of order ``drift_degree`` in addition to the residual.
    variogram_model:
        One of ``linear``, ``power``, ``gaussian``, ``spherical``,
        ``exponential``, ``hole-effect``.  The model is auto-fit to the data.
    return_variance:
        If ``True``, return ``(surface, variance)`` where ``variance`` is a
        :class:`RasterGrid` of the kriging estimation variance.
    """
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging

    if variogram_model not in _VARIOGRAMS:
        raise ValueError(
            f"Unknown variogram_model {variogram_model!r}; choose from {sorted(_VARIOGRAMS)}"
        )

    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )

    gx = grid.xmin + (np.arange(grid.ncols) + 0.5) * grid.cellsize
    gy = grid.ymax - (np.arange(grid.nrows) + 0.5) * grid.cellsize

    # Degenerate (zero-variance) data has no variogram to fit; the kriging
    # surface is simply that constant, with zero estimation variance.
    if np.ptp(zv) <= 1e-12:
        const = float(zv[0]) if len(zv) else np.nan
        surface = RasterGrid(
            np.full(grid.shape, const), grid,
            meta={"method": f"{method}_kriging", "variogram": "degenerate"},
        )
        if return_variance:
            return surface, RasterGrid(np.zeros(grid.shape), grid,
                                       meta={"method": "kriging_variance"})
        return surface

    if method == "ordinary":
        model = OrdinaryKriging(
            x, yv, zv, variogram_model=variogram_model, nlags=nlags, verbose=False,
            enable_plotting=False,
        )
        zhat, var = model.execute("grid", gx, gy)
    elif method == "universal":
        model = UniversalKriging(
            x, yv, zv, variogram_model=variogram_model, nlags=nlags,
            drift_terms=["regional_linear"] if drift_degree >= 1 else [],
            verbose=False, enable_plotting=False,
        )
        zhat, var = model.execute("grid", gx, gy)
    else:
        raise ValueError("method must be 'ordinary' or 'universal'.")

    # gy was supplied north-first, so PyKrige's row 0 is already the northern
    # edge — matching RasterGrid's north-up convention. No flip needed.
    zhat = np.asarray(zhat)
    var = np.asarray(var)

    surface = RasterGrid(
        zhat, grid,
        meta={"method": f"{method}_kriging", "variogram": variogram_model},
    )
    if return_variance:
        variance = RasterGrid(var, grid, meta={"method": "kriging_variance"})
        return surface, variance
    return surface
