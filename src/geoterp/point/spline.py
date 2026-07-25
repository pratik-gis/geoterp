"""Spline interpolation via radial basis functions.

SciPy's :class:`~scipy.interpolate.RBFInterpolator` is the best-in-class tool
here, so we wrap it rather than reinvent it.  The default ``thin_plate_spline``
kernel reproduces the classic "minimum curvature" spline that GIS users expect;
other kernels (``cubic``, ``multiquadric``, ``gaussian`` …) are exposed too.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RBFInterpolator

from ..core import GridSpec, RasterGrid
from ..io import extract_xyz

# Kernels that require an explicit shape/epsilon parameter.
_SHAPE_KERNELS = {"multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"}


def spline(
    points,
    value: Optional[str] = None,
    *,
    y=None,
    z=None,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
    neighbors: Optional[int] = None,
    epsilon: Optional[float] = None,
) -> RasterGrid:
    """Fit a radial-basis-function spline surface to scattered points.

    Parameters
    ----------
    kernel:
        RBF kernel name accepted by :class:`scipy.interpolate.RBFInterpolator`,
        e.g. ``"thin_plate_spline"`` (default), ``"cubic"``, ``"quintic"``,
        ``"linear"``, ``"multiquadric"``, ``"gaussian"``.
    smoothing:
        Regularisation.  ``0`` interpolates exactly; larger values approximate,
        which is useful for noisy data.
    neighbors:
        If set, use a local RBF built from the ``neighbors`` nearest samples per
        query point (much faster and more stable for large datasets).
    epsilon:
        Shape parameter for scale-dependent kernels.  Defaults to the mean
        nearest-neighbour spacing when required and not supplied.
    """
    x, yv, zv, crs = extract_xyz(points, value=value, y=y, z=z)
    if grid is None:
        grid = GridSpec.from_points(
            x, yv, resolution=resolution, cellsize=cellsize, crs=crs
        )

    obs = np.column_stack([x, yv])
    if epsilon is None and kernel in _SHAPE_KERNELS:
        epsilon = _default_epsilon(obs)

    kwargs = dict(smoothing=smoothing, kernel=kernel)
    if neighbors is not None:
        kwargs["neighbors"] = int(min(neighbors, len(x)))
    if epsilon is not None:
        kwargs["epsilon"] = epsilon

    interp = RBFInterpolator(obs, zv, **kwargs)

    X, Y = grid.cell_centers()
    q = np.column_stack([X.ravel(), Y.ravel()])
    out = interp(q).reshape(grid.shape)
    return RasterGrid(out, grid, meta={"method": "spline", "kernel": kernel})


def _default_epsilon(obs: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if len(obs) < 2:
        return 1.0
    tree = cKDTree(obs)
    d, _ = tree.query(obs, k=2)
    spacing = float(np.mean(d[:, 1]))
    return spacing if spacing > 0 else 1.0
