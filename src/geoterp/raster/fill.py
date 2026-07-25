"""Filling voids / NoData holes in a raster by interpolation.

DEMs and other rasters routinely arrive with holes — cloud gaps, water masks,
sensor dropouts, or the "no returns" voids of SRTM.  This module fills those
cells from the surrounding valid data using the same interpolation ideas as the
rest of the package, but applied *within* a single raster instead of from
scattered points.

Methods
-------
``idw`` (default)
    GDAL's ``FillNodata`` (via rasterio) — an inverse-distance-weighted fill
    along the void edges with optional smoothing.  Fast, robust, best-in-class
    for large rasters, so it is wrapped rather than reimplemented.
``nearest`` / ``linear`` / ``cubic``
    ``scipy.interpolate.griddata`` treating every valid cell as a sample.
    ``linear``/``cubic`` leave voids outside the valid convex hull, which are
    then closed with a nearest-neighbour pass so no hole survives.
``rbf``
    A locally-neighboured radial-basis-function (thin-plate) fill — the
    smoothest option for isolated voids.
``laplace``
    An iterative diffusion / minimum-curvature inpaint built from scratch: hole
    cells relax to the average of their neighbours until convergence, giving a
    smooth, artefact-free fill that honours the void boundary exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import RasterGrid
from ..io import read_raster


def _hole_mask(arr: np.ndarray) -> np.ndarray:
    return ~np.isfinite(arr)


def fill_nodata(
    raster,
    *,
    method: str = "idw",
    max_search_distance: float = 100.0,
    smoothing_iterations: int = 0,
    neighbors: int = 16,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> RasterGrid:
    """Fill NoData/NaN holes in a raster by interpolation.

    Parameters
    ----------
    raster:
        A :class:`RasterGrid` or a path to a single-band raster.  Holes are the
        cells that are NaN (or equal to the file's NoData value on read).
    method:
        ``idw`` (default), ``nearest``, ``linear``, ``cubic``, ``rbf`` or
        ``laplace`` — see the module docstring.
    max_search_distance:
        (``idw`` only) maximum number of cells to search outward for values.
    smoothing_iterations:
        (``idw`` only) post-fill 3×3 smoothing passes over the filled voids.
    neighbors:
        (``rbf`` only) number of nearest valid cells used per void cell.
    max_iter, tol:
        (``laplace`` only) iteration budget and convergence threshold.

    Returns
    -------
    RasterGrid
        A copy with every hole filled.  Cells that were already valid are
        untouched.
    """
    if isinstance(raster, str):
        raster = read_raster(raster)

    arr = raster.filled(np.nan).astype(float)
    holes = _hole_mask(arr)
    if not holes.any():
        return RasterGrid(arr.copy(), raster.spec, meta={"method": f"fill_{method}",
                                                         "filled": 0})
    if holes.all():
        raise ValueError("Raster is entirely NoData; nothing to interpolate from.")

    if method == "idw":
        out = _fill_gdal(arr, holes, max_search_distance, smoothing_iterations)
    elif method in ("nearest", "linear", "cubic"):
        out = _fill_griddata(arr, holes, method)
    elif method in ("rbf", "spline"):
        out = _fill_rbf(arr, holes, neighbors)
    elif method == "laplace":
        out = _fill_laplace(arr, holes, max_iter, tol)
    else:
        raise ValueError(
            "method must be one of: idw, nearest, linear, cubic, rbf, laplace"
        )

    return RasterGrid(
        out, raster.spec,
        meta={"method": f"fill_{method}", "filled": int(holes.sum())},
    )


# --- implementations -----------------------------------------------------
def _fill_gdal(arr, holes, max_search_distance, smoothing_iterations):
    from rasterio.fill import fillnodata

    # fillnodata treats mask==0 as holes to fill.
    mask = (~holes).astype(np.uint8)
    filled = fillnodata(
        arr.astype("float64"),
        mask=mask,
        max_search_distance=float(max_search_distance),
        smoothing_iterations=int(smoothing_iterations),
    )
    return np.asarray(filled, dtype=float)


def _fill_griddata(arr, holes, method):
    from scipy.interpolate import griddata

    valid = ~holes
    ys, xs = np.nonzero(valid)
    vals = arr[valid]
    hy, hx = np.nonzero(holes)

    out = arr.copy()
    interp = griddata((ys, xs), vals, (hy, hx), method=method)
    out[hy, hx] = interp

    # linear/cubic leave NaN outside the convex hull — close them with nearest.
    still = ~np.isfinite(out)
    if still.any():
        ny, nx = np.nonzero(still)
        near = griddata((ys, xs), vals, (ny, nx), method="nearest")
        out[ny, nx] = near
    return out


def _fill_rbf(arr, holes, neighbors):
    from scipy.interpolate import RBFInterpolator

    valid = ~holes
    ys, xs = np.nonzero(valid)
    obs = np.column_stack([xs, ys]).astype(float)
    vals = arr[valid]
    hy, hx = np.nonzero(holes)
    query = np.column_stack([hx, hy]).astype(float)

    k = int(min(neighbors, len(vals)))
    interp = RBFInterpolator(obs, vals, kernel="thin_plate_spline", neighbors=k)
    out = arr.copy()
    out[hy, hx] = interp(query)
    return out


def _fill_laplace(arr, holes, max_iter, tol):
    """Iterative Laplace/diffusion inpaint (minimum-curvature-like).

    Hole cells relax toward the mean of their 4-neighbours while valid cells
    stay pinned, so the fill joins the boundary smoothly with no new extrema.
    """
    out = arr.copy()
    # Seed holes with a global mean so the relaxation starts from a sane value.
    out[holes] = np.nanmean(arr)
    scale = max(np.nanmax(arr) - np.nanmin(arr), 1e-9)

    for it in range(max_iter):
        up = np.vstack([out[:1], out[:-1]])
        down = np.vstack([out[1:], out[-1:]])
        left = np.hstack([out[:, :1], out[:, :-1]])
        right = np.hstack([out[:, 1:], out[:, -1:]])
        avg = 0.25 * (up + down + left + right)
        new = np.where(holes, avg, out)
        change = np.abs(new[holes] - out[holes]).max() / scale
        out = new
        if change < tol:
            break
    return out
