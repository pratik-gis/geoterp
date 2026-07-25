"""Raster resampling with interpolation kernels.

GDAL (through rasterio) already ships fast, correct implementations of every
kernel we need — nearest, bilinear, cubic convolution and Lanczos/sinc — so we
drive ``rasterio.warp.reproject`` rather than hand-roll convolutions.  The
function changes a raster's resolution while keeping its extent and CRS fixed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core import GridSpec, RasterGrid
from ..io import read_raster

_METHODS = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "cubic": "cubic",            # cubic convolution
    "cubic_spline": "cubic_spline",
    "lanczos": "lanczos",        # sinc-based
    "average": "average",
    "mode": "mode",
}


def _resolve_method(name: str):
    from rasterio.enums import Resampling

    if name not in _METHODS:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(_METHODS)}")
    return getattr(Resampling, _METHODS[name])


def resample(
    raster,
    *,
    scale: Optional[float] = None,
    cellsize: Optional[float] = None,
    shape: Optional[Tuple[int, int]] = None,
    method: str = "bilinear",
) -> RasterGrid:
    """Resample a raster to a new resolution.

    Provide exactly one of ``scale`` (e.g. ``2`` doubles resolution / halves the
    cell size), ``cellsize`` (target cell size in CRS units), or ``shape``
    (target ``(rows, cols)``).

    Parameters
    ----------
    raster:
        A :class:`RasterGrid` or a path to a single-band raster.
    method:
        ``nearest``, ``bilinear``, ``cubic`` (convolution), ``cubic_spline``,
        or ``lanczos`` (sinc).
    """
    import rasterio
    from rasterio.warp import reproject

    if isinstance(raster, str):
        raster = read_raster(raster)

    spec = raster.spec
    given = [v is not None for v in (scale, cellsize, shape)]
    if sum(given) != 1:
        raise ValueError("Pass exactly one of scale, cellsize, or shape.")

    if scale is not None:
        new_cell = spec.cellsize / float(scale)
        new_spec = GridSpec(spec.xmin, spec.ymin, spec.xmax, spec.ymax, new_cell, spec.crs)
        dst_shape = new_spec.shape
    elif cellsize is not None:
        new_spec = GridSpec(spec.xmin, spec.ymin, spec.xmax, spec.ymax, float(cellsize), spec.crs)
        dst_shape = new_spec.shape
    else:
        nrows, ncols = shape
        new_cell = (spec.xmax - spec.xmin) / ncols
        new_spec = GridSpec(spec.xmin, spec.ymin, spec.xmax, spec.ymax, new_cell, spec.crs)
        dst_shape = (int(nrows), int(ncols))

    src = raster.data.astype("float32")
    src_nodata = np.nan
    dst = np.full(dst_shape, np.nan, dtype="float32")

    reproject(
        source=src,
        destination=dst,
        src_transform=spec.transform,
        src_crs=spec.crs or "EPSG:4326",
        dst_transform=new_spec.transform,
        dst_crs=new_spec.crs or "EPSG:4326",
        src_nodata=src_nodata,
        dst_nodata=np.nan,
        resampling=_resolve_method(method),
    )

    return RasterGrid(dst, new_spec, meta={"method": f"resample_{method}"})
