"""Raster-to-raster interpolation and aggregation.

:func:`resample`
    Change resolution with an interpolation kernel (nearest, bilinear, cubic
    convolution, cubic spline, Lanczos/sinc).
:func:`aggregate`
    Coarsen by integer block reduction (average, sum, min, max, median, mode).
:func:`fill_nodata`
    Fill NoData/void holes by interpolation (idw, nearest, linear, cubic, rbf,
    laplace).
"""

from .aggregate import aggregate
from .fill import fill_nodata
from .resample import resample

__all__ = ["resample", "aggregate", "fill_nodata"]
