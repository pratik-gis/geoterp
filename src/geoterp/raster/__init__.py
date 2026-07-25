"""Raster-to-raster interpolation and aggregation.

:func:`resample`
    Change resolution with an interpolation kernel (nearest, bilinear, cubic
    convolution, cubic spline, Lanczos/sinc).
:func:`aggregate`
    Coarsen by integer block reduction (average, sum, min, max, median, mode).
"""

from .aggregate import aggregate
from .resample import resample

__all__ = ["resample", "aggregate"]
