"""Polygon-to-polygon (areal) interpolation methods.

:func:`areal_weighting`
    Plain area-of-overlap redistribution (extensive or intensive).
:func:`dasymetric`
    Area weighting refined by ancillary density zones.
:func:`pycnophylactic` / :func:`pycnophylactic_to_polygons`
    Tobler's smooth, mass-preserving surface (and its re-aggregation).
"""

from .areal import areal_weighting
from .dasymetric import dasymetric
from .pycnophylactic import pycnophylactic, pycnophylactic_to_polygons

__all__ = [
    "areal_weighting",
    "dasymetric",
    "pycnophylactic",
    "pycnophylactic_to_polygons",
]
