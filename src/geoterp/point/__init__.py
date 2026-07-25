"""Point-to-surface interpolation methods.

Deterministic
    :func:`idw`, :func:`nearest` (Thiessen), :func:`natural_neighbor`,
    :func:`spline`, :func:`trend_surface`, :func:`local_polynomial`
Geostatistical
    :func:`kriging`
Vector output
    :func:`voronoi_polygons`
"""

from .idw import idw
from .kriging import kriging
from .local_poly import local_polynomial
from .natural_neighbor import natural_neighbor
from .nearest import nearest, voronoi_polygons
from .spline import spline
from .trend import trend_surface

__all__ = [
    "idw",
    "nearest",
    "voronoi_polygons",
    "natural_neighbor",
    "spline",
    "trend_surface",
    "local_polynomial",
    "kriging",
]
