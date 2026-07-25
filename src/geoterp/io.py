"""Input coercion helpers.

The public interpolation functions are deliberately liberal in what they
accept: a GeoDataFrame of points, three plain arrays, an ``(N, 2)`` coordinate
array plus values, or a path to any OGR-readable vector file.  Everything is
funnelled through :func:`extract_xyz` so the numerical code downstream only
ever sees clean ``float`` arrays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _is_geodataframe(obj) -> bool:
    try:
        import geopandas as gpd
    except Exception:  # pragma: no cover
        return False
    return isinstance(obj, gpd.GeoDataFrame)


def read_vector(path: str):
    """Read a vector file into a GeoDataFrame (thin wrapper around geopandas)."""
    import geopandas as gpd

    return gpd.read_file(path)


def extract_xyz(
    points,
    value: Optional[str] = None,
    y=None,
    z=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Normalise a point dataset into ``(x, y, z, crs)`` arrays.

    Accepted call patterns
    ----------------------
    * ``extract_xyz(gdf, value="temp")`` – GeoDataFrame with point geometry.
    * ``extract_xyz("points.geojson", value="temp")`` – a vector file path.
    * ``extract_xyz(x, y=y, z=z)`` – three 1-D array-likes.
    * ``extract_xyz(coords, z=z)`` – ``coords`` an ``(N, 2)`` array.

    Returns
    -------
    x, y, z : 1-D float arrays of equal length (rows with NaN values dropped)
    crs : the coordinate reference system if one could be determined, else None
    """
    crs = None

    if isinstance(points, str):
        points = read_vector(points)

    if _is_geodataframe(points):
        gdf = points
        crs = gdf.crs
        geom = gdf.geometry
        if not (geom.geom_type == "Point").all():
            # Fall back to representative points for non-point geometries.
            geom = geom.representative_point()
        x = geom.x.to_numpy(dtype=float)
        yv = geom.y.to_numpy(dtype=float)
        if value is None:
            raise ValueError("A 'value' column name is required for GeoDataFrame input.")
        if value not in gdf.columns:
            raise KeyError(f"Column {value!r} not found. Available: {list(gdf.columns)}")
        zv = np.asarray(gdf[value].to_numpy(), dtype=float)
        return _finite(x, yv, zv, crs)

    # Array-based inputs -------------------------------------------------
    arr = np.asarray(points, dtype=float)
    if y is None and arr.ndim == 2 and arr.shape[1] >= 2:
        x = arr[:, 0]
        yv = arr[:, 1]
        if z is None:
            if arr.shape[1] >= 3:
                zv = arr[:, 2]
            else:
                raise ValueError("No z/values supplied for coordinate array.")
        else:
            zv = np.asarray(z, dtype=float)
        return _finite(x, yv, zv, crs)

    if y is None or z is None:
        raise ValueError(
            "Provide either a GeoDataFrame/path with 'value', an (N,2/3) array, "
            "or x, y=..., z=... arrays."
        )
    x = arr.ravel()
    yv = np.asarray(y, dtype=float).ravel()
    zv = np.asarray(z, dtype=float).ravel()
    return _finite(x, yv, zv, crs)


def _finite(x, y, z, crs):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (len(x) == len(y) == len(z)):
        raise ValueError(
            f"x, y, z must be equal length (got {len(x)}, {len(y)}, {len(z)})."
        )
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not mask.any():
        raise ValueError("No finite (x, y, z) samples remain after cleaning.")
    return x[mask], y[mask], z[mask], crs


def read_raster(path: str):
    """Read a single-band raster, returning ``(array, RasterGrid-spec)``.

    Returns the data as a masked-aware float array plus a
    :class:`~geoterp.core.RasterGrid`.
    """
    import rasterio

    from .core import GridSpec, RasterGrid

    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        t = src.transform
        cellsize = abs(t.a)
        spec = GridSpec(
            xmin=src.bounds.left,
            ymin=src.bounds.bottom,
            xmax=src.bounds.right,
            ymax=src.bounds.top,
            cellsize=cellsize,
            crs=src.crs,
        )
    return RasterGrid(arr, spec, nodata=np.nan, meta={"source": path})
