"""Core data structures shared across all interpolation methods.

The two workhorses here are :class:`GridSpec` (a description of a regular
output grid: extent + cell size + CRS) and :class:`RasterGrid` (an actual
2-D array of values living on such a grid).  Every point-to-surface method in
:mod:`geoterp` returns a :class:`RasterGrid`, and every raster method both
consumes and returns one, so these two classes are the lingua franca of the
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union

import numpy as np

try:  # affine ships with rasterio; keep the import soft for lightweight installs
    from affine import Affine
except Exception:  # pragma: no cover - affine is a rasterio dependency
    Affine = None  # type: ignore

CRSLike = Union[str, int, None]


def _to_affine(xmin: float, ymax: float, cellsize: float) -> "Affine":
    if Affine is None:  # pragma: no cover
        raise ImportError("The 'affine' package is required for raster transforms.")
    # North-up transform: origin at the *top-left* corner of the grid.
    return Affine(cellsize, 0.0, xmin, 0.0, -cellsize, ymax)


@dataclass
class GridSpec:
    """Description of a regular, axis-aligned output grid.

    Parameters
    ----------
    xmin, ymin, xmax, ymax:
        Bounding box of the grid, expressed in CRS units and interpreted as the
        *outer* edges of the corner cells.
    cellsize:
        Side length of a (square) grid cell in CRS units.
    crs:
        Coordinate reference system (anything ``pyproj``/``rasterio`` accept,
        e.g. ``"EPSG:4326"`` or ``32643``).  May be ``None`` for
        unprojected/synthetic data.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    cellsize: float
    crs: CRSLike = None

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_bounds(
        cls,
        bounds: Tuple[float, float, float, float],
        *,
        cellsize: Optional[float] = None,
        resolution: Optional[int] = None,
        buffer: float = 0.0,
        crs: CRSLike = None,
    ) -> "GridSpec":
        """Build a grid from a ``(xmin, ymin, xmax, ymax)`` bounding box.

        Exactly one of ``cellsize`` or ``resolution`` must be given.
        ``resolution`` is the approximate number of cells along the *longer*
        side of the extent; the cell size is derived from it so cells stay
        square.  ``buffer`` (in CRS units) pads the extent on every side.
        """
        xmin, ymin, xmax, ymax = bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            raise ValueError(f"Degenerate bounds: {bounds!r}")

        if cellsize is None and resolution is None:
            resolution = 100
        if cellsize is not None and resolution is not None:
            raise ValueError("Pass either 'cellsize' or 'resolution', not both.")
        if cellsize is None:
            if resolution is None or resolution < 1:
                raise ValueError("resolution must be a positive integer.")
            cellsize = max(width, height) / float(resolution)
        if cellsize <= 0:
            raise ValueError("cellsize must be positive.")
        return cls(xmin, ymin, xmax, ymax, float(cellsize), crs)

    @classmethod
    def from_points(
        cls,
        x: Iterable[float],
        y: Iterable[float],
        *,
        cellsize: Optional[float] = None,
        resolution: Optional[int] = None,
        buffer: Optional[float] = None,
        crs: CRSLike = None,
    ) -> "GridSpec":
        """Build a grid that covers a point cloud.

        If ``buffer`` is ``None`` a default pad of 5% of the extent is applied,
        which keeps edge samples away from the very border of the surface.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            raise ValueError("No points supplied.")
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if buffer is None:
            span = max(xmax - xmin, ymax - ymin)
            buffer = 0.05 * span if span > 0 else 1.0
        return cls.from_bounds(
            (xmin, ymin, xmax, ymax),
            cellsize=cellsize,
            resolution=resolution,
            buffer=buffer,
            crs=crs,
        )

    # -- geometry --------------------------------------------------------
    @property
    def ncols(self) -> int:
        return max(1, int(np.ceil((self.xmax - self.xmin) / self.cellsize)))

    @property
    def nrows(self) -> int:
        return max(1, int(np.ceil((self.ymax - self.ymin) / self.cellsize)))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nrows, self.ncols)

    @property
    def transform(self) -> "Affine":
        return _to_affine(self.xmin, self.ymax, self.cellsize)

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` mesh-grids of cell-centre coordinates.

        Row 0 is the *northern-most* row so the arrays line up with an image
        whose origin is top-left, matching :attr:`transform`.
        """
        cx = self.xmin + (np.arange(self.ncols) + 0.5) * self.cellsize
        cy = self.ymax - (np.arange(self.nrows) + 0.5) * self.cellsize
        return np.meshgrid(cx, cy)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GridSpec(bounds=({self.xmin:.4g}, {self.ymin:.4g}, "
            f"{self.xmax:.4g}, {self.ymax:.4g}), cellsize={self.cellsize:.4g}, "
            f"shape={self.shape}, crs={self.crs})"
        )


@dataclass
class RasterGrid:
    """A 2-D array of values living on a :class:`GridSpec`.

    ``data`` is stored north-up (row 0 = top).  ``nodata`` marks empty cells;
    the default of ``NaN`` plays nicely with the interpolators, which fill
    unreachable cells with ``NaN``.
    """

    data: np.ndarray
    spec: GridSpec
    nodata: float = np.nan
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError("RasterGrid.data must be 2-D (rows, cols).")

    @property
    def transform(self) -> "Affine":
        return self.spec.transform

    @property
    def crs(self) -> CRSLike:
        return self.spec.crs

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.spec.xmin, self.spec.ymin, self.spec.xmax, self.spec.ymax)

    def filled(self, value: float = np.nan) -> np.ndarray:
        """Return the data array with ``nodata`` replaced by ``value``."""
        arr = np.array(self.data, dtype=float)
        if np.isnan(self.nodata):
            mask = np.isnan(arr)
        else:
            mask = arr == self.nodata
        arr[mask] = value
        return arr

    def to_geotiff(self, path: str, dtype: str = "float32") -> str:
        """Write the grid to a GeoTIFF using rasterio.  Returns ``path``."""
        import rasterio

        arr = self.data.astype(dtype)
        nodata = self.nodata
        if np.isnan(nodata) and not np.issubdtype(np.dtype(dtype), np.floating):
            nodata = 0
        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
            "compress": "deflate",
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        return path

    def to_xarray(self):  # pragma: no cover - optional dependency
        """Return an ``xarray.DataArray`` view (requires xarray)."""
        import xarray as xr

        X, Y = self.spec.cell_centers()
        return xr.DataArray(
            self.data,
            dims=("y", "x"),
            coords={"y": Y[:, 0], "x": X[0, :]},
            name=self.meta.get("name", "value"),
        )

    def plot(self, ax=None, cmap: str = "viridis", **kwargs):  # pragma: no cover
        """Quick-look plot via matplotlib.  Returns the image handle."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        extent = (self.spec.xmin, self.spec.xmax, self.spec.ymin, self.spec.ymax)
        im = ax.imshow(self.filled(np.nan), extent=extent, origin="upper", cmap=cmap, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return im

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        valid = np.count_nonzero(~np.isnan(self.filled(np.nan)))
        return (
            f"RasterGrid(shape={self.shape}, cellsize={self.spec.cellsize:.4g}, "
            f"valid={valid}/{self.data.size}, crs={self.crs})"
        )
