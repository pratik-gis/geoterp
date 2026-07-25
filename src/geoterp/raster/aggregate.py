"""Raster aggregation (downsampling by summary statistics).

Coarsen a raster by an integer factor, replacing each block of cells with a
single summary value: mean, mode, min, max, sum or median.  Implemented as an
exact NumPy block reduction so results are deterministic and NoData (NaN) cells
are ignored per block — no GDAL-version quirks for ``sum``/``mode``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import GridSpec, RasterGrid
from ..io import read_raster

_FUNCS = ("average", "mean", "sum", "min", "max", "median", "mode")


def _block_view(a: np.ndarray, fr: int, fc: int) -> np.ndarray:
    """Trim ``a`` to a multiple of the factors and reshape into blocks.

    Returns an array of shape ``(nR, nC, fr*fc)`` where the last axis holds the
    cells of each block.
    """
    nr = (a.shape[0] // fr) * fr
    nc = (a.shape[1] // fc) * fc
    a = a[:nr, :nc]
    R = nr // fr
    C = nc // fc
    return a.reshape(R, fr, C, fc).transpose(0, 2, 1, 3).reshape(R, C, fr * fc)


def aggregate(
    raster,
    *,
    factor: int = 2,
    factor_y: Optional[int] = None,
    method: str = "average",
) -> RasterGrid:
    """Aggregate (coarsen) a raster by integer block reduction.

    Parameters
    ----------
    raster:
        A :class:`RasterGrid` or a path to a single-band raster.
    factor:
        Block size in the x-direction (and y, unless ``factor_y`` is given).
    factor_y:
        Optional separate block size in the y-direction.
    method:
        ``average``/``mean``, ``sum``, ``min``, ``max``, ``median`` or ``mode``.
    """
    if isinstance(raster, str):
        raster = read_raster(raster)
    if method not in _FUNCS:
        raise ValueError(f"Unknown method {method!r}; choose from {_FUNCS}")

    fr = int(factor_y or factor)
    fc = int(factor)
    if fr < 1 or fc < 1:
        raise ValueError("factors must be >= 1")

    a = raster.filled(np.nan).astype(float)
    blocks = _block_view(a, fr, fc)  # (R, C, fr*fc)

    with np.errstate(invalid="ignore", all="ignore"):
        if method in ("average", "mean"):
            out = np.nanmean(blocks, axis=2)
        elif method == "sum":
            out = np.nansum(blocks, axis=2)
            out[np.all(np.isnan(blocks), axis=2)] = np.nan
        elif method == "min":
            out = np.nanmin(blocks, axis=2)
        elif method == "max":
            out = np.nanmax(blocks, axis=2)
        elif method == "median":
            out = np.nanmedian(blocks, axis=2)
        elif method == "mode":
            out = _nanmode(blocks)

    spec = raster.spec
    new = GridSpec(
        spec.xmin,
        spec.ymax - out.shape[0] * spec.cellsize * fr,
        spec.xmin + out.shape[1] * spec.cellsize * fc,
        spec.ymax,
        spec.cellsize * fc,  # square cells assume fr == fc; documented below
        spec.crs,
    )
    if fr != fc:
        # Non-square aggregation: keep x cell size; note y differs.
        new.ymin = spec.ymax - out.shape[0] * spec.cellsize * fr
    return RasterGrid(out, new, meta={"method": f"aggregate_{method}", "factor": (fr, fc)})


def _nanmode(blocks: np.ndarray) -> np.ndarray:
    """Per-block most common value, ignoring NaN.  For categorical rasters."""
    R, C, _ = blocks.shape
    out = np.full((R, C), np.nan)
    for i in range(R):
        for j in range(C):
            vals = blocks[i, j]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            uniq, counts = np.unique(vals, return_counts=True)
            out[i, j] = uniq[np.argmax(counts)]
    return out
