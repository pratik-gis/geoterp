"""Pycnophylactic interpolation (Tobler, 1979), from scratch.

Tobler's method turns choropleth counts into a *smooth* density surface that

* has no sharp discontinuities at zone borders, and
* is **mass-preserving** ("pycnophylactic"): the values inside each source zone
  still sum to that zone's original total.

The algorithm rasterises the zones, seeds every cell with its zone's uniform
density, then repeats {smooth with a Laplacian filter → restore each zone's mass}
until the surface stops changing.  There is no maintained library for this on
modern Python, so it is implemented here on plain NumPy arrays.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import GridSpec, RasterGrid


def pycnophylactic(
    source,
    value: str,
    *,
    grid: Optional[GridSpec] = None,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    relaxation: float = 0.2,
    non_negative: bool = True,
    connectivity: int = 4,
):
    """Smooth, mass-preserving density surface from polygon counts.

    Parameters
    ----------
    source:
        Polygon GeoDataFrame carrying an extensive ``value`` (a count/total).
    value:
        Column of per-zone totals to redistribute.
    grid, resolution, cellsize:
        Output grid (see :class:`GridSpec`).  Finer grids resolve the smooth
        surface better at higher cost.
    max_iter, tol:
        Iteration budget and convergence threshold (max relative cell change).
    relaxation:
        Blend factor for each smoothing step (0–1); smaller is more stable.
    non_negative:
        Clamp densities at 0 each iteration (recommended for counts).
    connectivity:
        ``4`` or ``8`` neighbour Laplacian smoothing.

    Returns
    -------
    RasterGrid
        Per-cell **density** (value per cell).  Summing the cells of a zone
        reproduces that zone's input total; multiply by nothing further — the
        surface is already in "count per cell" units.
    """
    import rasterio.features

    gdf = source
    if grid is None:
        b = gdf.total_bounds  # xmin, ymin, xmax, ymax
        grid = GridSpec.from_bounds(
            tuple(b), resolution=resolution, cellsize=cellsize, buffer=0.0,
            crs=gdf.crs,
        )

    nrows, ncols = grid.shape
    transform = grid.transform

    # Rasterise zone ids (1-based; 0 = background).
    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry.values))
    zone = rasterio.features.rasterize(
        shapes, out_shape=(nrows, ncols), transform=transform, fill=0,
        dtype="int32", all_touched=False,
    )

    totals = gdf[value].to_numpy(dtype=float)
    n_zones = len(totals)

    # Precompute per-zone cell masks and counts.
    zone_masks = [zone == (i + 1) for i in range(n_zones)]
    zone_counts = np.array([m.sum() for m in zone_masks], dtype=float)

    # Seed with uniform density = total / (#cells in zone).
    surf = np.zeros((nrows, ncols), dtype=float)
    for i, m in enumerate(zone_masks):
        if zone_counts[i] > 0:
            surf[m] = totals[i] / zone_counts[i]

    active = zone > 0

    def smooth(a: np.ndarray) -> np.ndarray:
        # Neighbour sum with edge replication.
        up = np.vstack([a[:1], a[:-1]])
        down = np.vstack([a[1:], a[-1:]])
        left = np.hstack([a[:, :1], a[:, :-1]])
        right = np.hstack([a[:, 1:], a[:, -1:]])
        s = up + down + left + right
        cnt = 4.0
        if connectivity == 8:
            # Diagonal neighbours via shifting the already-shifted arrays.
            upleft = np.hstack([up[:, :1], up[:, :-1]])
            upright = np.hstack([up[:, 1:], up[:, -1:]])
            downleft = np.hstack([down[:, :1], down[:, :-1]])
            downright = np.hstack([down[:, 1:], down[:, -1:]])
            s = s + upleft + upright + downleft + downright
            cnt = 8.0
        return s / cnt

    for it in range(max_iter):
        prev = surf
        smoothed = smooth(surf)
        surf = surf + relaxation * (smoothed - surf)
        # Keep background cells from drifting; they stay at 0.
        surf[~active] = 0.0
        if non_negative:
            np.clip(surf, 0.0, None, out=surf)

        # Restore each zone's mass (additive correction over its cells).
        for i, m in enumerate(zone_masks):
            if zone_counts[i] == 0:
                continue
            cur = surf[m].sum()
            surf[m] += (totals[i] - cur) / zone_counts[i]
        if non_negative:
            np.clip(surf, 0.0, None, out=surf)
            surf[~active] = 0.0

        denom = np.maximum(np.abs(prev[active]).max(), 1e-12)
        change = np.abs(surf[active] - prev[active]).max() / denom
        if change < tol:
            break

    surf[~active] = np.nan
    return RasterGrid(
        surf, grid,
        meta={"method": "pycnophylactic", "iterations": it + 1, "value": value},
    )


def pycnophylactic_to_polygons(
    source,
    target,
    value: str,
    *,
    resolution: Optional[int] = None,
    cellsize: Optional[float] = None,
    suffix: str = "_est",
    **kwargs,
):
    """Run pycnophylactic interpolation and re-aggregate onto ``target`` polygons.

    Convenience wrapper: builds the smooth surface, then sums the density cells
    whose centres fall in each target polygon.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    surface = pycnophylactic(
        source, value, resolution=resolution, cellsize=cellsize, **kwargs
    )
    X, Y = surface.spec.cell_centers()
    vals = surface.data
    mask = np.isfinite(vals)
    pts = gpd.GeoDataFrame(
        {"__v": vals[mask]},
        geometry=[Point(xy) for xy in zip(X[mask], Y[mask])],
        crs=source.crs,
    )
    tgt = target.copy()
    tgt["__tgt_id"] = np.arange(len(tgt))
    joined = gpd.sjoin(pts, tgt[["__tgt_id", "geometry"]], predicate="within")
    grp = joined.groupby("__tgt_id")["__v"].sum()
    out = np.zeros(len(target), dtype=float)
    for tid, v in grp.items():
        out[int(tid)] = v
    result = target.copy()
    result[f"{value}{suffix}"] = out
    return result
