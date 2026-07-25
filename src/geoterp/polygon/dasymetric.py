"""Dasymetric mapping.

Dasymetric interpolation refines plain areal weighting with *ancillary*
information about where the quantity actually concentrates.  Instead of
splitting a source value in proportion to overlap **area**, it splits it in
proportion to a **weight** derived from ancillary zones (e.g. land-cover
classes each carrying a relative population density).

The overlay is a three-way intersection ``source × ancillary × target``.  Each
resulting piece receives a weight ``area · density(ancillary_class)``, and each
source's total is distributed across its pieces in proportion to those weights,
then summed back up to the target zones.  If a source has no positive ancillary
weight anywhere, it falls back to pure area weighting so no mass is lost.

Implemented directly on GeoPandas overlays (no raster dependency).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def dasymetric(
    source,
    target,
    value: str,
    ancillary,
    weight_col: str,
    *,
    suffix: str = "_est",
):
    """Dasymetric interpolation of an extensive ``value``.

    Parameters
    ----------
    source:
        Polygons carrying the extensive ``value`` (e.g. population counts).
    target:
        Polygons to estimate the value over.
    ancillary:
        Polygons describing relative density, e.g. land-use classes.
    weight_col:
        Column in ``ancillary`` giving each class's *relative* density weight
        (any non-negative scale; only ratios matter).
    suffix:
        Appended to the ``value`` column name in the output.

    Returns
    -------
    GeoDataFrame
        A copy of ``target`` with the estimated ``value + suffix`` column.  Total
        mass is preserved: the sum over targets equals the sum over sources
        (up to overlaps outside the target extent).
    """
    import geopandas as gpd

    for name, gdf in (("source", source), ("target", target), ("ancillary", ancillary)):
        if gdf.crs is None:
            raise ValueError(f"{name} has no CRS; set one before interpolating.")

    src = source.copy()
    src["__src_id"] = np.arange(len(src))
    tgt = target.copy()
    tgt["__tgt_id"] = np.arange(len(tgt))
    anc = ancillary.copy()
    anc["__w"] = anc[weight_col].to_numpy(dtype=float)

    # source ∩ ancillary, then ∩ target
    sa = gpd.overlay(
        src[["__src_id", value, "geometry"]],
        anc[["__w", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    pieces = gpd.overlay(
        sa,
        tgt[["__tgt_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    pieces["__area"] = pieces.geometry.area
    pieces["__weight"] = pieces["__area"] * pieces["__w"].clip(lower=0)

    # Per-source normaliser; fall back to area if all ancillary weights are zero.
    wsum = pieces.groupby("__src_id")["__weight"].transform("sum")
    asum = pieces.groupby("__src_id")["__area"].transform("sum")
    use_area = wsum <= 0
    norm = np.where(use_area, pieces["__area"] / asum.replace(0, np.nan),
                    pieces["__weight"] / wsum.replace(0, np.nan))
    pieces["__alloc"] = pieces[value].to_numpy(dtype=float) * np.nan_to_num(norm)

    grp = pieces.groupby("__tgt_id")["__alloc"].sum()
    out = np.zeros(len(target), dtype=float)
    for tid, val in grp.items():
        out[int(tid)] = val

    result = target.copy()
    result[f"{value}{suffix}"] = out
    return result
