"""Areal (area-weighted) interpolation between incompatible polygon zonings.

Given a value measured over *source* zones (e.g. population per census tract),
estimate it over an unrelated set of *target* zones.  The workhorse is the
area of overlap between each source and target polygon.

* **Extensive** variables (counts, totals) are split in proportion to the
  fraction of each source that falls in a target:
  ``v_t = Σ_s v_s · area(s ∩ t) / area(s)``.
* **Intensive** variables (densities, rates, means) are combined as an
  area-weighted average of the overlapping sources.

Implemented directly on top of GeoPandas overlays.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np


def areal_weighting(
    source,
    target,
    value: Union[str, Iterable[str]],
    *,
    extensive: bool = True,
    suffix: str = "_est",
):
    """Interpolate ``value`` from ``source`` polygons onto ``target`` polygons.

    Parameters
    ----------
    source, target:
        GeoDataFrames of polygons.  They should share a CRS; if ``target`` has
        none it inherits ``source``'s.
    value:
        Column name (or list of names) in ``source`` to interpolate.
    extensive:
        ``True`` for count-like variables that must be split and summed;
        ``False`` for density/rate variables that are averaged.
    suffix:
        Appended to each value column name in the returned target GeoDataFrame.

    Returns
    -------
    GeoDataFrame
        A copy of ``target`` with one estimated column per input ``value``.
    """
    import geopandas as gpd

    values = [value] if isinstance(value, str) else list(value)
    target = target.copy()
    if target.crs is None:
        target = target.set_crs(source.crs, allow_override=True)

    src = source.copy()
    src["__src_id"] = np.arange(len(src))
    src["__src_area"] = src.geometry.area
    tgt = target.copy()
    tgt["__tgt_id"] = np.arange(len(tgt))

    inter = gpd.overlay(
        src[["__src_id", "__src_area", *values, "geometry"]],
        tgt[["__tgt_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    inter["__piece_area"] = inter.geometry.area

    for col in values:
        out = np.zeros(len(target), dtype=float)
        if extensive:
            frac = np.where(
                inter["__src_area"] > 0,
                inter["__piece_area"] / inter["__src_area"],
                0.0,
            )
            contrib = inter[col].to_numpy(dtype=float) * frac
            grp = inter.assign(__c=contrib).groupby("__tgt_id")["__c"].sum()
        else:
            w = inter["__piece_area"].to_numpy(dtype=float)
            wv = inter[col].to_numpy(dtype=float) * w
            df = inter.assign(__wv=wv, __w=w).groupby("__tgt_id")[["__wv", "__w"]].sum()
            grp = df["__wv"] / df["__w"].replace(0, np.nan)
        for tid, val in grp.items():
            out[int(tid)] = val
        target[f"{col}{suffix}"] = out

    return target
