"""Command-line interface for geoterp.

Grouped as ``geoterp <family> <method> ...``::

    geoterp point idw stations.geojson --value temperature -o t.tif
    geoterp point kriging stations.geojson --value temperature --variogram spherical
    geoterp polygon areal source.geojson target.geojson --value population
    geoterp polygon pycno source.geojson --value population -o pycno.tif
    geoterp raster resample dem.tif --scale 2 --method bilinear -o hi.tif
    geoterp raster aggregate dem.tif --factor 4 --method average -o lo.tif
    geoterp sample --outdir ./sample_data

Run ``geoterp <family> --help`` for the methods in each family.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def _add_grid_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("output grid")
    g.add_argument("--resolution", type=int, default=None,
                   help="approx. number of cells along the longer side (default 100)")
    g.add_argument("--cellsize", type=float, default=None,
                   help="output cell size in CRS units (overrides --resolution)")


def _default_out(inp: str, method: str, ext: str) -> str:
    import os

    base = os.path.splitext(os.path.basename(inp))[0]
    return f"{base}_{method}.{ext}"


def _save_grid(grid, output: Optional[str], inp: str, method: str) -> str:
    out = output or _default_out(inp, method, "tif")
    grid.to_geotiff(out)
    return out


def _save_vector(gdf, output: Optional[str], inp: str, method: str) -> str:
    out = output or _default_out(inp, method, "geojson")
    gdf.to_file(out, driver="GeoJSON")
    return out


# --- point family --------------------------------------------------------
def _cmd_point(args) -> int:
    import geoterp

    method = args.method

    if method == "voronoi":
        gdf = geoterp.voronoi_polygons(args.input, value=args.value)
        out = _save_vector(gdf, args.output, args.input, "voronoi")
        print(f"wrote {out}  ({len(gdf)} polygons)")
        return 0

    common = dict(value=args.value, resolution=args.resolution, cellsize=args.cellsize)

    if method == "idw":
        grid = geoterp.idw(args.input, power=args.power, neighbors=args.neighbors, **common)
    elif method == "nearest":
        grid = geoterp.nearest(args.input, **common)
    elif method == "natural":
        grid = geoterp.natural_neighbor(args.input, max_neighbors=args.neighbors, **common)
    elif method == "spline":
        grid = geoterp.spline(args.input, kernel=args.kernel, smoothing=args.smoothing, **common)
    elif method == "trend":
        grid = geoterp.trend_surface(args.input, degree=args.degree, **common)
    elif method == "local":
        grid = geoterp.local_polynomial(
            args.input, degree=args.degree, neighbors=args.neighbors, **common)
    elif method == "kriging":
        grid = geoterp.kriging(
            args.input, method=args.kmethod, variogram_model=args.variogram, **common)
    else:  # pragma: no cover
        raise SystemExit(f"unknown point method {method!r}")

    out = _save_grid(grid, args.output, args.input, method)
    print(f"wrote {out}  {grid!r}")
    return 0


# --- polygon family ------------------------------------------------------
def _cmd_polygon(args) -> int:
    import geoterp
    from geoterp.io import read_vector

    if args.method == "areal":
        src = read_vector(args.source)
        tgt = read_vector(args.target)
        res = geoterp.areal_weighting(src, tgt, args.value, extensive=not args.intensive)
        out = _save_vector(res, args.output, args.source, "areal")
    elif args.method == "dasymetric":
        src = read_vector(args.source)
        tgt = read_vector(args.target)
        anc = read_vector(args.ancillary)
        res = geoterp.dasymetric(src, tgt, args.value, anc, args.weight_col)
        out = _save_vector(res, args.output, args.source, "dasymetric")
    elif args.method == "pycno":
        src = read_vector(args.source)
        if args.target:
            tgt = read_vector(args.target)
            res = geoterp.pycnophylactic_to_polygons(
                src, tgt, args.value, resolution=args.resolution, cellsize=args.cellsize)
            out = _save_vector(res, args.output, args.source, "pycno")
        else:
            grid = geoterp.pycnophylactic(
                src, args.value, resolution=args.resolution, cellsize=args.cellsize)
            out = _save_grid(grid, args.output, args.source, "pycno")
            print(f"wrote {out}  {grid!r}")
            return 0
    else:  # pragma: no cover
        raise SystemExit(f"unknown polygon method {args.method!r}")

    print(f"wrote {out}")
    return 0


# --- raster family -------------------------------------------------------
def _cmd_raster(args) -> int:
    import geoterp

    if args.method == "resample":
        grid = geoterp.resample(
            args.input, scale=args.scale, cellsize=args.cellsize,
            shape=tuple(args.shape) if args.shape else None, method=args.kernel)
    elif args.method == "aggregate":
        grid = geoterp.aggregate(
            args.input, factor=args.factor, factor_y=args.factor_y, method=args.stat)
    else:  # pragma: no cover
        raise SystemExit(f"unknown raster method {args.method!r}")

    out = _save_grid(grid, args.output, args.input, args.method)
    print(f"wrote {out}  {grid!r}")
    return 0


def _cmd_sample(args) -> int:
    from geoterp.datasets import make_sample_dataset

    paths = make_sample_dataset(args.outdir)
    print(f"wrote sample dataset to {args.outdir}:")
    for name, p in paths.items():
        print(f"  {name:14s} {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geoterp", description=__doc__.split("\n")[0])
    from . import __version__

    p.add_argument("--version", action="version", version=f"geoterp {__version__}")
    sub = p.add_subparsers(dest="family", required=True)

    # point ---------------------------------------------------------------
    pt = sub.add_parser("point", help="point → surface interpolation")
    ptsub = pt.add_subparsers(dest="method", required=True)

    def _point_base(name, help_):
        sp = ptsub.add_parser(name, help=help_)
        sp.add_argument("input", help="point vector file (GeoJSON, SHP, CSV, ...)")
        sp.add_argument("--value", required=(name != "voronoi"),
                        help="attribute column to interpolate")
        sp.add_argument("-o", "--output", default=None, help="output path")
        if name != "voronoi":
            _add_grid_args(sp)
        return sp

    sp = _point_base("idw", "inverse distance weighting")
    sp.add_argument("--power", type=float, default=2.0)
    sp.add_argument("--neighbors", type=int, default=12)
    _point_base("nearest", "nearest neighbour / Thiessen raster")
    _point_base("voronoi", "Thiessen (Voronoi) polygons → vector")
    sp = _point_base("natural", "natural-neighbour (Sibson)")
    sp.add_argument("--neighbors", type=int, default=32, help="max candidate neighbours")
    sp = _point_base("spline", "radial basis function spline")
    sp.add_argument("--kernel", default="thin_plate_spline")
    sp.add_argument("--smoothing", type=float, default=0.0)
    sp = _point_base("trend", "global polynomial trend surface")
    sp.add_argument("--degree", type=int, default=2)
    sp = _point_base("local", "local polynomial (LOESS-style)")
    sp.add_argument("--degree", type=int, default=1)
    sp.add_argument("--neighbors", type=int, default=16)
    sp = _point_base("kriging", "ordinary/universal kriging")
    sp.add_argument("--kmethod", default="ordinary", choices=["ordinary", "universal"])
    sp.add_argument("--variogram", default="spherical")
    pt.set_defaults(func=_cmd_point)

    # polygon -------------------------------------------------------------
    pg = sub.add_parser("polygon", help="polygon → polygon interpolation")
    pgsub = pg.add_subparsers(dest="method", required=True)
    ar = pgsub.add_parser("areal", help="areal weighting")
    ar.add_argument("source"); ar.add_argument("target")
    ar.add_argument("--value", required=True)
    ar.add_argument("--intensive", action="store_true", help="treat value as a rate/density")
    ar.add_argument("-o", "--output", default=None)
    da = pgsub.add_parser("dasymetric", help="dasymetric mapping")
    da.add_argument("source"); da.add_argument("target")
    da.add_argument("--value", required=True)
    da.add_argument("--ancillary", required=True)
    da.add_argument("--weight-col", required=True, dest="weight_col")
    da.add_argument("-o", "--output", default=None)
    py = pgsub.add_parser("pycno", help="pycnophylactic (Tobler) interpolation")
    py.add_argument("source")
    py.add_argument("--value", required=True)
    py.add_argument("--target", default=None, help="optional: re-aggregate to these polygons")
    _add_grid_args(py)
    py.add_argument("-o", "--output", default=None)
    pg.set_defaults(func=_cmd_polygon)

    # raster --------------------------------------------------------------
    rs = sub.add_parser("raster", help="raster → raster resample/aggregate")
    rssub = rs.add_subparsers(dest="method", required=True)
    re = rssub.add_parser("resample", help="change resolution with a kernel")
    re.add_argument("input")
    re.add_argument("--scale", type=float, default=None)
    re.add_argument("--cellsize", type=float, default=None)
    re.add_argument("--shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"))
    re.add_argument("--kernel", default="bilinear",
                    choices=["nearest", "bilinear", "cubic", "cubic_spline", "lanczos"])
    re.add_argument("-o", "--output", default=None)
    ag = rssub.add_parser("aggregate", help="coarsen by block reduction")
    ag.add_argument("input")
    ag.add_argument("--factor", type=int, default=2)
    ag.add_argument("--factor-y", type=int, default=None, dest="factor_y")
    ag.add_argument("--stat", default="average",
                    choices=["average", "mean", "sum", "min", "max", "median", "mode"])
    ag.add_argument("-o", "--output", default=None)
    rs.set_defaults(func=_cmd_raster)

    # sample --------------------------------------------------------------
    sm = sub.add_parser("sample", help="write the bundled sample dataset")
    sm.add_argument("--outdir", default="sample_data")
    sm.set_defaults(func=_cmd_sample)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
