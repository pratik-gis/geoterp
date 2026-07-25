"""Generate the figures used in the geoterp blog post.

Run from the repo root inside the project venv::

    python examples/make_blog_figures.py /path/to/blog/assets/images/blog

All figures use the site's dark palette so they sit cleanly on the blog.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import geoterp
from geoterp import datasets as ds

BG = "#0a0f14"
CARD = "#121a24"
TEXT = "#dbe4ee"
DIM = "#8fa1b3"
ACCENT = "#34d399"
CMAP = "magma"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": CARD,
    "text.color": TEXT,
    "axes.labelcolor": DIM,
    "axes.edgecolor": "#1f2b38",
    "xtick.color": DIM,
    "ytick.color": DIM,
    "font.size": 11,
    "axes.titlesize": 12,
})


def _clean(ax, title):
    ax.set_title(title, color=TEXT, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#1f2b38")


def header(outdir, stations):
    grid = geoterp.kriging(stations, "temperature", resolution=200)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    im = grid.plot(ax=ax, cmap=CMAP)
    ax.scatter(stations.geometry.x, stations.geometry.y, s=12,
               facecolor="none", edgecolor=ACCENT, linewidths=0.8, alpha=0.9)
    _clean(ax, "")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("temperature (°C)", color=DIM)
    cb.ax.yaxis.set_tick_params(color=DIM)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=DIM)
    ax.set_title("geoterp · ordinary kriging of 80 weather stations",
                 color=TEXT, fontsize=15, pad=12)
    fig.tight_layout()
    p = os.path.join(outdir, "geoterp-header.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def point_methods(outdir, stations):
    methods = [
        ("IDW (power 2)", lambda: geoterp.idw(stations, "temperature", resolution=90)),
        ("Nearest / Thiessen", lambda: geoterp.nearest(stations, "temperature", resolution=90)),
        ("Natural neighbour", lambda: geoterp.natural_neighbor(stations, "temperature", resolution=70)),
        ("Spline (thin-plate)", lambda: geoterp.spline(stations, "temperature", resolution=90)),
        ("Trend surface (deg 2)", lambda: geoterp.trend_surface(stations, "temperature", resolution=90, degree=2)),
        ("Ordinary kriging", lambda: geoterp.kriging(stations, "temperature", resolution=70)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    vmin = stations.temperature.min()
    vmax = stations.temperature.max()
    for ax, (name, fn) in zip(axes.ravel(), methods):
        g = fn()
        im = g.plot(ax=ax, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.scatter(stations.geometry.x, stations.geometry.y, s=4,
                   color=ACCENT, alpha=0.6)
        _clean(ax, name)
    fig.suptitle("Six ways to turn points into a surface — same 80 samples",
                 color=TEXT, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(outdir, "geoterp-point-methods.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def pycno(outdir):
    src = ds.load_source_zones()
    grid = geoterp.pycnophylactic(src, "population", resolution=140, max_iter=250)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    src.plot(column="population", ax=axes[0], cmap=CMAP, edgecolor=BG, linewidth=1.5,
             legend=True, legend_kwds={"shrink": 0.6})
    _clean(axes[0], "Choropleth counts (16 tracts)")
    im = grid.plot(ax=axes[1], cmap=CMAP)
    src.boundary.plot(ax=axes[1], edgecolor=ACCENT, linewidth=0.6, alpha=0.5)
    _clean(axes[1], "Pycnophylactic surface (mass-preserving)")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("people / cell", color=DIM)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=DIM)
    fig.suptitle("Tobler's pycnophylactic interpolation removes the block artefacts",
                 color=TEXT, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(outdir, "geoterp-pycno.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def areal(outdir):
    src, tgt, lu = ds.load_source_zones(), ds.load_target_zones(), ds.load_landuse()
    aw = geoterp.areal_weighting(src, tgt, "population")
    dm = geoterp.dasymetric(src, tgt, "population", lu, "weight")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    src.plot(column="population", ax=axes[0], cmap=CMAP, edgecolor=BG, linewidth=1.2)
    _clean(axes[0], "Source tracts (4×4)")
    aw.plot(column="population_est", ax=axes[1], cmap=CMAP, edgecolor=BG, linewidth=1.2)
    _clean(axes[1], "Areal weighting → districts (3×3)")
    dm.plot(column="population_est", ax=axes[2], cmap=CMAP, edgecolor=BG, linewidth=1.2)
    _clean(axes[2], "Dasymetric (land-use weighted)")
    fig.suptitle("Reprojecting population onto an incompatible zoning",
                 color=TEXT, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, "geoterp-areal.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def raster(outdir):
    dem = ds.load_dem(resolution=40)
    variants = [
        ("Source (40×40)", dem),
        ("Nearest ×4", geoterp.resample(dem, scale=4, method="nearest")),
        ("Bilinear ×4", geoterp.resample(dem, scale=4, method="bilinear")),
        ("Lanczos ×4", geoterp.resample(dem, scale=4, method="lanczos")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (name, g) in zip(axes, variants):
        g.plot(ax=ax, cmap="terrain")
        _clean(ax, name)
    fig.suptitle("Raster resampling kernels — upsampling a coarse DEM ×4",
                 color=TEXT, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(outdir, "geoterp-raster.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fill(outdir):
    from geoterp.core import RasterGrid

    dem = ds.load_dem(resolution=90)
    truth = dem.data.copy()
    rng = np.random.default_rng(3)
    holed = truth.copy()
    holed[28:46, 30:52] = np.nan               # big void
    idx = rng.choice(holed.size, size=500, replace=False)
    holed.flat[idx] = np.nan                   # scattered dropouts
    r = RasterGrid(holed, dem.spec)
    filled = geoterp.fill_nodata(r, method="idw", smoothing_iterations=2)

    vmin, vmax = np.nanmin(truth), np.nanmax(truth)
    panels = [
        ("DEM with voids (10% missing)", holed),
        ("Filled — GDAL IDW", filled.data),
        ("Original (reference)", truth),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, (name, arr) in zip(axes, panels):
        ax.imshow(arr, cmap="terrain", vmin=vmin, vmax=vmax)
        _clean(ax, name)
    fig.suptitle("Filling NoData holes in a DEM by interpolation",
                 color=TEXT, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(outdir, "geoterp-fill.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "examples/figures"
    os.makedirs(outdir, exist_ok=True)
    stations = ds.load_stations()
    header(outdir, stations)
    point_methods(outdir, stations)
    pycno(outdir)
    areal(outdir)
    raster(outdir)
    fill(outdir)


if __name__ == "__main__":
    main()
