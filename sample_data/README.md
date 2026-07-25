# geoterp sample dataset

A small, self-consistent scenario for trying out `geoterp` — a fictional
10 km × 10 km city ("Terraville") projected in UTM zone 43N (`EPSG:32643`), so
all distances and areas are in metres. Everything here is generated
deterministically by `geoterp.datasets`; regenerate it any time with:

```bash
geoterp sample --outdir sample_data
# or:  python -c "from geoterp.datasets import make_sample_dataset; make_sample_dataset('sample_data')"
```

| File | Geometry | Key attributes | Use it for |
| --- | --- | --- | --- |
| `stations.geojson` | 80 points | `temperature` (°C), `elevation` (m) | point → surface (IDW, kriging, spline, …) |
| `stations.csv` | x/y columns | same | the "plain arrays" workflow |
| `source_zones.geojson` | 4×4 polygons | `population` | polygon → polygon (source) |
| `target_zones.geojson` | 3×3 polygons | `district_id` | polygon → polygon (target) |
| `landuse.geojson` | 6×6 polygons | `landuse`, `weight` | dasymetric ancillary weights |
| `dem.tif` | raster | elevation | raster resample / aggregate |

Temperature falls with both elevation and latitude, and population concentrates
toward the low-lying south-west — so the interpolated surfaces show real,
interpretable structure rather than noise.

## Quick start

```python
import geoterp
from geoterp.io import read_vector

stations = read_vector("sample_data/stations.geojson")
geoterp.kriging(stations, "temperature", resolution=200).to_geotiff("temp.tif")
```
