def generate_clustered_points(raster_path, n, n_clusters=5, seed=42):
    """Points clustered in a few spatial areas → fewer tiles touched."""
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None
    rng = np.random.default_rng(seed)
    cluster_cols = rng.uniform(0, nx, n_clusters)
    cluster_rows = rng.uniform(0, ny, n_clusters)
    spread = 200  # pixels
    all_cols, all_rows = [], []
    per_cluster = n // n_clusters
    for cc, cr in zip(cluster_cols, cluster_rows):
        cols = rng.normal(cc, spread, per_cluster).clip(0, nx - 1)
        rows = rng.normal(cr, spread, per_cluster).clip(0, ny - 1)
        all_cols.append(cols)
        all_rows.append(rows)
    cols = np.concatenate(all_cols)
    rows = np.concatenate(all_rows)
    xs = gt[0] + cols * gt[1] + rows * gt[2]
    ys = gt[3] + cols * gt[4] + rows * gt[5]
    return xs, ys

from osgeo import gdal
import numpy as np

def random_points(dsn, n, seed=42):
    ds = gdal.Open(dsn)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None
    rng = np.random.default_rng(seed)
    cols = rng.uniform(0, nx - 1, n)
    rows = rng.uniform(0, ny - 1, n)
    xs = gt[0] + cols * gt[1] + rows * gt[2]
    ys = gt[3] + cols * gt[4] + rows * gt[5]
    return xs, ys


def make_traverse(x0, y0, x1, y1, step=1000):
    """Points along a line at `step` metre intervals."""
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    n = int(dist / step)
    t = np.linspace(0, 1, n)
    return x0 + t * dx, y0 + t * dy


# # Roughly Casey to South Pole direction — ~2500 km
# xs, ys = make_traverse(500000, -500000, -500000, -2000000)
# print(f"Transect: {len(xs)} points")

from pixtract import extract_points
from time import time
from osgeo import gdal
gdal.UseExceptions()
import numpy as np
dsn = "/vsicurl/https://raw.githubusercontent.com/mdsumner/rema-ovr/main/REMA-2m_dem_ovr.vrt"

# Full diagonal — ~8000 km, 8000 points
xs, ys = make_traverse(-2700000, -2500000, 2750000, 3342000)
print(f"Diagonal: {len(xs)} points, {len(xs)}km")



t0 = time()
xs,ys = random_points(dsn, 10000)
time() - t0

t0 = time()
dsn = "GTI:rema_v2_tiles.gti.parquet"
x = extract_points(dsn, xs, ys, max_workers = 24)
time() - t0

# import numpy as np
# import pyarrow as pa
# import pyarrow.parquet as pq
# 
# tbl = pa.table({"x": xs, "y": ys, "elevation": x})
# pq.write_table(tbl, "rema_extract.parquet")








>>> dsn = "/vsicurl/https://raw.githubusercontent.com/mdsumner/rema-ovr/main/REMA-2m_dem_ovr.vrt"
>>> xs,ys = generate_clustered_points(dsn, 1000)
>>>
>>> t0 = time()
>>> dsn = "GTI:rema_v2_tiles.gti.parquet"
>>> x = extract_points(dsn, xs, ys)

>>> time() - t0
25.236035585403442
>>>
>>>
>>> xs.min()
np.float64(-2187789.1688080314)
>>> xs.max()
np.float64(1980701.924572763)
>>> ys.min()
np.float64(-2358842.929599952)
>>> ys.max()
np.float64(2594816.782712664)

x = extract_points(dsn, xs, ys, max_workers = 24)

>>> time() - t0
3.786593198776245
