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


from pixtract import extract_points
from time import time
from osgeo import gdal
gdal.UseExceptions()
import numpy as np
dsn = "/vsicurl/https://s3.us-west-2.amazonaws.com/pgc-opendata-dems/rema/mosaics/v2.0/2m/22_34/22_34_2_1_2m_v2.0_dem.tif"
xs,ys = generate_clustered_points(dsn, 100)


t0 = time()
dsn = "/vsicurl/https://s3.us-west-2.amazonaws.com/pgc-opendata-dems/rema/mosaics/v2.0/2m/22_34/22_34_2_1_2m_v2.0_dem.tif"
x = extract_points(dsn, xs, ys)
time() - t0
# 6.369928359985352

t0 = time()
dsn = "vrt:///vsicurl/https://raw.githubusercontent.com/mdsumner/rema-ovr/main/REMA-2m_dem_ovr.vrt?projwin=299900,-799900,350100,-850100"
extract_points(dsn, xs, ys)
time() - t0
#11.18113112449646

t0 = time()
dsn = "/vsicurl/https://raw.githubusercontent.com/mdsumner/rema-ovr/main/REMA-2m_dem_ovr.vrt"
extract_points(dsn, xs, ys)
time() - t0
#14.485344886779785
