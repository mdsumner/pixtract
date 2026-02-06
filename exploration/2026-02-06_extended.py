"""
Extended benchmarks: larger rasters, compression, point clustering.
Tests where tile-grouped extraction really shines.
"""

import numpy as np
from osgeo import gdal, osr
import time
import os
from concurrent.futures import ThreadPoolExecutor

gdal.UseExceptions()


def create_raster(path, nx, ny, tile_x, tile_y, compress="DEFLATE"):
    drv = gdal.GetDriverByName("GTiff")
    opts = [f"TILED=YES", f"BLOCKXSIZE={tile_x}", f"BLOCKYSIZE={tile_y}"]
    if compress:
        opts.append(f"COMPRESS={compress}")
    ds = drv.Create(path, nx, ny, 1, gdal.GDT_Float32, options=opts)
    gt = (100.0, 0.0001, 0.0, -30.0, 0.0, -0.0001)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    # Write tile-by-tile to avoid huge memory
    for yoff in range(0, ny, tile_y):
        h = min(tile_y, ny - yoff)
        for xoff in range(0, nx, tile_x):
            w = min(tile_x, nx - xoff)
            rows = np.arange(yoff, yoff + h, dtype=np.float32)
            cols = np.arange(xoff, xoff + w, dtype=np.float32)
            data = rows[:, None] * 100000 + cols[None, :]
            band.WriteArray(data, xoff, yoff)
    band.FlushCache()
    ds.FlushCache()
    ds = None
    size_mb = os.path.getsize(path) / 1024 / 1024
    return path, size_mb


def geo_to_pixel(xs, ys, gt):
    det = gt[1] * gt[5] - gt[2] * gt[4]
    col = (gt[5] * (xs - gt[0]) - gt[2] * (ys - gt[3])) / det
    row = (-gt[4] * (xs - gt[0]) + gt[1] * (ys - gt[3])) / det
    return col, row


def extract_tile_grouped(raster_path, xs, ys, band_idx=1):
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(band_idx)
    bx, by = band.GetBlockSize()
    ntx = (nx + bx - 1) // bx
    nty = (ny + by - 1) // by

    col, row = geo_to_pixel(xs, ys, gt)
    icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
    irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)
    tx = icol // bx
    ty = irow // by
    lc = icol - tx * bx
    lr = irow - ty * by

    tile_key = ty * ntx + tx
    sort_idx = np.argsort(tile_key)
    sorted_keys = tile_key[sort_idx]
    breaks = np.nonzero(np.diff(sorted_keys))[0] + 1
    groups = np.split(sort_idx, breaks)
    unique_keys = sorted_keys[np.concatenate([[0], breaks])]

    result = np.empty(len(xs), dtype=np.float64)
    for key, group in zip(unique_keys, groups):
        t_y, t_x = divmod(int(key), ntx)
        xoff = t_x * bx
        yoff = t_y * by
        win_x = min(bx, nx - xoff)
        win_y = min(by, ny - yoff)
        tile_data = band.ReadAsArray(xoff, yoff, win_x, win_y)
        result[group] = tile_data[lr[group], lc[group]]

    ds = None
    return result, len(unique_keys)


def extract_full_read(raster_path, xs, ys, band_idx=1):
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(band_idx)
    data = band.ReadAsArray()
    col, row = geo_to_pixel(xs, ys, gt)
    icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
    irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)
    result = data[irow, icol].astype(np.float64)
    ds = None
    return result


def extract_readraster_bulk(raster_path, xs, ys, band_idx=1):
    """Use GDAL's ReadRaster with struct unpacking - avoids gdal_array entirely.
    Still per-point but tests raw C API speed."""
    import struct
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(band_idx)

    col, row = geo_to_pixel(xs, ys, gt)
    icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
    irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)

    result = np.empty(len(xs), dtype=np.float64)
    for i in range(len(xs)):
        buf = band.ReadRaster(int(icol[i]), int(irow[i]), 1, 1, buf_type=gdal.GDT_Float64)
        result[i] = struct.unpack('d', buf)[0]
    ds = None
    return result


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


def generate_uniform_points(raster_path, n, seed=42):
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None
    rng = np.random.default_rng(seed)
    cols = rng.uniform(0, nx - 1, n)
    rows = rng.uniform(0, ny - 1, n)
    xs = gt[0] + cols * gt[1] + rows * gt[2]
    ys = gt[3] + cols * gt[4] + rows * gt[5]
    return xs, ys


def run_benchmark(name, raster_path, xs, ys, do_full=True, do_naive_n=0):
    """Run tile-grouped and optionally full-read and naive extraction."""
    import struct
    print(f"\n--- {name} ---")

    t0 = time.perf_counter()
    res_tile, n_tiles_read = extract_tile_grouped(raster_path, xs, ys)
    t_tile = time.perf_counter() - t0
    print(f"  Tile-grouped : {t_tile:.4f}s  ({len(xs):,} pts, {n_tiles_read} tiles read)")

    if do_full:
        t0 = time.perf_counter()
        res_full = extract_full_read(raster_path, xs, ys)
        t_full = time.perf_counter() - t0
        print(f"  Full-read    : {t_full:.4f}s  ({len(xs):,} pts)")
        assert np.allclose(res_tile, res_full), "MISMATCH!"
        print(f"  ✓ Results match")
    
    if do_naive_n > 0:
        t0 = time.perf_counter()
        res_naive = extract_readraster_bulk(raster_path, xs[:do_naive_n], ys[:do_naive_n])
        t_naive = time.perf_counter() - t0
        rate = do_naive_n / t_naive
        print(f"  Naive (ReadRaster): {t_naive:.4f}s ({do_naive_n:,} pts)")
        print(f"    → extrapolated {len(xs):,} pts: ~{len(xs)/rate:.1f}s")


def main():
    n_points = 500_000

    # -------------------------------------------------------------------
    # Test 1: Moderate raster, compressed, uniform points
    # -------------------------------------------------------------------
    print("=" * 65)
    path1, sz1 = create_raster("/tmp/medium_deflate.tif", 8192, 8192, 256, 256, "DEFLATE")
    print(f"Test 1: 8192×8192 DEFLATE ({sz1:.1f} MB)")
    xs, ys = generate_uniform_points(path1, n_points)
    run_benchmark("Uniform points, DEFLATE compressed", path1, xs, ys, do_naive_n=5000)

    # -------------------------------------------------------------------
    # Test 2: Same raster, clustered points (fewer tiles)
    # -------------------------------------------------------------------
    xs_c, ys_c = generate_clustered_points(path1, n_points, n_clusters=5)
    run_benchmark("Clustered points (5 clusters), DEFLATE", path1, xs_c, ys_c, do_naive_n=5000)

    # -------------------------------------------------------------------
    # Test 3: Large raster (16k×16k) - full read becomes expensive
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    path2, sz2 = create_raster("/tmp/large_deflate.tif", 16384, 16384, 512, 512, "DEFLATE")
    print(f"Test 2: 16384×16384 DEFLATE tiles=512 ({sz2:.1f} MB)")
    xs_l, ys_l = generate_uniform_points(path2, n_points)
    run_benchmark("Large raster, uniform", path2, xs_l, ys_l, do_full=True)

    xs_lc, ys_lc = generate_clustered_points(path2, n_points, n_clusters=3)
    run_benchmark("Large raster, clustered (3)", path2, xs_lc, ys_lc, do_full=False)

    # -------------------------------------------------------------------
    # Test 4: Huge raster (32k×32k) — can't reasonably full-read
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    path3, sz3 = create_raster("/tmp/huge_deflate.tif", 32768, 32768, 512, 512, "DEFLATE")
    print(f"Test 3: 32768×32768 DEFLATE tiles=512 ({sz3:.1f} MB)")
    xs_h, ys_h = generate_uniform_points(path3, n_points)
    run_benchmark("Huge raster, uniform, 500k pts", path3, xs_h, ys_h, do_full=False)

    xs_h1m, ys_h1m = generate_uniform_points(path3, 2_000_000, seed=99)
    run_benchmark("Huge raster, uniform, 2M pts", path3, xs_h1m, ys_h1m, do_full=False)

    print("\n" + "=" * 65)
    print("DONE")


if __name__ == "__main__":
    main()
