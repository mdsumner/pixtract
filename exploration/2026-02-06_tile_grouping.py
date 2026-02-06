"""
Fast raster point extraction via tile-aware grouped reads.

Strategy:
  1. GeoTransform → map (x, y) to fractional pixel (col, row)
  2. GetBlockSize() → tile dims (tw, th)
  3. floor(col / tw), floor(row / th) → tile index per point
  4. Group points by tile, ReadAsArray per tile, look up local offsets

No GEOS needed — tile membership is integer arithmetic on a regular grid.
"""

import numpy as np
from osgeo import gdal, osr
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# 1. Create a test tiled raster
# ---------------------------------------------------------------------------
def create_test_raster(path, nx=4096, ny=4096, tile_x=256, tile_y=256, nbands=1):
    """Create a tiled GeoTIFF with predictable values for verification."""
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        path, nx, ny, nbands, gdal.GDT_Float32,
        options=[
            f"TILED=YES",
            f"BLOCKXSIZE={tile_x}",
            f"BLOCKYSIZE={tile_y}",
            "COMPRESS=NONE",
        ],
    )
    # Arbitrary geotransform: origin at (100, -30), 0.001 deg pixels
    gt = (100.0, 0.001, 0.0, -30.0, 0.0, -0.001)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    # Fill with row * 10000 + col so we can verify extractions
    for b in range(nbands):
        band = ds.GetRasterBand(b + 1)
        for row_start in range(0, ny, tile_y):
            rows = min(tile_y, ny - row_start)
            cols_arr = np.arange(nx, dtype=np.float32)
            rows_arr = np.arange(row_start, row_start + rows, dtype=np.float32)
            data = rows_arr[:, None] * 10000 + cols_arr[None, :]
            band.WriteArray(data, 0, row_start)
        band.FlushCache()
    ds.FlushCache()
    ds = None
    return path


# ---------------------------------------------------------------------------
# 2. Generate random query points within the raster extent
# ---------------------------------------------------------------------------
def random_points_in_raster(raster_path, n=100_000, seed=42):
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None

    rng = np.random.default_rng(seed)
    # Random fractional pixel positions, convert to map coords
    cols = rng.uniform(0, nx - 1, n)
    rows = rng.uniform(0, ny - 1, n)
    xs = gt[0] + cols * gt[1] + rows * gt[2]
    ys = gt[3] + cols * gt[4] + rows * gt[5]
    return xs, ys


# ---------------------------------------------------------------------------
# 3. Core: geo coords → pixel coords → tile indices
# ---------------------------------------------------------------------------
def geo_to_pixel(xs, ys, gt):
    """Inverse geotransform: map coords → fractional pixel coords."""
    # For north-up rasters (gt[2]==0, gt[4]==0) this simplifies, but
    # let's do the general case via the inverse matrix.
    det = gt[1] * gt[5] - gt[2] * gt[4]
    col = (gt[5] * (xs - gt[0]) - gt[2] * (ys - gt[3])) / det
    row = (-gt[4] * (xs - gt[0]) + gt[1] * (ys - gt[3])) / det
    return col, row


def classify_to_tiles(col, row, block_x, block_y, n_tiles_x, n_tiles_y):
    """Integer divide pixel coords by block size → tile (tx, ty) index.
    Returns tile keys and per-point local offsets within tile."""
    # Floor to integer pixel
    icol = np.floor(col).astype(np.int64)
    irow = np.floor(row).astype(np.int64)

    # Tile indices
    tx = icol // block_x
    ty = irow // block_y

    # Clamp to valid range
    np.clip(tx, 0, n_tiles_x - 1, out=tx)
    np.clip(ty, 0, n_tiles_y - 1, out=ty)

    # Local offsets within each tile
    local_col = icol - tx * block_x
    local_row = irow - ty * block_y

    return tx, ty, local_col, local_row


# ---------------------------------------------------------------------------
# 4a. TILE-GROUPED extraction (the fast way)
# ---------------------------------------------------------------------------
def extract_tile_grouped(raster_path, xs, ys, band_idx=1):
    """Extract raster values by grouping points into tiles, one read per tile."""
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(band_idx)
    block_x, block_y = band.GetBlockSize()

    n_tiles_x = (nx + block_x - 1) // block_x
    n_tiles_y = (ny + block_y - 1) // block_y

    col, row = geo_to_pixel(xs, ys, gt)
    tx, ty, local_col, local_row = classify_to_tiles(
        col, row, block_x, block_y, n_tiles_x, n_tiles_y
    )

    # Group point indices by tile key
    # Encode tile key as single int for fast grouping
    tile_key = ty * n_tiles_x + tx
    sort_idx = np.argsort(tile_key)
    sorted_keys = tile_key[sort_idx]

    # Find boundaries between groups
    breaks = np.nonzero(np.diff(sorted_keys))[0] + 1
    groups = np.split(sort_idx, breaks)
    unique_keys = sorted_keys[np.concatenate([[0], breaks])]

    result = np.empty(len(xs), dtype=np.float64)

    for key, group in zip(unique_keys, groups):
        t_y = key // n_tiles_x
        t_x = key % n_tiles_x

        xoff = int(t_x * block_x)
        yoff = int(t_y * block_y)
        win_x = min(block_x, nx - xoff)
        win_y = min(block_y, ny - yoff)

        tile_data = band.ReadAsArray(xoff, yoff, win_x, win_y)

        lc = local_col[group]
        lr = local_row[group]
        result[group] = tile_data[lr, lc]

    ds = None
    return result


# ---------------------------------------------------------------------------
# 4b. TILE-GROUPED with parallel reads (ThreadPoolExecutor)
# ---------------------------------------------------------------------------
def extract_tile_grouped_parallel(raster_path, xs, ys, band_idx=1, max_workers=4):
    """Same as above but parallel tile reads using thread pool."""
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(band_idx)
    block_x, block_y = band.GetBlockSize()
    ds = None  # close; we'll reopen per-thread

    n_tiles_x = (nx + block_x - 1) // block_x
    n_tiles_y = (ny + block_y - 1) // block_y

    col, row = geo_to_pixel(xs, ys, gt)
    tx, ty, local_col, local_row = classify_to_tiles(
        col, row, block_x, block_y, n_tiles_x, n_tiles_y
    )

    tile_key = ty * n_tiles_x + tx
    sort_idx = np.argsort(tile_key)
    sorted_keys = tile_key[sort_idx]
    breaks = np.nonzero(np.diff(sorted_keys))[0] + 1
    groups = np.split(sort_idx, breaks)
    unique_keys = sorted_keys[np.concatenate([[0], breaks])]

    result = np.empty(len(xs), dtype=np.float64)

    def read_tile(key, group):
        t_y = key // n_tiles_x
        t_x = key % n_tiles_x
        xoff = int(t_x * block_x)
        yoff = int(t_y * block_y)
        win_x = min(block_x, nx - xoff)
        win_y = min(block_y, ny - yoff)
        # Each thread opens its own dataset handle
        thread_ds = gdal.Open(raster_path)
        tile_data = thread_ds.GetRasterBand(band_idx).ReadAsArray(xoff, yoff, win_x, win_y)
        thread_ds = None
        lc = local_col[group]
        lr = local_row[group]
        result[group] = tile_data[lr, lc]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_tile, k, g) for k, g in zip(unique_keys, groups)]
        for f in futures:
            f.result()

    return result


# ---------------------------------------------------------------------------
# 4c. NAIVE per-point extraction (baseline for comparison)
# ---------------------------------------------------------------------------
def extract_naive(raster_path, xs, ys, band_idx=1):
    """Read one pixel at a time — the slow baseline."""
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(band_idx)
    nx, ny = ds.RasterXSize, ds.RasterYSize

    col, row = geo_to_pixel(xs, ys, gt)
    icol = np.floor(col).astype(np.int64)
    irow = np.floor(row).astype(np.int64)
    np.clip(icol, 0, nx - 1, out=icol)
    np.clip(irow, 0, ny - 1, out=irow)

    result = np.empty(len(xs), dtype=np.float64)
    for i in range(len(xs)):
        val = band.ReadAsArray(int(icol[i]), int(irow[i]), 1, 1)
        result[i] = val[0, 0]

    ds = None
    return result


# ---------------------------------------------------------------------------
# 4d. FULL-BAND read then index (simple but memory-hungry)
# ---------------------------------------------------------------------------
def extract_full_read(raster_path, xs, ys, band_idx=1):
    """Read entire band into memory, then numpy fancy-index."""
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(band_idx)
    nx, ny = ds.RasterXSize, ds.RasterYSize

    data = band.ReadAsArray()  # entire band
    col, row = geo_to_pixel(xs, ys, gt)
    icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
    irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)

    result = data[irow, icol].astype(np.float64)
    ds = None
    return result


# ---------------------------------------------------------------------------
# 5. Benchmark runner
# ---------------------------------------------------------------------------
def benchmark():
    raster_path = "/tmp/test_tiled.tif"
    n_points = 500_000
    n_points_naive = 10_000  # naive is too slow for 500k

    print("=" * 65)
    print(f"Creating test raster (4096×4096, 256×256 tiles)...")
    create_test_raster(raster_path)

    print(f"Generating {n_points:,} random points...")
    xs, ys = random_points_in_raster(raster_path, n=n_points)
    xs_small, ys_small = xs[:n_points_naive], ys[:n_points_naive]

    print("=" * 65)

    # --- Full read ---
    t0 = time.perf_counter()
    res_full = extract_full_read(raster_path, xs, ys)
    t_full = time.perf_counter() - t0
    print(f"Full-band read + index   : {t_full:.4f}s  ({n_points:,} pts)")

    # --- Tile-grouped serial ---
    t0 = time.perf_counter()
    res_tiled = extract_tile_grouped(raster_path, xs, ys)
    t_tiled = time.perf_counter() - t0
    print(f"Tile-grouped (serial)    : {t_tiled:.4f}s  ({n_points:,} pts)")

    # --- Tile-grouped parallel ---
    for nw in [2, 4]:
        t0 = time.perf_counter()
        res_par = extract_tile_grouped_parallel(raster_path, xs, ys, max_workers=nw)
        t_par = time.perf_counter() - t0
        print(f"Tile-grouped (parallel/{nw}): {t_par:.4f}s  ({n_points:,} pts)")

    # --- Naive (small subset) ---
    t0 = time.perf_counter()
    res_naive = extract_naive(raster_path, xs_small, ys_small)
    t_naive = time.perf_counter() - t0
    print(f"Naive per-pixel          : {t_naive:.4f}s  ({n_points_naive:,} pts)")
    naive_rate = n_points_naive / t_naive
    print(f"  → extrapolated for {n_points:,}: ~{n_points / naive_rate:.1f}s")

    # --- Verify correctness ---
    print("=" * 65)
    match_full_vs_tiled = np.allclose(res_full, res_tiled)
    match_full_vs_naive = np.allclose(res_full[:n_points_naive], res_naive)
    print(f"Full vs Tile-grouped match: {match_full_vs_tiled}")
    print(f"Full vs Naive match:        {match_full_vs_naive}")

    # --- Stats ---
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    bx, by = band.GetBlockSize()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    n_tiles = ((nx + bx - 1) // bx) * ((ny + by - 1) // by)
    ds = None

    # How many unique tiles were touched?
    col, row = geo_to_pixel(xs, ys, ds_gt := (100.0, 0.001, 0.0, -30.0, 0.0, -0.001))
    tx = (np.floor(col).astype(int) // bx)
    ty = (np.floor(row).astype(int) // by)
    unique_tiles = len(set(zip(tx.tolist(), ty.tolist())))
    print(f"\nRaster: {nx}×{ny}, tile: {bx}×{by}, total tiles: {n_tiles}")
    print(f"Unique tiles touched by {n_points:,} points: {unique_tiles}/{n_tiles}")
    print(f"  → {unique_tiles} ReadAsArray calls vs {n_points:,} naive calls")
    print("=" * 65)


if __name__ == "__main__":
    benchmark()
