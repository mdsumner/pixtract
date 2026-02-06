"""
tile_extract — Fast raster point extraction via tile-aware grouped reads.

No GEOS or spatial predicates needed. Tile membership is integer arithmetic
on pixel coordinates derived from the GeoTransform:

  1. Inverse GeoTransform: (x, y) → fractional (col, row)
  2. floor(col / block_x), floor(row / block_y) → tile index
  3. argsort + split to group points by tile
  4. One ReadAsArray per tile, fancy-index local offsets

Works with both osgeo.gdal and rasterio backends.
"""

from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

__all__ = ["extract_points"]


# ── Core geometry (shared by both backends) ─────────────────────────────

def _inverse_geotransform(xs, ys, gt):
    """Map coordinates → fractional pixel coordinates via inverse geotransform.
    gt is a 6-tuple: (origin_x, pixel_w, rot_x, origin_y, rot_y, pixel_h)
    """
    det = gt[1] * gt[5] - gt[2] * gt[4]
    col = (gt[5] * (xs - gt[0]) - gt[2] * (ys - gt[3])) / det
    row = (-gt[4] * (xs - gt[0]) + gt[1] * (ys - gt[3])) / det
    return col, row


def _group_by_tile(icol, irow, block_x, block_y, n_tiles_x, n_tiles_y):
    """Classify pixel coords into tile groups. Returns:
      - unique_keys: 1D array of tile keys (ty * n_tiles_x + tx)
      - groups: list of index arrays (into the original point array)
      - local_col, local_row: per-point offsets within their tile
    """
    tx = np.clip(icol // block_x, 0, n_tiles_x - 1)
    ty = np.clip(irow // block_y, 0, n_tiles_y - 1)
    local_col = icol - tx * block_x
    local_row = irow - ty * block_y

    tile_key = ty * n_tiles_x + tx
    sort_idx = np.argsort(tile_key, kind="mergesort")
    sorted_keys = tile_key[sort_idx]

    breaks = np.nonzero(np.diff(sorted_keys))[0] + 1
    groups = np.split(sort_idx, breaks)
    unique_keys = sorted_keys[np.concatenate([[0], breaks])]

    return unique_keys, groups, local_col, local_row, n_tiles_x


# ── GDAL backend ───────────────────────────────────────────────────────

def _extract_gdal(path, xs, ys, band_idx, max_workers):
    from osgeo import gdal
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(band_idx)
    block_x, block_y = band.GetBlockSize()
    nodata = band.GetNoDataValue()
    ds = None  # close; reopen per-thread if parallel

    n_tiles_x = (nx + block_x - 1) // block_x
    n_tiles_y = (ny + block_y - 1) // block_y

    col, row = _inverse_geotransform(xs, ys, gt)
    icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
    irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)

    unique_keys, groups, lc, lr, ntx = _group_by_tile(
        icol, irow, block_x, block_y, n_tiles_x, n_tiles_y
    )

    result = np.full(len(xs), np.nan, dtype=np.float64)

    def _read_tile(key, group):
        t_y, t_x = divmod(int(key), ntx)
        xoff, yoff = t_x * block_x, t_y * block_y
        win_x = min(block_x, nx - xoff)
        win_y = min(block_y, ny - yoff)
        thread_ds = gdal.Open(path)
        tile = thread_ds.GetRasterBand(band_idx).ReadAsArray(xoff, yoff, win_x, win_y)
        thread_ds = None
        vals = tile[lr[group], lc[group]]
        if nodata is not None:
            vals = np.where(vals == nodata, np.nan, vals)
        result[group] = vals

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_read_tile, k, g) for k, g in zip(unique_keys, groups)]
            for f in futs:
                f.result()
    else:
        for k, g in zip(unique_keys, groups):
            _read_tile(k, g)

    return result, len(unique_keys)


# ── rasterio backend ──────────────────────────────────────────────────

def _extract_rasterio(path, xs, ys, band_idx, max_workers):
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        gt = src.transform.to_gdal()
        nx, ny = src.width, src.height
        block_x, block_y = src.block_shapes[band_idx - 1]  # (rows, cols)
        block_y, block_x = block_x, block_y  # rasterio gives (height, width)
        nodata = src.nodata

    n_tiles_x = (nx + block_x - 1) // block_x
    n_tiles_y = (ny + block_y - 1) // block_y

    col, row = _inverse_geotransform(xs, ys, gt)
    icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
    irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)

    unique_keys, groups, lc, lr, ntx = _group_by_tile(
        icol, irow, block_x, block_y, n_tiles_x, n_tiles_y
    )

    result = np.full(len(xs), np.nan, dtype=np.float64)

    def _read_tile(key, group):
        t_y, t_x = divmod(int(key), ntx)
        xoff, yoff = t_x * block_x, t_y * block_y
        win_x = min(block_x, nx - xoff)
        win_y = min(block_y, ny - yoff)
        with rasterio.open(path) as src:
            tile = src.read(band_idx, window=Window(xoff, yoff, win_x, win_y))
        vals = tile[lr[group], lc[group]]
        if nodata is not None:
            vals = np.where(vals == nodata, np.nan, vals)
        result[group] = vals

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_read_tile, k, g) for k, g in zip(unique_keys, groups)]
            for f in futs:
                f.result()
    else:
        for k, g in zip(unique_keys, groups):
            _read_tile(k, g)

    return result, len(unique_keys)


# ── Public API ─────────────────────────────────────────────────────────

def extract_points(
    raster_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    band: int = 1,
    backend: Literal["gdal", "rasterio"] = "gdal",
    max_workers: int | None = None,
) -> np.ndarray:
    """Extract raster values at (x, y) coordinates using tile-grouped reads.

    Parameters
    ----------
    raster_path : str
        Path to raster (local path, /vsicurl/..., /vsis3/..., etc.)
    xs, ys : array-like
        Map coordinates in the raster's CRS.
    band : int
        Band index (1-based).
    backend : "gdal" | "rasterio"
        Which library to use for I/O.
    max_workers : int or None
        Thread pool size for parallel tile reads. None or 1 for serial.

    Returns
    -------
    np.ndarray
        Extracted values (float64). NoData pixels → NaN.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if backend == "gdal":
        result, n_tiles = _extract_gdal(raster_path, xs, ys, band, max_workers)
    elif backend == "rasterio":
        result, n_tiles = _extract_rasterio(raster_path, xs, ys, band, max_workers)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return result
