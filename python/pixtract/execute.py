"""
Execute a plan: read each touched block once, in plan order, and hand the
values of its cells to a reducer.

Blocks come back from GDAL as 2D arrays in raster order (row-major from the
top-left); they are used flat, so a segment is the slice
block.ravel()[r * width + c0 : r * width + c1].
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .plan import _expand, block_groups

__all__ = ["execute", "GdalReader", "RasterioReader"]


class GdalReader:
    """Block reads through osgeo.gdal, one dataset handle per thread per path."""

    def __init__(self):
        from osgeo import gdal
        gdal.UseExceptions()
        self._gdal = gdal
        self._local = threading.local()

    def read(self, path, band, xoff, yoff, xsize, ysize):
        cache = self._local.__dict__.setdefault("ds", {})
        ds = cache.get(path)
        if ds is None:
            ds = cache[path] = self._gdal.Open(path)
        return ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xsize, ysize)


class RasterioReader:
    """Block reads through rasterio, one handle per thread per path."""

    def __init__(self):
        import rasterio
        from rasterio.windows import Window
        self._open, self._Window = rasterio.open, Window
        self._local = threading.local()

    def read(self, path, band, xoff, yoff, xsize, ysize):
        cache = self._local.__dict__.setdefault("ds", {})
        ds = cache.get(path)
        if ds is None:
            ds = cache[path] = self._open(path)
        return ds.read(band, window=self._Window(xoff, yoff, xsize, ysize))


def _reader(backend):
    if backend == "gdal":
        return GdalReader()
    if backend == "rasterio":
        return RasterioReader()
    if hasattr(backend, "read"):
        return backend
    raise ValueError(f"Unknown backend: {backend!r}")


def _fetch(plan, reader, a, b):
    """Values (float64, flat, plan order) for segments a:b, which share a block."""
    s, src = plan.seg, plan.sources
    n = s["c1"][a:b] - s["c0"][a:b]
    k = int(s["src"][a])
    if k < 0:
        v = np.full(int(n.sum()), src.fill, dtype=np.float64)
    else:
        bx, by = int(src.block_x[k]), int(src.block_y[k])
        xoff, yoff = int(s["bcol"][a]) * bx, int(s["brow"][a]) * by
        wx = min(bx, int(src.file_xsize[k]) - xoff)
        wy = min(by, int(src.file_ysize[k]) - yoff)
        block = reader.read(src.path[k], int(src.band[k]), xoff, yoff, wx, wy)
        flat = block.ravel()
        start = s["r"][a:b] * wx + s["c0"][a:b]
        _, idx = _expand(start, start + n)
        v = flat[idx].astype(np.float64)
        t = src.transparent[k]
        if t is not None:
            v[v == t] = src.fill
    if src.nodata is not None:
        v[v == src.nodata] = np.nan
    return v


def _cells(plan, a, b, v):
    s = plan.seg
    n = s["c1"][a:b] - s["c0"][a:b]
    which, col = _expand(s["col"][a:b], s["col"][a:b] + n)
    return {
        "id": s["id"][a:b][which],
        "w": s["w"][a:b][which],
        "run": s["run"][a:b][which],
        "row": s["row"][a:b][which],
        "col": col,
        "value": v,
    }


def execute(plan, reducer, backend="gdal", max_workers=None):
    """Run a plan. Returns reducer.result().

    Reads happen in plan order (source, block row, block col), each touched
    block exactly once. With max_workers > 1 blocks are fetched on a thread
    pool, a bounded number ahead of the reducer, and reduced in order on the
    calling thread.
    """
    reader = _reader(backend)
    starts, stops = block_groups(plan)
    reducer.start(plan)
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending = deque()
            it = iter(zip(starts, stops))
            for a, b in it:
                pending.append((a, b, pool.submit(_fetch, plan, reader, a, b)))
                if len(pending) >= 2 * max_workers:
                    a0, b0, f = pending.popleft()
                    reducer.add(_cells(plan, a0, b0, f.result()))
            while pending:
                a0, b0, f = pending.popleft()
                reducer.add(_cells(plan, a0, b0, f.result()))
    else:
        for a, b in zip(starts, stops):
            reducer.add(_cells(plan, a, b, _fetch(plan, reader, a, b)))
    return reducer.result()
