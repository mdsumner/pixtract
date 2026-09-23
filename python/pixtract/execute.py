"""
Execute a plan: fetch each read window once, in plan order, and hand the
values of its cells to a reducer.

A plan from plan_cells() is read block by block; plan_reads() groups blocks
into larger windows. Windows come back from GDAL as 2D arrays in raster order
(row-major from the top-left); they are used flat, so a segment on window row
lr starting at window column lc is the slice
window.ravel()[lr * width + lc : lr * width + lc + n].
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .plan import _expand
from .reads import block_reads

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


def read_groups(plan):
    """Start/stop offsets of each read in a plan with read windows."""
    r = plan.seg["read"]
    if r.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.concatenate([[0], np.nonzero(np.diff(r) != 0)[0] + 1])
    return starts, np.concatenate([starts[1:], [r.size]])


def _fetch(plan, reader, a, b):
    """Values (float64, flat, plan order) for segments a:b, which share a read."""
    s, src = plan.seg, plan.sources
    n = s["c1"][a:b] - s["c0"][a:b]
    k = int(s["src"][a])
    if k < 0:
        v = np.full(int(n.sum()), src.fill, dtype=np.float64)
    else:
        rd, j = plan.reads, int(s["read"][a])
        xoff, yoff = int(rd["xoff"][j]), int(rd["yoff"][j])
        wx, wy = int(rd["xsize"][j]), int(rd["ysize"][j])
        win = reader.read(src.path[k], int(src.band[k]), xoff, yoff, wx, wy)
        flat = win.ravel()
        # segment position inside the window: block origin + offset in block
        lr = s["brow"][a:b] * int(src.block_y[k]) + s["r"][a:b] - yoff
        lc = s["bcol"][a:b] * int(src.block_x[k]) + s["c0"][a:b] - xoff
        start = lr * wx + lc
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

    Reads happen in plan order, each read window exactly once: one per
    touched block for a plan from plan_cells(), or the windows chosen by
    plan_reads(). With max_workers > 1 windows are fetched on a thread pool,
    at most 2 * max_workers ahead of the reducer, and reduced in order on the
    calling thread.
    """
    if plan.reads is None:
        plan = block_reads(plan)
    reader = _reader(backend)
    starts, stops = read_groups(plan)
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
