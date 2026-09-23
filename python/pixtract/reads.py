"""
Source inspection and read planning: which sources a query touches, and
which read windows to fetch them with.

    inspect_source(dsn)        what the dataset is made of: grid, data type,
                               blocks, overviews, and (for a VRT mosaic) one
                               row per source with its extent and block layout
    source_usage(plan)         per source: cells, runs, blocks the query
                               touches, and how densely
    plan_reads(plan, mem=...)  group touched blocks into read windows that fit
                               a memory budget
    plan_extraction(dsn, cells, mem=...)
                               all of the above in one call; the result goes
                               straight to execute() with any reducer

Read windows ("meta-tiles"). Each source's block grid is covered by a
coarser grid of meta-tiles, kx by ky blocks each, aligned to block (0, 0).
Within a meta-tile the query touches some blocks; they are fetched either as
one window (the bounding box of the touched blocks, untouched blocks inside
it included) or block by block, whichever the cost model says is cheaper:

    cost(read) = request_bytes + decoded bytes of the read

request_bytes is the fixed cost of one read call expressed in bytes, so it
trades the number of reads against the bytes they fetch. It is larger for
remote sources (a round trip) than for local files. (kx, ky) is chosen per
source, from powers of two up to the whole file, as the shape with the
lowest total cost whose largest possible window fits the budget for one
read. Clustered queries get large windows, scattered ones get block reads,
and a dense query over a small file gets one read of the whole file.

The budget for one read is mem divided by the number of reads execute()
can hold at once: 1 serially, 2 * max_workers with threads.

Python conventions: 0-based, half-open. A read window is xoff, yoff, xsize,
ysize in the source file's own pixel grid, as GDAL takes them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .grid import gt_dim_to_extent
from .cells import cells_window
from .plan import block_groups, default_planner, plan_cells, plan_sources

__all__ = ["inspect_source", "source_table", "source_usage", "plan_reads",
           "block_reads", "plan_extraction", "request_bytes_for",
           "DEFAULT_MEM", "LOCAL_REQUEST_BYTES", "REMOTE_REQUEST_BYTES"]

#: default memory budget for pixel buffers in flight (256 MiB)
DEFAULT_MEM = 256 * 2**20
#: cost of one read call on a local file, in bytes (per-call overhead only)
LOCAL_REQUEST_BYTES = 64 * 2**10
#: cost of one read call on a remote source (about one round trip at tens
#: of MB/s)
REMOTE_REQUEST_BYTES = 2 * 2**20

_REMOTE = ("/vsicurl", "/vsis3", "/vsigs", "/vsiaz", "/vsiadls", "/vsioss",
           "/vsiswift", "/vsiwebhdfs", "/vsihdfs", "http://", "https://",
           "ftp://")


def request_bytes_for(path):
    """Default per-read cost for a path: remote (/vsicurl, /vsis3, http...)
    or local."""
    remote = any(t in path for t in _REMOTE)
    return REMOTE_REQUEST_BYTES if remote else LOCAL_REQUEST_BYTES


# -- Inspect ----------------------------------------------------------------

def _nblocks(lo, n, b):
    """Blocks of size b touched by [lo, lo + n)."""
    return (lo + n - 1) // b - lo // b + 1


def source_table(sources):
    """One row per source: where it sits in the mosaic, its map extent, and
    the block layout of the part of the file that the mosaic uses."""
    s = sources
    gt = s.grid.gt
    n = len(s)
    ext = np.full((n, 4), np.nan)
    if gt[2] == 0 and gt[4] == 0:
        for i in range(n):
            sub = (gt[0] + s.xoff[i] * gt[1], gt[1], 0.0,
                   gt[3] + s.yoff[i] * gt[5], 0.0, gt[5])
            ext[i] = gt_dim_to_extent(sub, (s.xsize[i], s.ysize[i]))
    nbx = _nblocks(s.src_xoff, s.xsize, s.block_x)
    nby = _nblocks(s.src_yoff, s.ysize, s.block_y)
    return {
        "src": np.arange(n), "path": list(s.path), "band": s.band,
        "xoff": s.xoff, "yoff": s.yoff, "xsize": s.xsize, "ysize": s.ysize,
        "xmin": ext[:, 0], "xmax": ext[:, 1], "ymin": ext[:, 2], "ymax": ext[:, 3],
        "file_xsize": s.file_xsize, "file_ysize": s.file_ysize,
        "block_x": s.block_x, "block_y": s.block_y,
        "nblocks_x": nbx, "nblocks_y": nby, "blocks": nbx * nby,
        "block_bytes": s.block_x * s.block_y * s.itemsize,
        "remote": np.array([request_bytes_for(p) == REMOTE_REQUEST_BYTES
                            for p in s.path]),
    }


def inspect_source(dsn, band=1, planner=None, overviews=True):
    """What a dataset is made of, from metadata only (no pixel reads).

    Returns a dict: dsn, driver, kind ("vrt" when the VRT was expanded into
    its sources, else "single"), ncol, nrow, gt, dtype, itemsize, block
    (the dataset's own block size), nodata, note, overviews (one dict per
    overview level of the dataset: ncol, nrow, factor, block), sources (the
    Sources object that plan_cells() takes) and table (source_table()).

    Overviews are reported, not used: extraction reads native-resolution
    cells. `planner` is as for plan_sources(); driver, block and overviews
    are only filled in by osgeo.gdal.
    """
    if planner is None:
        planner = default_planner()
    src = plan_sources(dsn, band, planner=planner)
    out = {
        "dsn": dsn, "driver": None, "kind": src.kind,
        "ncol": src.grid.ncol, "nrow": src.grid.nrow, "gt": src.grid.gt,
        "dtype": src.dtype, "itemsize": src.itemsize, "block": None,
        "nodata": src.nodata, "note": src.note, "overviews": [],
        "sources": src, "table": source_table(src),
    }
    if planner == "gdal":
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(dsn)
        b = ds.GetRasterBand(band)
        out["driver"] = ds.GetDriver().ShortName
        out["block"] = tuple(b.GetBlockSize())
        if overviews:
            for i in range(b.GetOverviewCount()):
                o = b.GetOverview(i)
                out["overviews"].append({
                    "level": i, "ncol": o.XSize, "nrow": o.YSize,
                    "factor": src.grid.ncol / o.XSize,
                    "block": tuple(o.GetBlockSize())})
    return out


# -- Usage --------------------------------------------------------------------

def _blocks(plan):
    """One row per touched block (plan order): src, brow, bcol, cells,
    segments, and the segment range [start, stop)."""
    starts, stops = block_groups(plan)
    s = plan.seg
    n = s["c1"] - s["c0"]
    cells = np.add.reduceat(n, starts) if starts.size else np.zeros(0, np.int64)
    return {"src": s["src"][starts], "brow": s["brow"][starts],
            "bcol": s["bcol"][starts], "cells": cells,
            "segments": stops - starts, "start": starts, "stop": stops}


def _count_distinct(group, value):
    """Number of distinct values per group (group ids 0..max)."""
    if group.size == 0:
        return np.zeros(0, np.int64)
    pairs = np.unique(np.stack([group, value]), axis=1)
    return np.bincount(pairs[0], minlength=int(group.max()) + 1)


def _block_extent(src, k, brow, bcol):
    """Decoded bytes of whole blocks, clipped at the file's right and bottom."""
    bx, by = src.block_x[k], src.block_y[k]
    w = np.minimum(bx, src.file_xsize[k] - bcol * bx)
    h = np.minimum(by, src.file_ysize[k] - brow * by)
    return w * h * src.itemsize


def source_usage(plan):
    """Per source the query touches: how much of it, and how densely.

    cells, runs, segments: of the query, in this source.
    blocks: blocks touched. span: blocks in the bounding box of the touched
    blocks (brow0..brow1, bcol0..bcol1, inclusive). occupancy = blocks /
    span: near 1 for a clustered query, near 0 for a scattered one.
    block_bytes: decoded bytes of the touched blocks.
    total_blocks: blocks of the source that the mosaic uses.
    A row with src -1 counts cells no source covers.
    """
    s, src = plan.seg, plan.sources
    b = _blocks(plan)
    ks = np.unique(b["src"])
    grp = np.searchsorted(ks, s["src"])
    runs = _count_distinct(grp, s["run"])
    tab = source_table(src)
    rows = []
    for i, k in enumerate(ks):
        m = b["src"] == k
        seg_m = grp == i
        row = {"src": int(k), "path": src.path[k] if k >= 0 else None,
               "cells": int(b["cells"][m].sum()), "runs": int(runs[i]),
               "segments": int(seg_m.sum())}
        if k < 0:
            row.update(blocks=0, span=0, occupancy=np.nan, brow0=-1, brow1=-1,
                       bcol0=-1, bcol1=-1, block_bytes=0, total_blocks=0)
        else:
            br, bc = b["brow"][m], b["bcol"][m]
            span = int((br.max() - br.min() + 1) * (bc.max() - bc.min() + 1))
            row.update(blocks=int(m.sum()), span=span,
                       occupancy=m.sum() / span,
                       brow0=int(br.min()), brow1=int(br.max()),
                       bcol0=int(bc.min()), bcol1=int(bc.max()),
                       block_bytes=int(_block_extent(src, k, br, bc).sum()),
                       total_blocks=int(tab["blocks"][k]))
        rows.append(row)
    keys = rows[0].keys() if rows else ("src", "path", "cells", "runs", "segments",
                                         "blocks", "span", "occupancy", "brow0",
                                         "brow1", "bcol0", "bcol1", "block_bytes",
                                         "total_blocks")
    return {k: ([r[k] for r in rows] if k == "path" else np.array([r[k] for r in rows]))
            for k in keys}


# -- Read planning -------------------------------------------------------------

def _pow2_upto(n):
    out, k = [], 1
    while k < n:
        out.append(k)
        k *= 2
    out.append(max(n, 1))
    return out


def _meta_cost(brow, bcol, bbytes, kx, ky, bx, by, fx, fy, isz, req):
    """Cost of meta-tiling touched blocks with kx x ky meta-tiles.

    Returns total cost, and per block: meta-tile id and whether its meta-tile
    is read as one window (True) or block by block (False).
    """
    mr, mc = brow // ky, bcol // kx
    key = mr * (int(mc.max()) + 1) + mc
    uk, g = np.unique(key, return_inverse=True)
    ng = uk.size
    r0 = np.full(ng, np.iinfo(np.int64).max)
    r1 = np.full(ng, -1)
    c0 = r0.copy()
    c1 = r1.copy()
    np.minimum.at(r0, g, brow)
    np.maximum.at(r1, g, brow)
    np.minimum.at(c0, g, bcol)
    np.maximum.at(c1, g, bcol)
    n = np.bincount(g, minlength=ng)
    wx = np.minimum((c1 + 1) * bx, fx) - c0 * bx
    wy = np.minimum((r1 + 1) * by, fy) - r0 * by
    wcost = req + wx * wy * isz
    pcost = n * req + np.bincount(g, weights=bbytes, minlength=ng)
    win = wcost < pcost
    return float(np.where(win, wcost, pcost).sum()), g, win[g]


def _window_budget(mem, max_workers):
    in_flight = 2 * max_workers if max_workers and max_workers > 1 else 1
    return max(int(mem) // in_flight, 1)


def block_reads(plan):
    """The plan with one read per touched block (what execute() does when a
    plan has no read windows)."""
    b = _blocks(plan)
    return _with_reads(plan, b, np.arange(b["src"].size))


def plan_reads(plan, mem=DEFAULT_MEM, request_bytes=None, max_workers=None):
    """Group a plan's touched blocks into read windows that fit `mem`.

    mem: bytes of decoded pixels execute() may hold at once (default 256 MiB).
    request_bytes: cost of one read in bytes; None picks it per source path
    (request_bytes_for()). A number, or a callable path -> number.
    max_workers: as passed to execute(); divides mem between reads in flight.

    Returns a new Plan with `reads` (one row per read, see _with_reads) and
    `extra["meta"]`: per source, the chosen meta-tile (kx, ky), the cost of
    the chosen reads and, for comparison, of reading block by block.
    """
    src = plan.sources
    b = _blocks(plan)
    budget = _window_budget(mem, max_workers)
    read_of = np.zeros(b["src"].size, np.int64)
    meta = []
    next_id = 0
    for k in np.unique(b["src"]):
        m = np.nonzero(b["src"] == k)[0]
        if k < 0:
            read_of[m] = next_id
            next_id += 1
            continue
        path = src.path[k]
        if request_bytes is None:
            req = request_bytes_for(path)
        elif callable(request_bytes):
            req = request_bytes(path)
        else:
            req = request_bytes
        bx, by = int(src.block_x[k]), int(src.block_y[k])
        fx, fy = int(src.file_xsize[k]), int(src.file_ysize[k])
        isz = int(src.itemsize)
        br, bc = b["brow"][m], b["bcol"][m]
        bb = _block_extent(src, k, br, bc).astype(np.float64)
        nbx, nby = -(-fx // bx), -(-fy // by)
        base = float((req + bb).sum())
        best = (base, 1, 1, np.arange(m.size), np.zeros(m.size, bool))
        for ky in _pow2_upto(nby):
            for kx in _pow2_upto(nbx):
                if kx == ky == 1:
                    continue
                if min(kx * bx, fx) * min(ky * by, fy) * isz > budget:
                    continue
                c, g, win = _meta_cost(br, bc, bb, kx, ky, bx, by, fx, fy, isz, req)
                if c < best[0]:
                    best = (c, kx, ky, g, win)
        c, kx, ky, g, win = best
        # read ids: one per windowed meta-tile, one per block otherwise
        rid = np.where(win, g, -1)
        ug, inv = np.unique(rid[win], return_inverse=True) if win.any() else (np.zeros(0), None)
        ids = np.empty(m.size, np.int64)
        if win.any():
            ids[win] = next_id + inv
        nb = int((~win).sum())
        ids[~win] = next_id + ug.size + np.arange(nb)
        # order reads by meta-tile (row-major), then block
        mt_order = (br // ky) * (nbx // kx + 1) + bc // kx
        blk = br * nbx + bc
        u, first = np.unique(ids[np.lexsort((blk, mt_order))], return_index=True)
        rank = np.argsort(np.argsort(first))
        read_of[m] = next_id + rank[np.searchsorted(u, ids)]
        n_reads = ug.size + nb
        next_id += n_reads
        meta.append({"src": int(k), "path": path, "block_x": bx, "block_y": by,
                     "kx": kx, "ky": ky, "budget": budget, "request_bytes": req,
                     "blocks": int(m.size), "reads": int(n_reads),
                     "windows": int(ug.size), "cost": c,
                     "block_cost": base})
    out = _with_reads(plan, b, read_of)
    out.extra = dict(plan.extra)
    out.extra["meta"] = {key: (np.array([r[key] for r in meta]) if key != "path"
                               else [r[key] for r in meta])
                         for key in (meta[0].keys() if meta else ())}
    out.extra["mem"] = int(mem)
    return out


def _with_reads(plan, b, read_of):
    """Attach a read table to a plan, given each touched block's read id.

    Read table (0-based, half-open, in the source file's grid):
      read, src, xoff, yoff, xsize, ysize   the window
      blocks    blocks inside the window; touched: blocks with query cells
      cells, segments, runs                 query content of the read
      bytes     decoded bytes of the window
    """
    src, s = plan.sources, plan.seg
    nr = int(read_of.max()) + 1 if read_of.size else 0
    ks = np.full(nr, -1, np.int64)
    ks[read_of] = b["src"]
    r0 = np.full(nr, np.iinfo(np.int64).max)
    r1 = np.full(nr, -1)
    c0 = r0.copy()
    c1 = r1.copy()
    np.minimum.at(r0, read_of, b["brow"])
    np.maximum.at(r1, read_of, b["brow"])
    np.minimum.at(c0, read_of, b["bcol"])
    np.maximum.at(c1, read_of, b["bcol"])
    cov = ks >= 0
    kk = np.where(cov, ks, 0)
    bx = np.where(cov, src.block_x[kk], 0)
    by = np.where(cov, src.block_y[kk], 0)
    fx = np.where(cov, src.file_xsize[kk], 0)
    fy = np.where(cov, src.file_ysize[kk], 0)
    xoff, yoff = c0 * bx, r0 * by
    xsize = np.minimum((c1 + 1) * bx, fx) - xoff
    ysize = np.minimum((r1 + 1) * by, fy) - yoff
    blocks = (c1 - c0 + 1) * (r1 - r0 + 1)
    for a in (xoff, yoff, xsize, ysize, blocks):
        a[~cov] = 0
    # per segment read id, then sort segments by read (stable keeps block order)
    seg_read = np.repeat(read_of, b["stop"] - b["start"])
    o = np.argsort(seg_read, kind="stable")
    seg = {k: v[o] for k, v in s.items()}
    seg["read"] = seg_read[o]
    runs = _count_distinct(seg["read"], seg["run"])
    reads = {
        "read": np.arange(nr), "src": ks,
        "xoff": xoff, "yoff": yoff, "xsize": xsize, "ysize": ysize,
        "blocks": blocks,
        "touched": np.bincount(read_of, minlength=nr),
        "cells": np.bincount(read_of, weights=b["cells"], minlength=nr).astype(np.int64),
        "segments": np.bincount(read_of, weights=b["segments"], minlength=nr).astype(np.int64),
        "runs": np.pad(runs, (0, nr - runs.size)),
        "bytes": xsize * ysize * src.itemsize,
    }
    return replace(plan, seg=seg, reads=reads, order="read")


def plan_extraction(dsn, cells, band=1, mem=DEFAULT_MEM, request_bytes=None,
                    max_workers=None, planner=None):
    """Sources, cell plan and read windows for a query, in one call.

    `cells` is a run table (for points, cells_from_points(xs, ys, grid) with
    the grid of plan_sources(dsn)). Pass the result to execute() with the
    same max_workers and whichever reader backend you like. `planner` is as
    for plan_sources().
    """
    sources = plan_sources(dsn, band, planner=planner, window=cells_window(cells))
    plan = plan_cells(cells, sources)
    return plan_reads(plan, mem=mem, request_bytes=request_bytes,
                      max_workers=max_workers)
