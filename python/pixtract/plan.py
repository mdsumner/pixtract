"""
Plan tables: grid, sources, cell-to-block segments, cost.

Python conventions: row/col are 0-based with row 0 at the top, and column
ranges are half-open [col_start, col_end). Everything is in the pixel space
of the dataset being queried (for a VRT, the mosaic grid).

A table is a plain dict of equal-length numpy arrays, so it converts to
pandas/arrow with one call and needs neither.

The pipeline is

    cells  ->  plan_cells(cells, sources)  ->  execute(plan, reducer)

where `cells` is a run table (row, col_start, col_end, id, w). A point is a
run of length 1; a boundary cell with a coverage fraction is a run of length
1 with w = fraction.
"""

from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .grid import gt_dim_to_extent

__all__ = ["Grid", "Sources", "plan_sources", "plan_cells", "cost"]


# -- Grid ---------------------------------------------------------------

@dataclass
class Grid:
    """Pixel grid of a dataset: GDAL geotransform and size."""

    gt: tuple
    ncol: int
    nrow: int

    @property
    def dimension(self):
        """(ncol, nrow), as vaster orders it."""
        return (self.ncol, self.nrow)

    def extent(self):
        """(xmin, xmax, ymin, ymax) of a north-up grid."""
        return gt_dim_to_extent(self.gt, self.dimension)


# -- Sources ------------------------------------------------------------

@dataclass
class Sources:
    """One row per source window in the queried dataset.

    For a plain file there is one source: the file itself. For a VRT whose
    sources are 1:1 pixel copies, there is one source per <SimpleSource> or
    <ComplexSource>, in VRT order (later sources are painted over earlier
    ones).

    xoff, yoff, xsize, ysize: the window in the queried grid (0-based).
    src_xoff, src_yoff: where that window starts inside the source file.
    file_xsize, file_ysize: size of the source file, to clip block reads.
    block_x, block_y: the source file's block size.
    transparent: per-source value treated as "not painted" (VRT <NODATA>).
    dtype, itemsize: data type name and bytes per cell of the queried band,
    used to size reads in memory.
    """

    path: list
    band: np.ndarray
    xoff: np.ndarray
    yoff: np.ndarray
    xsize: np.ndarray
    ysize: np.ndarray
    src_xoff: np.ndarray
    src_yoff: np.ndarray
    file_xsize: np.ndarray
    file_ysize: np.ndarray
    block_x: np.ndarray
    block_y: np.ndarray
    transparent: list
    grid: Grid
    nodata: float | None = None
    dsn: str = ""
    kind: str = "single"
    note: str = ""
    dtype: str = "Float64"
    itemsize: int = 8

    def __len__(self):
        return len(self.path)

    @property
    def fill(self):
        """Value GDAL returns for cells no source covers."""
        return 0.0 if self.nodata is None else self.nodata

    def as_table(self):
        return {k: getattr(self, k) for k in (
            "path", "band", "xoff", "yoff", "xsize", "ysize", "src_xoff",
            "src_yoff", "block_x", "block_y")}


def _i64(x):
    return np.asarray(x, dtype=np.int64)


def _single_source(dsn, band, grid, block, nodata, note="", dtype=("Float64", 8)):
    return Sources(
        path=[dsn], band=_i64([band]), xoff=_i64([0]), yoff=_i64([0]),
        xsize=_i64([grid.ncol]), ysize=_i64([grid.nrow]),
        src_xoff=_i64([0]), src_yoff=_i64([0]),
        file_xsize=_i64([grid.ncol]), file_ysize=_i64([grid.nrow]),
        block_x=_i64([block[0]]), block_y=_i64([block[1]]),
        transparent=[None], grid=grid, nodata=nodata, dsn=dsn,
        kind="single", note=note, dtype=dtype[0], itemsize=dtype[1],
    )


def plan_sources(dsn, band=1, backend="gdal", expand_vrt=True):
    """Build the source table for a dataset. Metadata only, no pixel reads.

    A VRT is expanded into its sources when every source is a 1:1 pixel copy
    (no resampling, scaling, LUT or mask use) of the same data type as the
    VRT band; anything else is read through the VRT itself as a single
    source, and `note` says why.
    """
    if backend == "rasterio":
        return _plan_sources_rasterio(dsn, band)
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(dsn)
    grid = Grid(tuple(ds.GetGeoTransform()), ds.RasterXSize, ds.RasterYSize)
    b = ds.GetRasterBand(band)
    nodata = b.GetNoDataValue()
    block = b.GetBlockSize()
    dtype = (gdal.GetDataTypeName(b.DataType), gdal.GetDataTypeSize(b.DataType) // 8)
    if expand_vrt and ds.GetDriver().ShortName == "VRT":
        try:
            src = _vrt_sources(ds, dsn, band, grid, nodata, b.DataType)
        except _Unsupported as e:
            return _single_source(dsn, band, grid, block, nodata,
                                  note=f"VRT read as one source: {e}", dtype=dtype)
        if src is not None:
            src.dtype, src.itemsize = dtype
            return src
    return _single_source(dsn, band, grid, block, nodata, dtype=dtype)


def _plan_sources_rasterio(dsn, band):
    import rasterio
    with rasterio.open(dsn) as src:
        grid = Grid(tuple(src.transform.to_gdal()), src.width, src.height)
        by, bx = src.block_shapes[band - 1]  # rasterio gives (rows, cols)
        nodata = src.nodata
        dt = np.dtype(src.dtypes[band - 1])
    return _single_source(dsn, band, grid, (bx, by), nodata,
                          note="rasterio backend reads the dataset as one source",
                          dtype=(src.dtypes[band - 1], dt.itemsize))


class _Unsupported(Exception):
    pass


_SOURCE_TAGS = {"SimpleSource", "ComplexSource"}
_OK_CHILDREN = {"SourceFilename", "SourceBand", "SourceProperties", "SrcRect",
                "DstRect", "NODATA", "OpenOptions"}


def _as_int(v, what):
    f = float(v)
    if f != int(f):
        raise _Unsupported(f"non-integer {what} {v}")
    return int(f)


def _vrt_sources(ds, dsn, band, grid, nodata, vrt_dtype):
    from osgeo import gdal
    xml = ds.GetMetadata("xml:VRT")
    if not xml:
        raise _Unsupported("no xml:VRT metadata")
    root = ET.fromstring(xml[0])
    vb = None
    for el in root.findall("VRTRasterBand"):
        if int(el.get("band", "0")) == band:
            vb = el
    if vb is None:
        raise _Unsupported(f"band {band} not found in VRT XML")
    if vb.get("subClass"):
        raise _Unsupported(f"band subClass {vb.get('subClass')}")
    vrt_dir = posixpath.dirname(dsn)

    rows = []
    for el in vb:
        if el.tag not in _SOURCE_TAGS:
            if el.tag.endswith("Source"):
                raise _Unsupported(f"source type {el.tag}")
            continue
        extra = {c.tag for c in el} - _OK_CHILDREN
        if extra:
            raise _Unsupported(f"{el.tag} with {sorted(extra)}")
        fn_el = el.find("SourceFilename")
        fn = fn_el.text
        if fn_el.get("relativeToVRT", "0") == "1":
            fn = posixpath.join(vrt_dir, fn)
        sband = el.findtext("SourceBand", "1")
        if not sband.isdigit():
            raise _Unsupported(f"SourceBand {sband}")
        src_r = el.find("SrcRect")
        dst_r = el.find("DstRect")
        if src_r is None or dst_r is None:
            raise _Unsupported("source without SrcRect/DstRect")
        sx, sy, sw, sh = (_as_int(src_r.get(k), "SrcRect")
                          for k in ("xOff", "yOff", "xSize", "ySize"))
        dx, dy, dw, dh = (_as_int(dst_r.get(k), "DstRect")
                          for k in ("xOff", "yOff", "xSize", "ySize"))
        if (sw, sh) != (dw, dh):
            raise _Unsupported("resampled source (SrcRect size != DstRect size)")
        props = el.find("SourceProperties")
        if props is not None and props.get("BlockXSize"):
            fx, fy = int(props.get("RasterXSize")), int(props.get("RasterYSize"))
            bx, by = int(props.get("BlockXSize")), int(props.get("BlockYSize"))
            dtype = gdal.GetDataTypeByName(props.get("DataType", ""))
        else:
            sds = gdal.Open(fn)
            sb = sds.GetRasterBand(int(sband))
            fx, fy = sds.RasterXSize, sds.RasterYSize
            bx, by = sb.GetBlockSize()
            dtype = sb.DataType
            sds = None
        if dtype != vrt_dtype:
            raise _Unsupported("source data type differs from the VRT band")
        tr = el.findtext("NODATA")
        transparent = None if tr is None else float(tr)
        # clip the destination window to the VRT grid
        x0, y0 = max(dx, 0), max(dy, 0)
        x1, y1 = min(dx + dw, grid.ncol), min(dy + dh, grid.nrow)
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append((fn, int(sband), x0, y0, x1 - x0, y1 - y0,
                     sx + (x0 - dx), sy + (y0 - dy), fx, fy, bx, by, transparent))

    if not rows:
        return None
    cols = list(zip(*rows))
    src = Sources(
        path=list(cols[0]), band=_i64(cols[1]), xoff=_i64(cols[2]),
        yoff=_i64(cols[3]), xsize=_i64(cols[4]), ysize=_i64(cols[5]),
        src_xoff=_i64(cols[6]), src_yoff=_i64(cols[7]),
        file_xsize=_i64(cols[8]), file_ysize=_i64(cols[9]),
        block_x=_i64(cols[10]), block_y=_i64(cols[11]),
        transparent=list(cols[12]), grid=grid, nodata=nodata, dsn=dsn,
        kind="vrt",
    )
    if any(t is not None for t in src.transparent) and _any_overlap(src):
        raise _Unsupported("overlapping sources with per-source NODATA")
    return src


def _any_overlap(src):
    n = len(src)
    x0, y0 = src.xoff, src.yoff
    x1, y1 = x0 + src.xsize, y0 + src.ysize
    for i in range(n - 1):
        j = slice(i + 1, n)
        hit = (x0[i] < x1[j]) & (x0[j] < x1[i]) & (y0[i] < y1[j]) & (y0[j] < y1[i])
        if hit.any():
            return True
    return False


# -- Cells -> plan --------------------------------------------------------

@dataclass
class Plan:
    """Cell-to-block plan: one row per segment, sorted by (src, brow, bcol).

    A segment is a contiguous piece of one run that lies inside one block of
    one source. `r`, `c0`, `c1` are the row and half-open column range
    inside the block; `row`, `col` locate the segment's first cell in the
    queried grid; `run` indexes the input cell table. src == -1 marks cells
    that no source covers.

    `reads` is None until plan_reads() groups blocks into read windows; it
    then holds the read table and every segment has a `read` column, with
    segments sorted by read.
    """

    seg: dict
    sources: Sources
    n_id: int
    n_run: int
    order: str = "src, brow, bcol"
    extra: dict = field(default_factory=dict)
    reads: dict | None = None

    def __len__(self):
        return len(self.seg["src"])

    @property
    def n_cells(self):
        return int((self.seg["c1"] - self.seg["c0"]).sum())


def _expand(lo, hi):
    """For ranges [lo, hi) return (which range, value) for every element."""
    n = hi - lo
    which = np.repeat(np.arange(len(lo)), n)
    first = np.repeat(np.cumsum(n) - n, n)
    return which, np.repeat(lo, n) + (np.arange(which.size) - first)


def _pieces(lo, hi):
    """_expand, but (None, lo) when every range has one element (the common
    case for points), so callers can skip the gathers."""
    if lo.size == 0 or ((hi - lo) == 1).all():
        return None, lo
    return _expand(lo, hi)


def _take(a, idx):
    return a if idx is None else a[idx]


def _per_source(values, src, covered, default):
    """values[src] where covered, else default; scalar when there is one source."""
    if values.size == 1 and covered.all():
        return values[0]
    return np.where(covered, values[np.where(covered, src, 0)], default)


def plan_cells(cells, sources):
    """Intersect a run table with the sources and their block grids.

    Pure integer arithmetic. Cost is O(runs x pieces per run), independent of
    the number of cells a run covers.
    """
    g = sources.grid
    row = _i64(cells["row"])
    c0 = _i64(cells["col_start"])
    c1 = _i64(cells["col_end"])
    ids = _i64(cells["id"])
    w = np.asarray(cells.get("w", np.ones(row.size)), dtype=np.float64)
    if w.shape == ():
        w = np.full(row.size, float(w))
    run = np.arange(row.size, dtype=np.int64)
    n_id = int(ids.max()) + 1 if ids.size else 0

    # clip to the grid
    c0c, c1c = np.maximum(c0, 0), np.minimum(c1, g.ncol)
    keep = (row >= 0) & (row < g.nrow) & (c1c > c0c)
    row, c0, c1, ids, w, run = row[keep], c0c[keep], c1c[keep], ids[keep], w[keep], run[keep]

    # 1. split runs at source column breaks and resolve which source wins
    xb = np.unique(np.concatenate([sources.xoff, sources.xoff + sources.xsize]))
    yb = np.unique(np.concatenate([sources.yoff, sources.yoff + sources.ysize]))
    claim = np.full((max(yb.size - 1, 0), max(xb.size - 1, 0)), -1, dtype=np.int64)
    for s in range(len(sources)):  # later sources overwrite earlier ones
        i0, i1 = np.searchsorted(yb, [sources.yoff[s], sources.yoff[s] + sources.ysize[s]])
        j0, j1 = np.searchsorted(xb, [sources.xoff[s], sources.xoff[s] + sources.xsize[s]])
        claim[i0:i1, j0:j1] = s

    # every source break inside a run becomes a cut
    breaks = np.concatenate([[np.iinfo(np.int64).min], xb, [np.iinfo(np.int64).max]])
    k0 = np.searchsorted(breaks, c0, side="right") - 1
    k1 = np.searchsorted(breaks, c1 - 1, side="right") - 1
    which, k = _pieces(k0, k1 + 1)
    p_lo = np.maximum(_take(c0, which), breaks[k])
    p_hi = np.minimum(_take(c1, which), breaks[k + 1])
    p_row = _take(row, which)
    ci = np.searchsorted(yb, p_row, side="right") - 1
    cj = k - 1  # index into xb intervals
    ok = (ci >= 0) & (ci < claim.shape[0]) & (cj >= 0) & (cj < claim.shape[1])
    src = np.full(k.size, -1, dtype=np.int64)
    src[ok] = claim[ci[ok], cj[ok]]

    # 2. split each piece at its source's block column boundaries
    covered = src >= 0
    bx = _per_source(sources.block_x, src, covered, 1 << 62)
    by = _per_source(sources.block_y, src, covered, 1 << 62)
    dxo = _per_source(sources.src_xoff - sources.xoff, src, covered, 0)
    dyo = _per_source(sources.src_yoff - sources.yoff, src, covered, 0)
    l_lo, l_hi = p_lo + dxo, p_hi + dxo  # columns in the source file
    l_row = p_row + dyo
    b0 = l_lo // bx
    b1 = (l_hi - 1) // bx
    w2, bcol = _pieces(b0, b1 + 1)
    bx2 = bx if np.ndim(bx) == 0 else _take(bx, w2)
    by2 = by if np.ndim(by) == 0 else _take(by, w2)
    dx2 = dxo if np.ndim(dxo) == 0 else _take(dxo, w2)
    l_row2 = _take(l_row, w2)
    s_lo = np.maximum(_take(l_lo, w2), bcol * bx2)
    s_hi = np.minimum(_take(l_hi, w2), (bcol + 1) * bx2)
    brow = l_row2 // by2
    seg = {
        "src": _take(src, w2),
        "brow": brow,
        "bcol": bcol,
        "r": l_row2 - brow * by2,
        "c0": s_lo - bcol * bx2,
        "c1": s_hi - bcol * bx2,
        "row": _take(p_row, w2),
        "col": s_lo - dx2,
        "id": _take(_take(ids, which), w2),
        "w": _take(_take(w, which), w2),
        "run": _take(_take(run, which), w2),
    }
    unc = seg["src"] < 0
    seg["brow"][unc] = 0
    seg["bcol"][unc] = 0
    seg["r"][unc] = 0
    # one integer sort key; uncovered (src -1) sorts first
    nbr = int(seg["brow"].max()) + 1 if seg["brow"].size else 1
    nbc = int(seg["bcol"].max()) + 1 if seg["bcol"].size else 1
    key = ((seg["src"] + 1) * nbr + seg["brow"]) * nbc + seg["bcol"]
    o = np.argsort(key, kind="stable")
    seg = {k: v[o] for k, v in seg.items()}
    return Plan(seg=seg, sources=sources, n_id=n_id, n_run=len(cells["row"]))


def block_groups(plan):
    """Start/stop offsets of each (src, brow, bcol) group in a sorted plan."""
    s = plan.seg
    if len(plan) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key_change = ((np.diff(s["src"]) != 0) | (np.diff(s["brow"]) != 0)
                  | (np.diff(s["bcol"]) != 0))
    starts = np.concatenate([[0], np.nonzero(key_change)[0] + 1])
    stops = np.concatenate([starts[1:], [len(plan)]])
    return starts, stops


def cost(plan, bytes=False):
    """What a plan will read, before any pixel I/O.

    blocks counts touched source blocks; with read windows from plan_reads(),
    reads and read_bytes (decoded, in memory) count what the windows fetch.

    With bytes=True, block byte counts are looked up from TIFF metadata
    (BLOCK_SIZE_x_y), which reads only the IFD; None where unavailable.
    """
    starts, _ = block_groups(plan)
    s = plan.seg
    covered = s["src"][starts] >= 0 if starts.size else np.zeros(0, bool)
    out = {
        "cells": plan.n_cells,
        "segments": len(plan),
        "sources": int(np.unique(s["src"][s["src"] >= 0]).size),
        "blocks": int(covered.sum()),
        "uncovered_cells": int((s["c1"] - s["c0"])[s["src"] < 0].sum()),
    }
    if plan.reads is not None:
        rd = plan.reads
        ok = rd["src"] >= 0
        out["reads"] = int(ok.sum())
        out["read_bytes"] = int(rd["bytes"][ok].sum())
    if bytes:
        out["bytes"] = _block_bytes(plan, starts[covered])
    return out


def _block_bytes(plan, starts):
    from osgeo import gdal
    gdal.UseExceptions()
    s, src = plan.seg, plan.sources
    total = 0
    handles = {}
    for i in starts:
        k = int(s["src"][i])
        if k not in handles:
            ds = gdal.Open(src.path[k])
            handles[k] = (ds, ds.GetRasterBand(int(src.band[k])))
        v = handles[k][1].GetMetadataItem(
            f"BLOCK_SIZE_{int(s['bcol'][i])}_{int(s['brow'][i])}", "TIFF")
        if v is None:
            return None
        total += int(v)
    return total
