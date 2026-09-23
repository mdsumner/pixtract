"""
Grid logic, with the names and conventions of vaster (hypertidy/vaster,
hypertidy/vaster-rs): dimension is (ncol, nrow), extent is
(xmin, xmax, ymin, ymax), row/col/cell are 0-based in raster order
(row-major from the top-left). R/grid.R has the same functions 1-based.
When a Python vaster exists these can be replaced by it.

Edge rule (as vaster and terra): a point on any edge of the grid is inside,
so a point exactly on the right or bottom edge falls in the last column or
row.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rowcol_from_xy", "cell_from_row_col", "gt_dim_to_extent",
           "extent_dim_to_gt"]


def rowcol_from_xy(gt, dimension, x, y):
    """0-based (row, col) of the cell containing each (x, y); -1 outside.

    Uses the full geotransform, so rotated grids work. A north-up grid is
    bounds-checked against gt_dim_to_extent() in coordinates, so a point
    taken from that extent is inside exactly.
    """
    nc, nr = int(dimension[0]), int(dimension[1])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        if gt[2] == 0 and gt[4] == 0:
            px = (x - gt[0]) / gt[1]
            py = (y - gt[3]) / gt[5]
            e = gt_dim_to_extent(gt, dimension)
            ok = (x >= e[0]) & (x <= e[1]) & (y >= e[2]) & (y <= e[3])
        else:
            det = gt[1] * gt[5] - gt[2] * gt[4]
            px = (gt[5] * (x - gt[0]) - gt[2] * (y - gt[3])) / det
            py = (-gt[4] * (x - gt[0]) + gt[1] * (y - gt[3])) / det
            ok = (px >= 0) & (px <= nc) & (py >= 0) & (py <= nr)
    col = np.clip(np.floor(np.where(ok, px, 0)), 0, nc - 1).astype(np.int64)
    row = np.clip(np.floor(np.where(ok, py, 0)), 0, nr - 1).astype(np.int64)
    col[~ok] = -1
    row[~ok] = -1
    return row, col


def cell_from_row_col(dimension, row, col):
    """0-based cell (flat raster-order index) of (row, col); -1 outside."""
    nc, nr = int(dimension[0]), int(dimension[1])
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    ok = (row >= 0) & (row < nr) & (col >= 0) & (col < nc)
    return np.where(ok, row * nc + col, -1)


def gt_dim_to_extent(gt, dimension):
    """(xmin, xmax, ymin, ymax) of a north-up grid."""
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError("gt_dim_to_extent() needs a north-up geotransform")
    x0, x1 = gt[0], gt[0] + dimension[0] * gt[1]
    y0, y1 = gt[3], gt[3] + dimension[1] * gt[5]
    return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def extent_dim_to_gt(extent, dimension):
    """North-up geotransform from extent and dimension."""
    return (extent[0], (extent[1] - extent[0]) / dimension[0], 0.0,
            extent[3], 0.0, -(extent[3] - extent[2]) / dimension[1])
