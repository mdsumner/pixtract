"""
Cell sets as run tables: row, col_start, col_end, id, w.

Python side: 0-based row/col, half-open [col_start, col_end). This is also
the on-disk convention, for every language; R converts on read and write.

Adapters:
  cells_from_points  - map coordinates -> single-cell runs, id = point index
  cells_from_burn    - controlledburn (Python) result -> runs + weighted edges
  read_cells/write_cells - CSV on disk (0-based, half-open)
"""

from __future__ import annotations

import numpy as np

from .grid import rowcol_from_xy

__all__ = ["cells_from_points", "cells_from_burn", "burn_args", "cells_window",
           "read_cells", "write_cells", "CELL_COLUMNS"]

CELL_COLUMNS = ("row", "col_start", "col_end", "id", "w")


def cells_from_points(xs, ys, grid):
    """Single-cell runs for points inside the grid; id is the point's index.

    Grid logic only, no burn. Points on the grid's edge are inside; points
    outside are left out of the table, so they come back as NaN from the
    Place reducer.
    """
    row, col = rowcol_from_xy(grid.gt, grid.dimension, xs, ys)
    ok = row >= 0
    icol = col[ok]
    irow = row[ok]
    ids = np.nonzero(ok)[0].astype(np.int64)
    return {"row": irow, "col_start": icol, "col_end": icol + 1,
            "id": ids, "w": np.ones(ids.size)}


def _field(t, name):
    return np.asarray(t[name])


def cells_from_burn(r, edges=True):
    """Run table from a controlledburn (Python) burn result.

    controlledburn's Python tables are already 0-based and half-open with
    id = input position, so runs pass through with w = 1. Boundary cells
    from coverage mode become single-cell runs weighted by their fraction.
    The burn must be on the raster's grid; see burn_args().
    """
    runs = r.runs
    parts = [{
        "row": _field(runs, "row"), "col_start": _field(runs, "col_start"),
        "col_end": _field(runs, "col_end"), "id": _field(runs, "id"),
        "w": np.ones(len(runs)),
    }]
    e = getattr(r, "edges", None)
    if edges and e is not None and len(e):
        col = _field(e, "col")
        parts.append({"row": _field(e, "row"), "col_start": col,
                      "col_end": col + 1, "id": _field(e, "id"),
                      "w": _field(e, "fraction").astype(np.float64)})
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]).astype(
        np.float64 if k == "w" else np.int64) for k in CELL_COLUMNS}


def burn_args(grid):
    """extent and shape for controlledburn.burn() so the burn is on `grid`."""
    return {"extent": grid.extent(), "shape": (grid.nrow, grid.ncol)}


def cells_window(cells):
    """Bounding window of a run table: (xoff, yoff, xsize, ysize), 0-based,
    in the grid the runs are on, or None when the table is empty.

    plan_sources(dsn, window=...) uses it to keep only the VRT sources a
    query can touch.
    """
    row = np.asarray(cells["row"])
    if row.size == 0:
        return None
    c0 = int(np.min(cells["col_start"]))
    c1 = int(np.max(cells["col_end"]))
    r0, r1 = int(row.min()), int(row.max()) + 1
    return (c0, r0, c1 - c0, r1 - r0)


def write_cells(path, cells):
    """Write a run table as CSV (0-based, half-open)."""
    arr = np.column_stack([np.asarray(cells[k], dtype=np.float64) for k in CELL_COLUMNS])
    fmt = ["%d", "%d", "%d", "%d", "%.17g"]
    np.savetxt(path, arr, delimiter=",", header=",".join(CELL_COLUMNS),
               comments="", fmt=fmt)


def read_cells(path):
    """Read a run table written by write_cells() or pix_write_cells() in R."""
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    out = {k: a[:, i].astype(np.int64) for i, k in enumerate(CELL_COLUMNS[:4])}
    out["w"] = a[:, 4]
    return out
