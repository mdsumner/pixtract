"""
Entry points built on the planner.

extract_points  - raster values at (x, y); the original pixtract API
extract_cells   - every cell of a run table with its value
zonal_stats     - grouped weighted statistics over a run table
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .cells import cells_from_points
from .execute import execute
from .plan import plan_cells, plan_sources
from .reduce import Cells, Place, Stats

__all__ = ["extract_points", "extract_cells", "zonal_stats"]


def extract_points(
    raster_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    band: int = 1,
    backend: Literal["gdal", "rasterio"] = "gdal",
    max_workers: int | None = None,
) -> np.ndarray:
    """Extract raster values at (x, y) coordinates using block-grouped reads.

    Parameters
    ----------
    raster_path : str
        Path to raster (local path, /vsicurl/..., /vsis3/..., VRT, etc.)
    xs, ys : array-like
        Map coordinates in the raster's CRS.
    band : int
        Band number (GDAL numbering, starting at 1).
    backend : "gdal" | "rasterio"
        Which library to use for I/O. With "gdal" a VRT of 1:1 sources is
        planned against its source files' own blocks.
    max_workers : int or None
        Thread pool size for parallel block reads. None or 1 for serial.

    Returns
    -------
    np.ndarray
        Extracted values (float64). NoData pixels and points outside the
        raster -> NaN.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    sources = plan_sources(raster_path, band, backend=backend)
    cells = cells_from_points(xs, ys, sources.grid)
    plan = plan_cells(cells, sources)
    plan.n_id = xs.size
    return execute(plan, Place(), backend=backend, max_workers=max_workers)


def extract_cells(raster_path, cells, band=1, backend="gdal", max_workers=None):
    """Every cell of a run table with its value (row, col, id, w, run, value)."""
    sources = plan_sources(raster_path, band, backend=backend)
    plan = plan_cells(cells, sources)
    return execute(plan, Cells(), backend=backend, max_workers=max_workers)


def zonal_stats(raster_path, cells, band=1, backend="gdal", max_workers=None):
    """count, weight, sum, mean, min, max per id over a run table."""
    sources = plan_sources(raster_path, band, backend=backend)
    plan = plan_cells(cells, sources)
    return execute(plan, Stats(), backend=backend, max_workers=max_workers)
