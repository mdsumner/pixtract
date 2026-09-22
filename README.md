# pixtract

<!-- badges: start -->
<!-- badges: end -->

Planned raster value extraction, in plain Python and R.

Point extraction and zonal statistics both come down to reading the values of a
known set of cells from a raster that is a mosaic of files, each tiled into
blocks. pixtract plans that read before doing it: it works out which file and
which block every cell lives in, then reads each touched block exactly once,
in order, and hands the values to a reducer. The plan says what a query costs
(files, blocks, bytes) before any pixel is read.

No spatial predicates, no GEOS: everything is integer arithmetic on the grid.

```
cells  ->  plan (source, block, segment)  ->  block-ordered reads  ->  reduce
```

- **cells** is a run table: `row, col_start, col_end, id, w`. A point is a run of
  length 1. Polygon interiors from controlledburn/cbr are runs; their boundary
  cells are length-1 runs weighted by coverage fraction.
- **plan** clips runs at source-file and block boundaries. A VRT whose sources are
  1:1 pixel copies is planned against the source files' own blocks, not the VRT's.
- **reduce** is `Place` (one value per point), `Cells` (every cell with its value)
  or `Stats` (count, weight, sum, mean, min, max per id).

## Python

Needs numpy and GDAL's Python bindings (`osgeo.gdal`); rasterio optional.
No install needed: put `python/` on the path, or `pip install -e .`.

```python
import sys; sys.path.insert(0, "python")
import pixtract

# points: the original API, unchanged
vals = pixtract.extract_points("/vsicurl/https://example.com/mosaic.vrt", xs, ys, max_workers=8)

# zonal stats from controlledburn runs on the raster's grid
import controlledburn as cb
src = pixtract.plan_sources(dsn)
cells = pixtract.cells_from_burn(cb.burn(polygons, **pixtract.burn_args(src.grid)))
plan = pixtract.plan_cells(cells, src)
pixtract.cost(plan, bytes=True)      # before any pixel I/O
stats = pixtract.execute(plan, pixtract.Stats(), max_workers=8)
```

## R

Needs gdalraster (and xml2, which gdalraster already uses). No package: source it.

```r
source("R/pixtract.R")

vals <- pix_extract_points(dsn, x, y)

library(cbr)
src <- pix_sources(dsn)
a <- pix_burn_args(attr(src, "grid"))
b <- cb_burn(polygons, extent = a$extent, dimension = a$dimension)
plan <- pix_plan(pix_cells_burn(b), src)
pix_cost(plan)
stats <- pix_execute(plan, pix_stats())
```

## Conventions

- **Index base**: 1-based in R (row 1 at the top, inclusive `col_end`, as cbr and
  controlledburn return), 0-based in Python and native code (half-open `col_end`).
  Cell tables written to disk are always 0-based and half-open;
  `pix_write_cells()`/`pix_read_cells()` convert in R. The places where R code
  crosses into 0-based offsets for GDAL are marked "0-based crossing" in the source.
- **Cell order**: pixel values are flat vectors in raster order (row-major from the
  top-left), as `gdalraster::read_ds()` returns them. R flips to a matrix only when
  something needs one, with `matrix(v, ncol = ncol, byrow = TRUE)`.

## Examples

Run from the repo root.

| | Python | R |
|---|---|---|
| points on a synthetic mosaic (offline) | `examples/points_synthetic.py` | `examples/points_synthetic.R` |
| zonal stats on a synthetic mosaic (offline) | `examples/zonal_synthetic.py` | `examples/zonal_synthetic.R` |
| Swiss cantons on Copernicus GLO-90 (S3) | `examples/cantons_glo90.py` | `examples/cantons_glo90.R` |

The cantons example reproduces the Apache Sedona "group by, but for pixels"
benchmark: 26 cantons over 18 GLO-90 tiles (a 7200 x 3600 mosaic) give
7,006,027 cells under the cell-centre rule, the same count Sedona reports, and
canton means that match its table (Valais 2138 m, Graubuenden 2023 m, Uri 1901 m,
Glarus 1584 m, Ticino 1400 m). The plan touches 14 blocks in 14 files, 53.5 MB
compressed.

## Tests

```
python -m pytest tests      # regression against the original extract_points, and dense reads
Rscript tests/test_r.R      # the same checks for the R side
```

`tests/legacy_extract.py` is the original implementation, kept as the oracle.

## Limits

- VRTs are expanded into sources only when every source is a 1:1 pixel copy of
  the band's data type (no resampling, scaling, LUT or mask). Otherwise, and for
  GTI, the dataset is read through GDAL as one source, which is still
  block-planned, just on the dataset's own blocks.
- Overlapping VRT sources: the last one wins, as in the VRT. Overlap combined with
  per-source NODATA falls back to reading through the VRT.
- In R, gdalraster returns a source file's own nodata as NA; inside a VRT that is
  treated as a transparent pixel (the VRT's fill value), which matches VRTs built
  by `gdalbuildvrt` with source nodata.

## Project structure

```
python/pixtract/   plan.py, cells.py, execute.py, reduce.py, extract.py
R/                 the same, as plain scripts; source("R/pixtract.R")
examples/          runnable Python and R examples
tests/             pytest suite, tests/test_r.R, synthetic fixtures
exploration/       dated lab notebook scripts from development
```

## Code of Conduct

Please note that the pixtract project is released with a [Contributor Code of Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html). By contributing to this project, you agree to abide by its terms.
