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
Planning (grid, blocks, VRT sources) uses `osgeo.gdal` whenever it is
installed; `backend="rasterio"` only switches the pixel reads. Without
`osgeo.gdal` everything falls back to rasterio, which plans a VRT as a single
source.
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

## Read planning: sources, memory and read windows

Block-by-block reads are exact but make one call per block; one big read of a
whole file makes one call but decodes everything. Which is better depends on
where the query falls, so the planner looks at the source first and then at
the query:

```python
info = pixtract.inspect_source(dsn)   # grid, dtype, blocks, overviews, one row per VRT source
plan = pixtract.plan_cells(cells, info["sources"])
pixtract.source_usage(plan)           # per source: cells, runs, blocks touched, occupancy
plan = pixtract.plan_reads(plan, mem=256 * 2**20, max_workers=8)
plan.reads                            # one row per read window: src, xoff, yoff, xsize, ysize, ...
plan.extra["meta"]                    # per source: meta-tile shape chosen, its cost, block-read cost
pixtract.execute(plan, pixtract.Stats(), max_workers=8)
```

```r
info <- pix_inspect(dsn)
plan <- pix_plan(cells, info$sources)
pix_usage(plan)
plan <- pix_plan_reads(plan, mem = 256 * 2^20)
attr(plan, "reads"); attr(plan, "meta")
pix_execute(plan, pix_stats())
```

- **Inspect** reads metadata only: the dataset's grid, data type, block size and
  overviews, and for a VRT mosaic one row per source with its place in the
  mosaic, its map extent and its own block layout.
- **Usage** says which sources the query implicates and how densely it hits each:
  blocks touched, the span of their bounding box, and occupancy (touched / span,
  near 1 for a clustered query, near 0 for a scattered one).
- **Read windows** ("meta-tiles"): each source's block grid is covered by a coarser
  grid of kx x ky blocks. In each meta-tile the touched blocks are read as one
  window (their bounding box) or block by block, whichever costs less under
  `cost(read) = request_bytes + decoded bytes`. `request_bytes` is the fixed cost
  of one read call, in bytes: 64 KiB for local files, 2 MiB for remote ones
  (`/vsicurl`, `/vsis3`, `http`, ...), or pass your own. (kx, ky) is chosen per
  source from powers of two up to the whole file, as the cheapest shape whose
  largest window fits the memory budget for one read: `mem` divided by the reads
  `execute()` holds at once (2 x `max_workers` with threads; the R executor is
  serial, so all of it).
- Every reducer works unchanged, and `extract_points()`, `extract_cells()` and
  `zonal_stats()` (and the `pix_` versions in R) take `mem=` to plan windows;
  without it they read block by block as before.
- Overviews are reported but not used: extraction reads native-resolution cells.

## Conventions

- **Index base**: 1-based in R (row 1 at the top, inclusive `col_end`, as cbr and
  controlledburn return), 0-based in Python and native code (half-open `col_end`).
  Cell tables written to disk are always 0-based and half-open;
  `pix_write_cells()`/`pix_read_cells()` convert in R. The places where R code
  crosses into 0-based offsets for GDAL are marked "0-based crossing" in the source.
- **Cell order**: pixel values are flat vectors in raster order (row-major from the
  top-left), as `gdalraster::read_ds()` returns them. R flips to a matrix only when
  something needs one, with `matrix(v, ncol = ncol, byrow = TRUE)`.
- **Grid logic**: `R/grid.R` and `python/pixtract/grid.py` use the names and
  conventions of [vaster](https://github.com/hypertidy/vaster) (`rowcol_from_xy`,
  `cell_from_row_col`, `gt_dim_to_extent`, `extent_dim_to_gt`; dimension is
  `(ncol, nrow)`, extent is `(xmin, xmax, ymin, ymax)`). Points are mapped to cells
  with this grid logic, not by burning. A point on any edge of the grid is inside,
  so one exactly on the right or bottom edge gets the last column or row; points
  outside give NA/NaN. `tests/grid_cases.csv` holds the edge cases both languages
  are tested against.

## Examples

Run from the repo root.

| | Python | R |
|---|---|---|
| points on a synthetic mosaic (offline) | `examples/points_synthetic.py` | `examples/points_synthetic.R` |
| zonal stats on a synthetic mosaic (offline) | `examples/zonal_synthetic.py` | `examples/zonal_synthetic.R` |
| Swiss cantons on Copernicus GLO-90 (S3) | `examples/cantons_glo90.py` | `examples/cantons_glo90.R` |
| read planning on a synthetic mosaic (offline) | `examples/read_plan_synthetic.py` | `examples/read_plan_synthetic.R` |
| read planning on Copernicus GLO-30 (S3) | `examples/read_plan_glo30.py` | `examples/read_plan_glo30.R` |
| Swiss cantons on the global COP30 VRT (26,475 tiles) | `examples/cantons_cop30_global.py` | |
| Wherobots WorldClim countries, night-light counties, points (S3) | `examples/wherobots_worldclim.py` | |
| human footprint (HFP-100) mean per PAD-US protected area (S3) | `examples/hfp_padus.py` | |

The cantons example reproduces the Apache Sedona "group by, but for pixels"
benchmark: 26 cantons over 18 GLO-90 tiles (a 7200 x 3600 mosaic) give
7,006,027 cells under the cell-centre rule, the same count Sedona reports, and
canton means that match its table (Valais 2138 m, Graubuenden 2023 m, Uri 1901 m,
Glarus 1584 m, Ticino 1400 m). The plan touches 14 blocks in 14 files, 53.5 MB
compressed.

The global version plans against OpenTopography's COP30_hh.vrt (26,475 sources)
without opening it or any tile: `scan_vrt()` reads the VRT's XML text (0.5 to
1 s), the cantons are burned on the 1296001 x 626401 global grid (0.3 s, the
burn is sparse), and `plan_sources(dsn, window=cells_window(cells))` keeps the 18
sources the burn's window touches. Only those have to be 1:1 copies; the 14,589
resampled high-latitude sources elsewhere no longer send the whole VRT to one
source. 63,054,224 cells, 36 reads, 357 MiB decoded.

The WorldClim example reruns the Wherobots "Raster Data Analysis With Spatial
SQL" post on the same public files. The point values match exactly (12 months
of precipitation and the night-light value at Missoula). Yearly precipitation
per country is the sum of 12 monthly country means; for large countries the
cell-centre rule lands within about 0.1 to 1 percent of the published values
(Colombia 2633.9 vs 2632.0, Malaysia 2879.5 vs 2881.2). Six of the 20
published values, all small islands, match exactly only when a cell counts
when its lower-right corner is inside, a half-cell offset in Sedona's
rasterization at the time (the post used allTouched = true). The published
county night-light means do not correspond to a mean over the county on this
raster under any rule tried (centre, corner, all touched, whole county or the
post's per-256-tile sum), so they are printed for reference only.

The HFP example is the workload from the Pangeo thread "Advice for scalable
raster-vector extraction": the coverage-weighted mean of a 14 GB global 100 m
COG under each of PAD-US 3.0's 438,113 polygons, where exactextract took about
10 minutes on one core with local files and xarray-based approaches ran out
of memory. Reading the COG remotely on 4 cores it takes about 10 minutes (half
of it for a dozen polygons that wrap around the globe in Mollweide), and 99.95%
of the means for polygons without holes equal the published
`pad_raster_means.csv` to float32 precision. Polygons with holes are off
because of a controlledburn bug, `exploration/2026-09-23_burn_hole_cells.py`.

## Tests

```
python -m pytest tests      # regression against the original extract_points, and dense reads
Rscript tests/test_r.R      # the same checks for the R side
```

Both run on GitHub Actions for every push and pull request
(`.github/workflows/tests.yml`, Ubuntu GDAL for Python, gdalraster for R).

`tests/legacy_extract.py` is the original implementation, kept as the oracle.

## Limits

- VRTs are expanded into sources only when every source is a 1:1 pixel copy of
  the band's data type (no resampling, scaling, LUT or mask). Otherwise, and for
  GTI, the dataset is read through GDAL as one source, which is still
  block-planned, just on the dataset's own blocks.
- Overlapping VRT sources: the last one wins, as in the VRT. Overlap combined with
  per-source NODATA falls back to reading through the VRT.
- The read-window cost model is a heuristic: decoded bytes stand in for bytes
  fetched and decompressed, and a window read counts as one request (GDAL merges
  the byte ranges of a COG window, but not always into one). Tune it with
  `request_bytes`.
- In R, gdalraster returns a source file's own nodata as NA; inside a VRT that is
  treated as a transparent pixel (the VRT's fill value), which matches VRTs built
  by `gdalbuildvrt` with source nodata.

## Project structure

```
python/pixtract/   grid.py, plan.py, cells.py, reads.py, execute.py, reduce.py, extract.py
R/                 the same, as plain scripts; source("R/pixtract.R")
examples/          runnable Python and R examples
tests/             pytest suite, tests/test_r.R, synthetic fixtures
exploration/       dated lab notebook scripts from development
```

## Code of Conduct

Please note that the pixtract project is released with a [Contributor Code of Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html). By contributing to this project, you agree to abide by its terms.
