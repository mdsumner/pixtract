## Entry points built on the planner.
##
##   pix_extract_points(dsn, x, y)  raster values at (x, y)
##   pix_extract_cells(dsn, cells)  every cell of a run table with its value
##   pix_zonal_stats(dsn, cells)    grouped weighted statistics by id
##
## With `mem` (bytes), blocks are grouped into read windows that fit it
## (pix_plan_reads()); NULL reads block by block.
##
## Load everything with source("R/pixtract.R").

pix_extract_points <- function(dsn, x, y, band = 1L, mem = NULL) {
  sources <- pix_sources(dsn, band)
  cells <- pix_cells_points(x, y, attr(sources, "grid"))
  plan <- .windows(pix_plan(cells, sources), mem)
  attr(plan, "n_id") <- length(x)
  pix_execute(plan, pix_place())
}

pix_extract_cells <- function(dsn, cells, band = 1L, mem = NULL) {
  pix_execute(.windows(pix_plan(cells, pix_sources(dsn, band)), mem), pix_cells())
}

pix_zonal_stats <- function(dsn, cells, band = 1L, mem = NULL) {
  pix_execute(.windows(pix_plan(cells, pix_sources(dsn, band)), mem), pix_stats())
}

.windows <- function(plan, mem) if (is.null(mem)) plan else pix_plan_reads(plan, mem = mem)
