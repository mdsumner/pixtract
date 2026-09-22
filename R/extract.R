## Entry points built on the planner.
##
##   pix_extract_points(dsn, x, y)  raster values at (x, y)
##   pix_extract_cells(dsn, cells)  every cell of a run table with its value
##   pix_zonal_stats(dsn, cells)    grouped weighted statistics by id
##
## Load everything with source("R/pixtract.R").

pix_extract_points <- function(dsn, x, y, band = 1L) {
  sources <- pix_sources(dsn, band)
  cells <- pix_cells_points(x, y, attr(sources, "grid"))
  plan <- pix_plan(cells, sources)
  attr(plan, "n_id") <- length(x)
  pix_execute(plan, pix_place())
}

pix_extract_cells <- function(dsn, cells, band = 1L) {
  pix_execute(pix_plan(cells, pix_sources(dsn, band)), pix_cells())
}

pix_zonal_stats <- function(dsn, cells, band = 1L) {
  pix_execute(pix_plan(cells, pix_sources(dsn, band)), pix_stats())
}
