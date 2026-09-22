## Grid logic, with the names and conventions of the vaster package
## (hypertidy/vaster): dimension is c(ncol, nrow), extent is
## c(xmin, xmax, ymin, ymax), row/col/cell are 1-based in raster order
## (row-major from the top-left). python/pixtract/grid.py has the same
## functions 0-based. When a Python vaster exists these can be replaced by
## vaster:: in R and the package in Python.
##
## Edge rule (as vaster and terra): a point on any edge of the grid is
## inside, so a point exactly on the right or bottom edge falls in the last
## column or row.

## 1-based (row, col) of the cell containing each (x, y); NA outside.
## Uses the full geotransform, so rotated grids work. A north-up grid is
## bounds-checked against gt_dim_to_extent() in coordinates, so a point
## taken from that extent is inside exactly.
rowcol_from_xy <- function(gt, dimension, x, y) {
  nc <- dimension[1L]; nr <- dimension[2L]
  if (gt[3L] == 0 && gt[5L] == 0) {
    px <- (x - gt[1L]) / gt[2L]
    py <- (y - gt[4L]) / gt[6L]
    e <- gt_dim_to_extent(gt, dimension)
    ok <- x >= e[1L] & x <= e[2L] & y >= e[3L] & y <= e[4L]
  } else {
    det <- gt[2L] * gt[6L] - gt[3L] * gt[5L]
    px <- (gt[6L] * (x - gt[1L]) - gt[3L] * (y - gt[4L])) / det
    py <- (-gt[5L] * (x - gt[1L]) + gt[2L] * (y - gt[4L])) / det
    ok <- px >= 0 & px <= nc & py >= 0 & py <= nr
  }
  ok <- !is.na(ok) & ok
  col <- pmin(pmax(floor(px), 0), nc - 1) + 1
  row <- pmin(pmax(floor(py), 0), nr - 1) + 1
  col[!ok] <- NA
  row[!ok] <- NA
  data.frame(row = as.integer(row), col = as.integer(col))
}

## 1-based cell (flat raster-order index) of (row, col); NA outside
cell_from_row_col <- function(dimension, row, col) {
  nc <- dimension[1L]; nr <- dimension[2L]
  ifelse(row < 1 | row > nr | col < 1 | col > nc, NA, (row - 1) * nc + col)
}

## c(xmin, xmax, ymin, ymax) of a north-up grid
gt_dim_to_extent <- function(gt, dimension) {
  if (gt[3L] != 0 || gt[5L] != 0) stop("gt_dim_to_extent() needs a north-up geotransform")
  x <- gt[1L] + c(0, dimension[1L]) * gt[2L]
  y <- gt[4L] + c(0, dimension[2L]) * gt[6L]
  c(min(x), max(x), min(y), max(y))
}

## north-up geotransform from extent and dimension
extent_dim_to_gt <- function(extent, dimension) {
  c(extent[1L], (extent[2L] - extent[1L]) / dimension[1L], 0,
    extent[4L], 0, -(extent[4L] - extent[3L]) / dimension[2L])
}
