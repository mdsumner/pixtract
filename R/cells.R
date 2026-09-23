## Cell sets as run tables: data.frame(row, col_start, col_end, id, w).
##
## In R: 1-based row/col, inclusive col_end, id = 1-based position. This is
## what cbr::cb_burn() and controlledburn::burn() return, so their tables
## pass straight through.
##
## On disk the convention is 0-based and half-open for every language
## (agreed); pix_write_cells() and pix_read_cells() convert.

## single-cell runs for points inside the grid (grid logic only, no burn);
## id is the point's position. Points on the grid's edge are inside.
pix_cells_points <- function(x, y, grid) {
  rc <- rowcol_from_xy(grid$gt, c(grid$ncol, grid$nrow), x, y)
  ok <- which(!is.na(rc$row))
  data.frame(row = rc$row[ok], col_start = rc$col[ok], col_end = rc$col[ok],
             id = ok, w = rep(1, length(ok)))
}

## run table from a cbr (cb_burn) or controlledburn (burn) result: interior
## runs with w = 1, plus boundary cells weighted by their coverage fraction.
## The burn must be on the raster's grid; see pix_burn_args().
pix_cells_burn <- function(b, edges = TRUE) {
  r <- b$runs
  out <- data.frame(row = r$row, col_start = r$col_start, col_end = r$col_end,
                    id = r$id, w = rep(1, nrow(r)))
  e <- b$edges
  if (edges && !is.null(e) && nrow(e) > 0L) {
    out <- rbind(out, data.frame(row = e$row, col_start = e$col, col_end = e$col,
                                 id = e$id, w = e$fraction))
  }
  out
}

## extent and dimension for cb_burn() / burn() so the burn is on `grid`
pix_burn_args <- function(grid) {
  list(extent = gt_dim_to_extent(grid$gt, c(grid$ncol, grid$nrow)), dimension = c(grid$ncol, grid$nrow))
}

## 0-based crossing (agreed): tables on disk are 0-based and half-open.
## A 1-based inclusive col_end is the same number as a 0-based exclusive one.
pix_write_cells <- function(cells, path) {
  d <- data.frame(row = cells$row - 1L, col_start = cells$col_start - 1L,
                  col_end = cells$col_end, id = cells$id - 1L,
                  w = if (is.null(cells$w)) 1 else cells$w)
  utils::write.csv(d, path, row.names = FALSE)
  invisible(path)
}

pix_read_cells <- function(path) {
  d <- utils::read.csv(path)
  data.frame(row = as.integer(d$row) + 1L, col_start = as.integer(d$col_start) + 1L,
             col_end = as.integer(d$col_end), id = as.integer(d$id) + 1L,
             w = as.numeric(d$w))
}
