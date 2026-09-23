## R tests: planned reads must equal a dense read of the same dataset.
## Run from the repo root:  Rscript tests/test_r.R

suppressPackageStartupMessages(library(gdalraster))
source("R/pixtract.R")

source("tests/synthetic.R")

fixtures <- list(
  single = make_tile(file.path(td, "single.tif"), 1000, 700, c(128, 64), c(100, -30), 0.01, 1, -9999),
  mosaic = make_mosaic("mosaic", blocks = list(c(64, 64), c(128, 32), c(256, 256))),
  overlap = make_mosaic("overlap", blocks = list(c(64, 64), c(32, 128)), overlap = c(17, 9)),
  gap = make_mosaic("gap", drop = 4),
  srcnodata = make_mosaic("srcnd", blocks = list(c(64, 64), c(128, 32)), drop = 1,
                          nodata = -1, vrt_nodata = -9999),
  srcnodata_novrt = make_mosaic("srcnd_novrt", drop = 1, nodata = -1, vrt_nodata = NA),
  overlap_nodata = make_mosaic("overlap_nd", layout = c(2, 2), overlap = c(17, 9), nodata = -1)
)

## dense reference: whole dataset as a flat raster-order vector
dense <- function(path) {
  ds <- new(GDALRaster, path, read_only = TRUE); on.exit(ds$close())
  d <- ds$dim()
  v <- ds$read(1L, 0L, 0L, d[1L], d[2L], d[1L], d[2L])
  nd <- ds$getNoDataValue(1L)
  if (!is.na(nd)) v[!is.na(v) & v == nd] <- NA
  structure(as.numeric(v), dim = d[1:2])
}

random_runs <- function(nrow, ncol, n, n_id, seed) {
  set.seed(seed)
  row <- sample(-2:(nrow + 3), n, replace = TRUE)
  c0 <- sample(-49:ncol, n, replace = TRUE)
  data.frame(row = row, col_start = c0, col_end = c0 + sample(0:398, n, replace = TRUE),
             id = sample(n_id, n, replace = TRUE), w = runif(n, 0.1, 1))
}

expand_ref <- function(cells, v) {
  ncol <- attr(v, "dim")[1L]; nrow <- attr(v, "dim")[2L]
  lo <- pmax(cells$col_start, 1L); hi <- pmin(cells$col_end, ncol)
  n <- ifelse(cells$row >= 1 & cells$row <= nrow & hi >= lo, hi - lo + 1L, 0L)
  run <- rep(seq_len(nrow(cells)), n)
  col <- sequence(n, from = lo)
  row <- cells$row[run]
  data.frame(run = run, row = row, col = col, id = cells$id[run], w = cells$w[run],
             value = v[cell_from_row_col(c(ncol, nrow), row, col)])
}

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

for (nm in names(fixtures)) {
  f <- fixtures[[nm]]
  v <- dense(f)
  d <- attr(v, "dim")
  cells <- random_runs(d[2L], d[1L], 300, 7, 11)
  got <- pix_extract_cells(f, cells)
  ref <- expand_ref(cells, v)
  got <- got[order(got$run, got$col), ]
  ok(nrow(got) == nrow(ref), paste(nm, "cell count"))
  ok(identical(got$col, ref$col) && identical(got$row, ref$row), paste(nm, "cells"))
  ok(identical(is.na(got$value), is.na(ref$value)) &&
       all(got$value == ref$value, na.rm = TRUE), paste(nm, "values"))

  st <- pix_zonal_stats(f, cells)
  r <- ref[!is.na(ref$value), ]
  for (k in 1:7) {
    m <- r$id == k
    ok(st$count[k] == sum(m), paste(nm, "count", k))
    if (any(m)) {
      ok(isTRUE(all.equal(st$mean[k], sum(r$w[m] * r$value[m]) / sum(r$w[m]))), paste(nm, "mean", k))
      ok(st$min[k] == min(r$value[m]) && st$max[k] == max(r$value[m]), paste(nm, "min/max", k))
    }
  }

  ## points, including some outside the extent
  g <- pix_grid(f)
  e <- gt_dim_to_extent(g$gt, c(g$ncol, g$nrow))
  set.seed(5)
  x <- runif(3000, e[1] - 0.2, e[2] + 0.2); y <- runif(3000, e[3] - 0.2, e[4] + 0.2)
  ## points on every edge are inside (along-edge coordinate at a cell centre)
  xm <- e[1] + (g$ncol %/% 3 + 0.5) * g$gt[2]; ym <- e[4] + (g$nrow %/% 3 + 0.5) * g$gt[6]
  x <- c(x, e[1], e[2], xm, xm, e[1], e[2], e[1], e[2])
  y <- c(y, ym, ym, e[3], e[4], e[3], e[3], e[4], e[4])
  pv <- pix_extract_points(f, x, y)
  rc <- rowcol_from_xy(g$gt, c(g$ncol, g$nrow), x, y)
  inside <- !is.na(rc$row)
  ok(all(inside[3001:3008]), paste(nm, "edge points are inside"))
  ref_p <- v[cell_from_row_col(c(g$ncol, g$nrow), rc$row[inside], rc$col[inside])]
  ok(all(is.na(pv[!inside])), paste(nm, "outside points are NA"))
  ok(identical(is.na(pv[inside]), is.na(ref_p)) && all(pv[inside] == ref_p, na.rm = TRUE),
     paste(nm, "points"))
}

## grid logic: the edge cases shared with tests/test_grid.py (0-based on disk)
gc <- utils::read.csv("tests/grid_cases.csv")
for (nm in unique(gc$grid)) {
  s <- gc[gc$grid == nm, ]
  gt <- unlist(s[1L, paste0("gt", 0:5)], use.names = FALSE)
  rc <- rowcol_from_xy(gt, c(s$ncol[1L], s$nrow[1L]), s$x, s$y)
  want_row <- ifelse(s$row < 0, NA_integer_, s$row + 1L)
  want_col <- ifelse(s$col < 0, NA_integer_, s$col + 1L)
  ok(identical(rc$row, as.integer(want_row)) && identical(rc$col, as.integer(want_col)),
     paste("grid cases", nm))
}
ok(identical(cell_from_row_col(c(10, 5), c(1, 1, 5, 3, 6, 1), c(1, 10, 10, 4, 1, 0)),
             c(1, 10, 50, 24, NA, NA)), "cell_from_row_col")
ok(isTRUE(all.equal(extent_dim_to_gt(c(100, 110, -37, -30), c(1000, 700)),
                    c(100, 0.01, 0, -30, 0, -0.01))), "extent_dim_to_gt")

src <- pix_sources(fixtures$mosaic)
ok(attr(src, "kind") == "vrt" && nrow(src) == 6L, "mosaic expands to 6 sources")
ok(attr(pix_sources(fixtures$overlap_nodata), "kind") == "single", "overlap + NODATA falls back")

## on-disk cells are 0-based half-open; R round-trips to 1-based inclusive
cells <- random_runs(50, 50, 20, 3, 13)
p <- file.path(td, "cells.csv")
pix_write_cells(cells, p)
raw <- utils::read.csv(p)
ok(all(raw$row == cells$row - 1L) && all(raw$col_end == cells$col_end), "disk is 0-based half-open")
back <- pix_read_cells(p)
ok(isTRUE(all.equal(back, cells, check.attributes = FALSE)), "cells round-trip")

cat(sprintf("R tests passed (%d fixtures x 3 checks, plus sources and cells I/O)\n", length(fixtures)))
