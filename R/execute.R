## Execute a plan: read each touched block once, in plan order, and hand the
## values of its cells to a reducer.
##
## GDALRaster$read() returns a block as a flat vector in raster order, so a
## segment on row_in_block r covering col0:col1 of a block of width wx is
## v[(r - 1) * wx + (col0:col1)]. No matrix is formed.

## 0-based crossing (agreed): GDAL takes 0-based pixel offsets. Block ids
## stay 1-based in R and are converted here, and only here.
.gdal_block_window <- function(block_col, block_row, bx, by, file_xsize, file_ysize) {
  xoff <- (block_col - 1L) * bx
  yoff <- (block_row - 1L) * by
  c(xoff = xoff, yoff = yoff,
    xsize = min(bx, file_xsize - xoff), ysize = min(by, file_ysize - yoff))
}

.fetch <- function(plan, a, b, open) {
  src <- attr(plan, "sources")
  nodata <- attr(src, "nodata")
  fill <- if (is.na(nodata)) 0 else NA_real_
  len <- plan$col1[a:b] - plan$col0[a:b] + 1L
  k <- plan$src[a]
  if (is.na(k)) return(rep(fill, sum(len)))
  win <- .gdal_block_window(plan$block_col[a], plan$block_row[a],
                            src$block_x[k], src$block_y[k],
                            src$file_xsize[k], src$file_ysize[k])
  ds <- open(src$path[k])
  blk <- ds$read(src$band[k], win[["xoff"]], win[["yoff"]], win[["xsize"]], win[["ysize"]],
                 win[["xsize"]], win[["ysize"]])
  v <- as.numeric(blk)[sequence(len, from = (plan$row_in_block[a:b] - 1L) * win[["xsize"]] +
                                  plan$col0[a:b])]
  ## gdalraster returns the source file's own nodata as NA; inside a VRT that
  ## is a transparent source pixel, which the VRT shows as its fill value
  if (attr(src, "kind") == "vrt") {
    tr <- src$transparent[k]
    v[is.na(v)] <- fill
    if (!is.na(tr)) v[!is.na(v) & v == tr] <- fill
  }
  if (!is.na(nodata)) v[!is.na(v) & v == nodata] <- NA_real_
  v
}

## Run a plan and return reducer$result(). Reads happen in plan order
## (source, block row, block col), each touched block exactly once.
pix_execute <- function(plan, reducer) {
  handles <- new.env()
  open <- function(path) {
    ds <- handles[[path]]
    if (is.null(ds)) {
      ds <- new(gdalraster::GDALRaster, path, read_only = TRUE)
      assign(path, ds, envir = handles)
    }
    ds
  }
  on.exit(for (p in ls(handles)) handles[[p]]$close())
  g <- pix_block_groups(plan)
  reducer$start(plan)
  for (i in seq_along(g$start)) {
    a <- g$start[i]; b <- g$stop[i]
    len <- plan$col1[a:b] - plan$col0[a:b] + 1L
    which <- rep.int(seq_len(b - a + 1L), len) + a - 1L
    reducer$add(list(id = plan$id[which], w = plan$w[which], run = plan$run[which],
                     row = plan$row[which],
                     col = sequence(len, from = plan$col[a:b]),
                     value = .fetch(plan, a, b, open)))
  }
  reducer$result()
}
