## Execute a plan: fetch each read window once, in plan order, and hand the
## values of its cells to a reducer.
##
## A plan from pix_plan() is read block by block; pix_plan_reads() groups
## blocks into larger windows. GDALRaster$read() returns a window as a flat
## vector in raster order, so a segment on window row lr starting at window
## column lc (both 1-based) is v[(lr - 1) * xsize + lc + 0:(n - 1)]. No matrix
## is formed.

## 0-based crossing (agreed): GDAL takes 0-based pixel offsets. Windows stay
## 1-based in R (src_col0, src_row0) and are converted here, and only here.
.gdal_window <- function(src_col0, src_row0, xsize, ysize) {
  c(xoff = src_col0 - 1L, yoff = src_row0 - 1L, xsize = xsize, ysize = ysize)
}

.fetch <- function(plan, a, b, open) {
  src <- attr(plan, "sources")
  nodata <- attr(src, "nodata")
  fill <- if (is.na(nodata)) 0 else NA_real_
  len <- plan$col1[a:b] - plan$col0[a:b] + 1L
  k <- plan$src[a]
  if (is.na(k)) return(rep(fill, sum(len)))
  rd <- attr(plan, "reads")[plan$read[a], ]
  win <- .gdal_window(rd$src_col0, rd$src_row0, rd$xsize, rd$ysize)
  ds <- open(src$path[k])
  v <- ds$read(src$band[k], win[["xoff"]], win[["yoff"]], win[["xsize"]], win[["ysize"]],
               win[["xsize"]], win[["ysize"]])
  ## segment position inside the window: block origin + offset in block
  lr <- (plan$block_row[a:b] - 1L) * src$block_y[k] + plan$row_in_block[a:b] - (rd$src_row0 - 1L)
  lc <- (plan$block_col[a:b] - 1L) * src$block_x[k] + plan$col0[a:b] - (rd$src_col0 - 1L)
  v <- as.numeric(v)[sequence(len, from = (lr - 1L) * rd$xsize + lc)]
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

## start/stop row of each read in a plan with read windows
pix_read_groups <- function(plan) {
  n <- nrow(plan)
  if (n == 0L) return(list(start = integer(), stop = integer()))
  start <- c(1L, which(diff(plan$read) != 0L) + 1L)
  list(start = start, stop = c(start[-1L] - 1L, n))
}

## Run a plan and return reducer$result(). Reads happen in plan order, each
## read window exactly once: one per touched block for a plan from
## pix_plan(), or the windows chosen by pix_plan_reads().
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
  if (is.null(attr(plan, "reads"))) plan <- pix_block_reads(plan)
  g <- pix_read_groups(plan)
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
