## Plan tables: grid, sources, cell-to-block segments, cost.
##
## R conventions: row/col are 1-based with row 1 at the top, and a run
## covers col_start:col_end inclusive (the layout cbr and controlledburn
## return). Pixel values are flat vectors in raster order (row-major from the
## top-left), as gdalraster::read_ds() and GDALRaster$read() return them.
##
## The pipeline is
##
##   cells  ->  pix_plan(cells, sources)  ->  pix_execute(plan, reducer)
##
## where `cells` is a run table data.frame(row, col_start, col_end, id, w).
## A point is a run of length 1; a boundary cell with a coverage fraction is
## a run of length 1 with w = fraction.
##
## Places where R code crosses into GDAL's 0-based offsets are marked
## "0-based crossing" and are agreed case by case.

## -- Grid ---------------------------------------------------------------

pix_grid <- function(dsn) {
  ds <- new(gdalraster::GDALRaster, dsn, read_only = TRUE)
  on.exit(ds$close())
  d <- ds$dim()
  list(gt = ds$getGeoTransform(), ncol = d[1L], nrow = d[2L])
}

## map coordinates -> 1-based (row, col); NA outside the grid
pix_cell_of <- function(grid, x, y) {
  gt <- grid$gt
  det <- gt[2L] * gt[6L] - gt[3L] * gt[5L]
  px <- (gt[6L] * (x - gt[1L]) - gt[3L] * (y - gt[4L])) / det
  py <- (-gt[5L] * (x - gt[1L]) + gt[2L] * (y - gt[4L])) / det
  col <- floor(px) + 1
  row <- floor(py) + 1
  bad <- is.na(col) | is.na(row) | col < 1 | col > grid$ncol | row < 1 | row > grid$nrow
  col[bad] <- NA
  row[bad] <- NA
  data.frame(row = as.integer(row), col = as.integer(col))
}

## flat index into a raster-order vector of the whole grid
pix_cell_index <- function(row, col, ncol) (row - 1) * ncol + col

## c(xmin, xmax, ymin, ymax) of a north-up grid
pix_extent <- function(grid) {
  gt <- grid$gt
  if (gt[3L] != 0 || gt[5L] != 0) stop("pix_extent() needs a north-up geotransform")
  x <- gt[1L] + c(0, grid$ncol) * gt[2L]
  y <- gt[4L] + c(0, grid$nrow) * gt[6L]
  c(min(x), max(x), min(y), max(y))
}

## -- Sources ------------------------------------------------------------

## One row per source window. For a plain file that is the file itself; a
## VRT of 1:1 pixel copies is expanded into its sources, in VRT order (later
## sources are painted over earlier ones). Anything else is read through the
## VRT as one source, and attr(, "note") says why.
##
## col0, row0: first cell of the window in the queried grid (1-based).
## src_col0, src_row0: that cell's position in the source file (1-based).
## transparent: VRT <NODATA> value of the source, NA if none.
pix_sources <- function(dsn, band = 1L, expand_vrt = TRUE) {
  ds <- new(gdalraster::GDALRaster, dsn, read_only = TRUE)
  on.exit(ds$close())
  d <- ds$dim()
  grid <- list(gt = ds$getGeoTransform(), ncol = d[1L], nrow = d[2L])
  nodata <- ds$getNoDataValue(band)
  bs <- ds$getBlockSize(band)
  single <- function(note = "") {
    .sources(data.frame(path = dsn, band = band, col0 = 1, row0 = 1,
                        xsize = grid$ncol, ysize = grid$nrow,
                        src_col0 = 1, src_row0 = 1,
                        file_xsize = grid$ncol, file_ysize = grid$nrow,
                        block_x = bs[1L], block_y = bs[2L],
                        transparent = NA_real_),
             grid, nodata, dsn, "single", note)
  }
  if (!expand_vrt || ds$getDriverShortName() != "VRT") return(single())
  res <- tryCatch(.vrt_sources(ds, dsn, band, grid, nodata),
                  pix_unsupported = function(e) conditionMessage(e))
  if (is.character(res)) return(single(paste("VRT read as one source:", res)))
  if (is.null(res)) single() else res
}

.sources <- function(df, grid, nodata, dsn, kind, note = "") {
  int <- setdiff(names(df), c("path", "transparent"))
  df[int] <- lapply(df[int], as.integer)
  structure(df, grid = grid, nodata = nodata, dsn = dsn, kind = kind, note = note)
}

.unsupported <- function(msg) {
  stop(structure(class = c("pix_unsupported", "error", "condition"),
                 list(message = msg, call = NULL)))
}

.ok_children <- c("SourceFilename", "SourceBand", "SourceProperties", "SrcRect",
                  "DstRect", "NODATA", "OpenOptions")

.vrt_sources <- function(ds, dsn, band, grid, nodata) {
  xml <- ds$getMetadata(band = 0L, domain = "xml:VRT")
  if (length(xml) == 0L || !nzchar(xml[1L])) .unsupported("no xml:VRT metadata")
  doc <- xml2::read_xml(xml[1L])
  vb <- xml2::xml_find_first(doc, sprintf("//VRTRasterBand[@band='%d']", band))
  if (inherits(vb, "xml_missing")) .unsupported(sprintf("band %d not in VRT XML", band))
  if (!is.na(xml2::xml_attr(vb, "subClass"))) .unsupported("VRT band subClass")
  vrt_type <- ds$getDataTypeName(band)
  vrt_dir <- dirname(dsn)
  num <- function(node, a, what) {
    v <- as.numeric(xml2::xml_attr(node, a))
    if (is.na(v) || v != round(v)) .unsupported(paste("non-integer", what))
    v
  }
  rows <- list()
  for (el in xml2::xml_children(vb)) {
    tag <- xml2::xml_name(el)
    if (!tag %in% c("SimpleSource", "ComplexSource")) {
      if (grepl("Source$", tag)) .unsupported(paste("source type", tag))
      next
    }
    kids <- xml2::xml_name(xml2::xml_children(el))
    extra <- setdiff(kids, .ok_children)
    if (length(extra)) .unsupported(paste(tag, "with", paste(extra, collapse = ", ")))
    fn_el <- xml2::xml_find_first(el, "SourceFilename")
    fn <- xml2::xml_text(fn_el)
    if (identical(xml2::xml_attr(fn_el, "relativeToVRT"), "1")) fn <- file.path(vrt_dir, fn)
    sband <- xml2::xml_text(xml2::xml_find_first(el, "SourceBand"))
    if (is.na(sband)) sband <- "1"
    if (!grepl("^[0-9]+$", sband)) .unsupported(paste("SourceBand", sband))
    sr <- xml2::xml_find_first(el, "SrcRect")
    dr <- xml2::xml_find_first(el, "DstRect")
    if (inherits(sr, "xml_missing") || inherits(dr, "xml_missing"))
      .unsupported("source without SrcRect/DstRect")
    s <- vapply(c("xOff", "yOff", "xSize", "ySize"), function(a) num(sr, a, "SrcRect"), 0)
    d <- vapply(c("xOff", "yOff", "xSize", "ySize"), function(a) num(dr, a, "DstRect"), 0)
    if (any(s[3:4] != d[3:4])) .unsupported("resampled source (SrcRect size != DstRect size)")
    pr <- xml2::xml_find_first(el, "SourceProperties")
    if (!inherits(pr, "xml_missing") && !is.na(xml2::xml_attr(pr, "BlockXSize"))) {
      fxy <- as.integer(xml2::xml_attr(pr, c("RasterXSize", "RasterYSize")))
      bxy <- as.integer(xml2::xml_attr(pr, c("BlockXSize", "BlockYSize")))
      dtype <- xml2::xml_attr(pr, "DataType")
    } else {
      sds <- new(gdalraster::GDALRaster, fn, read_only = TRUE)
      fxy <- sds$dim()[1:2]
      bxy <- sds$getBlockSize(as.integer(sband))
      dtype <- sds$getDataTypeName(as.integer(sband))
      sds$close()
    }
    if (!identical(dtype, vrt_type)) .unsupported("source data type differs from the VRT band")
    tr <- xml2::xml_text(xml2::xml_find_first(el, "NODATA"))
    ## 0-based crossing (agreed): VRT DstRect/SrcRect offsets are 0-based;
    ## stored here as the 1-based first cell, clipped to the VRT grid.
    x0 <- max(d[1L], 0) + 1; y0 <- max(d[2L], 0) + 1
    x1 <- min(d[1L] + d[3L], grid$ncol); y1 <- min(d[2L] + d[4L], grid$nrow)
    if (x1 < x0 || y1 < y0) next
    rows[[length(rows) + 1L]] <- data.frame(
      path = fn, band = as.integer(sband), col0 = x0, row0 = y0,
      xsize = x1 - x0 + 1, ysize = y1 - y0 + 1,
      src_col0 = s[1L] + 1 + (x0 - 1 - d[1L]), src_row0 = s[2L] + 1 + (y0 - 1 - d[2L]),
      file_xsize = fxy[1L], file_ysize = fxy[2L],
      block_x = bxy[1L], block_y = bxy[2L],
      transparent = if (is.na(tr)) NA_real_ else as.numeric(tr))
  }
  if (!length(rows)) return(NULL)
  src <- .sources(do.call(rbind, rows), grid, nodata, dsn, "vrt")
  if (any(!is.na(src$transparent)) && .any_overlap(src))
    .unsupported("overlapping sources with per-source NODATA")
  src
}

.any_overlap <- function(s) {
  n <- nrow(s)
  if (n < 2L) return(FALSE)
  x1 <- s$col0 + s$xsize; y1 <- s$row0 + s$ysize
  for (i in seq_len(n - 1L)) {
    j <- (i + 1L):n
    if (any(s$col0[i] < x1[j] & s$col0[j] < x1[i] & s$row0[i] < y1[j] & s$row0[j] < y1[i]))
      return(TRUE)
  }
  FALSE
}

## -- Cells -> plan --------------------------------------------------------

## Intersect a run table with the sources and their block grids.
##
## Returns a data.frame with one row per segment (a contiguous piece of a run
## inside one block of one source), sorted by (src, block_row, block_col):
##   src, block_row, block_col   which block (1-based; src NA = uncovered)
##   row_in_block, col0, col1    the segment inside the block (1-based, inclusive)
##   row, col                    the segment's first cell in the queried grid
##   id, w, run                  from the input run table (run = its row number)
## Integer arithmetic only; cost is O(runs x pieces per run).
pix_plan <- function(cells, sources) {
  g <- attr(sources, "grid")
  row <- as.integer(cells$row)
  c0 <- as.integer(cells$col_start)
  e <- as.integer(cells$col_end) + 1L        # exclusive end, internal only
  id <- as.integer(cells$id)
  w <- if (is.null(cells$w)) rep(1, length(row)) else rep_len(as.numeric(cells$w), length(row))
  run <- seq_along(row)
  n_id <- if (length(id)) max(id) else 0L

  ## clip to the grid
  c0 <- pmax(c0, 1L); e <- pmin(e, g$ncol + 1L)
  keep <- row >= 1L & row <= g$nrow & e > c0
  row <- row[keep]; c0 <- c0[keep]; e <- e[keep]; id <- id[keep]; w <- w[keep]; run <- run[keep]

  ## 1. split runs at source column breaks; resolve which source wins
  xb <- sort(unique(c(sources$col0, sources$col0 + sources$xsize)))
  yb <- sort(unique(c(sources$row0, sources$row0 + sources$ysize)))
  claim <- matrix(NA_integer_, max(length(yb) - 1L, 0L), max(length(xb) - 1L, 0L))
  for (s in seq_len(nrow(sources))) {   # later sources overwrite earlier ones
    i <- match(sources$row0[s], yb):(match(sources$row0[s] + sources$ysize[s], yb) - 1L)
    j <- match(sources$col0[s], xb):(match(sources$col0[s] + sources$xsize[s], xb) - 1L)
    claim[i, j] <- s
  }
  brk <- c(-Inf, xb, Inf)
  k0 <- findInterval(c0, brk)
  k1 <- findInterval(e - 1L, brk)
  np <- k1 - k0 + 1L
  wh <- rep.int(seq_along(k0), np)
  k <- sequence(np, from = k0)
  p_lo <- as.integer(pmax(c0[wh], brk[k]))
  p_hi <- as.integer(pmin(e[wh], brk[k + 1L]))
  p_row <- row[wh]
  ci <- findInterval(p_row, yb)
  cj <- k - 1L
  ok <- ci >= 1L & ci < length(yb) & cj >= 1L & cj < length(xb)
  src <- rep(NA_integer_, length(wh))
  src[ok] <- claim[cbind(ci[ok], cj[ok])]

  ## 2. split each piece at its source's block column boundaries
  cov <- !is.na(src)
  s <- ifelse(cov, src, 1L)
  big <- .Machine$integer.max %/% 2L
  bx <- ifelse(cov, sources$block_x[s], big)
  by <- ifelse(cov, sources$block_y[s], big)
  dx <- ifelse(cov, sources$src_col0[s] - sources$col0[s], 0L)
  dy <- ifelse(cov, sources$src_row0[s] - sources$row0[s], 0L)
  l_lo <- p_lo + dx; l_hi <- p_hi + dx    # columns in the source file, hi exclusive
  l_row <- p_row + dy
  b0 <- (l_lo - 1L) %/% bx + 1L
  b1 <- (l_hi - 2L) %/% bx + 1L
  nb <- b1 - b0 + 1L
  w2 <- rep.int(seq_along(b0), nb)
  bcol <- sequence(nb, from = b0)
  first <- (bcol - 1L) * bx[w2]           # cells before this block in the file
  s_lo <- pmax(l_lo[w2], first + 1L)
  s_hi <- pmin(l_hi[w2], first + bx[w2] + 1L)
  brow <- (l_row[w2] - 1L) %/% by[w2] + 1L
  seg <- data.frame(
    src = src[w2], block_row = brow, block_col = bcol,
    row_in_block = l_row[w2] - (brow - 1L) * by[w2],
    col0 = s_lo - first, col1 = s_hi - 1L - first,
    row = p_row[w2], col = s_lo - dx[w2],
    id = id[wh][w2], w = w[wh][w2], run = run[wh][w2])
  unc <- is.na(seg$src)
  seg$block_row[unc] <- 1L; seg$block_col[unc] <- 1L; seg$row_in_block[unc] <- 1L
  seg <- seg[order(seg$src, seg$block_row, seg$block_col, na.last = TRUE), , drop = FALSE]
  rownames(seg) <- NULL
  structure(seg, sources = sources, n_id = n_id, n_run = nrow(cells))
}

## start/stop row of each (src, block_row, block_col) group in a plan
pix_block_groups <- function(plan) {
  n <- nrow(plan)
  if (n == 0L) return(list(start = integer(), stop = integer()))
  s <- plan$src; s[is.na(s)] <- 0L
  ch <- which(diff(s) != 0L | diff(plan$block_row) != 0L | diff(plan$block_col) != 0L)
  start <- c(1L, ch + 1L)
  list(start = start, stop = c(start[-1L] - 1L, n))
}

## what a plan will read, before any pixel I/O
pix_cost <- function(plan) {
  g <- pix_block_groups(plan)
  len <- plan$col1 - plan$col0 + 1L
  list(cells = sum(len), segments = nrow(plan),
       sources = length(unique(stats::na.omit(plan$src))),
       blocks = sum(!is.na(plan$src[g$start])),
       uncovered_cells = sum(len[is.na(plan$src)]))
}
