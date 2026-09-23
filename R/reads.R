## Source inspection and read planning: which sources a query touches, and
## which read windows to fetch them with. Same design as
## python/pixtract/reads.py; see there for the cost model.
##
##   pix_inspect(dsn)             grid, data type, blocks, overviews and (for a
##                                VRT mosaic) one row per source with its extent
##                                and block layout
##   pix_usage(plan)              per source: cells, runs, blocks the query
##                                touches, and how densely
##   pix_plan_reads(plan, mem)    group touched blocks into read windows that
##                                fit a memory budget
##   pix_plan_extraction(dsn, cells, mem)
##                                all of the above; the result goes straight
##                                to pix_execute() with any reducer
##
## Read windows ("meta-tiles"): each source's block grid is covered by a
## coarser grid of kx by ky blocks, aligned to the first block. In each
## meta-tile the touched blocks are read as one window (their bounding box)
## or block by block, whichever costs less, with
##
##   cost(read) = request_bytes + decoded bytes of the read
##
## (kx, ky) is chosen per source, from powers of two up to the whole file, as
## the cheapest shape whose largest window fits `mem`. pix_execute() is
## serial, so all of `mem` goes to one read.
##
## R conventions: 1-based, inclusive. A read window is src_col0, src_row0
## (first cell of the window in the source file), xsize, ysize.

pix_default_mem <- 256 * 2^20           # bytes of decoded pixels per read
pix_local_request_bytes <- 64 * 2^10    # cost of one read on a local file
pix_remote_request_bytes <- 2 * 2^20    # cost of one read on a remote source

.remote_tokens <- c("/vsicurl", "/vsis3", "/vsigs", "/vsiaz", "/vsiadls", "/vsioss",
                    "/vsiswift", "/vsiwebhdfs", "/vsihdfs", "http://", "https://",
                    "ftp://")

## default per-read cost for each path: remote (/vsicurl, /vsis3, http...) or local
pix_request_bytes <- function(path) {
  remote <- vapply(path, function(p) any(vapply(.remote_tokens, grepl, NA, x = p,
                                                fixed = TRUE)), NA, USE.NAMES = FALSE)
  ifelse(remote, pix_remote_request_bytes, pix_local_request_bytes)
}

## -- Inspect ----------------------------------------------------------------

## blocks of size b touched by cells lo:(lo + n - 1), lo 1-based
.nblocks <- function(lo, n, b) (lo + n - 2L) %/% b - (lo - 1L) %/% b + 1L

## One row per source: where it sits in the mosaic, its map extent, and the
## block layout of the part of the file the mosaic uses.
pix_source_table <- function(sources) {
  s <- sources
  gt <- attr(s, "grid")$gt
  n <- nrow(s)
  ext <- matrix(NA_real_, n, 4L)
  if (gt[3L] == 0 && gt[5L] == 0) {
    for (i in seq_len(n)) {
      sub <- c(gt[1L] + (s$col0[i] - 1) * gt[2L], gt[2L], 0,
               gt[4L] + (s$row0[i] - 1) * gt[6L], 0, gt[6L])
      ext[i, ] <- gt_dim_to_extent(sub, c(s$xsize[i], s$ysize[i]))
    }
  }
  nbx <- .nblocks(s$src_col0, s$xsize, s$block_x)
  nby <- .nblocks(s$src_row0, s$ysize, s$block_y)
  data.frame(src = seq_len(n), path = s$path, band = s$band,
             col0 = s$col0, row0 = s$row0, xsize = s$xsize, ysize = s$ysize,
             xmin = ext[, 1L], xmax = ext[, 2L], ymin = ext[, 3L], ymax = ext[, 4L],
             file_xsize = s$file_xsize, file_ysize = s$file_ysize,
             block_x = s$block_x, block_y = s$block_y,
             nblocks_x = nbx, nblocks_y = nby, blocks = nbx * nby,
             block_bytes = as.numeric(s$block_x) * s$block_y * attr(s, "itemsize"),
             remote = pix_request_bytes(s$path) == pix_remote_request_bytes)
}

## What a dataset is made of, from metadata only (no pixel reads). A list:
## dsn, driver, kind ("vrt" when expanded into sources, else "single"),
## ncol, nrow, gt, dtype, itemsize, block, nodata, note, overviews (a
## data.frame: level, ncol, nrow, factor, block_x, block_y), sources (what
## pix_plan() takes) and table (pix_source_table()). Overviews are reported,
## not used: extraction reads native-resolution cells.
pix_inspect <- function(dsn, band = 1L, overviews = TRUE) {
  src <- pix_sources(dsn, band)
  g <- attr(src, "grid")
  ds <- new(gdalraster::GDALRaster, dsn, read_only = TRUE)
  on.exit(ds$close())
  ovr <- data.frame(level = integer(), ncol = integer(), nrow = integer(),
                    factor = numeric(), block_x = integer(), block_y = integer())
  if (overviews) {
    for (i in seq_len(ds$getOverviewCount(band))) {
      ## 0-based crossing (new, to discuss): the OVERVIEW_LEVEL open option
      ## counts from 0; `level` here is 1-based (level 1 = first overview)
      o <- new(gdalraster::GDALRaster, dsn, TRUE, paste0("OVERVIEW_LEVEL=", i - 1L))
      d <- o$dim(); b <- o$getBlockSize(band)
      o$close()
      ovr[i, ] <- list(i, d[1L], d[2L], g$ncol / d[1L], b[1L], b[2L])
    }
  }
  list(dsn = dsn, driver = ds$getDriverShortName(), kind = attr(src, "kind"),
       ncol = g$ncol, nrow = g$nrow, gt = g$gt,
       dtype = attr(src, "dtype"), itemsize = attr(src, "itemsize"),
       block = ds$getBlockSize(band), nodata = attr(src, "nodata"),
       note = attr(src, "note"), overviews = ovr,
       sources = src, table = pix_source_table(src))
}

## -- Usage ------------------------------------------------------------------

## one row per touched block, in plan order, with its range of plan rows
.blocks <- function(plan) {
  g <- pix_block_groups(plan)
  cs <- cumsum(as.numeric(plan$col1 - plan$col0 + 1L))
  end <- cs[g$stop]
  data.frame(src = plan$src[g$start], block_row = plan$block_row[g$start],
             block_col = plan$block_col[g$start],
             cells = end - c(0, end[-length(end)]),
             segments = g$stop - g$start + 1L, start = g$start, stop = g$stop)
}

## decoded bytes of whole blocks, clipped at the file's right and bottom
.block_bytes <- function(src, k, brow, bcol) {
  bx <- src$block_x[k]; by <- src$block_y[k]
  w <- pmin(bx, src$file_xsize[k] - (bcol - 1L) * bx)
  h <- pmin(by, src$file_ysize[k] - (brow - 1L) * by)
  as.numeric(w) * h * attr(src, "itemsize")
}

## Per source the query touches: how much of it, and how densely.
## cells, runs, segments: of the query, in this source. blocks: blocks
## touched; span: blocks in their bounding box (block_row0..block_row1,
## block_col0..block_col1); occupancy = blocks / span, near 1 for a clustered
## query and near 0 for a scattered one. block_bytes: decoded bytes of the
## touched blocks; total_blocks: blocks of the source the mosaic uses.
## A row with src NA counts cells no source covers.
pix_usage <- function(plan) {
  src <- attr(plan, "sources")
  b <- .blocks(plan)
  tab <- pix_source_table(src)
  ks <- sort(unique(b$src), na.last = TRUE)
  rows <- lapply(ks, function(k) {
    m <- if (is.na(k)) is.na(b$src) else !is.na(b$src) & b$src == k
    pm <- if (is.na(k)) is.na(plan$src) else !is.na(plan$src) & plan$src == k
    out <- data.frame(src = k, path = if (is.na(k)) NA_character_ else src$path[k],
                      cells = sum(b$cells[m]), runs = length(unique(plan$run[pm])),
                      segments = sum(pm))
    if (is.na(k)) {
      cbind(out, blocks = 0L, span = 0L, occupancy = NA_real_,
            block_row0 = NA_integer_, block_row1 = NA_integer_,
            block_col0 = NA_integer_, block_col1 = NA_integer_,
            block_bytes = 0, total_blocks = 0L)
    } else {
      br <- b$block_row[m]; bc <- b$block_col[m]
      span <- (max(br) - min(br) + 1L) * (max(bc) - min(bc) + 1L)
      cbind(out, blocks = sum(m), span = span, occupancy = sum(m) / span,
            block_row0 = min(br), block_row1 = max(br),
            block_col0 = min(bc), block_col1 = max(bc),
            block_bytes = sum(.block_bytes(src, k, br, bc)),
            total_blocks = tab$blocks[k])
    }
  })
  do.call(rbind, rows)
}

## -- Read planning ------------------------------------------------------------

.pow2_upto <- function(n) {
  k <- 2^(0:max(0, ceiling(log2(n))))
  unique(c(k[k < n], max(n, 1)))
}

## cost of meta-tiling touched blocks with kx x ky meta-tiles; returns the
## total and, per block, its meta-tile and whether that is read as a window
.meta_cost <- function(brow, bcol, bbytes, kx, ky, bx, by, fx, fy, isz, req) {
  mr <- (brow - 1L) %/% ky; mc <- (bcol - 1L) %/% kx
  key <- mr * (max(mc) + 1) + mc
  g <- match(key, unique(key))
  r0 <- tapply(brow, g, min); r1 <- tapply(brow, g, max)
  c0 <- tapply(bcol, g, min); c1 <- tapply(bcol, g, max)
  n <- tabulate(g)
  wx <- pmin(c1 * bx, fx) - (c0 - 1) * bx
  wy <- pmin(r1 * by, fy) - (r0 - 1) * by
  wcost <- req + as.numeric(wx) * wy * isz
  pcost <- n * req + as.vector(rowsum(bbytes, g, reorder = TRUE))
  win <- wcost < pcost
  list(cost = sum(ifelse(win, wcost, pcost)), g = g, win = as.vector(win[g]))
}

## The plan with one read per touched block (what pix_execute() does when a
## plan has no read windows).
pix_block_reads <- function(plan) {
  b <- .blocks(plan)
  .with_reads(plan, b, seq_len(nrow(b)))
}

## Group a plan's touched blocks into read windows that fit `mem`.
##
## mem: bytes of decoded pixels one read may hold (default 256 MiB).
## request_bytes: cost of one read in bytes; NULL picks it per source path
## (pix_request_bytes()); or a number, or a function(path).
##
## Returns the plan with attr "reads" (one row per read, see .with_reads) and
## attr "meta": per source, the chosen meta-tile (kx, ky), the cost of the
## chosen reads and, for comparison, of reading block by block.
pix_plan_reads <- function(plan, mem = pix_default_mem, request_bytes = NULL) {
  src <- attr(plan, "sources")
  isz <- attr(src, "itemsize")
  b <- .blocks(plan)
  read_of <- integer(nrow(b))
  meta <- list()
  next_id <- 0L
  for (k in sort(unique(b$src), na.last = TRUE)) {
    if (is.na(k)) {
      next_id <- next_id + 1L
      read_of[is.na(b$src)] <- next_id
      next
    }
    m <- which(!is.na(b$src) & b$src == k)
    path <- src$path[k]
    req <- if (is.null(request_bytes)) pix_request_bytes(path) else
      if (is.function(request_bytes)) request_bytes(path) else request_bytes
    bx <- src$block_x[k]; by <- src$block_y[k]
    fx <- src$file_xsize[k]; fy <- src$file_ysize[k]
    br <- b$block_row[m]; bc <- b$block_col[m]
    bb <- .block_bytes(src, k, br, bc)
    nbx <- -(-fx %/% bx); nby <- -(-fy %/% by)
    base <- sum(req + bb)
    best <- list(cost = base, kx = 1, ky = 1, g = seq_along(m), win = rep(FALSE, length(m)))
    for (ky in .pow2_upto(nby)) for (kx in .pow2_upto(nbx)) {
      if (kx == 1 && ky == 1) next
      if (as.numeric(min(kx * bx, fx)) * min(ky * by, fy) * isz > mem) next
      r <- .meta_cost(br, bc, bb, kx, ky, bx, by, fx, fy, isz, req)
      if (r$cost < best$cost) best <- c(r, kx = kx, ky = ky)
    }
    win <- best$win
    ## one read per windowed meta-tile, one per block otherwise, ordered by
    ## meta-tile (row-major) then block
    ids <- ifelse(win, best$g, -seq_along(m))
    mt <- ((br - 1L) %/% best$ky) * (nbx %/% best$kx + 1) + (bc - 1L) %/% best$kx
    o <- order(mt, (br - 1) * nbx + bc)
    u <- unique(ids[o])
    read_of[m] <- next_id + match(ids, u)
    next_id <- next_id + length(u)
    meta[[length(meta) + 1L]] <- data.frame(
      src = k, path = path, block_x = bx, block_y = by, kx = best$kx, ky = best$ky,
      budget = mem, request_bytes = req, blocks = length(m), reads = length(u),
      windows = length(unique(best$g[win])), cost = best$cost, block_cost = base)
  }
  out <- .with_reads(plan, b, read_of)
  attr(out, "meta") <- do.call(rbind, meta)
  attr(out, "mem") <- mem
  out
}

## Attach a read table to a plan, given each touched block's read id.
## Read table (1-based, inclusive, in the source file's grid):
##   read, src, src_col0, src_row0, xsize, ysize   the window
##   blocks    blocks inside the window; touched: blocks with query cells
##   cells, segments, runs                         query content of the read
##   bytes     decoded bytes of the window
.with_reads <- function(plan, b, read_of) {
  src <- attr(plan, "sources")
  nr <- if (length(read_of)) max(read_of) else 0L
  f <- factor(read_of, levels = seq_len(nr))
  first <- match(seq_len(nr), read_of)
  k <- b$src[first]
  r0 <- as.vector(tapply(b$block_row, f, min)); r1 <- as.vector(tapply(b$block_row, f, max))
  c0 <- as.vector(tapply(b$block_col, f, min)); c1 <- as.vector(tapply(b$block_col, f, max))
  kk <- ifelse(is.na(k), 1L, k)
  bx <- src$block_x[kk]; by <- src$block_y[kk]
  xsize <- pmin(c1 * bx, src$file_xsize[kk]) - (c0 - 1L) * bx
  ysize <- pmin(r1 * by, src$file_ysize[kk]) - (r0 - 1L) * by
  unc <- is.na(k)
  xsize[unc] <- 0L; ysize[unc] <- 0L
  seg_read <- rep.int(read_of, b$stop - b$start + 1L)
  o <- order(seg_read, method = "radix")   # stable: keeps block order in a read
  seg <- plan[o, , drop = FALSE]
  seg$read <- seg_read[o]
  rownames(seg) <- NULL
  pairs <- unique(data.frame(read = seg$read, run = seg$run))
  reads <- data.frame(
    read = seq_len(nr), src = k,
    src_col0 = ifelse(unc, NA_integer_, (c0 - 1L) * bx + 1L),
    src_row0 = ifelse(unc, NA_integer_, (r0 - 1L) * by + 1L),
    xsize = xsize, ysize = ysize,
    blocks = ifelse(unc, 0L, (c1 - c0 + 1L) * (r1 - r0 + 1L)),
    touched = tabulate(read_of, nr),
    cells = as.vector(rowsum(b$cells, f, reorder = TRUE)),
    segments = as.vector(rowsum(b$segments, f, reorder = TRUE)),
    runs = tabulate(pairs$read, nr),
    bytes = as.numeric(xsize) * ysize * attr(src, "itemsize"))
  structure(seg, sources = attr(plan, "sources"), n_id = attr(plan, "n_id"),
            n_run = attr(plan, "n_run"), reads = reads)
}

## Sources, cell plan and read windows for a query, in one call. `cells` is
## a run table (for points, pix_cells_points(x, y, attr(pix_sources(dsn), "grid"))).
pix_plan_extraction <- function(dsn, cells, band = 1L, mem = pix_default_mem,
                                request_bytes = NULL) {
  pix_plan_reads(pix_plan(cells, pix_sources(dsn, band)), mem = mem,
                 request_bytes = request_bytes)
}
