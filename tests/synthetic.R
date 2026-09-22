## Synthetic rasters for R tests and examples; no network needed.
## Tile k holds values k * 1e6 + row * 1000 + col (0-based row/col in the value
## only, so the numbers match the Python fixtures in synthetic.py).

td <- tempfile("pixtract"); dir.create(td)

make_tile <- function(path, nx, ny, block, origin, res, k, nodata = NA) {
  ds <- create("GTiff", path, nx, ny, 1L, "Float32", return_obj = TRUE,
               options = c("TILED=YES", paste0("BLOCKXSIZE=", block[1L]),
                           paste0("BLOCKYSIZE=", block[2L]), "COMPRESS=DEFLATE"))
  ds$setGeoTransform(c(origin[1L], res, 0, origin[2L], 0, -res))
  ds$setProjection(srs_to_wkt("EPSG:4326"))
  ## raster order: row-major from the top-left
  v <- k * 1e6 + rep((seq_len(ny) - 1) * 1000, each = nx) + rep(seq_len(nx) - 1, ny)
  if (!is.na(nodata)) {
    set.seed(k)
    v[runif(length(v)) < 0.02] <- nodata
    ds$setNoDataValue(1L, nodata)
  }
  ds$write(1L, 0L, 0L, nx, ny, v)
  ds$close()
  path
}

make_mosaic <- function(name, layout = c(3, 2), size = c(300, 200), blocks = list(c(64, 64)),
                        overlap = c(0, 0), drop = integer(), nodata = NA, vrt_nodata = nodata) {
  base <- file.path(td, name); dir.create(base)
  paths <- character(); k <- 0
  for (j in seq_len(layout[2L]) - 1) for (i in seq_len(layout[1L]) - 1) {
    if (!k %in% drop) {
      o <- c(100 + i * (size[1L] - overlap[1L]) * 0.01, -30 - j * (size[2L] - overlap[2L]) * 0.01)
      p <- file.path(base, sprintf("tile_%d_%d.tif", j, i))
      paths <- c(paths, make_tile(p, size[1L], size[2L], blocks[[k %% length(blocks) + 1]],
                                  o, 0.01, k, nodata))
    }
    k <- k + 1
  }
  vrt <- file.path(base, "mosaic.vrt")
  args <- c("-vrtnodata", if (is.na(vrt_nodata)) "None" else vrt_nodata,
            "-srcnodata", if (is.na(nodata)) "None" else nodata)
  buildVRT(vrt, paths, cl_arg = args, quiet = TRUE)
  vrt
}
