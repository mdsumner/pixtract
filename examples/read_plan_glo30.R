## Read planning on Copernicus GLO-30 over Switzerland (network needed).
##
## A VRT of 18 GLO-30 COGs (N45-N47 x E005-E010, 3600 x 3600 each, 1024 x
## 1024 blocks) over /vsicurl. Two point queries: a GPS-track-like cluster
## near Zermatt and a handful of points spread over the whole mosaic. For
## each, which sources it implicates, the read windows pix_plan_reads()
## picks, and the time to run it block by block and with those windows.
##
## Run from the repo root:  Rscript examples/read_plan_glo30.R

suppressPackageStartupMessages(library(gdalraster))
source("R/pixtract.R")
set_config_option("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
set_config_option("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
dir.create("examples/data", showWarnings = FALSE)

vrt <- "examples/data/ch_glo30.vrt"
if (!file.exists(vrt)) {
  url <- paste0("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/",
                "Copernicus_DSM_COG_10_%s_%s_DEM/Copernicus_DSM_COG_10_%s_%s_DEM.tif")
  ne <- expand.grid(e = sprintf("E%03d_00", 5:10), n = sprintf("N%02d_00", 45:47))
  buildVRT(vrt, sprintf(url, ne$n, ne$e, ne$n, ne$e), quiet = TRUE)
}

info <- pix_inspect(vrt)
tab <- info$table
cat(sprintf("%s %d x %d %s, %d sources, blocks %d x %d (%d in all), remote: %s\n",
            info$driver, info$ncol, info$nrow, info$dtype, nrow(tab), tab$block_x[1],
            tab$block_y[1], sum(tab$blocks), all(tab$remote)))
src <- info$sources

set.seed(1)
queries <- list(
  ## 20k points along a wandering track near Zermatt
  "track near Zermatt" = list(x = 7.75 + cumsum(rnorm(2e4, 0, 0.002)),
                              y = 46.02 + cumsum(rnorm(2e4, 0, 0.002))),
  ## 12 points anywhere in the mosaic
  "12 scattered points" = list(x = runif(12, 5, 11), y = runif(12, 45, 48)))

for (nm in names(queries)) {
  q <- queries[[nm]]
  plan <- pix_plan(pix_cells_points(q$x, q$y, attr(src, "grid")), src)
  u <- pix_usage(plan)
  cat(sprintf("\n== %s: %d cells in %d of %d sources, %d blocks (occupancy %s)\n", nm,
              sum(u$cells), nrow(u), nrow(tab), sum(u$blocks),
              paste(sprintf("%.2f", u$occupancy), collapse = ", ")))
  for (mem in c(2^22, 2^24, 2^26)) {
    cst <- pix_cost(pix_plan_reads(plan, mem = mem))
    cat(sprintf("  mem %4.0f MiB: %3d reads, %7.1f MiB decoded (block reads: %d, %.1f MiB)\n",
                mem / 2^20, cst$reads, cst$read_bytes / 2^20, cst$blocks,
                sum(u$block_bytes) / 2^20))
  }
  p <- pix_plan_reads(plan, mem = 2^26)
  for (run in list(list("block reads", plan), list("windows", p))) {
    vsi_curl_clear_cache()
    t0 <- proc.time()[["elapsed"]]
    v <- pix_execute(run[[2]], pix_place())
    cat(sprintf("  %-12s %6.1f s, mean elevation %.0f m\n", run[[1]],
                proc.time()[["elapsed"]] - t0, mean(v, na.rm = TRUE)))
  }
}
