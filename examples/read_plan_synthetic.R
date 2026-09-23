## Read planning on a synthetic 3 x 2 mosaic (no network).
##
## Inspects the mosaic, then for three queries (clustered points, scattered
## points, a dense block of runs) shows which sources each implicates and
## which read windows pix_plan_reads() picks at a few memory budgets. Values
## are checked against plain block-by-block reads.
##
## Run from the repo root:  Rscript examples/read_plan_synthetic.R

suppressPackageStartupMessages(library(gdalraster))
source("R/pixtract.R")
source("tests/synthetic.R")

vrt <- make_mosaic("readplan", size = c(1200, 800), blocks = list(c(256, 256), c(512, 128)))

info <- pix_inspect(vrt)
cat(sprintf("%s %d x %d %s, %d sources (%s), %d overviews\n", info$driver, info$ncol,
            info$nrow, info$dtype, nrow(info$sources), info$kind, nrow(info$overviews)))
tab <- info$table
tab$file <- basename(tab$path)
print(tab[, c("src", "file", "col0", "row0", "block_x", "block_y", "nblocks_x",
              "nblocks_y", "xmin", "ymax")], row.names = FALSE)

src <- info$sources
g <- attr(src, "grid")
set.seed(1)
## 50k points in three tight clusters
k <- sample(3, 5e4, replace = TRUE)
clustered <- pix_cells_points(c(102, 110, 131)[k] + rnorm(length(k), 0, 0.1),
                              c(-33, -40, -45)[k] + rnorm(length(k), 0, 0.1), g)
## 40 points anywhere
scattered <- pix_cells_points(runif(40, 100, 136), runif(40, -46, -30), g)
## a 600-row block of runs, 1300 cells wide (a polygon, as runs)
dense <- data.frame(row = 101:700, col_start = 201L, col_end = 1500L, id = 1L, w = 1)

queries <- list("clustered points" = clustered, "scattered points" = scattered,
                "dense runs" = dense)
for (nm in names(queries)) {
  plan <- pix_plan(queries[[nm]], src)
  cat(sprintf("\n== %s: %s\n", nm, paste(names(pix_cost(plan)), unlist(pix_cost(plan)),
                                         sep = " ", collapse = ", ")))
  print(pix_usage(plan)[, c("src", "cells", "runs", "blocks", "span", "occupancy")],
        row.names = FALSE, digits = 3)
  ref <- pix_execute(plan, pix_stats())
  for (mem in c(2^18, 2^22, 2^28)) {
    p <- pix_plan_reads(plan, mem = mem)
    cst <- pix_cost(p)
    m <- attr(p, "meta")
    cat(sprintf("  mem %6.2f MiB: %4d reads, %7.2f MiB (meta-tile per source %s)\n",
                mem / 2^20, cst$reads, cst$read_bytes / 2^20,
                paste0(m$src, ":", m$kx, "x", m$ky, collapse = ", ")))
    got <- pix_execute(p, pix_stats())
    stopifnot(identical(got$count, ref$count), isTRUE(all.equal(got$sum, ref$sum)))
  }
}
cat("\nread windows give the same values as block reads\n")
