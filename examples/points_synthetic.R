## Point extraction on a synthetic 3 x 2 mosaic (no network).
## Run from the repo root:  Rscript examples/points_synthetic.R

suppressPackageStartupMessages(library(gdalraster))
source("R/pixtract.R")
source("tests/synthetic.R")

vrt <- make_mosaic("points", size = c(1200, 800), blocks = list(c(256, 256), c(512, 128)))

## 200k points in three clusters, plus a few outside the mosaic
set.seed(1)
k <- sample(3, 2e5, replace = TRUE)
x <- c(102, 110, 131)[k] + rnorm(length(k), 0, 0.4)
y <- c(-33, -40, -45)[k] + rnorm(length(k), 0, 0.4)

## the plan, before any pixel I/O
src <- pix_sources(vrt)
plan <- pix_plan(pix_cells_points(x, y, attr(src, "grid")), src)
cat(nrow(src), " sources (", attr(src, "kind"), "); cost: ", sep = "")
str(pix_cost(plan))

vals <- pix_extract_points(vrt, x, y)
cat(sum(!is.na(vals)), "of", length(vals), "points have values; first five:", head(vals, 5), "\n")
