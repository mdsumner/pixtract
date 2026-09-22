## Zonal statistics from cbr runs on a synthetic mosaic (no network).
## Needs cbr (mdsumner/cbr) and wk. Run from the repo root:
##   Rscript examples/zonal_synthetic.R

suppressPackageStartupMessages({library(gdalraster); library(cbr)})
source("R/pixtract.R")
source("tests/synthetic.R")

vrt <- make_mosaic("zonal", size = c(1200, 800), blocks = list(c(256, 256), c(512, 128)))
src <- pix_sources(vrt)

circle <- function(x, y, r) {
  a <- seq(0, 2 * pi, length.out = 181)
  sprintf("POLYGON ((%s))", paste(x + r * cos(a), y + r * sin(a), collapse = ", "))
}
polys <- wk::wkt(c(circle(104, -34, 1.5), circle(115, -41, 3),
                   "POLYGON ((128 -46, 133 -46, 133 -44, 128 -44, 128 -46))"))

## burn on the raster's own grid; tables are 1-based, inclusive, id = position
a <- pix_burn_args(attr(src, "grid"))
b <- cb_burn(polys, extent = a$extent, dimension = a$dimension)
cells <- pix_cells_burn(b)                    # runs w = 1, edges w = fraction
str(pix_cost(pix_plan(cells, src)))

print(pix_zonal_stats(vrt, cells), row.names = FALSE)
