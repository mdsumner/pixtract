## Swiss cantons on Copernicus GLO-90: zonal stats from S3 (network needed).
##
## Mirrors the Apache Sedona "group by, but for pixels" benchmark: 26 cantons
## (Overture 2026-08-19.0) over the 18 GLO-90 tiles N45-N47 x E005-E010.
## Sedona reports 7,006,027 cells under cantons (cell-centre rule).
##
## Cantons come from examples/data/cantons.gpkg if the Python example has
## written it, otherwise from Overture via duckdb (needs the duckdb package).
## Run from the repo root:  Rscript examples/cantons_glo90.R

suppressPackageStartupMessages({library(gdalraster); library(cbr)})
source("R/pixtract.R")

data <- file.path("examples", "data")
dir.create(data, showWarnings = FALSE)

cantons <- function() {
  gpkg <- file.path(data, "cantons.gpkg")
  if (file.exists(gpkg)) {
    v <- new(GDALVector, gpkg)
    on.exit(v$close())
    f <- v$fetch(-1)
    nm <- f$name
    Encoding(nm) <- "UTF-8"
    return(list(name = nm, geom = wk::wkb(f$geom)))
  }
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  DBI::dbExecute(con, "INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
  d <- DBI::dbGetQuery(con, "
    SELECT names.primary AS name, ST_AsWKB(geometry) AS wkb
    FROM read_parquet('s3://overturemaps-us-west-2/release/2026-08-19.0/theme=divisions/type=division_area/*', hive_partitioning = 1)
    WHERE bbox.xmin BETWEEN 5.5 AND 10.7 AND bbox.ymin BETWEEN 45.7 AND 47.9
      AND country = 'CH' AND subtype = 'region'")
  list(name = d$name, geom = wk::wkb(d$wkb))
}

dem_vrt <- function() {
  vrt <- file.path(data, "ch_dem.vrt")
  if (!file.exists(vrt)) {
    key <- function(n, e) sprintf("Copernicus_DSM_COG_30_%s_%s_DEM", n, e)
    tiles <- as.vector(outer(sprintf("N%02d_00", 45:47), sprintf("E%03d_00", 5:10), function(n, e)
      sprintf("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/%s/%s.tif", key(n, e), key(n, e))))
    buildVRT(vrt, tiles, quiet = TRUE)
  }
  vrt
}

ch <- cantons()
vrt <- dem_vrt()
src <- pix_sources(vrt)
a <- pix_burn_args(attr(src, "grid"))       # 7200 x 3600, half-pixel offset origin

t0 <- proc.time()[["elapsed"]]
b_approx <- cb_burn(ch$geom, extent = a$extent, dimension = a$dimension, coverage = FALSE)
b_cov <- cb_burn(ch$geom, extent = a$extent, dimension = a$dimension, coverage = TRUE)
t_burn <- proc.time()[["elapsed"]] - t0

## cbr runs are 1-based with an inclusive col_end
r <- b_approx$runs
cat(sprintf("cells under cantons (cell-centre): %s  (Sedona: 7,006,027)\n",
            format(sum(r$col_end - r$col_start + 1L), big.mark = ",")))

approx <- pix_cells_burn(b_approx)
str(pix_cost(pix_plan(approx, src)))

t0 <- proc.time()[["elapsed"]]
st <- pix_zonal_stats(vrt, approx)
sc <- pix_zonal_stats(vrt, pix_cells_burn(b_cov))
t_stats <- proc.time()[["elapsed"]] - t0
cat(sprintf("burn %.2f s, planned reads + stats (both modes) %.1f s\n", t_burn, t_stats))

out <- data.frame(canton = ch$name, cells = st$count, mean = round(st$mean, 1),
                  cov_mean = round(sc$mean, 1))
print(out[order(-out$mean), ], row.names = FALSE)
