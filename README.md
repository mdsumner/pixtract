# pixtract

<!-- badges: start -->
<!-- badges: end -->


Fast raster point extraction via tile-grouped reads using GDAL.

No spatial predicates, no GEOS. Tile membership is integer arithmetic on pixel coordinates from the GeoTransform:

```
col, row = inverse_geotransform(x, y)
tile_x, tile_y = col // block_size_x, row // block_size_y
```

Group points by tile, one `ReadAsArray` per tile, fancy-index local offsets. Works on anything GDAL can open — local files, COGs over `/vsicurl/`, VRTs, `/vsis3/`, whatever.

## Install

```bash
uv pip install -e .
```

Requires GDAL Python bindings (`osgeo.gdal`) installed via system package or conda. Not pip-installable.

## Usage

```python
from pixtract import extract_points

# Local file
vals = extract_points("dem.tif", xs, ys)

# COG over HTTP
vals = extract_points("/vsicurl/https://example.com/cog.tif", xs, ys)

# VRT mosaic — GDAL resolves sources internally
vals = extract_points("/vsicurl/https://example.com/mosaic.vrt", xs, ys)

# Threaded reads (for network/COG sources)
vals = extract_points(dsn, xs, ys, max_workers=8)

# rasterio backend
vals = extract_points(dsn, xs, ys, backend="rasterio")
```

`xs` and `ys` are arrays of map coordinates in the raster's CRS. Returns `float64` array with `NaN` for nodata pixels.

<!--
This entire block of text and the following code block will be hidden.

## Why

| Method | 500k points, 8k×8k DEFLATE |
|---|---|
| **pixtract** | **0.07–0.8s** |
| rasterio.sample() | 11s |
| GDAL per-pixel | ~26s |
| rasterstats | ~1400s |

Clustered points (the real-world case) touch fewer tiles and run faster. 500k points hitting 47 tiles on a 16k raster: 0.07s.
-->

## How it works

1. Inverse GeoTransform: `(x, y)` → fractional `(col, row)`
2. Integer divide by block size → tile index per point
3. `argsort` + `split` → group points by tile
4. One `ReadAsArray` per unique tile touched
5. Fancy-index local offsets within each tile

For compressed rasters, COGs, or network sources, reading 47 tiles instead of 500,000 individual pixels is the entire difference.

## Project structure

```
src/pixtract/       # package
exploration/        # dated lab notebook scripts from development
```

## Status

Early development. Works, tested against REMA 2m Antarctic elevation COGs and VRTs.

## Code of Conduct
  
Please note that the pixtract project is released with a [Contributor Code of Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html). By contributing to this project, you agree to abide by its terms.
