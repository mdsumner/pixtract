# Tile-Grouped Raster Point Extraction — Benchmark Results

## The Idea

No GEOS needed. Tile membership is **integer arithmetic** on pixel coordinates:

```
col, row = inverse_geotransform(x, y)
tile_x = col // block_size_x
tile_y = row // block_size_y
```

Then `argsort` + `split` to group points by tile → one `ReadAsArray` per tile → fancy-index local offsets.

## Results: 500,000 points, 8192×8192 DEFLATE-compressed, 256×256 tiles

| Method                        | Time     | Factor |
|-------------------------------|----------|--------|
| **Tile-grouped (GDAL)**       | **0.78s** | **1×** |
| Full-band read + index        | 0.73s    | ~1×    |
| rasterio.sample()             | 11.35s   | 15×    |
| GDAL naive per-pixel          | ~25.7s*  | 33×    |
| rasterstats.point_query       | ~1395s*  | 1788×  |

*extrapolated from subset

## When tile-grouped really wins

### Clustered points (real-world: penguin colonies, vessel tracks, field sites)

| Scenario                          | Tiles read | Time   |
|-----------------------------------|-----------|--------|
| 500k uniform → 8k raster         | 1024      | 0.78s  |
| **500k clustered (5 clusters)**   | **206**   | **0.18s** |
| **500k clustered (3 clusters) → 16k raster** | **47** | **0.14s** |

Clustered points touch fewer tiles → massive I/O savings. This is the common case for real extraction queries.

### Scaling to large rasters

| Raster size  | Tile size | Method        | Time   |
|-------------|-----------|---------------|--------|
| 16k × 16k  | 512       | Tile-grouped  | 1.32s  |
| 16k × 16k  | 512       | Full-read     | 1.88s  |
| 32k × 32k  | 512       | Tile-grouped (500k pts)  | 3.46s  |
| 32k × 32k  | 512       | Tile-grouped (2M pts)    | 3.86s  |

Note: 500k → 2M points barely changes time — it's I/O bound on tile reads, not point count.
Full-band read becomes impractical for large rasters (>1 GB uncompressed).

### Where this approach is essential

- **COG over /vsicurl/**: each tile = one HTTP range request. 47 requests (clustered) vs 500,000 = game over.
- **Compressed rasters**: decompress 47 tiles vs 1024 or vs entire band.
- **Multi-band extraction**: read the same tile set from multiple bands.
- **Rasters too large for memory**: tile-grouped works within bounded memory.

## Key design notes

1. **No spatial library needed** — point-in-tile is just integer division on a regular grid
2. **`argsort` + `split`** is the fast numpy idiom for groupby (no Python dict overhead)
3. **Thread parallelism** helps for network I/O (COG) but limited benefit for local SSD
4. **CRS reprojection** can be done with `osgeo.osr.CoordinateTransformation` or `pyproj` before calling extract
5. For truly massive point sets on COGs, could combine with `asyncio` + GDAL's `/vsicurl/` or `fsspec`

## Existing Python tools checked

- `rasterio.sample()` — per-point reads, not tile-aware (15× slower)
- `rasterstats.point_query` — wraps rasterio + shapely, absurdly slow (~1800×)
- `xarray.Dataset.sel()` — loads lazily but not tile-aligned point extraction
- `exactextract` — polygon-focused, not relevant for points
- No existing library implements tile-grouped point extraction.
