"""Synthetic rasters for tests and examples; no network needed."""

import os

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


def make_tile(path, nx, ny, block, origin, res, seed=0, nodata=None,
              value_offset=0.0, nodata_frac=0.02):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, nx, ny, 1, gdal.GDT_Float32,
                    options=["TILED=YES", f"BLOCKXSIZE={block[0]}",
                             f"BLOCKYSIZE={block[1]}", "COMPRESS=DEFLATE"])
    ds.SetGeoTransform((origin[0], res, 0, origin[1], 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    rng = np.random.default_rng(seed)
    r = np.arange(ny, dtype=np.float32)[:, None]
    c = np.arange(nx, dtype=np.float32)[None, :]
    data = (value_offset + r * 1000 + c).astype(np.float32)
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
        data[rng.random((ny, nx)) < nodata_frac] = nodata
    band.WriteArray(data)
    ds = None
    return path


def make_mosaic(base, layout=(3, 2), size=(300, 200), blocks=((64, 64),),
                overlap=(0, 0), drop=(), nodata=None, res=0.01,
                origin=(100.0, -30.0), vrt_nodata=None):
    """Tiles laid out on one grid plus a VRT over them. Returns the VRT path.

    Tile k (row-major in the layout) holds values k * 1e6 + row * 1000 + col.
    """
    os.makedirs(base, exist_ok=True)
    nxl, nyl = layout
    sx, sy = size
    paths = []
    k = 0
    for j in range(nyl):
        for i in range(nxl):
            if k not in drop:
                ox = origin[0] + i * (sx - overlap[0]) * res
                oy = origin[1] - j * (sy - overlap[1]) * res
                p = os.path.join(base, f"tile_{j}_{i}.tif")
                make_tile(p, sx, sy, blocks[k % len(blocks)], (ox, oy), res,
                          seed=k, nodata=nodata, value_offset=k * 1e6)
                paths.append(p)
            k += 1
    vrt = os.path.join(base, "mosaic.vrt")
    if vrt_nodata is None:
        vrt_nodata = nodata if nodata is not None else "None"
    opts = gdal.BuildVRTOptions(VRTNodata=vrt_nodata,
                                srcNodata=nodata if nodata is not None else "None")
    gdal.BuildVRT(vrt, paths, options=opts).FlushCache()
    return vrt
