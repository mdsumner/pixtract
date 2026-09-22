import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

gdal = pytest.importorskip("osgeo.gdal")
gdal.UseExceptions()

from synthetic import make_tile, make_mosaic  # noqa: E402


@pytest.fixture(scope="session")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("pixtract")


@pytest.fixture(scope="session")
def single(tmp):
    """One 1000 x 700 tiled DEFLATE GeoTIFF, 128 x 64 blocks, with nodata."""
    return make_tile(str(tmp / "single.tif"), 1000, 700, (128, 64),
                     origin=(100.0, -30.0), res=0.01, seed=1, nodata=-9999)


@pytest.fixture(scope="session")
def mosaic(tmp):
    """3 x 2 mosaic of tiles with different block sizes, as a VRT."""
    return make_mosaic(str(tmp / "mosaic"), layout=(3, 2), size=(300, 200),
                       blocks=[(64, 64), (128, 32), (256, 256)], nodata=None)


@pytest.fixture(scope="session")
def mosaic_overlap(tmp):
    """Mosaic whose tiles overlap by 17 columns and 9 rows (last one wins)."""
    return make_mosaic(str(tmp / "overlap"), layout=(3, 2), size=(300, 200),
                       blocks=[(64, 64), (32, 128)], overlap=(17, 9), nodata=None)


@pytest.fixture(scope="session")
def mosaic_overlap_nodata(tmp):
    """Overlapping tiles with per-source NODATA: read through the VRT itself."""
    return make_mosaic(str(tmp / "overlap_nd"), layout=(2, 2), size=(300, 200),
                       blocks=[(64, 64)], overlap=(17, 9), nodata=-1)


@pytest.fixture(scope="session")
def mosaic_srcnodata(tmp):
    """Source NODATA -1 differs from the VRT's nodata -9999, and a tile is missing."""
    return make_mosaic(str(tmp / "srcnd"), layout=(3, 2), size=(300, 200),
                       blocks=[(64, 64), (128, 32)], drop=[1], nodata=-1,
                       vrt_nodata=-9999)


@pytest.fixture(scope="session")
def mosaic_srcnodata_novrt(tmp):
    """Source NODATA -1 but no VRT nodata: transparent pixels read as 0."""
    return make_mosaic(str(tmp / "srcnd_novrt"), layout=(3, 2), size=(300, 200),
                       blocks=[(64, 64)], drop=[1], nodata=-1, vrt_nodata="None")


@pytest.fixture(scope="session")
def mosaic_gap(tmp):
    """Mosaic with a missing tile (a hole the VRT fills with nodata or 0)."""
    return make_mosaic(str(tmp / "gap"), layout=(3, 2), size=(300, 200),
                       blocks=[(64, 64)], drop=[4], nodata=None)


def dense(path, band=1):
    ds = gdal.Open(path)
    b = ds.GetRasterBand(band)
    v = b.ReadAsArray().astype(np.float64)
    nd = b.GetNoDataValue()
    if nd is not None:
        v[v == nd] = np.nan
    return v
