"""extract_points must give what the original implementation gave."""

import numpy as np
import pytest

import legacy_extract
import pixtract
from conftest import dense


def _points(path, n, seed, pad=0.0, clustered=False):
    from osgeo import gdal
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    xmin, xmax = gt[0], gt[0] + gt[1] * ds.RasterXSize
    ymax, ymin = gt[3], gt[3] + gt[5] * ds.RasterYSize
    rng = np.random.default_rng(seed)
    if clustered:
        cx = rng.uniform(xmin, xmax, 4)
        cy = rng.uniform(ymin, ymax, 4)
        k = rng.integers(0, 4, n)
        xs = cx[k] + rng.normal(0, (xmax - xmin) / 40, n)
        ys = cy[k] + rng.normal(0, (ymax - ymin) / 40, n)
    else:
        dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
        xs = rng.uniform(xmin - dx, xmax + dx, n)
        ys = rng.uniform(ymin - dy, ymax + dy, n)
    inside = (xs >= xmin) & (xs < xmax) & (ys > ymin) & (ys <= ymax)
    return xs, ys, inside


def _inside_strict(path, xs, ys):
    """Points whose pixel is inside the raster (same test the planner uses)."""
    from osgeo import gdal
    ds = gdal.Open(path)
    g = pixtract.Grid(ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize)
    col, row = g.pixel(xs, ys)
    return (col >= 0) & (col < g.ncol) & (row >= 0) & (row < g.nrow)


@pytest.mark.parametrize("fixture", ["single", "mosaic", "mosaic_overlap",
                                     "mosaic_gap", "mosaic_overlap_nodata",
                                     "mosaic_srcnodata", "mosaic_srcnodata_novrt"])
@pytest.mark.parametrize("clustered", [False, True])
def test_matches_legacy_inside(request, fixture, clustered):
    path = request.getfixturevalue(fixture)
    xs, ys, _ = _points(path, 20000, seed=3, clustered=clustered)
    inside = _inside_strict(path, xs, ys)
    new = pixtract.extract_points(path, xs, ys)
    old = legacy_extract.extract_points(path, xs, ys)
    np.testing.assert_array_equal(new[inside], old[inside])
    assert np.isnan(new[~inside]).all()


@pytest.mark.parametrize("fixture", ["single", "mosaic", "mosaic_overlap",
                                     "mosaic_gap", "mosaic_srcnodata", "mosaic_srcnodata_novrt"])
def test_matches_dense(request, fixture):
    path = request.getfixturevalue(fixture)
    xs, ys, _ = _points(path, 5000, seed=4)
    inside = _inside_strict(path, xs, ys)
    g = pixtract.plan_sources(path).grid
    col, row = g.pixel(xs[inside], ys[inside])
    d = dense(path)[np.floor(row).astype(int), np.floor(col).astype(int)]
    np.testing.assert_array_equal(pixtract.extract_points(path, xs, ys)[inside], d)


def test_outside_is_nan_not_edge_value(single):
    """The original clipped out-of-extent points to the edge pixel."""
    xs, ys, _ = _points(single, 5000, seed=5, pad=0.3)
    inside = _inside_strict(single, xs, ys)
    assert (~inside).sum() > 100
    new = pixtract.extract_points(single, xs, ys)
    assert np.isnan(new[~inside]).all()


def test_threads_match_serial(mosaic):
    xs, ys, _ = _points(mosaic, 20000, seed=6)
    a = pixtract.extract_points(mosaic, xs, ys)
    b = pixtract.extract_points(mosaic, xs, ys, max_workers=4)
    np.testing.assert_array_equal(a, b)


def test_rasterio_backend(single):
    pytest.importorskip("rasterio")
    xs, ys, _ = _points(single, 5000, seed=7)
    inside = _inside_strict(single, xs, ys)
    a = pixtract.extract_points(single, xs, ys, backend="rasterio")
    b = legacy_extract.extract_points(single, xs, ys, backend="rasterio")
    np.testing.assert_array_equal(a[inside], b[inside])


def test_empty_and_all_outside(single):
    assert pixtract.extract_points(single, [], []).size == 0
    v = pixtract.extract_points(single, [0.0, 1.0], [0.0, 1.0])
    assert np.isnan(v).all() and v.size == 2
