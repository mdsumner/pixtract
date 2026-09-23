"""Read windows: same values as block reads, within budget, shaped by the query."""

import numpy as np
import pytest

import pixtract
from conftest import dense
from synthetic import make_tile
from test_plan_equiv import FIX, expand_dense, random_runs

MEMS = [1, 2**14, 2**17, 2**20, 2**30]


@pytest.mark.parametrize("fixture", FIX)
@pytest.mark.parametrize("mem", MEMS)
def test_windows_equal_dense(request, fixture, mem):
    path = request.getfixturevalue(fixture)
    v = dense(path)
    cells = random_runs(*v.shape, n=300, n_id=7, seed=21)
    got = pixtract.extract_cells(path, cells, mem=mem, max_workers=2)
    ref = expand_dense(cells, v)
    assert got["value"].size == ref.shape[0]
    o_got = np.lexsort((got["col"], got["row"], got["run"]))
    run_ref = np.repeat(np.arange(len(cells["row"])), _lengths(cells, v.shape))
    o_ref = np.lexsort((ref[:, 1], ref[:, 0], run_ref))
    np.testing.assert_array_equal(got["col"][o_got], ref[o_ref, 1])
    np.testing.assert_array_equal(got["value"][o_got], ref[o_ref, 4])


def _lengths(cells, shape):
    nrow, ncol = shape
    r = np.asarray(cells["row"])
    n = np.minimum(cells["col_end"], ncol) - np.maximum(cells["col_start"], 0)
    return np.where((r >= 0) & (r < nrow) & (n > 0), n, 0)


@pytest.mark.parametrize("fixture", FIX)
def test_points_windows_equal_blocks(request, fixture):
    path = request.getfixturevalue(fixture)
    src = pixtract.plan_sources(path)
    e = src.grid.extent()
    rng = np.random.default_rng(3)
    xs = rng.uniform(e[0], e[1], 4000)
    ys = rng.uniform(e[2], e[3], 4000)
    a = pixtract.extract_points(path, xs, ys)
    for mem in MEMS:
        b = pixtract.extract_points(path, xs, ys, mem=mem)
        np.testing.assert_array_equal(a, b)


def test_reads_cover_plan_and_fit_budget(mosaic):
    src = pixtract.plan_sources(mosaic)
    cells = random_runs(src.grid.nrow, src.grid.ncol, n=2000, n_id=5, seed=4)
    plan = pixtract.plan_cells(cells, src)
    for mem, workers in [(2**15, None), (2**17, 4), (2**30, None)]:
        p = pixtract.plan_reads(plan, mem=mem, max_workers=workers)
        rd = p.reads
        budget = mem // (2 * workers if workers else 1)
        ok = rd["src"] >= 0
        # a read is over budget only when it is one block that alone is
        one_block = rd["blocks"] == 1
        assert np.all((rd["bytes"][ok] <= budget) | one_block[ok])
        # every segment's read is from its own source, and inside the window
        s = p.seg
        k = s["src"]
        cov = k >= 0
        np.testing.assert_array_equal(rd["src"][s["read"]], k)
        bx, by = src.block_x[k[cov]], src.block_y[k[cov]]
        col = s["bcol"][cov] * bx + s["c0"][cov]
        row = s["brow"][cov] * by + s["r"][cov]
        j = s["read"][cov]
        assert np.all(col >= rd["xoff"][j]) and np.all(row >= rd["yoff"][j])
        assert np.all(s["bcol"][cov] * bx + s["c1"][cov] <= rd["xoff"][j] + rd["xsize"][j])
        assert np.all(row < rd["yoff"][j] + rd["ysize"][j])
        # segments are grouped by read, and cell totals are preserved
        assert np.all(np.diff(s["read"]) >= 0)
        assert rd["cells"].sum() == plan.n_cells
        assert rd["touched"].sum() == pixtract.cost(plan)["blocks"] + (
            1 if (plan.seg["src"] < 0).any() else 0)


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """4096 x 4096 Float32, 256 x 256 blocks (16 x 16 blocks, 256 KiB each)."""
    p = str(tmp_path_factory.mktemp("big") / "big.tif")
    return make_tile(p, 4096, 4096, (256, 256), origin=(0.0, 0.0), res=1.0, seed=0)


def _points_plan(path, xs, ys, **kw):
    src = pixtract.plan_sources(path)
    plan = pixtract.plan_cells(pixtract.cells_from_points(xs, ys, src.grid), src)
    return plan, pixtract.plan_reads(plan, **kw)


def test_clustered_query_gets_windows(big):
    rng = np.random.default_rng(1)
    # a tight cluster covering about 3 x 3 blocks
    xs = 1500 + rng.uniform(0, 700, 20000)
    ys = -1500 - rng.uniform(0, 700, 20000)
    plan, p = _points_plan(big, xs, ys, mem=2**30)
    u = pixtract.source_usage(plan)
    assert u["occupancy"][0] == 1.0
    assert pixtract.cost(p)["reads"] < pixtract.cost(plan)["blocks"]
    assert pixtract.cost(p)["reads"] == 1


def test_scattered_query_gets_block_reads(big):
    # one point in each of 8 blocks spread along a diagonal: a window over
    # them would read 28x the bytes
    b = np.arange(0, 16, 2)
    xs = b * 256 + 10.5
    ys = -(b * 256 + 10.5)
    plan, p = _points_plan(big, xs, ys, mem=2**30)
    u = pixtract.source_usage(plan)
    assert u["blocks"][0] == 8 and u["span"][0] == 15 * 15
    assert pixtract.cost(p)["reads"] == 8
    assert pixtract.cost(p)["read_bytes"] == 8 * 256 * 256 * 4


def test_remote_cost_favours_fewer_reads(big):
    # every other block in a 4 x 4 patch: locally block reads win, with a
    # remote-sized request cost one window wins
    br, bc = np.meshgrid(np.arange(4, 8), np.arange(4, 8), indexing="ij")
    m = ((br + bc) % 2) == 0
    xs = bc[m] * 256 + 10.5
    ys = -(br[m] * 256 + 10.5)
    _, local = _points_plan(big, xs, ys, mem=2**30)
    _, remote = _points_plan(big, xs, ys, mem=2**30,
                             request_bytes=pixtract.reads.REMOTE_REQUEST_BYTES)
    assert pixtract.cost(local)["reads"] == 8
    assert pixtract.cost(remote)["reads"] == 1


def test_budget_caps_window_size(big):
    rng = np.random.default_rng(2)
    xs = rng.uniform(0, 4096, 50000)
    ys = -rng.uniform(0, 4096, 50000)
    for mem in (2**18, 2**20, 2**22, 2**30):
        _, p = _points_plan(big, xs, ys, mem=mem)
        rd = p.reads
        assert rd["bytes"].max() <= mem
        assert rd["touched"].sum() == 256
    # the whole file (64 MiB as Float32) fits in 1 GiB: one read
    assert p.reads["read"].size == 1


def test_inspect_source(mosaic, single, tmp):
    info = pixtract.inspect_source(mosaic)
    assert info["driver"] == "VRT" and info["kind"] == "vrt"
    assert info["dtype"] == "Float32" and info["itemsize"] == 4
    t = info["table"]
    assert len(t["path"]) == 6
    # tile 0: 300 x 200 with 64 x 64 blocks -> 5 x 4 blocks
    assert (t["nblocks_x"][0], t["nblocks_y"][0]) == (5, 4)
    np.testing.assert_allclose((t["xmin"][1], t["xmax"][1], t["ymin"][1], t["ymax"][1]),
                               (103.0, 106.0, -32.0, -30.0))
    # overviews are reported
    from osgeo import gdal
    p = str(tmp / "ovr.tif")
    gdal.Translate(p, single, creationOptions=["TILED=YES"])
    ds = gdal.Open(p, gdal.GA_Update)
    ds.BuildOverviews("NEAREST", [2, 4])
    ds = None
    ov = pixtract.inspect_source(p)["overviews"]
    assert [o["ncol"] for o in ov] == [500, 250] and ov[0]["factor"] == 2


def test_source_usage_counts(mosaic):
    src = pixtract.plan_sources(mosaic)
    # one run across the whole top row: sources 0, 1, 2
    cells = {"row": [0], "col_start": [0], "col_end": [900], "id": [0]}
    u = pixtract.source_usage(pixtract.plan_cells(cells, src))
    np.testing.assert_array_equal(u["src"], [0, 1, 2])
    np.testing.assert_array_equal(u["cells"], [300, 300, 300])
    # block widths 64, 128, 256 -> 5, 3, 2 blocks along the row
    np.testing.assert_array_equal(u["blocks"], [5, 3, 2])
    np.testing.assert_array_equal(u["runs"], [1, 1, 1])


def test_plan_extraction_executes(mosaic):
    src = pixtract.plan_sources(mosaic)
    cells = random_runs(src.grid.nrow, src.grid.ncol, n=500, n_id=9, seed=8)
    p = pixtract.plan_extraction(mosaic, cells, mem=2**18)
    a = pixtract.execute(p, pixtract.Stats())
    b = pixtract.zonal_stats(mosaic, cells)
    for k in ("count", "min", "max"):
        np.testing.assert_array_equal(a[k], b[k])
    for k in ("weight", "sum"):  # summed in a different order
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12)
    assert set(p.extra["meta"]["src"].tolist()) <= set(range(len(src)))
