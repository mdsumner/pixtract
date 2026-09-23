"""Planned reads over run tables must equal a dense read + index."""

import numpy as np
import pytest

import pixtract
from conftest import dense


def random_runs(nrow, ncol, n, n_id, seed, max_len=400):
    rng = np.random.default_rng(seed)
    row = rng.integers(-3, nrow + 3, n)
    c0 = rng.integers(-50, ncol, n)
    c1 = c0 + rng.integers(1, max_len, n)
    return {"row": row, "col_start": c0, "col_end": c1,
            "id": rng.integers(0, n_id, n), "w": rng.uniform(0.1, 1, n)}


def expand_dense(cells, v):
    """Reference: expand runs cell by cell and index the dense array."""
    nrow, ncol = v.shape
    out = []
    for i in range(len(cells["row"])):
        r = cells["row"][i]
        if r < 0 or r >= nrow:
            continue
        for c in range(max(cells["col_start"][i], 0), min(cells["col_end"][i], ncol)):
            out.append((r, c, cells["id"][i], cells["w"][i], v[r, c]))
    return np.array(out, dtype=np.float64).reshape(-1, 5)


FIX = ["single", "mosaic", "mosaic_overlap", "mosaic_gap", "mosaic_overlap_nodata",
       "mosaic_srcnodata", "mosaic_srcnodata_novrt"]


@pytest.mark.parametrize("fixture", FIX)
def test_extract_cells_equals_dense(request, fixture):
    path = request.getfixturevalue(fixture)
    v = dense(path)
    cells = random_runs(*v.shape, n=300, n_id=7, seed=11)
    got = pixtract.extract_cells(path, cells)
    ref = expand_dense(cells, v)
    o_got = np.lexsort((got["col"], got["row"], got["run"]))
    assert got["value"].size == ref.shape[0]
    run_ref = np.repeat(np.arange(len(cells["row"])), _run_lengths(cells, v.shape))
    o_ref = np.lexsort((ref[:, 1], ref[:, 0], run_ref))
    np.testing.assert_array_equal(got["row"][o_got], ref[o_ref, 0])
    np.testing.assert_array_equal(got["col"][o_got], ref[o_ref, 1])
    np.testing.assert_array_equal(got["id"][o_got], ref[o_ref, 2])
    np.testing.assert_array_equal(got["value"][o_got], ref[o_ref, 4])


def _run_lengths(cells, shape):
    nrow, ncol = shape
    r = np.asarray(cells["row"])
    n = np.minimum(cells["col_end"], ncol) - np.maximum(cells["col_start"], 0)
    return np.where((r >= 0) & (r < nrow) & (n > 0), n, 0)


@pytest.mark.parametrize("fixture", FIX)
def test_zonal_stats_equals_dense(request, fixture):
    path = request.getfixturevalue(fixture)
    v = dense(path)
    cells = random_runs(*v.shape, n=500, n_id=9, seed=12)
    st = pixtract.zonal_stats(path, cells, max_workers=3)
    ref = expand_dense(cells, v)
    ok = ~np.isnan(ref[:, 4])
    ref = ref[ok]
    for k in range(9):
        m = ref[:, 2] == k
        assert st["count"][k] == m.sum()
        if m.any():
            np.testing.assert_allclose(st["weight"][k], ref[m, 3].sum())
            np.testing.assert_allclose(
                st["mean"][k], (ref[m, 3] * ref[m, 4]).sum() / ref[m, 3].sum())
            assert st["min"][k] == ref[m, 4].min()
            assert st["max"][k] == ref[m, 4].max()


def test_zonal_stats_n_id_keeps_empty_zones(single):
    cells = {"row": np.array([0]), "col_start": np.array([0]),
             "col_end": np.array([4]), "id": np.array([1]), "w": np.ones(1)}
    st = pixtract.zonal_stats(single, cells, n_id=4)
    assert st["count"].tolist() == [0, 4, 0, 0]
    assert np.isnan(st["mean"][[0, 2, 3]]).all()


def test_vrt_is_planned_on_source_blocks(mosaic, mosaic_overlap_nodata):
    s = pixtract.plan_sources(mosaic)
    assert s.kind == "vrt" and len(s) == 6
    assert set(zip(s.block_x.tolist(), s.block_y.tolist())) == {(64, 64), (128, 32), (256, 256)}
    s2 = pixtract.plan_sources(mosaic_overlap_nodata)
    assert s2.kind == "single" and "NODATA" in s2.note


def test_cost_counts_blocks(single):
    s = pixtract.plan_sources(single)
    # one run across the whole first row: ceil(1000 / 128) = 8 blocks
    cells = {"row": [0], "col_start": [0], "col_end": [1000], "id": [0]}
    c = pixtract.cost(pixtract.plan_cells(cells, s), bytes=True)
    assert c["blocks"] == 8 and c["cells"] == 1000 and c["segments"] == 8
    assert c["bytes"] > 0


def test_cells_csv_roundtrip(tmp):
    cells = random_runs(50, 50, n=20, n_id=3, seed=13)
    p = str(tmp / "cells.csv")
    pixtract.write_cells(p, cells)
    back = pixtract.read_cells(p)
    for k in ("row", "col_start", "col_end", "id"):
        np.testing.assert_array_equal(back[k], cells[k])
    np.testing.assert_array_equal(back["w"], cells["w"])
