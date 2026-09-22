"""Grid logic: the edge cases in grid_cases.csv, shared with tests/test_r.R.

Expected row/col are 0-based, -1 for outside; R adds 1 and uses NA.
"""

import csv
import os

import numpy as np
import pytest

from pixtract import (rowcol_from_xy, cell_from_row_col, gt_dim_to_extent,
                      extent_dim_to_gt)

CASES = os.path.join(os.path.dirname(__file__), "grid_cases.csv")


def _cases():
    with open(CASES) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c['grid']}-{c['x']}-{c['y']}")
def test_rowcol_from_xy_cases(case):
    gt = tuple(float(case[f"gt{i}"]) for i in range(6))
    dim = (int(case["ncol"]), int(case["nrow"]))
    row, col = rowcol_from_xy(gt, dim, [float(case["x"])], [float(case["y"])])
    assert (row[0], col[0]) == (int(case["row"]), int(case["col"]))


def test_rowcol_from_xy_vectorised():
    c = _cases()
    for name in {r["grid"] for r in c}:
        s = [r for r in c if r["grid"] == name]
        gt = tuple(float(s[0][f"gt{i}"]) for i in range(6))
        dim = (int(s[0]["ncol"]), int(s[0]["nrow"]))
        row, col = rowcol_from_xy(gt, dim, [float(r["x"]) for r in s],
                                  [float(r["y"]) for r in s])
        np.testing.assert_array_equal(row, [int(r["row"]) for r in s])
        np.testing.assert_array_equal(col, [int(r["col"]) for r in s])


def test_cell_from_row_col():
    np.testing.assert_array_equal(
        cell_from_row_col((10, 5), [0, 0, 4, 2, 5, 0], [0, 9, 9, 3, 0, -1]),
        [0, 9, 49, 23, -1, -1])


def test_extent_gt_round_trip():
    gt = (100.0, 0.01, 0.0, -30.0, 0.0, -0.01)
    e = gt_dim_to_extent(gt, (1000, 700))
    np.testing.assert_allclose(e, (100.0, 110.0, -37.0, -30.0))
    np.testing.assert_allclose(extent_dim_to_gt(e, (1000, 700)), gt)
    with pytest.raises(ValueError):
        gt_dim_to_extent((0, 1, 0.5, 10, 0.25, -1), (10, 5))
