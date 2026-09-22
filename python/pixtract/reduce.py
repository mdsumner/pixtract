"""
Reducers: what to do with the values of each block's cells.

A reducer has start(plan), add(cells) and result(). `cells` is a dict of
flat arrays for one block: id, w, run, row, col, value (NaN for nodata).
"""

from __future__ import annotations

import numpy as np

__all__ = ["Place", "Cells", "Stats"]


class Place:
    """One value per id (points): out[id] = value. Ids with no cell are NaN."""

    def start(self, plan):
        self.out = np.full(plan.n_id, np.nan)

    def add(self, cells):
        self.out[cells["id"]] = cells["value"]

    def result(self):
        return self.out


class Cells:
    """Every cell with its value, in plan order: row, col, id, w, run, value."""

    def start(self, plan):
        self.parts = []

    def add(self, cells):
        self.parts.append(cells)

    def result(self):
        keys = ("row", "col", "id", "w", "run", "value")
        if not self.parts:
            return {k: np.zeros(0) for k in keys}
        return {k: np.concatenate([p[k] for p in self.parts]) for k in keys}


class Stats:
    """Grouped weighted statistics by id, NaN values excluded.

    count: number of valid cells; weight: sum of w over valid cells;
    sum: sum of w * value; mean: sum / weight; min, max: unweighted.
    With coverage-fraction weights, mean is the coverage-weighted mean.
    """

    def start(self, plan):
        n = plan.n_id
        self.count = np.zeros(n, np.int64)
        self.weight = np.zeros(n)
        self.sum = np.zeros(n)
        self.min = np.full(n, np.inf)
        self.max = np.full(n, -np.inf)

    def add(self, cells):
        v = cells["value"]
        ok = ~np.isnan(v)
        if not ok.any():
            return
        i, w, v = cells["id"][ok], cells["w"][ok], v[ok]
        n = self.count.size
        self.count += np.bincount(i, minlength=n)
        self.weight += np.bincount(i, weights=w, minlength=n)
        self.sum += np.bincount(i, weights=w * v, minlength=n)
        np.fmin.at(self.min, i, v)
        np.fmax.at(self.max, i, v)

    def result(self):
        empty = self.count == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.sum / self.weight
        mn, mx = self.min.copy(), self.max.copy()
        mn[empty] = np.nan
        mx[empty] = np.nan
        mean[empty] = np.nan
        return {
            "id": np.arange(self.count.size),
            "count": self.count,
            "weight": self.weight,
            "sum": self.sum,
            "mean": mean,
            "min": mn,
            "max": mx,
        }
