"""Point extraction on a synthetic 3 x 2 mosaic (no network).

Run from the repo root:  python examples/points_synthetic.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "python"),
                os.path.join(os.path.dirname(__file__), "..", "tests")]
import pixtract  # noqa: E402
from synthetic import make_mosaic  # noqa: E402

vrt = make_mosaic(tempfile.mkdtemp(), layout=(3, 2), size=(1200, 800),
                  blocks=[(256, 256), (512, 128)])

# 200k points in three clusters, plus a few outside the mosaic
rng = np.random.default_rng(1)
k = rng.integers(0, 3, 200_000)
xs = np.array([102.0, 110.0, 131.0])[k] + rng.normal(0, 0.4, k.size)
ys = np.array([-33.0, -40.0, -45.0])[k] + rng.normal(0, 0.4, k.size)

# the plan, before any pixel I/O
src = pixtract.plan_sources(vrt)
plan = pixtract.plan_cells(pixtract.cells_from_points(xs, ys, src.grid), src)
print(f"{len(src)} sources ({src.kind}); cost: {pixtract.cost(plan)}")

vals = pixtract.extract_points(vrt, xs, ys, max_workers=4)
print(f"{np.isfinite(vals).sum()} of {vals.size} points have values; first five: {vals[:5]}")
