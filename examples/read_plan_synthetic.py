"""Read planning on a synthetic 3 x 2 mosaic (no network).

Inspects the mosaic, then for three queries (clustered points, scattered
points, a dense block of runs) shows which sources each implicates and which
read windows plan_reads() picks at a few memory budgets. Values are checked
against plain block-by-block reads.

Run from the repo root:  python examples/read_plan_synthetic.py
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


def show(d, cols):
    """Print a dict-of-arrays table."""
    w = {c: max(len(c), *(len(f"{v:.3g}" if isinstance(v, float) else str(v))
                          for v in d[c])) for c in cols}
    print("  " + " ".join(c.rjust(w[c]) for c in cols))
    for i in range(len(d[cols[0]])):
        print("  " + " ".join((f"{d[c][i]:.3g}" if isinstance(d[c][i], float)
                               else str(d[c][i])).rjust(w[c]) for c in cols))


info = pixtract.inspect_source(vrt)
print(f"{info['driver']} {info['ncol']} x {info['nrow']} {info['dtype']}, "
      f"{len(info['sources'])} sources ({info['kind']}), "
      f"{len(info['overviews'])} overviews")
t = info["table"]
t["file"] = [os.path.basename(p) for p in t["path"]]
show(t, ["src", "file", "xoff", "yoff", "block_x", "block_y", "nblocks_x",
         "nblocks_y", "xmin", "ymax"])

src = info["sources"]
rng = np.random.default_rng(1)
# 50k points in three tight clusters
k = rng.integers(0, 3, 50_000)
xs = np.array([102.0, 110.0, 131.0])[k] + rng.normal(0, 0.1, k.size)
ys = np.array([-33.0, -40.0, -45.0])[k] + rng.normal(0, 0.1, k.size)
clustered = pixtract.cells_from_points(xs, ys, src.grid)
# 40 points anywhere
scattered = pixtract.cells_from_points(rng.uniform(100, 136, 40),
                                       rng.uniform(-46, -30, 40), src.grid)
# a 600-row block of runs, 1300 cells wide (a polygon, as runs)
rows = np.arange(100, 700)
dense = {"row": rows, "col_start": np.full(rows.size, 200),
         "col_end": np.full(rows.size, 1500), "id": np.zeros(rows.size, np.int64),
         "w": np.ones(rows.size)}

for name, cells in [("clustered points", clustered), ("scattered points", scattered),
                    ("dense runs", dense)]:
    plan = pixtract.plan_cells(cells, src)
    print(f"\n== {name}: {pixtract.cost(plan)}")
    u = pixtract.source_usage(plan)
    show(u, ["src", "cells", "runs", "blocks", "span", "occupancy"])
    ref = pixtract.execute(plan, pixtract.Stats())
    for mem in (2**18, 2**22, 2**28):
        p = pixtract.plan_reads(plan, mem=mem)
        c = pixtract.cost(p)
        m = p.extra["meta"]
        shapes = ", ".join(f"{s}:{kx}x{ky}" for s, kx, ky in zip(m["src"], m["kx"], m["ky"]))
        print(f"  mem {mem / 2**20:6.2f} MiB: {c['reads']:4d} reads, "
              f"{c['read_bytes'] / 2**20:7.2f} MiB (meta-tile per source {shapes})")
        got = pixtract.execute(p, pixtract.Stats())
        assert np.array_equal(got["count"], ref["count"])
        assert np.allclose(got["sum"], ref["sum"], rtol=1e-12)
print("\nread windows give the same values as block reads")
