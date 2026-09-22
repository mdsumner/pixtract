"""Zonal statistics from controlledburn runs on a synthetic mosaic (no network).

Needs the controlledburn Python package (hypertidy/controlledburn, python/)
and shapely. Run from the repo root:  python examples/zonal_synthetic.py
"""
import os
import sys
import tempfile

import numpy as np
import shapely
import controlledburn as cb

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "python"),
                os.path.join(os.path.dirname(__file__), "..", "tests")]
import pixtract  # noqa: E402
from synthetic import make_mosaic  # noqa: E402

vrt = make_mosaic(tempfile.mkdtemp(), layout=(3, 2), size=(1200, 800),
                  blocks=[(256, 256), (512, 128)])
src = pixtract.plan_sources(vrt)

polys = [shapely.Point(104, -34).buffer(1.5), shapely.Point(115, -41).buffer(3),
         shapely.box(128, -46, 133, -44)]
# burn on the raster's own grid; tables are 0-based, half-open, id = position
r = cb.burn(polys, **pixtract.burn_args(src.grid))
cells = pixtract.cells_from_burn(r)          # runs w = 1, edges w = fraction
print("cost:", pixtract.cost(pixtract.plan_cells(cells, src)))

st = pixtract.zonal_stats(vrt, cells)
for i in range(len(polys)):
    print(f"polygon {i}: {st['count'][i]} cells, weight {st['weight'][i]:.1f}, "
          f"mean {st['mean'][i]:.1f}, min {st['min'][i]:.0f}, max {st['max'][i]:.0f}")
