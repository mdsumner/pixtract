"""Read planning on Copernicus GLO-30 over Switzerland (network needed).

A VRT of 18 GLO-30 COGs (N45-N47 x E005-E010, 3600 x 3600 each, 1024 x 1024
blocks) over /vsicurl. Two point queries: a GPS-track-like cluster near
Zermatt and a handful of points spread over the whole mosaic. For each, which
sources it implicates, the read windows plan_reads() picks, and the time to
run it block by block and with those windows.

Run from the repo root:  python examples/read_plan_glo30.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pixtract  # noqa: E402
from osgeo import gdal  # noqa: E402

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
data = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data, exist_ok=True)

vrt = os.path.join(data, "ch_glo30.vrt")
if not os.path.exists(vrt):
    url = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{n}_{e}_DEM/Copernicus_DSM_COG_10_{n}_{e}_DEM.tif")
    tiles = [url.format(n=f"N{a:02d}_00", e=f"E{b:03d}_00")
             for a in range(45, 48) for b in range(5, 11)]
    gdal.BuildVRT(vrt, tiles).FlushCache()

info = pixtract.inspect_source(vrt)
t = info["table"]
print(f"{info['driver']} {info['ncol']} x {info['nrow']} {info['dtype']}, "
      f"{len(t['path'])} sources, blocks {t['block_x'][0]} x {t['block_y'][0]} "
      f"({t['blocks'].sum()} in all), remote: {bool(t['remote'].all())}")
src = info["sources"]

rng = np.random.default_rng(1)
queries = {
    # 20k points along a wandering track near Zermatt
    "track near Zermatt": (7.75 + np.cumsum(rng.normal(0, 0.002, 20_000)),
                           46.02 + np.cumsum(rng.normal(0, 0.002, 20_000))),
    # 12 points anywhere in the mosaic
    "12 scattered points": (rng.uniform(5, 11, 12), rng.uniform(45, 48, 12)),
}

for name, (xs, ys) in queries.items():
    plan = pixtract.plan_cells(pixtract.cells_from_points(xs, ys, src.grid), src)
    u = pixtract.source_usage(plan)
    print(f"\n== {name}: {int(u['cells'].sum())} cells in {len(u['src'])} of "
          f"{len(t['path'])} sources, {int(u['blocks'].sum())} blocks "
          f"(occupancy {', '.join(f'{o:.2f}' for o in u['occupancy'])})")
    for mem in (2**24, 2**26, 2**28):
        p = pixtract.plan_reads(plan, mem=mem, max_workers=4)
        c = pixtract.cost(p)
        print(f"  mem {mem / 2**20:4.0f} MiB: {c['reads']:3d} reads, "
              f"{c['read_bytes'] / 2**20:7.1f} MiB decoded "
              f"(block reads: {c['blocks']}, {u['block_bytes'].sum() / 2**20:.1f} MiB)")
    p = pixtract.plan_reads(plan, mem=2**28, max_workers=4)
    for label, pl in (("block reads", plan), ("windows", p)):
        gdal.VSICurlClearCache()
        t0 = time.time()
        v = pixtract.execute(pl, pixtract.Place(), max_workers=4)
        print(f"  {label:12s} {time.time() - t0:6.1f} s, "
              f"mean elevation {np.nanmean(v):.0f} m")
