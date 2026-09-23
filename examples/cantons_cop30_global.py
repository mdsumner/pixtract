"""Swiss cantons on a global Copernicus GLO-30 VRT (network needed).

The same 26 cantons as cantons_glo90.py, but the DEM is one global VRT of
26,475 tiles instead of a VRT built from the 18 tiles we know we need:

    /vsicurl/https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh.vrt

Steps, with timings:
  1. scan_vrt(): read the VRT's XML text and list every source (no tile is
     opened, and GDAL does not open the VRT)
  2. burn the cantons on the VRT's grid (1296001 x 626401 cells; the burn is
     sparse, so the size of the grid barely matters)
  3. plan_sources(window=cells_window(cells)): keep only the sources the
     burn's bounding window touches, check just those are 1:1 copies
  4. plan cells and read windows, then read and reduce

--aws reads the pixels from the Copernicus COGs on AWS instead of the
OpenTopography tiles the VRT points at. The two share origin and cell size;
an OpenTopography tile is 3601 x 3601 (its last row and column repeat the
neighbour's first), an AWS tile 3600 x 3600, so each source is cut to
3600 x 3600 and the neighbouring tile supplies the shared edge. Use it where
opentopography.s3.sdsc.edu is out of reach.

Run from the repo root:
  python examples/cantons_cop30_global.py [VRT] [--aws]
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
import controlledburn as cb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pixtract  # noqa: E402
from osgeo import gdal  # noqa: E402

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.vrt")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
aws = "--aws" in sys.argv
vrt = args[0] if args else \
    "/vsicurl/https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh.vrt"
gpkg = os.path.join(os.path.dirname(__file__), "data", "cantons.gpkg")
if not os.path.exists(gpkg):
    sys.exit("examples/data/cantons.gpkg not found: run examples/cantons_glo90.py once")

gds = gdal.OpenEx(gpkg)
lyr = gds.GetLayer(0)
feats = [(f.GetField("name"), bytes(f.GetGeometryRef().ExportToWkb())) for f in lyr]
names, wkb = [f[0] for f in feats], [f[1] for f in feats]


def tick(label, t0):
    print(f"  {label:44s} {time.time() - t0:7.2f} s")
    return time.time()


print(f"VRT: {vrt}")
t0 = time.time()
scan = pixtract.scan_vrt(vrt)
t0 = tick(f"scan_vrt ({len(scan['table']['path'])} sources)", t0)
tb = scan["table"]
print(f"  grid {scan['ncol']} x {scan['nrow']}, {scan['dtype']}, "
      f"resampled sources: {int((tb['src_xsize'] != np.round(tb['dst_xsize'])).sum())}, "
      f"non-integer DstRect offsets: {int((tb['dst_xoff'] != np.round(tb['dst_xoff'])).sum())}")

grid = pixtract.Grid(scan["gt"], scan["ncol"], scan["nrow"])
approx = pixtract.cells_from_burn(cb.burn(wkb, mode="approx", **pixtract.burn_args(grid)))
cover = pixtract.cells_from_burn(cb.burn(wkb, **pixtract.burn_args(grid)))
t0 = tick("burn, both modes, on the global grid", t0)
win = pixtract.cells_window(approx)
print(f"  {len(approx['row']):,} runs, {int((approx['col_end'] - approx['col_start']).sum()):,} "
      f"cells (cell-centre), window {win}")

src = pixtract.plan_sources(vrt, window=win)
t0 = tick("plan_sources(window=)", t0)
print(f"  kind {src.kind}: {src.note}")
for p in src.path:
    print("   ", os.path.basename(p))

plan = pixtract.plan_cells(approx, src)
t0 = tick("plan_cells", t0)
print("  cost:", pixtract.cost(plan))

if aws:
    url = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{k}/{k}.tif")
    paths = [url.format(k=os.path.basename(p)[:-4].replace("DSM_10", "DSM_COG_10"))
             for p in src.path]
    info = gdal.Info(paths[0], format="json")
    bx, by = info["bands"][0]["block"]
    n = len(paths)
    src = replace(src, path=paths,
                  xsize=np.minimum(src.xsize, 3600 - src.src_xoff),
                  ysize=np.minimum(src.ysize, 3600 - src.src_yoff),
                  file_xsize=np.full(n, 3600), file_ysize=np.full(n, 3600),
                  block_x=np.full(n, bx), block_y=np.full(n, by))
    print(f"  reading from AWS COGs instead ({bx} x {by} blocks)")

rows = []
t_read = time.time()
for label, cells in (("cell-centre", approx), ("coverage", cover)):
    t1 = time.time()
    p = pixtract.plan_reads(pixtract.plan_cells(cells, src), max_workers=8)
    c = pixtract.cost(p)
    rows.append(pixtract.execute(p, pixtract.Stats(), max_workers=8))
    print(f"  {label:12s} {c['reads']} reads, {c['read_bytes'] / 2**20:.0f} MiB decoded, "
          f"{c['uncovered_cells']} uncovered cells, {time.time() - t1:.1f} s")
tick("planned reads + stats (both modes)", t_read)

st, sc = rows
print(f"\ncells under cantons (cell-centre): {int(st['count'].sum()):,}  "
      f"(GLO-90: 7,006,027; x 9 = {9 * 7006027:,})")
print(f"{'canton':34s} {'cells':>9s} {'mean':>8s} {'cov-mean':>8s}")
for i in np.argsort(-st["mean"]):
    print(f"{names[i]:34s} {st['count'][i]:9d} {st['mean'][i]:8.1f} {sc['mean'][i]:8.1f}")
