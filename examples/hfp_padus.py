"""Human footprint (HFP-100 2021) mean per PAD-US 3.0 protected area.

The workload from the Pangeo thread "Advice for scalable raster-vector
extraction" (https://discourse.pangeo.io/t/advice-for-scalable-raster-vector-extraction/4129):
the coverage-weighted mean of a 100 m global COG (360802 x 176500 UInt16,
Mollweide, 512 blocks, ZSTD, 14 GB) under each of the 438,113 polygons of
PAD-US 3.0's Combined_DOD_TRIB_Fee_Designation_Easement layer. Published
there: exactextract ~10 min on 1 core with 3-4 GB (local files); rasterstats +
joblib under 3 min on 24 cores; xarray/geocube and xvec ran out of memory.

The means are checked against pad_raster_means.csv (column
hfp_2021_100m_v1-2_cog), published beside the polygons on source.coop; its
row i is the layer's i-th feature (FID i + 1).

Result (2026-09-23, 4 cores, 15 GB, COG read remotely over /vsicurl):
611 s in all (burn 145 s, plan + read + reduce 455 s), 3.4 billion cells
(runs + edges), 19,161 blocks. About 2.9 billion of those cells and half the
time come from a dozen polygons that cross the antimeridian and wrap around
the globe once projected to Mollweide (the published means use the same
geometry). Of the 425,483 polygons whose burned coverage sums to their
area, 99.95% of means equal the published ones to float32 precision; the
other 220 are slivers of less than about one cell that differ by at most
0.5%. The remaining 12,630 polygons have holes and hit a controlledburn
bug that drops every cell a hole touches
(exploration/2026-09-23_burn_hole_cells.py).

Steps:
  1. polygons -> raster CRS (ogr2ogr, cached as FlatGeobuf, FID order kept)
  2. polygons sorted by position, burned in batches with controlledburn
     (coverage fractions), one zonal_stats() per batch
  3. compare with the published means

Inputs (paths or URLs; source.coop is read through its S3 endpoint):
  --gpkg  PADUS3_0Geopackage.gpkg (4.2 GB; downloading it first is faster)
  --cog   hfp_2021_100m_v1-2_cog.tif, read in place over /vsicurl
  --csv   pad_raster_means.csv

Run from the repo root:  python examples/hfp_padus.py --gpkg path/to/PADUS3_0Geopackage.gpkg
"""
import argparse
import os
import sys
import time

import numpy as np
import controlledburn as cb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pixtract  # noqa: E402
from osgeo import gdal, ogr  # noqa: E402

gdal.UseExceptions()
ogr.UseExceptions()

COOP = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
LAYER = "PADUS3_0Combined_DOD_TRIB_Fee_Designation_Easement"

ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", default="/vsicurl/" + COOP + "cboettig/pad-us-3/PADUS3_0Geopackage.gpkg")
ap.add_argument("--cog", default="/vsicurl/" + COOP + "vizzuality/hfp-100/hfp_2021_100m_v1-2_cog.tif")
ap.add_argument("--csv", default=COOP + "cboettig/pad-us-3/pad_raster_means.csv")
ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data"))
ap.add_argument("--batch", type=int, default=20000, help="polygons per burn")
ap.add_argument("--mem", type=float, default=512, help="read budget, MiB")
ap.add_argument("--workers", type=int, default=os.cpu_count())
ap.add_argument("--limit", type=int, default=None, help="first n polygons only")
a = ap.parse_args()
os.makedirs(a.data, exist_ok=True)

t_all = time.time()
src = pixtract.plan_sources(a.cog)
grid = src.grid

# 1. polygons in the raster's CRS, FID order
fgb = os.path.join(a.data, "padus3_combined_moll.fgb")
if not os.path.exists(fgb):
    t0 = time.time()
    gdal.VectorTranslate(fgb, a.gpkg, layers=[LAYER], format="FlatGeobuf",
                         dstSRS=gdal.Open(a.cog).GetSpatialRef(),
                         layerCreationOptions=["SPATIAL_INDEX=NO"])
    print(f"reprojected {LAYER} in {time.time() - t0:.0f} s")
t0 = time.time()
vds = ogr.Open(fgb)
lyr = vds.GetLayer(0)
wkb, cx, cy, area = [], [], [], []
for f in lyr:
    g = f.GetGeometryRef()
    if g is None:
        wkb.append(None)
        cx.append(np.nan)
        cy.append(np.nan)
        area.append(0.0)
        continue
    e = g.GetEnvelope()
    wkb.append(bytes(g.ExportToWkb()))
    area.append(g.GetArea() / (grid.gt[1] * -grid.gt[5]))
    cx.append((e[0] + e[1]) / 2)
    cy.append((e[2] + e[3]) / 2)
    if a.limit and len(wkb) >= a.limit:
        break
n = len(wkb)
print(f"{n:,} polygons read in {time.time() - t0:.0f} s")

# 2. batches of nearby polygons, so each batch reads a compact set of blocks
row, col = pixtract.rowcol_from_xy(grid.gt, grid.dimension, np.array(cx), np.array(cy))
key = np.where(row >= 0, (row // 512) * 1_000_000 + (col // 512), -1)
order = np.argsort(key, kind="stable")
order = order[[wkb[i] is not None for i in order]]
args = pixtract.burn_args(grid)
out = {k: np.full(n, np.nan) for k in ("mean", "weight")}
burned = np.zeros(n)  # sum of coverage per polygon, from the burn alone
count = np.zeros(n, np.int64)
t_burn = t_read = 0.0
blocks = cells_total = 0
for b0 in range(0, order.size, a.batch):
    idx = order[b0:b0 + a.batch]
    t0 = time.time()
    cells = pixtract.cells_from_burn(cb.burn([wkb[i] for i in idx], **args))
    t_burn += time.time() - t0
    burned[idx] = np.bincount(cells["id"], minlength=idx.size,
                              weights=cells["w"] * (cells["col_end"] - cells["col_start"]))
    t0 = time.time()
    plan = pixtract.plan_reads(pixtract.plan_cells(cells, src), mem=int(a.mem * 2**20),
                               max_workers=a.workers)
    plan.n_id = idx.size
    c = pixtract.cost(plan)
    blocks += c["blocks"]
    cells_total += c["cells"]
    st = pixtract.execute(plan, pixtract.Stats(), max_workers=a.workers)
    t_read += time.time() - t0
    out["mean"][idx] = st["mean"]
    out["weight"][idx] = st["weight"]
    count[idx] = st["count"]
    print(f"  batch {b0 // a.batch + 1}: {idx.size} polygons, {c['cells']:,} cells, "
          f"{c['blocks']} blocks, {c.get('reads', 0)} reads, "
          f"burn {t_burn:.0f} s, reads {t_read:.0f} s", flush=True)

print(f"\n{n:,} polygons, {cells_total:,} cells (runs + edges), {blocks:,} blocks read")
print(f"burn {t_burn:.0f} s, plan + read + reduce {t_read:.0f} s, "
      f"total {time.time() - t_all:.0f} s, {a.workers} workers")
np.savez(os.path.join(a.data, "hfp_padus.npz"), mean=out["mean"], weight=out["weight"],
         count=count, burned=burned, area=np.array(area))

# 3. compare with the published means
import csv  # noqa: E402
import io  # noqa: E402
import urllib.request  # noqa: E402

if a.csv.startswith("http"):
    text = urllib.request.urlopen(a.csv).read().decode()
else:
    text = open(a.csv).read()
rd = csv.reader(io.StringIO(text))
hdr = next(rd)
j = hdr.index("hfp_2021_100m_v1-2_cog")
pub = np.array([float(r[j]) if r[j] else np.nan for r in rd])[:n]
print(f"\npublished means: {pub.size:,}; empty here only: "
      f"{(np.isnan(out['mean']) & ~np.isnan(pub)).sum():,}, empty there only: "
      f"{(~np.isnan(out['mean']) & np.isnan(pub)).sum():,}")
# The CSV holds float32 means, so "equal" is to float32 precision.
# Polygons whose burned coverage falls short of their area hit a
# controlledburn bug (cells touched by a small hole are dropped).
area = np.array(area)
full = np.abs(burned - area) <= 1e-6 * np.maximum(area, 1) + 1e-3
for label, sel in (("coverage sums to area", full), ("coverage short of area", ~full)):
    both = sel & ~np.isnan(pub) & ~np.isnan(out["mean"])
    d = np.abs(out["mean"][both] - pub[both])
    eq = d <= 4e-7 * np.maximum(np.abs(pub[both]), 1)
    print(f"{label}: {sel.sum():,} polygons, {both.sum():,} with both means, "
          f"{eq.mean():.3%} equal, max |difference| {d.max() if d.size else 0:.4g} (score x 1000)")
