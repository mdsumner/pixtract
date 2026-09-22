"""Swiss cantons on Copernicus GLO-90: zonal stats from S3 (network needed).

Mirrors the Apache Sedona "group by, but for pixels" benchmark: 26 cantons
(Overture 2026-08-19.0) over the 18 GLO-90 tiles N45-N47 x E005-E010.
Sedona reports 7,006,027 cells under cantons (cell-centre rule).

Needs the controlledburn Python package and pyarrow.
Run from the repo root:  python examples/cantons_glo90.py
"""
import os
import sys
import time

import numpy as np
import controlledburn as cb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pixtract  # noqa: E402
from osgeo import gdal  # noqa: E402

gdal.UseExceptions()
data = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data, exist_ok=True)


def cantons():
    """(names, wkb) for the 26 cantons, cached as a GeoPackage R can read too."""
    gpkg = os.path.join(data, "cantons.gpkg")
    if not os.path.exists(gpkg):
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.fs as fs
        s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
        d = ds.dataset("overturemaps-us-west-2/release/2026-08-19.0/theme=divisions/"
                       "type=division_area/", filesystem=s3, format="parquet")
        flt = ((pc.field("country") == "CH") & (pc.field("subtype") == "region")
               & (pc.field("bbox", "xmin") >= 5.5) & (pc.field("bbox", "xmin") <= 10.7)
               & (pc.field("bbox", "ymin") >= 45.7) & (pc.field("bbox", "ymin") <= 47.9))
        tb = d.to_table(columns={"name": pc.field("names", "primary"),
                                 "geometry": pc.field("geometry")}, filter=flt)
        from osgeo import ogr, osr
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        out = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg)
        lyr = out.CreateLayer("cantons", srs, ogr.wkbMultiPolygon)
        lyr.CreateField(ogr.FieldDefn("name", ogr.OFTString))
        for n, g in zip(tb.column("name").to_pylist(), tb.column("geometry").to_pylist()):
            f = ogr.Feature(lyr.GetLayerDefn())
            f.SetField("name", n)
            f.SetGeometry(ogr.ForceToMultiPolygon(ogr.CreateGeometryFromWkb(bytes(g))))
            lyr.CreateFeature(f)
        out = None
    src = gdal.OpenEx(gpkg)
    lyr = src.GetLayer(0)
    feats = [(f.GetField("name"), bytes(f.GetGeometryRef().ExportToWkb())) for f in lyr]
    return [f[0] for f in feats], [f[1] for f in feats]


def dem_vrt():
    vrt = os.path.join(data, "ch_dem.vrt")
    if not os.path.exists(vrt):
        url = ("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/"
               "Copernicus_DSM_COG_30_{n}_{e}_DEM/Copernicus_DSM_COG_30_{n}_{e}_DEM.tif")
        tiles = [url.format(n=f"N{a:02d}_00", e=f"E{b:03d}_00")
                 for a in range(45, 48) for b in range(5, 11)]
        gdal.BuildVRT(vrt, tiles).FlushCache()
    return vrt


names, wkb = cantons()
vrt = dem_vrt()
src = pixtract.plan_sources(vrt)
args = pixtract.burn_args(src.grid)          # 7200 x 3600, half-pixel offset origin

t0 = time.time()
approx = pixtract.cells_from_burn(cb.burn(wkb, mode="approx", **args))
cover = pixtract.cells_from_burn(cb.burn(wkb, **args))
t_burn = time.time() - t0

plan = pixtract.plan_cells(approx, src)
print("cost:", pixtract.cost(plan, bytes=True))

t0 = time.time()
st = pixtract.zonal_stats(vrt, approx, max_workers=8)
sc = pixtract.zonal_stats(vrt, cover, max_workers=8)
t_stats = time.time() - t0

print(f"cells under cantons (cell-centre): {int(st['count'].sum()):,}  (Sedona: 7,006,027)")
print(f"burn {t_burn:.2f} s, planned reads + stats (both modes) {t_stats:.1f} s")
print(f"{'canton':34s} {'cells':>8s} {'mean':>8s} {'cov-mean':>8s}")
for i in np.argsort(-st["mean"]):
    print(f"{names[i]:34s} {st['count'][i]:8d} {st['mean'][i]:8.1f} {sc['mean'][i]:8.1f}")
