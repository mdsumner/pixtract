"""Wherobots "Raster Data Analysis With Spatial SQL": WorldClim and night lights.

Reproduces the published numbers in
https://wherobots.com/blog/raster-data-analysis-spatial-sql-wherobots-apache-sedona/
with the same public inputs (s3://wherobots-examples/data/examples/):

- point values at (-113.9940, 46.8721): WorldClim 2.1 10 arc-minute monthly
  precipitation (12 files) and DMSP-OLS 1993 stable lights
- yearly precipitation per Natural Earth 10m country: the sum over months of
  the country's mean monthly precipitation (Sedona: RS_ZonalStats 'avg' with
  allTouched = true), top 20 published
- mean 1993 night lights per US county (Natural Earth 10m admin-2), first 12
  published

Cell rules compared for countries:
  centre   - cells whose centre is inside (controlledburn mode="approx")
  touched  - every cell the polygon covers at all (coverage mode, w = 1)
  coverage - coverage-fraction weighted mean
  corner   - cells whose lower-right corner is inside: a centre burn on the
             grid shifted half a cell right and down. Not a rule to use; it
             is here because it reproduces several published values exactly,
             so the gaps come from Sedona's rasterization, not the reads.

Needs the controlledburn Python package. Network: reads the rasters and
shapefiles over /vsicurl from the wherobots-examples bucket.
Run from the repo root:  python examples/wherobots_worldclim.py
"""
import os
import sys
import time

import numpy as np
import controlledburn as cb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pixtract  # noqa: E402
from osgeo import gdal, ogr  # noqa: E402

gdal.UseExceptions()
BUCKET = "/vsicurl/https://wherobots-examples.s3.us-west-2.amazonaws.com/data/examples/"
PREC = [BUCKET + f"world_clim/wc2.1_10m_prec/wc2.1_10m_prec_{m:02d}.tif" for m in range(1, 13)]
LIGHTS = BUCKET + "DMSP_OLS/F101993.v4b.global.stable_lights.avg_vis.tif"
MEM = 64 << 20

# Published values (Wherobots blog, copied verbatim).
PUB_POINT_PREC = [43, 27, 31, 31, 50, 49, 27, 31, 31, 29, 35, 40]
PUB_POINT_LIGHTS = 63.0
PUB_COUNTRY = {
    "Micronesia": 4937.5, "Palau": 3526.0, "Samoa": 3378.75,
    "Brunei": 3345.6875, "Solomon Is.": 3234.474358974358,
    "Saint Helena": 3111.0, "Wallis and Futuna Is.": 3008.0,
    "Papua New Guinea": 2988.5508720930234, "Vanuatu": 2934.628571428572,
    "Malaysia": 2881.220250521921, "Costa Rica": 2831.0000000000005,
    "Faeroe Is.": 2807.0000000000005, "Indonesia": 2733.146385760997,
    "Sierra Leone": 2717.5348837209303, "Panama": 2675.306306306307,
    "Liberia": 2656.3900709219856, "Colombia": 2631.9744668068433,
    "Fiji": 2560.3461538461534, "S\u00e3o Tom\u00e9 and Principe": 2555.6666666666665,
    "Philippines": 2525.6388261851025,
}
PUB_COUNTY = {
    "US01003": 6.563388510224063, "US01019": 5.303058346553825,
    "US01021": 7.112594570538514, "US01025": 2.5223492723492806,
    "US01031": 9.564617731305812, "US01033": 13.433770014555993,
    "US01037": 8.240051020408188, "US01039": 2.301078582434537,
    "US01041": 0.9495387954422182, "US01045": 8.3112128146453,
    "US01047": 2.008212672420753, "US01053": 4.339487179487201,
}


def layer(name, field):
    ds = ogr.Open(BUCKET + f"natural_earth/{name}/{name}.shp")
    feats = [(f.GetField(field), bytes(f.GetGeometryRef().ExportToWkb()))
             for f in ds.GetLayer(0)]
    return [f[0] for f in feats], [f[1] for f in feats]


def burn(wkb, grid, mode, shift=(0.0, 0.0)):
    """Run table for polygons on `grid`; shift moves the grid by cells (x, y)."""
    args = pixtract.burn_args(grid)
    xmin, xmax, ymin, ymax = args["extent"]
    dx, dy = shift[0] * grid.gt[1], shift[1] * -grid.gt[5]
    args["extent"] = (xmin + dx, xmax + dx, ymin + dy, ymax + dy)
    if mode == "approx":
        return pixtract.cells_from_burn(cb.burn(wkb, mode="approx", **args))
    cells = pixtract.cells_from_burn(cb.burn(wkb, **args))
    if mode == "touched":
        cells["w"] = np.ones_like(cells["w"])
    return cells


# ---- points ------------------------------------------------------------
x, y = np.array([-113.9940]), np.array([46.8721])
t0 = time.time()
prec = [pixtract.extract_points(p, x, y)[0] for p in PREC]
lights = pixtract.extract_points(LIGHTS, x, y)[0]
print(f"point (-113.9940, 46.8721), {time.time() - t0:.1f} s")
print("  precipitation by month:", [int(v) for v in prec])
print("  published:             ", PUB_POINT_PREC)
print(f"  night lights {lights:.1f} (published {PUB_POINT_LIGHTS})")

# ---- countries ---------------------------------------------------------
names, wkb = layer("ne_10m_admin_0_countries", "NAME")
grid = pixtract.plan_sources(PREC[0]).grid
rules = {
    "centre": burn(wkb, grid, "approx"),
    "touched": burn(wkb, grid, "touched"),
    "coverage": burn(wkb, grid, "coverage"),
    "corner": burn(wkb, grid, "approx", shift=(0.5, -0.5)),
}
yearly = {}
t0 = time.time()
for rule, cells in rules.items():
    tot = np.zeros(len(names))
    for p in PREC:
        tot += pixtract.zonal_stats(p, cells, n_id=len(names), mem=MEM,
                                    max_workers=8)["mean"]
    yearly[rule] = tot
print(f"\n{len(names)} countries x 12 months x {len(rules)} rules, "
      f"{time.time() - t0:.1f} s")
print(f"{'country':24s} {'published':>10s}" + "".join(f"{r:>10s}" for r in rules))
exact = dict.fromkeys(rules, 0)
for k, v in PUB_COUNTRY.items():
    i = names.index(k) if k in names else None
    row = f"{k[:24]:24s} {v:10.2f}"
    for r in rules:
        s = np.nan if i is None else yearly[r][i]
        exact[r] += bool(abs(s - v) < 1e-6)
        row += f"{s:10.2f}"
    print(row)
print("exact matches:", exact)

# ---- counties ----------------------------------------------------------
fips, wkb = layer("ne_10m_admin_2_counties", "FIPS")
grid = pixtract.plan_sources(LIGHTS).grid
t0 = time.time()
cells = burn(wkb, grid, "approx")
st = pixtract.zonal_stats(LIGHTS, cells, n_id=len(fips), mem=MEM, max_workers=8)
print(f"\n{len(fips)} counties, night lights, {int(st['count'].sum()):,} cells, "
      f"{time.time() - t0:.1f} s")
print(f"{'FIPS':8s} {'published':>10s} {'mean':>10s} {'cells':>8s}")
for k, v in PUB_COUNTY.items():
    i = fips.index(k)
    print(f"{k:8s} {v:10.3f} {st['mean'][i]:10.3f} {st['count'][i]:8d}")
