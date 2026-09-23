"""controlledburn: a run fills to the grid's right edge when a polygon vertex
sits exactly on a row's cell-centre y (found 2026-09-23 in the Swiss cantons
on GLO-30).

Aargau (Overture 2026-08-19.0) has a vertex at (7.9240396, 47.2375). On the
18-tile GLO-30 grid (origin 4.99986111, 48.00013889, 1 arcsec) row 2745 has
its centre at 47.2375, and the approx burn returns one run from col 10527 to
the grid's last column instead of stopping at the boundary: 10,536 extra
cells, and Aargau's mean elevation 483.4 m instead of 480.2 m. On the
COP30_hh.vrt grid (origin 1.1e-8 deg off) the same polygon burns correctly.

Below, the three vertices around that point closed off to the west, burned
on a 100 x 10 window of the same grid. Expected: row 2745 runs 10500-10527.

Run: python exploration/2026-09-23_burn_vertex_on_centre.py
"""
import controlledburn as cb
import numpy as np
from osgeo import ogr

res = 1 / 3600
x0, y0 = 4.9998611111111115, 48.00013888888889
c0, r0, nc, nr = 10500, 2740, 100, 10
ext = (x0 + c0 * res, x0 + (c0 + nc) * res, y0 - (r0 + nr) * res, y0 - r0 * res)
pts = [(7.9226077, 47.2377078), (7.9240396, 47.2375), (7.9252611, 47.2373227)]

for dy in (0.0, 1e-9):
    p = [(x, y + dy) for x, y in pts]
    ring = p + [(p[-1][0] - 0.05, p[-1][1]), (p[0][0] - 0.05, p[0][1]), p[0]]
    wkt = "POLYGON ((" + ", ".join(f"{x!r} {y!r}" for x, y in ring) + "))"
    g = ogr.CreateGeometryFromWkt(wkt)
    r = cb.burn([bytes(g.ExportToWkb())], mode="approx", extent=ext, shape=(nr, nc))
    rows = np.asarray(r.runs["row"]) + r0
    a = np.asarray(r.runs["col_start"]) + c0
    b = np.asarray(r.runs["col_end"]) + c0
    print(f"vertex moved {dy:g} deg:",
          [(int(i), int(j), int(k)) for i, j, k in zip(rows, a, b)])
# vertex moved 0 deg: [(2745, 10500, 10600)]      <- runs to the window edge
# vertex moved 1e-09 deg: [(2745, 10500, 10527)]
