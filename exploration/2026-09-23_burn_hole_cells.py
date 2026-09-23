"""controlledburn coverage mode drops every cell a hole touches.

Found reproducing the Pangeo HFP x PAD-US means (examples/hfp_padus.py):
polygons whose burned coverage sums to less than their area. The cells an
interior ring passes through vanish from the output: they are neither in a
run nor listed as edges, so their coverage (up to 1.0) is lost. Ring
orientation makes no difference.

A 10 x 10 square on a 10 x 10 grid of unit cells, with a 1.4 x 0.4 hole
across columns 3 and 4 of row 4, has area 99.44 but burns to 98.0; the two
cells the hole touches are missing (expected fractions 0.8 and 0.76). A
3.1 x 3.1 hole (area 90.39) burns to 84.0: all 16 cells it touches are gone.

Run:  python exploration/2026-09-23_burn_hole_cells.py
"""
import controlledburn as cb
from osgeo import ogr


def burn(wkt):
    g = ogr.CreateGeometryFromWkt(wkt)
    r = cb.burn([bytes(g.ExportToWkb())], extent=(0, 10, 0, 10), shape=(10, 10))
    cov = {}
    for row, c0, c1, _ in r.runs:
        for c in range(int(c0), int(c1)):
            cov[(int(row), c)] = cov.get((int(row), c), 0.0) + 1.0
    for row, col, frac, _ in r.edges:
        cov[(int(row), int(col))] = cov.get((int(row), int(col)), 0.0) + float(frac)
    print(f"{wkt}\n  area {g.GetArea():.4f}  burned {sum(cov.values()):.4f}")
    for cell in [(4, 3), (4, 4)]:
        print(f"  row, col {cell}: {cov.get(cell, 'missing')}")


burn("POLYGON ((0 0,10 0,10 10,0 10,0 0),(3.2 5.2,4.6 5.2,4.6 5.6,3.2 5.6,3.2 5.2))")
burn("POLYGON ((0 0,10 0,10 10,0 10,0 0),(3.3 3.3,6.4 3.3,6.4 6.4,3.3 6.4,3.3 3.3))")
burn("POLYGON ((0 0,10 0,10 10,0 10,0 0),(4.2 5.2,4.6 5.2,4.6 5.6,4.2 5.6,4.2 5.2))")
