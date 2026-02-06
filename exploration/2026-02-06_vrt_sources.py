"""
Exploring how to get VRT source-file-to-bbox mappings from GDAL.

Question: can we do the tile-grouping trick at the VRT level?
  - Level 1 (super-tile): classify points → source file
  - Level 2 (tile): classify points → internal tile within source

We need: (xmin, ymin, xmax, ymax, source_dsn) for each VRT source.
"""

import numpy as np
from osgeo import gdal, osr
import xml.etree.ElementTree as ET
import json
import os
import tempfile
import time

gdal.UseExceptions()

# ── Create a test scenario: multiple tiled GeoTIFFs + a VRT ──────────

def create_source_tifs(base_dir, n_sources=6, nx=2048, ny=2048,
                       tile_x=256, tile_y=256):
    """Create a grid of source GeoTIFFs simulating a mosaic."""
    paths = []
    # Arrange in a 3×2 grid
    cols, rows = 3, 2
    pixel_size = 0.001  # degrees
    origin_x, origin_y = 100.0, -30.0

    for r in range(rows):
        for c in range(cols):
            fname = os.path.join(base_dir, f"source_{r}_{c}.tif")
            drv = gdal.GetDriverByName("GTiff")
            ds = drv.Create(fname, nx, ny, 1, gdal.GDT_Float32,
                           options=[f"TILED=YES",
                                    f"BLOCKXSIZE={tile_x}",
                                    f"BLOCKYSIZE={tile_y}",
                                    "COMPRESS=DEFLATE"])
            ox = origin_x + c * nx * pixel_size
            oy = origin_y - r * ny * pixel_size
            ds.SetGeoTransform((ox, pixel_size, 0, oy, 0, -pixel_size))
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())

            band = ds.GetRasterBand(1)
            # Encode source index in the data for verification
            val = (r * cols + c) * 1000000
            data = np.full((ny, nx), val, dtype=np.float32)
            rows_arr = np.arange(ny, dtype=np.float32)
            cols_arr = np.arange(nx, dtype=np.float32)
            data += rows_arr[:, None] * 1000 + cols_arr[None, :]
            band.WriteArray(data)
            band.FlushCache()
            ds.FlushCache()
            ds = None
            paths.append(fname)
            print(f"  Created {fname}: origin=({ox:.3f}, {oy:.3f}), "
                  f"extent=({ox:.3f},{oy - ny*pixel_size:.3f},"
                  f"{ox + nx*pixel_size:.3f},{oy:.3f})")

    return paths


def build_vrt(source_paths, vrt_path):
    """Build a VRT from source files."""
    vrt_ds = gdal.BuildVRT(vrt_path, source_paths)
    vrt_ds.FlushCache()
    vrt_ds = None
    return vrt_path


# ── Method 1: Parse VRT XML directly ─────────────────────────────────

def get_sources_from_vrt_xml(vrt_path):
    """Parse the VRT XML to extract source file extents.
    
    Each <SimpleSource> has:
      <SourceFilename>path</SourceFilename>
      <SrcRect xOff="0" yOff="0" xSize="2048" ySize="2048"/>
      <DstRect xOff="0" yOff="0" xSize="2048" ySize="2048"/>
    
    DstRect gives the pixel position in the VRT.
    Combined with VRT's geotransform → geographic bbox.
    """
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    # VRT geotransform
    gt_elem = root.find(".//GeoTransform")
    gt = tuple(float(x.strip()) for x in gt_elem.text.split(","))

    sources = []
    for band in root.findall(".//VRTRasterBand"):
        for source in band.findall("SimpleSource"):
            fname = source.find("SourceFilename").text
            relative = source.find("SourceFilename").get("relativeToVRT", "0")

            dst = source.find("DstRect")
            dx = float(dst.get("xOff"))
            dy = float(dst.get("yOff"))
            dw = float(dst.get("xSize"))
            dh = float(dst.get("ySize"))

            # Convert DstRect pixel coords → geographic bbox using VRT geotransform
            xmin = gt[0] + dx * gt[1] + dy * gt[2]
            xmax = gt[0] + (dx + dw) * gt[1] + (dy + dh) * gt[2]
            ymax = gt[3] + dx * gt[4] + dy * gt[5]
            ymin = gt[3] + (dx + dw) * gt[4] + (dy + dh) * gt[5]

            # Ensure min < max
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin

            sources.append({
                "filename": fname,
                "relative": relative == "1",
                "dst_xoff": dx, "dst_yoff": dy,
                "dst_xsize": dw, "dst_ysize": dh,
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax,
            })

    return sources, gt


# ── Method 2: gdal.Info JSON ─────────────────────────────────────────

def get_sources_from_gdal_info(vrt_path):
    """Use gdal.Info() to get source metadata."""
    info = gdal.Info(vrt_path, format='json')
    print(f"\n  gdal.Info keys: {list(info.keys())}")
    if 'bands' in info:
        for b in info['bands']:
            if 'metadata' in b:
                print(f"  Band metadata keys: {list(b['metadata'].keys())}")
    # gdal.Info doesn't directly give per-source extents for VRTs
    # It gives the overall extent
    print(f"  cornerCoordinates: {info.get('cornerCoordinates', {}).keys()}")
    return info


# ── Method 3: Open each source individually ──────────────────────────

def get_sources_by_opening(source_paths):
    """Just open each source and read its geotransform."""
    sources = []
    for path in source_paths:
        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        nx, ny = ds.RasterXSize, ds.RasterYSize
        xmin = gt[0]
        xmax = gt[0] + nx * gt[1]
        ymax = gt[3]
        ymin = gt[3] + ny * gt[5]
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        band = ds.GetRasterBand(1)
        bx, by = band.GetBlockSize()
        ds = None
        sources.append({
            "filename": path,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "nx": nx, "ny": ny, "gt": gt, "block": (bx, by),
        })
    return sources


# ── Super-tile grouping: classify points to source files ─────────────

def classify_points_to_sources(xs, ys, sources):
    """Classify each point to the source file(s) it falls in.
    
    For non-overlapping sources (typical mosaic), each point → 0 or 1 source.
    Uses vectorized bbox tests (no GEOS needed).
    """
    # Build arrays of source extents
    n_src = len(sources)
    xmins = np.array([s["xmin"] for s in sources])
    xmaxs = np.array([s["xmax"] for s in sources])
    ymins = np.array([s["ymin"] for s in sources])
    ymaxs = np.array([s["ymax"] for s in sources])

    # For each point, test against all sources
    # Shape: (n_points, n_sources)
    in_x = (xs[:, None] >= xmins[None, :]) & (xs[:, None] < xmaxs[None, :])
    in_y = (ys[:, None] >= ymins[None, :]) & (ys[:, None] < ymaxs[None, :])
    inside = in_x & in_y  # (n_points, n_sources)

    # For non-overlapping mosaic, argmax gives the source index
    # Points outside all sources → handle separately
    any_match = inside.any(axis=1)
    source_idx = np.full(len(xs), -1, dtype=np.int64)
    source_idx[any_match] = inside[any_match].argmax(axis=1)

    return source_idx


def classify_points_to_sources_fast(xs, ys, sources):
    """Faster version for many sources: use the VRT geotransform to
    compute which grid cell each point is in, if sources form a regular grid.
    
    For irregular mosaics, fall back to the bbox test above.
    For REMA-style (regular grid of COGs), this is O(1) per point.
    """
    # Check if sources form a regular grid
    # ... (would need to detect grid structure)
    # For now, fall back to bbox test but with early exit
    return classify_points_to_sources(xs, ys, sources)


def classify_points_to_sources_sorted(xs, ys, sources):
    """For many sources (1500+ REMA tiles): sort sources by xmin,
    use searchsorted to narrow candidates per point.
    
    Much faster than broadcasting (n_points, n_sources) for large n_sources.
    """
    n_src = len(sources)
    xmins = np.array([s["xmin"] for s in sources])
    xmaxs = np.array([s["xmax"] for s in sources])
    ymins = np.array([s["ymin"] for s in sources])
    ymaxs = np.array([s["ymax"] for s in sources])

    # Sort sources by xmin for searchsorted
    sort_x = np.argsort(xmins)
    xmins_s = xmins[sort_x]
    xmaxs_s = xmaxs[sort_x]
    ymins_s = ymins[sort_x]
    ymaxs_s = ymaxs[sort_x]

    source_idx = np.full(len(xs), -1, dtype=np.int64)

    # For each point, searchsorted gives the rightmost source whose xmin <= x
    # Then scan backwards to check bbox containment
    right = np.searchsorted(xmins_s, xs, side='right')  # first idx where xmin > x

    # Vectorized: for each point, check candidates from right-1 backwards
    # In practice, for a non-overlapping grid, there are at most ~2-3 candidates
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        for j in range(right[i] - 1, -1, -1):
            if xmins_s[j] > x:
                continue
            if xmaxs_s[j] <= x:
                # Past this source's extent, and all further sources have smaller xmin
                # But they might have larger xmax... can't break early without more info
                continue
            if ymins_s[j] <= y < ymaxs_s[j]:
                source_idx[i] = sort_x[j]
                break
    return source_idx


# ── Two-level extraction: super-tile then tile ───────────────────────

def extract_vrt_two_level(vrt_path, xs, ys, band_idx=1):
    """Two-level grouped extraction:
    Level 1: classify points → VRT source file
    Level 2: within each source, classify points → internal tile
    """
    from tile_extract_module import _inverse_geotransform, _group_by_tile

    # Get source extents from VRT XML
    sources_xml, vrt_gt = get_sources_from_vrt_xml(vrt_path)

    # Also open each source to get block sizes
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    sources = []
    for s in sources_xml:
        if s.get("relative"):
            s["filename"] = os.path.join(vrt_dir, s["filename"])
        ds = gdal.Open(s["filename"])
        gt = ds.GetGeoTransform()
        nx, ny = ds.RasterXSize, ds.RasterYSize
        bx, by = ds.GetRasterBand(band_idx).GetBlockSize()
        ds = None
        sources.append({
            **s, "gt": gt, "nx": nx, "ny": ny, "block": (bx, by)
        })

    # Level 1: classify points to source files
    source_idx = classify_points_to_sources(xs, ys, sources)
    n_matched = np.sum(source_idx >= 0)
    n_missed = np.sum(source_idx < 0)

    result = np.full(len(xs), np.nan, dtype=np.float64)
    tiles_read_total = 0

    # Level 2: for each source, do tile-grouped extraction
    for src_i in np.unique(source_idx):
        if src_i < 0:
            continue
        mask = source_idx == src_i
        src = sources[src_i]
        src_xs = xs[mask]
        src_ys = ys[mask]

        gt = src["gt"]
        nx, ny = src["nx"], src["ny"]
        bx, by = src["block"]
        ntx = (nx + bx - 1) // bx
        nty = (ny + by - 1) // by

        col, row = _inverse_geotransform(src_xs, src_ys, gt)
        icol = np.clip(np.floor(col).astype(np.int64), 0, nx - 1)
        irow = np.clip(np.floor(row).astype(np.int64), 0, ny - 1)

        unique_keys, groups, lc, lr, _ = _group_by_tile(
            icol, irow, bx, by, ntx, nty
        )
        tiles_read_total += len(unique_keys)

        ds = gdal.Open(src["filename"])
        band = ds.GetRasterBand(band_idx)

        # Map group indices back to original array positions
        orig_indices = np.where(mask)[0]

        for key, group in zip(unique_keys, groups):
            t_y, t_x = divmod(int(key), ntx)
            xoff, yoff = t_x * bx, t_y * by
            win_x = min(bx, nx - xoff)
            win_y = min(by, ny - yoff)
            tile_data = band.ReadAsArray(xoff, yoff, win_x, win_y)
            result[orig_indices[group]] = tile_data[lr[group], lc[group]]

        ds = None

    return result, n_matched, n_missed, tiles_read_total


def extract_vrt_naive(vrt_path, xs, ys, band_idx=1):
    """Read directly from VRT — let GDAL resolve source files internally.
    Uses tile-grouped approach on the VRT itself."""
    from tile_extract_module import extract_points
    return extract_points(vrt_path, xs, ys, band=band_idx, backend="gdal")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    tmpdir = "/tmp/vrt_test"
    os.makedirs(tmpdir, exist_ok=True)

    print("=" * 65)
    print("Creating 6 source GeoTIFFs (3×2 grid, each 2048×2048)...")
    paths = create_source_tifs(tmpdir)

    vrt_path = os.path.join(tmpdir, "mosaic.vrt")
    print(f"\nBuilding VRT: {vrt_path}")
    build_vrt(paths, vrt_path)

    # ── Examine VRT structure ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("METHOD 1: Parse VRT XML")
    sources, vrt_gt = get_sources_from_vrt_xml(vrt_path)
    for s in sources:
        print(f"  {os.path.basename(s['filename'])}: "
              f"bbox=({s['xmin']:.3f},{s['ymin']:.3f},{s['xmax']:.3f},{s['ymax']:.3f})")

    print(f"\nVRT GeoTransform: {vrt_gt}")

    print("\n" + "-" * 65)
    print("METHOD 2: gdal.Info JSON")
    info = get_sources_from_gdal_info(vrt_path)

    print("\n" + "-" * 65)
    print("METHOD 3: Open each source")
    src_info = get_sources_by_opening(paths)
    for s in src_info:
        print(f"  {os.path.basename(s['filename'])}: "
              f"bbox=({s['xmin']:.3f},{s['ymin']:.3f},{s['xmax']:.3f},{s['ymax']:.3f}) "
              f"block={s['block']}")

    # ── Print VRT XML for inspection ─────────────────────────────────
    print("\n" + "=" * 65)
    print("VRT XML (first 60 lines):")
    with open(vrt_path) as f:
        for i, line in enumerate(f):
            if i >= 60: break
            print(f"  {line.rstrip()}")

    # ── Generate test points ─────────────────────────────────────────
    print("\n" + "=" * 65)
    n_points = 500_000
    ds = gdal.Open(vrt_path)
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None

    rng = np.random.default_rng(42)
    cols = rng.uniform(0, nx - 1, n_points)
    rows = rng.uniform(0, ny - 1, n_points)
    xs = gt[0] + cols * gt[1]
    ys = gt[3] + rows * gt[5]
    print(f"Generated {n_points:,} points across VRT extent")
    print(f"  x range: [{xs.min():.4f}, {xs.max():.4f}]")
    print(f"  y range: [{ys.min():.4f}, {ys.max():.4f}]")

    # ── Benchmark: two-level vs VRT-direct ───────────────────────────
    print("\n" + "=" * 65)
    print("BENCHMARK: Two-level vs VRT-direct extraction")

    t0 = time.perf_counter()
    res_2level, n_match, n_miss, tiles_read = extract_vrt_two_level(
        vrt_path, xs, ys
    )
    t_2level = time.perf_counter() - t0
    print(f"\n  Two-level (source→tile): {t_2level:.4f}s")
    print(f"    matched: {n_match:,}, missed: {n_miss:,}, tiles read: {tiles_read}")

    t0 = time.perf_counter()
    res_direct = extract_vrt_naive(vrt_path, xs, ys)
    t_direct = time.perf_counter() - t0
    print(f"\n  VRT-direct (tile-grouped on VRT): {t_direct:.4f}s")

    match = np.allclose(res_2level, res_direct, equal_nan=True)
    print(f"\n  Results match: {match}")

    # ── Classify timing ──────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("Point classification timing:")

    t0 = time.perf_counter()
    for _ in range(100):
        classify_points_to_sources(xs, ys, sources)
    t_classify = (time.perf_counter() - t0) / 100
    print(f"  Vectorized bbox (6 sources): {t_classify*1000:.2f}ms per call")

    # Simulate REMA-scale: 1500 sources
    print("\n  Simulating 1500-source mosaic classification...")
    fake_sources = []
    for i in range(1500):
        r, c = divmod(i, 50)
        fake_sources.append({
            "xmin": c * 2.0, "xmax": (c+1) * 2.0,
            "ymin": -r * 2.0 - 2.0, "ymax": -r * 2.0,
        })
    # Points within the fake mosaic
    xs_fake = rng.uniform(0, 100, 100_000)
    ys_fake = rng.uniform(-60, 0, 100_000)

    t0 = time.perf_counter()
    classify_points_to_sources(xs_fake, ys_fake, fake_sources)
    t_big = time.perf_counter() - t0
    print(f"  Vectorized bbox (1500 sources, 100k pts): {t_big*1000:.1f}ms")
    print(f"    → broadcasting shape: (100000, 1500) = "
          f"{100000*1500*8/1024/1024:.0f} MB per comparison")

    t0 = time.perf_counter()
    classify_points_to_sources_sorted(xs_fake, ys_fake, fake_sources)
    t_sorted = time.perf_counter() - t0
    print(f"  Searchsorted (1500 sources, 100k pts): {t_sorted*1000:.1f}ms")

    print("\n" + "=" * 65)
    print("DONE")


if __name__ == "__main__":
    main()
