#!/usr/bin/env python3
"""Crop a 20×20 km AOI subset and generate an RGB overview from the source TIFF."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AOI_META, AOI_OVERVIEW, AOI_SIZE_PX, AOI_SUBSET, DATA_DIR, SOURCE_TIFF


def gdal_info(path: Path) -> dict:
    result = subprocess.run(
        ["gdalinfo", "-json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def crop_aoi(
    src: Path,
    dst: Path,
    size_px: int = AOI_SIZE_PX,
    col_off: int | None = None,
    row_off: int | None = None,
) -> dict:
    info = gdal_info(src)
    w = info["size"][0]
    h = info["size"][1]

    if col_off is None:
        col_off = (w - size_px) // 2
    if row_off is None:
        row_off = (h - size_px) // 2

    col_off = max(0, min(col_off, w - size_px))
    row_off = max(0, min(row_off, h - size_px))

    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gdal_translate",
            "-co", "COMPRESS=ZSTD",
            "-co", "TILED=YES",
            "-srcwin", str(col_off), str(row_off), str(size_px), str(size_px),
            str(src),
            str(dst),
        ],
        check=True,
    )

    gt = info["geoTransform"]
    # pixel -> geo for upper-left of window
    ulx = gt[0] + col_off * gt[1] + row_off * gt[2]
    uly = gt[3] + col_off * gt[4] + row_off * gt[5]
    lrx = ulx + size_px * gt[1]
    lry = uly + size_px * gt[5]
    bbox_src = [min(ulx, lrx), min(lry, uly), max(ulx, lrx), max(lry, uly)]

    epsg = _extract_epsg(info)
    crs_wkt = info.get("coordinateSystem", {}).get("wkt", "")
    crs_auth = info.get("coordinateSystem", {}).get("data", {})
    if not epsg and isinstance(crs_auth, dict):
        # gdalinfo sometimes nests EPSG under wkt; try rasterio as fallback
        epsg = _epsg_via_rasterio(src)

    meta = {
        "source": str(src),
        "subset": str(dst),
        "width_px": size_px,
        "height_px": size_px,
        "col_off": col_off,
        "row_off": row_off,
        "pixel_size_m": abs(gt[1]),
        "extent_m": size_px * abs(gt[1]),
        "crs": crs_wkt,
        "epsg": epsg,
        "bbox_utm": bbox_src,
        "bbox_wgs84": _bbox_to_wgs84(bbox_src, epsg=epsg, crs_wkt=crs_wkt),
    }
    return meta


def _extract_epsg(info: dict) -> int | None:
    """Pull EPSG code from gdalinfo -json if present."""
    # Prefer authority codes if exposed
    for key in ("stac", "metadata"):
        pass
    wkt = info.get("coordinateSystem", {}).get("wkt", "") or ""
    # Look for AUTHORITY["EPSG","32643"]
    import re

    matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
    if matches:
        return int(matches[-1])  # last AUTHORITY is usually the CRS itself
    return None


def _epsg_via_rasterio(src: Path) -> int | None:
    try:
        import rasterio

        with rasterio.open(src) as ds:
            if ds.crs is None:
                return None
            return ds.crs.to_epsg()
    except Exception:
        return None


def _bbox_to_wgs84(
    bbox: list[float],
    epsg: int | None = None,
    crs_wkt: str = "",
) -> list[float]:
    """Transform source CRS bbox to WGS84 [min_lon, min_lat, max_lon, max_lat]."""
    try:
        from pyproj import Transformer

        if epsg is not None:
            src_crs = f"EPSG:{epsg}"
        elif crs_wkt:
            src_crs = crs_wkt
        else:
            raise ValueError("No CRS available for bbox transform")

        t = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        lons, lats = [], []
        for x, y in [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]:
            lon, lat = t.transform(x, y)
            lons.append(lon)
            lats.append(lat)
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception as e:
        print(f"Warning: bbox WGS84 transform failed ({e}); returning source bbox")
        return bbox


def create_overview(src: Path, dst: Path, max_size: int = 2048) -> None:
    info = gdal_info(src)
    w, h = info["size"]

    scale = max(w, h) / max_size
    out_w = int(w / scale)
    out_h = int(h / scale)

    # Read downsampled RGB via gdal_translate to VRT then numpy
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as tmp:
        vrt = tmp.name

    subprocess.run(
        ["gdal_translate", "-of", "VRT", "-outsize", str(out_w), str(out_h), str(src), vrt],
        check=True,
    )

    try:
        import rasterio
        with rasterio.open(vrt) as ds:
            rgb = ds.read([1, 2, 3])
            rgb = np.transpose(rgb, (1, 2, 0))
    except ImportError:
        # Fallback: read center crop at lower res via gdal
        subprocess.run(
            ["gdal_translate", "-of", "PNG", "-outsize", str(out_w), str(out_h), str(src), str(dst)],
            check=True,
        )
        Path(vrt).unlink(missing_ok=True)
        print(f"Overview saved (gdal PNG): {dst}")
        return
    finally:
        Path(vrt).unlink(missing_ok=True)

    # Stretch for display
    valid = rgb > 0
    if valid.any():
        for c in range(3):
            band = rgb[:, :, c]
            v = band[valid[:, :, c]]
            p2, p98 = np.percentile(v, [2, 98])
            rgb[:, :, c] = np.clip((band - p2) / max(p98 - p2, 1) * 255, 0, 255)

    img = Image.fromarray(rgb.astype(np.uint8))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    print(f"Overview saved: {dst} ({out_w}×{out_h})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Crop AOI subset and generate RGB overview")
    parser.add_argument("--input", type=Path, default=SOURCE_TIFF, help="Source GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Directory for subset, overview, meta")
    parser.add_argument("--size-px", type=int, default=AOI_SIZE_PX, help="AOI window size in pixels")
    parser.add_argument("--col-off", type=int, default=None, help="Column offset (default: center)")
    parser.add_argument("--row-off", type=int, default=None, help="Row offset (default: center)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Source TIFF not found: {args.input}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset = args.output_dir / "aoi_subset.tif"
    overview = args.output_dir / "overview.png"
    meta_path = args.output_dir / "aoi_meta.json"

    print(f"Cropping {args.size_px}×{args.size_px} px AOI from {args.input}")
    meta = crop_aoi(args.input, subset, size_px=args.size_px, col_off=args.col_off, row_off=args.row_off)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"AOI subset: {subset}")
    print(f"BBox WGS84: {meta['bbox_wgs84']}")

    print("Generating overview...")
    create_overview(args.input, overview)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
