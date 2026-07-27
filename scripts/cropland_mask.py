#!/usr/bin/env python3
"""Build an ESA WorldCover agricultural mask aligned to a reference GeoTIFF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PC_STAC_URL,
    WORLDCOVER_AG_CLASSES,
    WORLDCOVER_EXCLUDE_CLASSES,
)


def _open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def build_cropland_mask(
    reference_tiff: Path,
    output_tiff: Path,
    ag_classes: tuple[int, ...] = WORLDCOVER_AG_CLASSES,
    mode: str = "ag",
) -> Path:
    """
    Download ESA WorldCover for the reference extent and write a binary mask
    matching the reference grid.

    mode:
      - "ag": 1 where class in ag_classes (cropland+grassland)
      - "not_exclude": 1 everywhere except water/wetland/built/trees
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    import odc.stac

    with rasterio.open(reference_tiff) as ref:
        bbox = list(rasterio.warp.transform_bounds(ref.crs, "EPSG:4326", *ref.bounds))
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        height, width = ref.height, ref.width
        res = abs(ref_transform.a)

    catalog = _open_catalog()
    items = list(
        catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
            max_items=50,
        ).items()
    )
    if not items:
        raise RuntimeError(f"No ESA WorldCover items found for bbox {bbox}")

    # Prefer a single year mosaic (2021), keep all tiles covering the bbox
    year_items = [i for i in items if "2021" in (i.id or "")]
    if not year_items:
        year_items = items
    # Deduplicate by id
    seen = set()
    mosaic_items = []
    for i in year_items:
        if i.id in seen:
            continue
        seen.add(i.id)
        mosaic_items.append(i)
    print(f"Using {len(mosaic_items)} WorldCover item(s): {[i.id for i in mosaic_items]}")

    data = odc.stac.load(
        mosaic_items,
        bands=["map"],
        bbox=bbox,
        crs=str(ref_crs),
        resolution=res,
        chunks={},
        groupby="solar_day",
    )
    da = data["map"]
    # If time dim present (multiple days), take first / mode
    if "time" in da.dims:
        da = da.isel(time=0)
    da = da.squeeze(drop=True)
    arr = np.asarray(da.values, dtype=np.uint8)

    try:
        src_transform = da.rio.transform()
    except Exception:
        from affine import Affine

        xs = np.asarray(da["x"].values)
        ys = np.asarray(da["y"].values)
        res_x = float(xs[1] - xs[0]) if len(xs) > 1 else res
        res_y = float(ys[1] - ys[0]) if len(ys) > 1 else -res
        src_transform = Affine(
            res_x, 0.0, float(xs[0]) - res_x / 2,
            0.0, res_y, float(ys[0]) - res_y / 2,
        )

    if mode == "not_exclude":
        binary = (~np.isin(arr, WORLDCOVER_EXCLUDE_CLASSES)).astype(np.uint8)
    else:
        binary = np.isin(arr, list(ag_classes)).astype(np.uint8)

    out_mask = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=binary,
        destination=out_mask,
        src_transform=src_transform,
        src_crs=ref_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )

    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(count=1, dtype="uint8", compress="zstd", nodata=0)
    for k in ("photometric",):
        profile.pop(k, None)
    profile["tiled"] = True
    with rasterio.open(output_tiff, "w", **profile) as dst:
        dst.write(out_mask, 1)

    frac = float(out_mask.mean())
    print(f"Ag mask saved: {output_tiff} (ag fraction={frac:.3f}, mode={mode})")
    meta = {
        "reference": str(reference_tiff),
        "output": str(output_tiff),
        "worldcover_items": [i.id for i in mosaic_items],
        "mode": mode,
        "ag_classes": list(ag_classes),
        "ag_fraction": frac,
        "bbox_wgs84": bbox,
    }
    output_tiff.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return output_tiff


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ESA WorldCover agricultural mask")
    parser.add_argument("--reference", type=Path, required=True, help="Reference GeoTIFF grid")
    parser.add_argument("--output", type=Path, required=True, help="Output mask GeoTIFF")
    parser.add_argument(
        "--mode",
        choices=["ag", "not_exclude"],
        default="ag",
        help="ag=cropland+grassland; not_exclude=drop water/wetland/built/trees only",
    )
    args = parser.parse_args()
    if not args.reference.exists():
        print(f"Reference not found: {args.reference}")
        sys.exit(1)
    build_cropland_mask(args.reference, args.output, mode=args.mode)


if __name__ == "__main__":
    main()
