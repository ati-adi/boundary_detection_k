#!/usr/bin/env python3
"""Render RGB + DelAny polygon overlays for the vendor Sample tiles (verification)."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SAMPLE = PROJECT_ROOT / "vendor" / "Delineate-Anything" / "data" / "images" / "Sample"
OUT_DIR = PROJECT_ROOT / "outputs" / "verification" / "delany"

GPKGS = {
    "v2": OUT_DIR / "delany_v2_sample.gpkg",
    "v1": OUT_DIR / "delany_v1_sample.gpkg",
    "authors": OUT_DIR / "authors_sample.gpkg",
}


def read_rgb(tiff: Path) -> tuple[np.ndarray, list]:
    with rasterio.open(tiff) as src:
        rgb = src.read([1, 2, 3])
        transform = src.transform
        crs = src.crs
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    for c in range(3):
        band = rgb[:, :, c]
        valid = band > 0
        if valid.any():
            p2, p98 = np.percentile(band[valid], [2, 98])
            rgb[:, :, c] = np.clip((band - p2) / max(p98 - p2, 1) * 255, 0, 255)
    h, w = rgb.shape[:2]
    extent = [transform.c, transform.c + w * transform.a,
              transform.f + h * transform.e, transform.f]
    return rgb.astype(np.uint8), extent


def main() -> None:
    tiles = sorted(VENDOR_SAMPLE.glob("*.tif"))
    gdfs = {}
    for key, path in GPKGS.items():
        if path.exists():
            gdfs[key] = gpd.read_file(path)
        else:
            print(f"WARNING: missing {path}")

    for tiff in tiles:
        rgb, extent = read_rgb(tiff)
        n = len(gdfs)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        if n == 1:
            axes = [axes]
        for ax, (key, gdf) in zip(axes, gdfs.items()):
            ax.imshow(rgb, extent=extent)
            sub = gdf.cx[extent[0]:extent[1], extent[2]:extent[3]]
            if not sub.empty:
                sub.plot(ax=ax, facecolor="none", edgecolor="lime", linewidth=0.8)
            ax.set_title(f"{tiff.name} — {key} ({len(sub)} polys in tile)")
            ax.set_axis_off()
        out_png = OUT_DIR / f"overlay_{tiff.stem}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    sys.exit(main())
