#!/usr/bin/env python3
"""Generate QA overlay for full-tile boundary output on RGB overview."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, SOURCE_TIFF
from visualize import load_boundaries, plot_overlay, read_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-tile boundary QA overlay")
    parser.add_argument("--input", type=Path, default=SOURCE_TIFF, help="Full RGB GeoTIFF")
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=OUTPUTS_DIR / "delany" / "boundaries_full.gpkg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "compare" / "full_tile_overlay.png",
    )
    args = parser.parse_args()

    if not args.boundaries.exists():
        print(f"Boundaries not found: {args.boundaries}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgb, (transform, crs) = read_rgb(args.input, max_size=2048)
    gdf = load_boundaries(args.boundaries)
    if gdf is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    n = len(gdf) if gdf is not None else 0
    plot_overlay(rgb, gdf, f"DelAny Full — Full Tile ({n} fields)", args.output, transform)
    print(f"QA map saved: {args.output}")


if __name__ == "__main__":
    main()
