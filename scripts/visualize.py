#!/usr/bin/env python3
"""Generate comparison overlays and boundary quality metrics across models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AOI_SUBSET, OUTPUTS_DIR, PROJECT_ROOT, SIMPLIFY_TOLERANCE_M

COMPARE_DIR = OUTPUTS_DIR / "compare"
MODEL_OUTPUTS = {
    "Delineate Anything": OUTPUTS_DIR / "delany" / "boundaries.gpkg",
    "Delineate Anything-S": OUTPUTS_DIR / "delany_s" / "boundaries.gpkg",
    "Delineate Anything v2": OUTPUTS_DIR / "delany_v2" / "boundaries.gpkg",
    "FTW PRUE B5": OUTPUTS_DIR / "ftw_prue" / "boundaries.gpkg",
    "Ensemble": OUTPUTS_DIR / "ensemble" / "boundaries.gpkg",
}


def read_rgb(tiff_path: Path, max_size: int = 2000) -> tuple[np.ndarray, object]:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(tiff_path) as src:
        scale = max(src.width, src.height) / max_size
        if scale > 1:
            out_h = int(src.height / scale)
            out_w = int(src.width / scale)
            rgb = src.read(
                [1, 2, 3],
                out_shape=(3, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            transform = src.transform * src.transform.scale(
                (src.width / out_w), (src.height / out_h)
            )
        else:
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
    return rgb.astype(np.uint8), (transform, crs)


def load_boundaries(gpkg: Path, fallback_crs=None):
    import geopandas as gpd
    if not gpkg.exists():
        return None
    gdf = gpd.read_file(gpkg)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        if fallback_crs is not None:
            gdf = gdf.set_crs(fallback_crs)
        else:
            print(f"Warning: {gpkg} has no CRS; leaving unset")
    return gdf


def compute_metrics(gdf, label: str) -> dict:
    if gdf is None or gdf.empty:
        return {"model": label, "polygon_count": 0}

    areas_m2 = gdf.geometry.area
    perimeters_m = gdf.geometry.length
    return {
        "model": label,
        "polygon_count": len(gdf),
        "total_area_km2": float(areas_m2.sum() / 1e6),
        "mean_area_ha": float(areas_m2.mean() / 1e4),
        "median_area_ha": float(np.median(areas_m2) / 1e4),
        "min_area_ha": float(areas_m2.min() / 1e4),
        "max_area_ha": float(areas_m2.max() / 1e4),
        "mean_perimeter_area_ratio": float((perimeters_m / areas_m2).mean()),
    }


def edge_alignment_score(rgb: np.ndarray, gdf, transform) -> float | None:
    """Proxy: mean distance from polygon boundaries to Canny edges (lower = better)."""
    if gdf is None or gdf.empty:
        return None
    try:
        import cv2
        import geopandas as gpd
        from rasterio.features import rasterize
        from shapely.ops import unary_union

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape

        boundaries = unary_union(gdf.geometry.boundary)
        boundary_raster = rasterize(
            [(boundaries, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        edge_pts = edges > 0
        boundary_pts = boundary_raster > 0
        if not boundary_pts.any():
            return None
        # Fraction of boundary pixels that coincide with Canny edges
        overlap = (edge_pts & boundary_pts).sum() / boundary_pts.sum()
        return float(overlap)
    except Exception as e:
        print(f"Edge alignment skipped for metric: {e}")
        return None


def plot_overlay(rgb, gdf, title: str, out_path: Path, transform) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(rgb, extent=_extent_from_transform(rgb.shape, transform))
    ax.set_title(title)
    ax.set_axis_off()

    if gdf is not None and not gdf.empty:
        gdf.plot(ax=ax, facecolor="none", edgecolor="lime", linewidth=0.8, alpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _extent_from_transform(shape: tuple, transform) -> list[float]:
    h, w = shape[:2]
    left = transform.c
    top = transform.f
    right = left + w * transform.a
    bottom = top + h * transform.e
    return [left, right, bottom, top]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare field boundary model outputs")
    parser.add_argument("--input", type=Path, default=AOI_SUBSET, help="RGB GeoTIFF for basemap")
    parser.add_argument("--output-dir", type=Path, default=COMPARE_DIR, help="Comparison output folder")
    parser.add_argument("--delany", type=Path, default=MODEL_OUTPUTS["Delineate Anything"])
    parser.add_argument("--delany-s", type=Path, default=MODEL_OUTPUTS["Delineate Anything-S"])
    parser.add_argument("--delany-v2", type=Path, default=MODEL_OUTPUTS["Delineate Anything v2"])
    parser.add_argument("--ftw", type=Path, default=MODEL_OUTPUTS["FTW PRUE B5"])
    parser.add_argument("--ensemble", type=Path, default=MODEL_OUTPUTS["Ensemble"])
    args = parser.parse_args()

    model_outputs = {
        "Delineate Anything v2": args.delany_v2,
        "Delineate Anything": args.delany,
        "Delineate Anything-S": args.delany_s,
        "FTW PRUE B5": args.ftw,
        "Ensemble": args.ensemble,
    }
    # Drop missing paths and de-duplicate identical files
    seen = set()
    filtered = {}
    for label, gpkg in model_outputs.items():
        if not gpkg.exists():
            continue
        key = str(gpkg.resolve())
        if key in seen:
            continue
        seen.add(key)
        filtered[label] = gpkg
    model_outputs = filtered

    if not args.input.exists():
        print(f"AOI subset not found: {args.input}")
        sys.exit(1)

    compare_dir = args.output_dir
    compare_dir.mkdir(parents=True, exist_ok=True)
    rgb, (transform, crs) = read_rgb(args.input)

    all_metrics = []
    panels = []

    for label, gpkg in model_outputs.items():
        if not gpkg.exists():
            print(f"Skipping {label}: {gpkg} not found")
            continue
        gdf = load_boundaries(gpkg, fallback_crs=crs)
        if gdf is not None and gdf.crs is not None and crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        metrics = compute_metrics(gdf, label)
        metrics["edge_alignment"] = edge_alignment_score(rgb, gdf, transform)
        all_metrics.append(metrics)

        out_png = compare_dir / f"{label.lower().replace(' ', '_')}_overlay.png"
        plot_overlay(rgb, gdf, f"{label} — {metrics['polygon_count']} fields", out_png, transform)
        panels.append((label, gdf))

    if not panels:
        print("No model outputs found to compare.")
        sys.exit(1)

    # Side-by-side comparison grid
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    extent = _extent_from_transform(rgb.shape, transform)
    for ax, (label, gdf) in zip(axes, panels):
        ax.imshow(rgb, extent=extent)
        if gdf is not None and not gdf.empty:
            gdf.plot(ax=ax, facecolor="none", edgecolor="lime", linewidth=0.6)
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle("Field Boundary Comparison — AOI Subset", fontsize=14)
    comparison_path = compare_dir / "side_by_side.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {comparison_path}")

    metrics_path = compare_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Metrics: {metrics_path}")

    # Print QA checklist summary
    print("\n=== Visual QA Checklist (manual review of PNGs) ===")
    print("1. Field corners sharp (not rounded blobs)?")
    print("2. Adjacent fields separated (no merged polygons)?")
    print("3. Small fields (0.5–2 ha) preserved?")
    print("4. Roads/rivers excluded?")
    print("5. Polygon edges align with visible crop/soil lines?")
    print(f"\nReview images in: {compare_dir}")


if __name__ == "__main__":
    main()
