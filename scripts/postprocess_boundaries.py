#!/usr/bin/env python3
"""
DelAnyFlow-style post-processing for field boundary polygons.

- Fix invalid geometries / fill small holes / drop slivers
- Clip / filter by cropland mask overlap
- Min/max area filter
- Optional RDP simplify (Shapely — safe on macOS)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CROPLAND_OVERLAP_MIN,
    MAX_AREA_HA,
    MIN_AREA_HA,
    SIMPLIFY_TOLERANCE_M,
)


def _ensure_crs(gdf, fallback_crs=None):
    if gdf.crs is None and fallback_crs is not None:
        return gdf.set_crs(fallback_crs)
    return gdf


def _fix_geometries(gdf):
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid

    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            geoms.append(None)
            continue
        g = make_valid(geom)
        if g.geom_type == "GeometryCollection":
            polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                geoms.append(None)
                continue
            g = polys[0] if len(polys) == 1 else MultiPolygon(
                [p for poly in polys for p in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly])]
            )
        if g.geom_type == "Polygon":
            # Fill small holes (< 5% of area or < 0.1 ha)
            if g.interiors:
                max_hole = max(0.05 * g.area, 1000.0)
                shells = [g.exterior]
                holes = [h for h in g.interiors if Polygon(h).area >= max_hole]
                g = Polygon(shells[0], holes)
        geoms.append(g)

    out = gdf.copy()
    out["geometry"] = geoms
    out = out[out.geometry.notnull() & ~out.geometry.is_empty].copy()
    out = out.explode(index_parts=False).reset_index(drop=True)
    return out


def _cropland_overlap_fraction(gdf, mask_path: Path) -> np.ndarray:
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    work = gdf
    if work.crs is not None and crs is not None and work.crs != crs:
        work = work.to_crs(crs)

    ids = np.arange(1, len(work) + 1, dtype=np.int32)
    shapes = [
        (geom, int(i))
        for geom, i in zip(work.geometry, ids)
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros(len(work), dtype=np.float64)

    id_raster = rasterize(
        shapes,
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    flat_ids = id_raster.ravel()
    flat_mask = mask.ravel().astype(np.float64)
    valid = flat_ids > 0
    sums = np.bincount(flat_ids[valid], weights=flat_mask[valid], minlength=len(ids) + 1)
    counts = np.bincount(flat_ids[valid], minlength=len(ids) + 1)
    fracs = np.zeros(len(ids) + 1, dtype=np.float64)
    nz = counts > 0
    fracs[nz] = sums[nz] / counts[nz]
    return fracs[ids]


def postprocess_boundaries(
    input_gpkg: Path,
    output_gpkg: Path,
    cropland_mask: Path | None = None,
    min_area_ha: float = MIN_AREA_HA,
    max_area_ha: float = MAX_AREA_HA,
    cropland_min: float = CROPLAND_OVERLAP_MIN,
    simplify_m: float = SIMPLIFY_TOLERANCE_M,
    reference_crs=None,
) -> dict:
    import geopandas as gpd

    if not input_gpkg.exists():
        raise FileNotFoundError(input_gpkg)

    gdf = gpd.read_file(input_gpkg)
    n_in = len(gdf)
    if gdf.empty:
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_gpkg, driver="GPKG", layer="fields")
        return {"input": str(input_gpkg), "output": str(output_gpkg), "n_in": 0, "n_out": 0}

    gdf = _ensure_crs(gdf, reference_crs)
    gdf = _fix_geometries(gdf)

    # Area filter (projected CRS assumed in meters)
    areas_ha = gdf.geometry.area / 1e4
    gdf = gdf[(areas_ha >= min_area_ha) & (areas_ha <= max_area_ha)].copy()
    n_after_area = len(gdf)

    n_after_crop = n_after_area
    if cropland_mask is not None and cropland_mask.exists() and not gdf.empty:
        fracs = _cropland_overlap_fraction(gdf, cropland_mask)
        gdf = gdf[fracs >= cropland_min].copy()
        n_after_crop = len(gdf)

    if simplify_m > 0 and not gdf.empty:
        gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True)
        gdf = _fix_geometries(gdf)

    # Drop tiny leftovers after simplify
    if not gdf.empty:
        areas_ha = gdf.geometry.area / 1e4
        gdf = gdf[areas_ha >= min_area_ha].copy()

    gdf = gdf.reset_index(drop=True)
    if "field_id" not in gdf.columns:
        gdf["field_id"] = range(1, len(gdf) + 1)
    gdf["area_ha"] = gdf.geometry.area / 1e4

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_gpkg, driver="GPKG", layer="fields")

    stats = {
        "input": str(input_gpkg),
        "output": str(output_gpkg),
        "n_in": n_in,
        "n_after_geom_area": n_after_area,
        "n_after_cropland": n_after_crop,
        "n_out": len(gdf),
        "min_area_ha": min_area_ha,
        "max_area_ha": max_area_ha,
        "cropland_min": cropland_min if cropland_mask else None,
        "simplify_m": simplify_m,
        "median_area_ha": float(gdf["area_ha"].median()) if len(gdf) else None,
    }
    output_gpkg.with_name(output_gpkg.stem + "_post_meta.json").write_text(
        json.dumps(stats, indent=2)
    )
    print(
        f"Postprocess {input_gpkg.name}: {n_in} → {len(gdf)} "
        f"(area={n_after_area}, cropland={n_after_crop})"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process field boundary polygons")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cropland-mask", type=Path, default=None)
    parser.add_argument("--min-area-ha", type=float, default=MIN_AREA_HA)
    parser.add_argument("--max-area-ha", type=float, default=MAX_AREA_HA)
    parser.add_argument("--cropland-min", type=float, default=CROPLAND_OVERLAP_MIN)
    parser.add_argument("--simplify-m", type=float, default=SIMPLIFY_TOLERANCE_M)
    args = parser.parse_args()

    stats = postprocess_boundaries(
        args.input,
        args.output,
        cropland_mask=args.cropland_mask,
        min_area_ha=args.min_area_ha,
        max_area_ha=args.max_area_ha,
        cropland_min=args.cropland_min,
        simplify_m=args.simplify_m,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
