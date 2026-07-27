#!/usr/bin/env python3
"""
Ensemble DelAny instance polygons with FTW semantic polygons.

Rules:
- Keep DelAny polys with enough cropland overlap and area in range
- Prefer DelAny where it overlaps FTW (source=both); else delany
- Add FTW-only regions poorly covered by DelAny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CROPLAND_OVERLAP_MIN, MAX_AREA_HA, MIN_AREA_HA


def _load(gpkg: Path, crs=None):
    import geopandas as gpd

    if not gpkg.exists():
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gpd.read_file(gpkg)
    if gdf.empty:
        return gdf.set_crs(crs) if crs and gdf.crs is None else gdf
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    elif gdf.crs is None and crs is not None:
        gdf = gdf.set_crs(crs)
    return gdf


def _filter_by_cropland(gdf, mask_path: Path, min_frac: float):
    """Vectorized cropland filter via zonal mean on a coarse sample grid."""
    import rasterio
    from rasterio.features import rasterize

    if gdf.empty:
        return gdf

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Rasterize polygon IDs, then compute mean mask per id
    ids = np.arange(1, len(gdf) + 1, dtype=np.int32)
    shapes = [(geom, int(i)) for geom, i in zip(gdf.geometry, ids) if geom is not None and not geom.is_empty]
    if not shapes:
        return gdf.iloc[0:0].copy()

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
    if not valid.any():
        return gdf.iloc[0:0].copy()

    sums = np.bincount(flat_ids[valid], weights=flat_mask[valid], minlength=len(ids) + 1)
    counts = np.bincount(flat_ids[valid], minlength=len(ids) + 1)
    fracs = np.zeros(len(ids) + 1, dtype=np.float64)
    nz = counts > 0
    fracs[nz] = sums[nz] / counts[nz]
    keep = fracs[ids] >= min_frac
    return gdf.iloc[np.where(keep)[0]].copy().reset_index(drop=True)


def ensemble_fields(
    delany_gpkg: Path,
    ftw_gpkg: Path,
    output_gpkg: Path,
    cropland_mask: Path | None = None,
    min_area_ha: float = MIN_AREA_HA,
    max_area_ha: float = MAX_AREA_HA,
    cropland_min: float = CROPLAND_OVERLAP_MIN,
    reference_tiff: Path | None = None,
) -> dict:
    import geopandas as gpd
    import rasterio
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    crs = None
    if reference_tiff and reference_tiff.exists():
        with rasterio.open(reference_tiff) as src:
            crs = src.crs

    delany = _load(delany_gpkg, crs)
    if crs is None and delany.crs is not None:
        crs = delany.crs
    ftw = _load(ftw_gpkg, crs)

    # Area filter first (cheap)
    if not delany.empty:
        a = delany.geometry.area / 1e4
        delany = delany[(a >= min_area_ha) & (a <= max_area_ha)].copy()
    if not ftw.empty:
        a = ftw.geometry.area / 1e4
        ftw = ftw[(a >= min_area_ha) & (a <= max_area_ha)].copy()

    if cropland_mask and cropland_mask.exists():
        print(f"Filtering DelAny by cropland ({len(delany)} polys)...")
        delany = _filter_by_cropland(delany, cropland_mask, cropland_min)
        print(f"  → {len(delany)}")
        print(f"Filtering FTW by cropland ({len(ftw)} polys)...")
        ftw = _filter_by_cropland(ftw, cropland_mask, cropland_min)
        print(f"  → {len(ftw)}")

    records = []

    # Mark DelAny that overlap FTW as "both"
    if not delany.empty:
        delany = delany.copy()
        delany["geometry"] = delany.geometry.map(
            lambda g: make_valid(g) if g is not None else g
        )
        delany = delany[delany.geometry.notnull() & ~delany.geometry.is_empty]
        if not ftw.empty and not delany.empty:
            joined = gpd.sjoin(
                delany[["geometry"]].reset_index(drop=True),
                ftw[["geometry"]].reset_index(drop=True),
                how="left",
                predicate="intersects",
            )
            has_ftw = joined.groupby(joined.index)["index_right"].transform(
                lambda s: s.notna().any()
            )
            # joined may duplicate rows; rebuild flags on unique delany index
            overlap_flags = joined.groupby(level=0)["index_right"].apply(
                lambda s: bool(s.notna().any())
            )
            # Align to delany length after reset
            delany = delany.reset_index(drop=True)
            flags = np.zeros(len(delany), dtype=bool)
            for idx, val in overlap_flags.items():
                if idx < len(flags):
                    flags[idx] = bool(val)
        else:
            delany = delany.reset_index(drop=True)
            flags = np.zeros(len(delany), dtype=bool)

        for i, row in delany.iterrows():
            geom = row.geometry
            records.append(
                {
                    "geometry": geom,
                    "source": "both" if flags[i] else "delany",
                    "area_ha": geom.area / 1e4,
                }
            )

    delany_union = unary_union([r["geometry"] for r in records]) if records else None

    # Add FTW-only fields not well covered by DelAny
    if not ftw.empty:
        ftw = ftw.copy()
        ftw["geometry"] = ftw.geometry.map(
            lambda g: make_valid(g) if g is not None else g
        )
        ftw = ftw[ftw.geometry.notnull() & ~ftw.geometry.is_empty].reset_index(drop=True)
        for _, row in ftw.iterrows():
            geom = row.geometry
            if delany_union is not None and not delany_union.is_empty:
                try:
                    inter = geom.intersection(delany_union).area
                except Exception:
                    inter = 0.0
                if inter / max(geom.area, 1.0) >= 0.5:
                    continue
                uncovered = geom.difference(delany_union)
                if uncovered.is_empty:
                    continue
                if uncovered.geom_type == "Polygon":
                    cands = [uncovered]
                elif uncovered.geom_type == "MultiPolygon":
                    cands = list(uncovered.geoms)
                else:
                    continue
                for c in cands:
                    ha = c.area / 1e4
                    if min_area_ha <= ha <= max_area_ha:
                        records.append({"geometry": c, "source": "ftw", "area_ha": ha})
            else:
                records.append(
                    {
                        "geometry": geom,
                        "source": "ftw",
                        "area_ha": geom.area / 1e4,
                    }
                )

    if not records:
        out = gpd.GeoDataFrame(columns=["source", "area_ha", "field_id"], geometry=[], crs=crs)
    else:
        out = gpd.GeoDataFrame(records, crs=crs)
        out = out.explode(index_parts=False).reset_index(drop=True)
        out = out[out.geometry.notnull() & ~out.geometry.is_empty].copy()
        out = out[
            (out.geometry.area / 1e4 >= min_area_ha)
            & (out.geometry.area / 1e4 <= max_area_ha)
        ].copy()
        out["area_ha"] = out.geometry.area / 1e4
        out["field_id"] = range(1, len(out) + 1)
        out = out[["field_id", "source", "area_ha", "geometry"]]

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(output_gpkg, driver="GPKG", layer="fields")

    counts = out["source"].value_counts().to_dict() if len(out) else {}
    stats = {
        "delany_in": int(len(_load(delany_gpkg, crs))),
        "ftw_in": int(len(_load(ftw_gpkg, crs))),
        "n_out": len(out),
        "by_source": {str(k): int(v) for k, v in counts.items()},
        "output": str(output_gpkg),
        "median_area_ha": float(out["area_ha"].median()) if len(out) else None,
    }
    output_gpkg.with_name(output_gpkg.stem + "_meta.json").write_text(json.dumps(stats, indent=2))
    print(f"Ensemble → {len(out)} fields ({counts})")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble DelAny + FTW field boundaries")
    parser.add_argument("--delany", type=Path, required=True)
    parser.add_argument("--ftw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cropland-mask", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--min-area-ha", type=float, default=MIN_AREA_HA)
    parser.add_argument("--max-area-ha", type=float, default=MAX_AREA_HA)
    parser.add_argument("--cropland-min", type=float, default=CROPLAND_OVERLAP_MIN)
    args = parser.parse_args()

    stats = ensemble_fields(
        args.delany,
        args.ftw,
        args.output,
        cropland_mask=args.cropland_mask,
        min_area_ha=args.min_area_ha,
        max_area_ha=args.max_area_ha,
        cropland_min=args.cropland_min,
        reference_tiff=args.reference,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
