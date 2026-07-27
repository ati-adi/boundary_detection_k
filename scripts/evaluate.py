#!/usr/bin/env python3
"""
Object-level evaluation of predicted field polygons vs a reference layer.

Supports GeoPackage / GeoJSON / GeoParquet reference. Computes IoU-matched
precision, recall, F1 at a given IoU threshold, plus over/under-segmentation
proxies and cropland false-positive rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PC_STAC_URL


def _load_vectors(path: Path, crs=None):
    import geopandas as gpd

    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        elif gdf.crs != crs:
            gdf = gdf.to_crs(crs)
    return gdf


def _pairwise_iou(pred, ref) -> np.ndarray:
    """Compute IoU matrix (n_pred × n_ref) using shapely intersections."""
    n_p, n_r = len(pred), len(ref)
    iou = np.zeros((n_p, n_r), dtype=np.float64)
    if n_p == 0 or n_r == 0:
        return iou
    # Spatial index for speed
    try:
        sindex = ref.sindex
    except Exception:
        sindex = None

    for i, pgeom in enumerate(pred.geometry):
        if pgeom is None or pgeom.is_empty:
            continue
        if sindex is not None:
            cand = list(sindex.intersection(pgeom.bounds))
        else:
            cand = range(n_r)
        pa = pgeom.area
        for j in cand:
            rgeom = ref.geometry.iloc[j]
            if rgeom is None or rgeom.is_empty:
                continue
            inter = pgeom.intersection(rgeom).area
            if inter <= 0:
                continue
            union = pa + rgeom.area - inter
            if union > 0:
                iou[i, j] = inter / union
    return iou


def match_at_iou(iou: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns (tp, fp, fn)."""
    n_p, n_r = iou.shape
    if n_p == 0:
        return 0, 0, n_r
    if n_r == 0:
        return 0, n_p, 0

    # Sort candidate pairs by IoU descending
    pairs = [
        (iou[i, j], i, j)
        for i in range(n_p)
        for j in range(n_r)
        if iou[i, j] >= threshold
    ]
    pairs.sort(reverse=True)
    matched_p, matched_r = set(), set()
    tp = 0
    for _, i, j in pairs:
        if i in matched_p or j in matched_r:
            continue
        matched_p.add(i)
        matched_r.add(j)
        tp += 1
    fp = n_p - tp
    fn = n_r - tp
    return tp, fp, fn


def over_under_segmentation(iou: np.ndarray, threshold: float = 0.3) -> dict:
    """
    Rough proxies:
    - overseg: many preds map to one ref (ref matched by >1 pred above thr)
    - underseg: one pred covers many refs
    """
    n_p, n_r = iou.shape
    if n_p == 0 or n_r == 0:
        return {"overseg_refs": 0, "underseg_preds": 0}

    over = 0
    for j in range(n_r):
        if (iou[:, j] >= threshold).sum() > 1:
            over += 1
    under = 0
    for i in range(n_p):
        if (iou[i, :] >= threshold).sum() > 1:
            under += 1
    return {"overseg_refs": int(over), "underseg_preds": int(under)}


def cropland_false_rate(pred, mask_path: Path) -> float | None:
    if mask_path is None or not Path(mask_path).exists() or pred.empty:
        return None
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
    gdf = pred
    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    ids = np.arange(1, len(gdf) + 1, dtype=np.int32)
    shapes = [(geom, int(i)) for geom, i in zip(gdf.geometry, ids) if geom is not None and not geom.is_empty]
    if not shapes:
        return 1.0
    id_raster = rasterize(shapes, out_shape=mask.shape, transform=transform, fill=0, dtype=np.int32)
    flat_ids = id_raster.ravel()
    flat_mask = mask.ravel().astype(np.float64)
    valid = flat_ids > 0
    sums = np.bincount(flat_ids[valid], weights=flat_mask[valid], minlength=len(ids) + 1)
    counts = np.bincount(flat_ids[valid], minlength=len(ids) + 1)
    fracs = np.zeros(len(ids) + 1, dtype=np.float64)
    nz = counts > 0
    fracs[nz] = sums[nz] / counts[nz]
    false = int((fracs[ids] < 0.35).sum())
    return float(false / len(gdf))


def evaluate(
    pred_path: Path,
    ref_path: Path,
    iou_threshold: float = 0.5,
    cropland_mask: Path | None = None,
    reference_tiff: Path | None = None,
) -> dict:
    import rasterio

    crs = None
    if reference_tiff and reference_tiff.exists():
        with rasterio.open(reference_tiff) as src:
            crs = src.crs

    pred = _load_vectors(pred_path, crs)
    ref = _load_vectors(ref_path, crs or pred.crs)

    iou = _pairwise_iou(pred, ref)
    tp, fp, fn = match_at_iou(iou, iou_threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    mean_matched_iou = 0.0
    if tp > 0:
        # Recompute matched IoUs greedily
        pairs = [
            (iou[i, j], i, j)
            for i in range(iou.shape[0])
            for j in range(iou.shape[1])
            if iou[i, j] >= iou_threshold
        ]
        pairs.sort(reverse=True)
        used_p, used_r, vals = set(), set(), []
        for v, i, j in pairs:
            if i in used_p or j in used_r:
                continue
            used_p.add(i)
            used_r.add(j)
            vals.append(v)
        mean_matched_iou = float(np.mean(vals)) if vals else 0.0

    seg = over_under_segmentation(iou)
    stats = {
        "pred": str(pred_path),
        "ref": str(ref_path),
        "n_pred": len(pred),
        "n_ref": len(ref),
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_matched_iou,
        **seg,
        "cropland_false_rate": cropland_false_rate(pred, cropland_mask) if cropland_mask else None,
    }
    return stats


def try_download_ftw_global_for_bbox(
    bbox: list[float],
    output_path: Path,
) -> Path | None:
    """
    Best-effort download of FTW global polygons for a bbox via Source Cooperative
    / fiboa if available. Returns path or None if unavailable.
    """
    # fiboa-cli / source.coop paths evolve; try known STAC-like endpoints then give up.
    try:
        import planetary_computer
        import pystac_client

        catalog = pystac_client.Client.open(
            PC_STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
        # No stable PC collection for FTW global yet — skip gracefully
        _ = catalog
    except Exception:
        pass

    print(
        "FTW global auto-download is not wired to a stable public STAC collection. "
        "Pass --ref path/to/reference.gpkg (or GeoParquet from fieldsofthe.world) instead."
    )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted fields vs reference")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True, help="Reference GPKG/GeoJSON/GeoParquet")
    parser.add_argument("--output", type=Path, default=None, help="Write metrics JSON")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--cropland-mask", type=Path, default=None)
    parser.add_argument("--reference-tiff", type=Path, default=None)
    args = parser.parse_args()

    stats = evaluate(
        args.pred,
        args.ref,
        iou_threshold=args.iou,
        cropland_mask=args.cropland_mask,
        reference_tiff=args.reference_tiff,
    )
    print(json.dumps(stats, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
