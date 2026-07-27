#!/usr/bin/env python3
"""Acquire Sentinel-2 L2A cloudless imagery over Kazakhstan AOIs.

For each AOI (20 km x 20 km bbox, WGS84):
  1. Find the clearest peak-growing-season scene (June-Aug 2024/2025).
  2. Find a second low-cloud scene >=90 days apart (for the FTW bi-temporal stack).
  3. Download a 3-band RGB 10 m GeoTIFF  -> source_data/kz_<name>.tiff
  4. Download an 8-band bi-temporal FTW stack (B04,B03,B02,B08 x 2 dates)
     -> source_data/ftw/kz_<name>_ftw.tif
  5. Write an RGB overview PNG and a JSON metadata record.

Uses the same pystac-client / odc-stac / Planetary Computer stack as
find_stac_pair.py and build_ftw_input.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dask.diagnostics.progress
import odc.stac
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PC_STAC_URL, PROJECT_ROOT
from crop_aoi import create_overview

RGB_BANDS = ["B04", "B03", "B02"]
FTW_BANDS = ["B04", "B03", "B02", "B08"]

GROWING_RANGES = [
    ("2025-06-01", "2025-08-31"),
    ("2024-06-01", "2024-08-31"),
]
SECOND_RANGE = ("2024-01-01", "2026-04-30")
MIN_GAP_DAYS = 90

AOIS = {
    # Kostanay oblast wheat belt, ~20x20 km, UTM zone 41N
    "kostanay": [63.35, 52.91, 63.65, 53.09],
    # Akmola oblast steppe farmland west of Astana, ~20x20 km, UTM zone 42N
    # (shifted west of the town at ~69.8E after first-pass QC of the overview)
    # NB: scene footprint barely covers this bbox -> TIFF came back ~empty
    # (mean DN ~0.2). Kept for reference; superseded by akmola_atbasar.
    "akmola": [69.40, 51.44, 69.69, 51.62],
    # Akmola oblast, Atbasar district grain belt (~200 km NW of Astana),
    # ~20x20 km, UTM zone 42N — replacement AOI after kz_akmola.tiff QC failed.
    "akmola_atbasar": [68.05, 51.70, 68.34, 51.88],
}


def _open_catalog():
    return pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def _cloud(item) -> float:
    props = item.properties or {}
    return float(props.get("eo:cloud_cover", props.get("s2:cloud_cover", 100.0)))


def _dt(item) -> datetime:
    dt = item.datetime
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _search(catalog, bbox, dt_range, max_cloud):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{dt_range[0]}/{dt_range[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=100,
    )
    return list(search.items())


def pick_scenes(catalog, bbox, max_cloud=15.0):
    """Return (growing_item, second_item)."""
    growing = []
    for rng in GROWING_RANGES:
        growing.extend(_search(catalog, bbox, rng, max_cloud))
    if not growing:
        raise RuntimeError(f"No growing-season scenes <{max_cloud}% cloud over {bbox}")
    growing.sort(key=lambda i: (_cloud(i), _dt(i)))
    grow = growing[0]

    # Prefer a companion from the same MGRS tile so the stack aligns cleanly.
    grow_tile = grow.properties.get("s2:mgrs_tile")
    candidates = _search(catalog, bbox, SECOND_RANGE, max_cloud)
    candidates = [
        i for i in candidates
        if abs((_dt(i) - _dt(grow)).days) >= MIN_GAP_DAYS
    ]
    same_tile = [i for i in candidates if i.properties.get("s2:mgrs_tile") == grow_tile]
    pool = same_tile or candidates
    if not pool:
        raise RuntimeError(f"No second scene >= {MIN_GAP_DAYS}d apart over {bbox}")
    pool.sort(key=lambda i: (_cloud(i), -abs((_dt(i) - _dt(grow)).days)))
    second = pool[0]
    return grow, second


def _load(items, bands, bbox):
    return odc.stac.load(
        items,
        bands=bands,
        dtype="uint16",
        resampling="bilinear",
        bbox=bbox,
        chunks={"x": "auto", "y": "auto"},
    )


def _fix_offset(data, items):
    """Sentinel-2 processing baseline >= 04.00 adds a +1000 DN offset."""
    version = max(
        float(i.properties.get("processing:version", 0) or 0) for i in items
    )
    if version >= 4:
        data = (data.astype("int32") - 1000).clip(min=0).astype("uint16")
    return data


def _write(data, out: Path, timestamp: datetime) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out} ...")
    with dask.diagnostics.progress.ProgressBar():
        data.rio.to_raster(
            str(out),
            driver="GTiff",
            compress="deflate",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            tags={"TIFFTAG_DATETIME": timestamp.strftime("%Y:%m:%d %H:%M:%S")},
        )


def process_aoi(name: str, bbox: list[float], max_cloud: float, out_dir: Path) -> dict:
    catalog = _open_catalog()
    grow, second = pick_scenes(catalog, bbox, max_cloud=max_cloud)
    print(f"[{name}] growing: {grow.id} ({_dt(grow).date()}, cloud {_cloud(grow):.2f}%)")
    print(f"[{name}] second : {second.id} ({_dt(second).date()}, cloud {_cloud(second):.2f}%)")

    # win_a = earlier date, win_b = later date (build_ftw_input.py convention)
    pair = sorted([grow, second], key=_dt)
    win_a, win_b = pair

    tic = time.time()
    # (a) RGB from the growing-season scene
    rgb = _fix_offset(_load([grow], RGB_BANDS, bbox), [grow])
    rgb = rgb.isel(time=0).rio.write_crs(rgb.rio.crs)
    rgb_path = out_dir / f"kz_{name}.tiff"
    _write(rgb, rgb_path, _dt(grow))

    # (b) FTW 8-band bi-temporal stack (B04,B03,B02,B08 for win_a then win_b)
    stack = _fix_offset(_load([win_a, win_b], FTW_BANDS, bbox), [win_a, win_b])
    stack = (
        stack.to_array(dim="band")
        .stack(bands=("time", "band"))
        .drop_vars("band")
        .transpose("bands", "y", "x")
    )
    ftw_path = out_dir / "ftw" / f"kz_{name}_ftw.tif"
    _write(stack, ftw_path, max(_dt(win_a), _dt(win_b)))
    print(f"[{name}] downloads done in {time.time() - tic:.1f}s")

    # Overview PNG of the RGB tiff
    overview_path = out_dir / f"kz_{name}_overview.png"
    create_overview(rgb_path, overview_path)

    meta = {
        "aoi": name,
        "bbox_wgs84": bbox,
        "growing_scene": {
            "id": grow.id,
            "date": _dt(grow).date().isoformat(),
            "cloud_pct": round(_cloud(grow), 3),
            "mgrs_tile": grow.properties.get("s2:mgrs_tile"),
            "epsg": grow.properties.get("proj:epsg"),
        },
        "ftw_pair": {
            "win_a": win_a.id,
            "win_a_date": _dt(win_a).date().isoformat(),
            "win_a_cloud_pct": round(_cloud(win_a), 3),
            "win_b": win_b.id,
            "win_b_date": _dt(win_b).date().isoformat(),
            "win_b_cloud_pct": round(_cloud(win_b), 3),
            "gap_days": abs((_dt(win_a) - _dt(win_b)).days),
        },
        "rgb_tiff": str(rgb_path),
        "ftw_tiff": str(ftw_path),
        "overview_png": str(overview_path),
    }
    meta_path = out_dir / f"kz_{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aois", nargs="+", default=list(AOIS), choices=list(AOIS))
    parser.add_argument("--max-cloud", type=float, default=15.0)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "source_data")
    args = parser.parse_args()

    results = {}
    for name in args.aois:
        results[name] = process_aoi(name, AOIS[name], args.max_cloud, args.out_dir)
    print(json.dumps({k: {"growing": v["growing_scene"], "ftw_pair": v["ftw_pair"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
