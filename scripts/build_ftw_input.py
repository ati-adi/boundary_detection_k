#!/usr/bin/env python3
"""Build FTW bi-temporal input stack using pystac-client (avoids SSL issues with from_file)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import dask.diagnostics.progress
import odc.stac
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AOI_META, DATA_DIR

BANDS = ["B04", "B03", "B02", "B08"]
DEFAULT_WIN_A = "S2A_MSIL2A_20250619T053241_R105_T44TNS_20250619T073113"
DEFAULT_WIN_B = "S2A_MSIL2A_20241212T054231_R005_T44TNS_20241212T093149"
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def get_item(item_id: str):
    catalog = pystac_client.Client.open(
        CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(search.items())
    if not items:
        raise ValueError(f"STAC item not found: {item_id}")
    return items[0]


def build_stack(out: Path, bbox: list[float], win_a: str = DEFAULT_WIN_A, win_b: str = DEFAULT_WIN_B) -> Path:
    items = [get_item(win_a), get_item(win_b)]
    timestamp = max(i.datetime for i in items if i.datetime)

    tic = time.time()
    data = odc.stac.load(
        items,
        bands=BANDS,
        dtype="uint16",
        resampling="bilinear",
        bbox=bbox,
        chunks={"x": "auto", "y": "auto"},
    )
    data = (
        data.to_array(dim="band")
        .stack(bands=("time", "band"))
        .drop_vars("band")
        .transpose("bands", "y", "x")
    )

    version = float(items[0].properties.get("processing:version", 0) or 0)
    if version >= 4:
        data = (data.astype("int32") - 1000).clip(min=0).astype("uint16")

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
    print(f"Done in {time.time() - tic:.1f}s")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build FTW bi-temporal input stack")
    parser.add_argument("--meta", type=Path, default=AOI_META)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "ftw_input.tif")
    parser.add_argument("--win-a", type=str, default=DEFAULT_WIN_A)
    parser.add_argument("--win-b", type=str, default=DEFAULT_WIN_B)
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text())
    bbox = meta["bbox_wgs84"]
    build_stack(args.output, bbox, win_a=args.win_a, win_b=args.win_b)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
