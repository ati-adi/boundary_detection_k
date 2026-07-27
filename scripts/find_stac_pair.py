#!/usr/bin/env python3
"""Find a low-cloud bi-temporal Sentinel-2 L2A STAC pair over an AOI bbox."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AOI_META, PC_STAC_URL


def _open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def _item_cloud(item) -> float:
    props = item.properties or {}
    return float(props.get("eo:cloud_cover", props.get("s2:cloud_cover", 100.0)))


def _item_date(item) -> datetime:
    dt = item.datetime
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def find_stac_pair(
    bbox: list[float],
    max_cloud: float = 20.0,
    target_gap_days: int = 150,
    max_items: int = 40,
    prefer_year: int | None = None,
) -> tuple[str, str, dict]:
    """
    Return (win_a_id, win_b_id, meta) for two low-cloud scenes.

    Prefers pairs ~target_gap_days apart (growing vs dormant season contrast).
    """
    catalog = _open_catalog()
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        max_items=max_items,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if len(items) < 2:
        # Relax cloud threshold
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            max_items=max_items,
            query={"eo:cloud_cover": {"lt": 60.0}},
        )
        items = list(search.items())

    if len(items) < 2:
        raise RuntimeError(
            f"Need ≥2 Sentinel-2 L2A scenes over bbox {bbox}; found {len(items)}"
        )

    # Sort by cloud then date
    items = sorted(items, key=lambda i: (_item_cloud(i), _item_date(i)))

    if prefer_year is not None:
        year_items = [i for i in items if _item_date(i).year == prefer_year]
        if len(year_items) >= 2:
            items = year_items + [i for i in items if i not in year_items]

    best = None
    best_score = float("inf")
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            gap = abs((_item_date(a) - _item_date(b)).days)
            if gap < 30:
                continue
            # Score: closeness to target gap + cloud penalty
            score = abs(gap - target_gap_days) + 0.5 * (_item_cloud(a) + _item_cloud(b))
            if score < best_score:
                best_score = score
                # Earlier date as win_a conventionally
                if _item_date(a) <= _item_date(b):
                    best = (a, b)
                else:
                    best = (b, a)

    if best is None:
        # Fallback: two clearest scenes regardless of gap
        clearest = sorted(items, key=_item_cloud)[:2]
        clearest = sorted(clearest, key=_item_date)
        best = (clearest[0], clearest[1])

    win_a, win_b = best
    meta = {
        "win_a": win_a.id,
        "win_b": win_b.id,
        "win_a_date": _item_date(win_a).date().isoformat(),
        "win_b_date": _item_date(win_b).date().isoformat(),
        "win_a_cloud": _item_cloud(win_a),
        "win_b_cloud": _item_cloud(win_b),
        "gap_days": abs((_item_date(win_a) - _item_date(win_b)).days),
        "bbox_wgs84": bbox,
    }
    return win_a.id, win_b.id, meta


def find_pair_from_meta(meta_path: Path, **kwargs) -> tuple[str, str, dict]:
    meta = json.loads(meta_path.read_text())
    bbox = meta["bbox_wgs84"]
    return find_stac_pair(bbox, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find bi-temporal S2 STAC pair for AOI")
    parser.add_argument("--meta", type=Path, default=AOI_META)
    parser.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--gap-days", type=int, default=150)
    args = parser.parse_args()

    if args.bbox:
        bbox = list(args.bbox)
    else:
        bbox = json.loads(args.meta.read_text())["bbox_wgs84"]

    win_a, win_b, info = find_stac_pair(bbox, max_cloud=args.max_cloud, target_gap_days=args.gap_days)
    print(json.dumps(info, indent=2))
    print(f"\nUse: --win-a {win_a} --win-b {win_b}")


if __name__ == "__main__":
    main()
