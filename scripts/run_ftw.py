#!/usr/bin/env python3
"""Run FTW PRUE inference on AOI with auto bi-temporal Sentinel-2 download."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AOI_META,
    FTW_PRUE_B5,
    FTW_PRUE_B5_URL,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from find_stac_pair import find_pair_from_meta


def load_bbox_wgs84(meta_path: Path) -> list[float]:
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta["bbox_wgs84"]
    raise FileNotFoundError(f"AOI metadata not found: {meta_path}. Run crop_aoi.py first.")


def ensure_prue_compat() -> None:
    """Ensure ftw-tools can load PRUE logcoshdice checkpoints (PyPI 1.4.3)."""
    patch = Path(__file__).resolve().parent / "patch_ftw_prue.py"
    subprocess.run([sys.executable, str(patch)], check=False)


def ensure_prue_model(dest: Path = FTW_PRUE_B5) -> Path:
    ensure_prue_compat()
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FTW PRUE B5 → {dest}")
    subprocess.run(["curl", "-L", "-o", str(dest), FTW_PRUE_B5_URL], check=True)
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"Failed to download PRUE checkpoint: {FTW_PRUE_B5_URL}")
    return dest


def resolve_windows(
    meta_path: Path,
    win_a: str | None,
    win_b: str | None,
) -> tuple[str, str, dict | None]:
    if win_a and win_b:
        return win_a, win_b, None
    print("Auto-selecting bi-temporal STAC pair for AOI...")
    a, b, info = find_pair_from_meta(meta_path)
    print(
        f"Selected {a} ({info['win_a_date']}, cloud={info['win_a_cloud']:.1f}%) + "
        f"{b} ({info['win_b_date']}, cloud={info['win_b_cloud']:.1f}%), "
        f"gap={info['gap_days']}d"
    )
    return a, b, info


def run_ftw(
    meta_path: Path,
    out_dir: Path,
    win_a: str | None = None,
    win_b: str | None = None,
    use_mps: bool = True,
) -> Path:
    bbox = load_bbox_wgs84(meta_path)
    model = ensure_prue_model()
    win_a, win_b, stac_info = resolve_windows(meta_path, win_a, win_b)

    out_dir.mkdir(parents=True, exist_ok=True)
    ftw_input = out_dir / "ftw_input.tif"
    preds = out_dir / "preds.tif"
    boundaries_raw = out_dir / "boundaries_raw.gpkg"
    boundaries = out_dir / "boundaries.gpkg"

    # Prefer MPS when available; otherwise CPU
    if use_mps:
        try:
            import torch
            use_mps = bool(torch.backends.mps.is_available())
        except Exception:
            use_mps = False
    mps_flag = ["--mps_mode"] if use_mps else ["--gpu", "-1"]

    # Reuse existing stack when possible
    if not ftw_input.exists():
        print(f"Building bi-temporal S2 stack for bbox {bbox}...")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_ftw_input.py"),
                "--meta", str(meta_path),
                "--output", str(ftw_input),
                "--win-a", win_a,
                "--win-b", win_b,
            ],
            check=True,
        )
    else:
        print(f"Reusing existing stack: {ftw_input}")

    print(f"Running FTW PRUE B5 inference on {ftw_input}")
    subprocess.run(
        [
            "ftw", "inference", "run",
            str(ftw_input),
            "--model", str(model),
            "-o", str(preds),
            *mps_flag,
            "-f",
            "--batch_size", "2",
        ],
        check=True,
    )

    print("Polygonizing FTW output...")
    subprocess.run(
        [
            "ftw", "inference", "polygonize",
            str(preds),
            "-o", str(boundaries_raw),
            "-f",
        ],
        check=True,
    )
    # Keep raw + copy to boundaries for callers that expect the old name
    import shutil
    shutil.copy2(boundaries_raw, boundaries)

    meta = {
        "model": str(model),
        "model_name": "FTW_PRUE_EFNET_B5",
        "win_a": win_a,
        "win_b": win_b,
        "bbox_wgs84": bbox,
        "input": str(ftw_input),
        "preds": str(preds),
        "output_raw": str(boundaries_raw),
        "output": str(boundaries),
        "stac_selection": stac_info,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Boundaries saved: {boundaries}")
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FTW PRUE field boundary detection")
    parser.add_argument("--meta", type=Path, default=AOI_META, help="aoi_meta.json path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "ftw_prue")
    parser.add_argument("--win-a", type=str, default=None, help="STAC item ID (auto if omitted)")
    parser.add_argument("--win-b", type=str, default=None, help="STAC item ID (auto if omitted)")
    parser.add_argument("--no-mps", action="store_true")
    args = parser.parse_args()
    run_ftw(
        args.meta,
        args.output_dir,
        win_a=args.win_a,
        win_b=args.win_b,
        use_mps=not args.no_mps,
    )


if __name__ == "__main__":
    main()
