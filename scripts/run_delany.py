#!/usr/bin/env python3
"""Run Delineate Anything inference on AOI subset or full tile."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AOI_SUBSET,
    DELANY_DIR,
    DELANY_FULL,
    DELANY_S,
    DELANY_V2,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)

MODEL_MAP = {
    "full": ("large", DELANY_FULL),
    "s": ("small", DELANY_S),
    "v2": ("v2", DELANY_V2),
}


def ensure_delany_repo() -> Path:
    if not DELANY_DIR.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/Lavreniuk/Delineate-Anything.git", str(DELANY_DIR)],
            check=True,
        )
    return DELANY_DIR


def _deep_override(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_override(result[key], value)
        else:
            result[key] = value
    return result


def write_delany_configs(
    delany_root: Path,
    model_key: str,
    model_path: Path,
    entry_name: str,
    confidence: float = 0.05,
    batch_size: int = 4,
    region_size: int = 4096,
) -> Path:
    """Write batch config and conf override tuned for 10 m RGB on Apple Silicon."""
    conf_override = delany_root / f"conf_{entry_name}.yaml"

    base_config = yaml.safe_load((delany_root / "conf_sample.yaml").read_text())
    overrides = {
        "model": [str(model_path)],
        "super_resolution": 1,
        "data_loader": {
            "bands": [1, 2, 3],
            "nodata_value": [0, 0, 0],
        },
        "execution_planner": {
            "region_width": region_size,
            "region_height": region_size,
        },
        "postprocess_limits": {
            "num_workers": [2, 2],
            "queue_tiles_capacity": 8,
            "max_tiles_inflight": 16,
        },
        "simplification_args": {
            "simplify": False,
            "epsilon_scale": 1,
        },
    }
    config = _deep_override(base_config, overrides)
    config["model"] = [str(model_path)]
    config["passes"][0]["batch_size"] = batch_size
    config["passes"][0]["model_args"] = [
        {
            "name": str(model_path),
            "minimal_confidence": confidence,
            "use_half": False,
        }
    ]
    conf_override.write_text(yaml.dump(config, default_flow_style=False))

    batch_config = delany_root / f"batch_{entry_name}.yaml"
    batch = {
        "base_config": str(conf_override.name),
        "data_root": "data/images",
        "output_root": "data/delineated",
        "temp_root": "data/temp",
        "keep_temp": True,
        "mask_root": "data/masks",
        "include": [entry_name],
        "exclude": None,
        "override": None,
    }
    batch_config.write_text(yaml.dump(batch, default_flow_style=False))
    return batch_config


def prepare_image(input_tiff: Path, entry_name: str) -> None:
    img_dir = DELANY_DIR / "data" / "images" / entry_name
    img_dir.mkdir(parents=True, exist_ok=True)
    dest = img_dir / input_tiff.name
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(input_tiff.resolve(), dest)


def run_delany(
    input_tiff: Path,
    output_gpkg: Path,
    model_key: str = "full",
    entry_name: str = "aoi_subset",
    confidence: float | None = None,
    batch_size: int = 4,
    region_size: int = 4096,
) -> Path:
    ensure_delany_repo()
    _, model_path = MODEL_MAP[model_key]
    if not model_path.exists():
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "download_models.py")], check=True)

    # Higher confidence on large/full-tile runs avoids MPS NMS stalls
    if confidence is None:
        confidence = 0.05 if "full" in entry_name else 0.02

    prepare_image(input_tiff, entry_name)
    batch_config = write_delany_configs(
        DELANY_DIR,
        model_key,
        model_path,
        entry_name,
        confidence=confidence,
        batch_size=batch_size,
        region_size=region_size,
    )

    out_dir = DELANY_DIR / "data" / "delineated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (DELANY_DIR / "data" / "temp").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GTIFF_SRS_SOURCE"] = "EPSG"

    print(f"Running Delineate Anything ({model_key}) on {input_tiff}")
    result = subprocess.run(
        [sys.executable, "delineate.py", "-b", batch_config.name, "--verbose"],
        cwd=str(DELANY_DIR),
        env=env,
    )

    # Vendor may delete temp mid-run when keep_temp=False; recreate for any cleanup
    (DELANY_DIR / "data" / "temp").mkdir(parents=True, exist_ok=True)

    result_gpkg = out_dir / f"{entry_name}.gpkg"
    if not result_gpkg.exists():
        raise FileNotFoundError(f"Expected output not found: {result_gpkg}")
    if result.returncode != 0:
        print(f"Warning: delineate.py exited with code {result.returncode}, using {result_gpkg}")

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_gpkg, output_gpkg)
    print(f"Boundaries saved: {output_gpkg}")
    return output_gpkg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Delineate Anything field boundary detection")
    parser.add_argument("--model", choices=["full", "s", "v2"], default="v2")
    parser.add_argument("--input", type=Path, default=AOI_SUBSET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--entry-name", default=None, help="DelAny folder entry name")
    parser.add_argument("--confidence", type=float, default=None, help="YOLO minimal confidence")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--region-size", type=int, default=4096)
    args = parser.parse_args()

    entry = args.entry_name or ("full_tile" if "source_data" in str(args.input) else "aoi_subset")
    if args.output is None:
        subdir = {"full": "delany", "s": "delany_s", "v2": "delany_v2"}[args.model]
        fname = "boundaries_full.gpkg" if "full" in entry else "boundaries.gpkg"
        args.output = OUTPUTS_DIR / subdir / fname

    if not args.input.exists():
        print(f"Input not found: {args.input}. Run crop_aoi.py first.")
        sys.exit(1)

    meta_path = PROJECT_ROOT / "data" / "aoi_meta.json"
    run_delany(
        args.input,
        args.output,
        model_key=args.model,
        entry_name=entry,
        confidence=args.confidence,
        batch_size=args.batch_size,
        region_size=args.region_size,
    )

    summary = {
        "model": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "entry": entry,
    }
    if meta_path.exists():
        summary["aoi_meta"] = json.loads(meta_path.read_text())
    out_meta = args.output.parent / f"{args.output.stem}_meta.json"
    out_meta.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
