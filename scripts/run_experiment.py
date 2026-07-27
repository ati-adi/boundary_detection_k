#!/usr/bin/env python3
"""
Run the full field-boundary experiment on a new GeoTIFF.

Pipeline: crop → cropland mask → DelAny (v2 + S) → postprocess → FTW PRUE
→ ensemble → compare → evaluate (optional) → full-tile winner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT


def run(cmd: list[str], step: str) -> None:
    print(f"\n==> {step}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def pick_winner_model(metrics_path: Path) -> str:
    """Pick best postprocessed model by edge_alignment then plausible median area."""
    if not metrics_path.exists():
        return "v2"
    metrics = json.loads(metrics_path.read_text())
    # Prefer ensemble if present
    for m in metrics:
        if "ensemble" in m.get("model", "").lower():
            return "ensemble"
    best = None
    best_score = -1.0
    for m in metrics:
        name = m.get("model", "")
        if m.get("polygon_count", 0) <= 0:
            continue
        ea = m.get("edge_alignment") or 0.0
        med = m.get("median_area_ha") or 0.0
        # Prefer median in 1–100 ha range
        area_bonus = 0.05 if 1.0 <= med <= 100.0 else 0.0
        score = ea + area_bonus
        if score > best_score:
            best_score = score
            best = name
    if best and "delany-s" in best.lower():
        return "s"
    if best and "v2" in best.lower():
        return "v2"
    if best and "delineate anything" in best.lower() and "s" not in best.lower():
        return "full"
    return "v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run crop → DelAny → FTW PRUE → postprocess → ensemble → compare"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input RGB GeoTIFF")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run folder name under outputs/ (default: TIFF stem)",
    )
    parser.add_argument("--skip-ftw", action="store_true", help="Skip FTW (no STAC)")
    parser.add_argument("--skip-full-tile", action="store_true", help="Skip full-tile run")
    parser.add_argument("--skip-v1", action="store_true", help="Skip DelAny v1 full/S (v2 only)")
    parser.add_argument("--win-a", type=str, default=None, help="Sentinel-2 STAC ID for FTW window A")
    parser.add_argument("--win-b", type=str, default=None, help="Sentinel-2 STAC ID for FTW window B")
    parser.add_argument("--size-px", type=int, default=2000, help="AOI crop size in pixels")
    parser.add_argument("--ref", type=Path, default=None, help="Optional reference polygons for evaluate.py")
    parser.add_argument("--full-tile-model", type=str, default=None, choices=["v2", "s", "full", "ensemble"])
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    run_id = args.run_id or args.input.stem
    run_root = PROJECT_ROOT / "outputs" / run_id
    data_dir = run_root / "data"
    py = sys.executable

    run_root.mkdir(parents=True, exist_ok=True)

    # 1. Crop AOI + overview
    run(
        [
            py, "scripts/crop_aoi.py",
            "--input", str(args.input),
            "--output-dir", str(data_dir),
            "--size-px", str(args.size_px),
        ],
        "Crop AOI subset",
    )

    aoi_subset = data_dir / "aoi_subset.tif"
    aoi_meta = data_dir / "aoi_meta.json"
    cropland = data_dir / "cropland_mask.tif"

    # 2. Cropland mask
    run(
        [
            py, "scripts/cropland_mask.py",
            "--reference", str(aoi_subset),
            "--output", str(cropland),
        ],
        "Build cropland mask (ESA WorldCover)",
    )

    # 3. DelAny v2 (primary)
    delany_v2_raw = run_root / "delany_v2" / "boundaries_raw.gpkg"
    delany_v2 = run_root / "delany_v2" / "boundaries.gpkg"
    run(
        [
            py, "scripts/run_delany.py",
            "--model", "v2",
            "--input", str(aoi_subset),
            "--output", str(delany_v2_raw),
            "--entry-name", f"{run_id}_aoi_v2",
        ],
        "DelAny v2 on AOI",
    )
    run(
        [
            py, "scripts/postprocess_boundaries.py",
            "--input", str(delany_v2_raw),
            "--output", str(delany_v2),
            "--cropland-mask", str(cropland),
        ],
        "Postprocess DelAny v2",
    )

    delany_full = run_root / "delany" / "boundaries.gpkg"
    delany_s = run_root / "delany_s" / "boundaries.gpkg"

    if not args.skip_v1:
        delany_full_raw = run_root / "delany" / "boundaries_raw.gpkg"
        delany_s_raw = run_root / "delany_s" / "boundaries_raw.gpkg"
        run(
            [
                py, "scripts/run_delany.py",
                "--model", "full",
                "--input", str(aoi_subset),
                "--output", str(delany_full_raw),
                "--entry-name", f"{run_id}_aoi",
            ],
            "DelAny full on AOI",
        )
        run(
            [
                py, "scripts/postprocess_boundaries.py",
                "--input", str(delany_full_raw),
                "--output", str(delany_full),
                "--cropland-mask", str(cropland),
            ],
            "Postprocess DelAny full",
        )
        run(
            [
                py, "scripts/run_delany.py",
                "--model", "s",
                "--input", str(aoi_subset),
                "--output", str(delany_s_raw),
                "--entry-name", f"{run_id}_aoi_s",
            ],
            "DelAny-S on AOI",
        )
        run(
            [
                py, "scripts/postprocess_boundaries.py",
                "--input", str(delany_s_raw),
                "--output", str(delany_s),
                "--cropland-mask", str(cropland),
            ],
            "Postprocess DelAny-S",
        )

    # 4. FTW PRUE
    ftw_raw = run_root / "ftw_prue" / "boundaries_raw.gpkg"
    ftw = run_root / "ftw_prue" / "boundaries.gpkg"
    if not args.skip_ftw:
        ftw_cmd = [
            py, "scripts/run_ftw.py",
            "--meta", str(aoi_meta),
            "--output-dir", str(run_root / "ftw_prue"),
        ]
        if args.win_a:
            ftw_cmd += ["--win-a", args.win_a]
        if args.win_b:
            ftw_cmd += ["--win-b", args.win_b]
        run(ftw_cmd, "FTW PRUE on AOI")
        # Prefer raw from run_ftw; fall back to boundaries.gpkg
        produced_raw = run_root / "ftw_prue" / "boundaries_raw.gpkg"
        produced = run_root / "ftw_prue" / "boundaries.gpkg"
        if produced_raw.exists() and not ftw_raw.exists():
            pass  # already named correctly
        elif produced.exists() and produced.resolve() != ftw_raw.resolve():
            if not ftw_raw.exists():
                produced.replace(ftw_raw)
        src_for_post = ftw_raw if ftw_raw.exists() else produced
        run(
            [
                py, "scripts/postprocess_boundaries.py",
                "--input", str(src_for_post),
                "--output", str(ftw),
                "--cropland-mask", str(cropland),
            ],
            "Postprocess FTW",
        )

    # 5. Ensemble
    ensemble = run_root / "ensemble" / "boundaries.gpkg"
    if not args.skip_ftw and ftw.exists() and delany_v2.exists():
        run(
            [
                py, "scripts/ensemble_fields.py",
                "--delany", str(delany_v2),
                "--ftw", str(ftw),
                "--output", str(ensemble),
                "--cropland-mask", str(cropland),
                "--reference", str(aoi_subset),
            ],
            "Ensemble DelAny v2 + FTW",
        )

    # 6. Compare
    viz_cmd = [
        py, "scripts/visualize.py",
        "--input", str(aoi_subset),
        "--output-dir", str(run_root / "compare"),
        "--delany", str(delany_full if delany_full.exists() else delany_v2),
        "--delany-s", str(delany_s if delany_s.exists() else delany_v2),
        "--ftw", str(ftw if ftw.exists() else delany_v2),
        "--delany-v2", str(delany_v2),
    ]
    if ensemble.exists():
        viz_cmd += ["--ensemble", str(ensemble)]
    run(viz_cmd, "Generate comparison overlays")

    # 7. Optional reference evaluation
    if args.ref and args.ref.exists():
        eval_targets = [
            ("delany_v2", delany_v2),
            ("ensemble", ensemble),
            ("ftw", ftw),
        ]
        for name, path in eval_targets:
            if not path.exists():
                continue
            run(
                [
                    py, "scripts/evaluate.py",
                    "--pred", str(path),
                    "--ref", str(args.ref),
                    "--cropland-mask", str(cropland),
                    "--reference-tiff", str(aoi_subset),
                    "--output", str(run_root / "compare" / f"eval_{name}.json"),
                ],
                f"Evaluate {name} vs reference",
            )

    # 8. Full tile with AOI winner
    if not args.skip_full_tile:
        winner = args.full_tile_model or pick_winner_model(run_root / "compare" / "metrics.json")
        print(f"\nFull-tile winner model: {winner}")
        if winner == "ensemble":
            # Run v2 + FTW full is heavy; run v2 full then postprocess as practical winner
            winner = "v2"
            print("Note: full-tile ensemble falls back to DelAny v2 + postprocess")

        full_raw = run_root / f"delany_{winner}" / "boundaries_full_raw.gpkg"
        full_out = run_root / f"delany_{winner}" / "boundaries_full.gpkg"
        # normalize paths for full/s/v2
        model_key = winner if winner in ("full", "s", "v2") else "v2"
        sub = {"full": "delany", "s": "delany_s", "v2": "delany_v2"}[model_key]
        full_raw = run_root / sub / "boundaries_full_raw.gpkg"
        full_out = run_root / sub / "boundaries_full.gpkg"

        run(
            [
                py, "scripts/run_delany.py",
                "--model", model_key,
                "--input", str(args.input),
                "--output", str(full_raw),
                "--entry-name", f"{run_id}_full_{model_key}",
                "--confidence", "0.08",
                "--batch-size", "2",
            ],
            f"DelAny {model_key} on full tile",
        )

        # Full-tile cropland mask
        cropland_full = data_dir / "cropland_mask_full.tif"
        run(
            [
                py, "scripts/cropland_mask.py",
                "--reference", str(args.input),
                "--output", str(cropland_full),
            ],
            "Build full-tile cropland mask",
        )
        run(
            [
                py, "scripts/postprocess_boundaries.py",
                "--input", str(full_raw),
                "--output", str(full_out),
                "--cropland-mask", str(cropland_full),
            ],
            f"Postprocess full-tile {model_key}",
        )
        run(
            [
                py, "scripts/visualize_full_tile.py",
                "--input", str(args.input),
                "--boundaries", str(full_out),
                "--output", str(run_root / "compare" / "full_tile_overlay.png"),
            ],
            "Full-tile QA overlay",
        )

    summary = {
        "run_id": run_id,
        "input": str(args.input),
        "outputs": str(run_root),
        "compare": str(run_root / "compare" / "side_by_side.png"),
        "metrics": str(run_root / "compare" / "metrics.json"),
        "ensemble": str(ensemble) if ensemble.exists() else None,
        "delany_v2": str(delany_v2),
    }
    (run_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Results in {run_root}")
    print(f"Compare: {summary['compare']}")


if __name__ == "__main__":
    main()
