#!/usr/bin/env python3
"""Evaluate FTW PRUE B5 on FTW test-split chips (pixel IoU vs semantic_3class GT).

Reproduces the claimed benchmark quality (pixel IoU ~0.75) on a per-country
subset of the FTW test split (source.coop kerner-lab/fields-of-the-world).

Each chip: s2_images/window_{a,b}/<id>.tif (4 bands: B04,B03,B02,B08) stacked
to the 8-band bi-temporal model input; label: label_masks/semantic_3class/<id>.tif.

Usage:
    python scripts/eval_ftw_test_chips.py \
        --data-root outputs/verification/ftw/ftw_data/ftw \
        --countries rwanda vietnam --max-chips 70
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FTW_PRUE_B5

NUM_CLASSES = 3  # 0 background, 1 field, 2 boundary (FTW semantic_3class)


def load_chip(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()  # (4, H, W) uint16-ish


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--countries", nargs="+", default=["rwanda"])
    p.add_argument("--max-chips", type=int, default=70)
    p.add_argument("--model", type=Path, default=FTW_PRUE_B5)
    p.add_argument("--output", type=Path, default=None, help="JSON results path")
    args = p.parse_args()

    from ftw.trainers import CustomSemanticSegmentationTask

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        str(args.model), map_location="cpu"
    )
    model = task.model.eval().to(device)

    results = {}
    for country in args.countries:
        cdir = args.data_root / country
        wa_dir = cdir / "s2_images" / "window_a"
        wb_dir = cdir / "s2_images" / "window_b"
        gt_dir = cdir / "label_masks" / "semantic_3class"
        chips = sorted(wa_dir.glob("*.tif"))[: args.max_chips]
        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # rows=gt, cols=pred
        n_used = 0
        for wa in chips:
            wb, gt = wb_dir / wa.name, gt_dir / wa.name
            if not (wb.exists() and gt.exists()):
                continue
            img = np.concatenate([load_chip(wa), load_chip(wb)], axis=0)  # (8,H,W)
            label = load_chip(gt)[0]
            x = torch.from_numpy(img.astype(np.float32) / 3000.0).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred = model(x).argmax(axis=1)[0].cpu().numpy()
            valid = label < NUM_CLASSES
            for g in range(NUM_CLASSES):
                for q in range(NUM_CLASSES):
                    conf[g, q] += int(((label == g) & (pred == q) & valid).sum())
            n_used += 1

        iou = {}
        for c in range(NUM_CLASSES):
            tp = conf[c, c]
            denom = conf[c, :].sum() + conf[:, c].sum() - tp
            iou[f"class_{c}"] = float(tp / denom) if denom else None
        valid_ious = [v for v in iou.values() if v is not None]
        iou["mean"] = float(np.mean(valid_ious)) if valid_ious else None
        px_acc = float(np.trace(conf) / conf.sum()) if conf.sum() else None
        results[country] = {
            "chips": n_used,
            "iou": iou,
            "pixel_accuracy": px_acc,
            "confusion_gt_rows_pred_cols": conf.tolist(),
        }
        print(f"[{country}] chips={n_used} meanIoU={iou['mean']:.4f} "
              f"fieldIoU={iou['class_1']:.4f} boundaryIoU={iou['class_2']:.4f} pxAcc={px_acc:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
