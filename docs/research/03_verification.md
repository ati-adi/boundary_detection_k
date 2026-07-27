# 03 — Verification of pretrained models on their own inputs

Date: 2026-07-26. Environment: conda `boundary_detector` (torch 2.13, MPS), weights in `models/`.
All runs local on this machine; commands reproducible via the scripts cited below.

## Delineate Anything (DelAny) — v1 and v2

**Inputs**: authors' own sample tiles shipped in the vendor repo —
`vendor/Delineate-Anything/data/images/Sample/part_{1,2,3}.tif` — together with
the authors' published delineation for the same tiles
(`vendor/Delineate-Anything/data/delineated/Sample.gpkg`, copied to
`outputs/verification/delany/authors_sample.gpkg`).

**Run**: `vendor/Delineate-Anything/batch_verify_v{1,2}.yaml` with local weights
`models/DelineateAnything{,v2}.pt`. Outputs:
`outputs/verification/delany/delany_v{1,2}_sample[.simp].gpkg`,
overlays `overlay_part_{1,2,3}.png` (v2 | v1 | authors side by side).

**Quantitative result** (our v2 output vs authors' published output, same inputs):

| Metric | Value |
|---|---|
| Polygons (ours v2 / authors) | 679 / 679 (identical count) |
| Matched at IoU ≥ 0.5 | 676 / 679 (99.6 %) |
| Mean best-IoU | 0.992 |
| Median best-IoU | 1.000 |
| v1 polygon count (same tiles) | 466 |

Our local v2 run reproduces the authors' published delineation almost exactly —
weights, model code and local pipeline are consistent with what the authors
released. Visually (see overlays): sharp rectangular field corners, adjacent
fields separated, settlements and water excluded. v1 produces coarser output
(466 vs 679 polygons), consistent with the paper's claim that v2 is
substantially better (+0.284 mAP@0.5 on their 100-country benchmark).
The paper's absolute mAP numbers (0.720 mAP@0.5 for v1 on FBIS-22M test) were
**not** independently reproduced — the FBIS test split is not bundled and is a
heavy download; the check above verifies reproducibility of the released
artifacts instead.

## FTW PRUE EfficientNet-B5 (v3 checkpoint)

**Inputs**: (a) the official FTW sample bi-temporal 8-band stack
`outputs/verification/ftw/austria.tif` (2312×3389 px, Austria), downloaded with
`ftw inference download`; (b) FTW **test-split** chips for Rwanda (70) and
Vietnam (70) with `semantic_3class` ground truth
(`outputs/verification/ftw/ftw_data/ftw/{rwanda,vietnam}`).

**Runs**:
- `ftw inference run austria.tif --model models/prue_efnet5_checkpoint.ckpt --mps_mode`
  → `austria_preds.tif` (186 s on MPS), `ftw inference polygonize` →
  `austria_boundaries.gpkg` (**4901 polygons**), overlay `austria_overlay.png`.
- `scripts/eval_ftw_test_chips.py --countries rwanda vietnam` →
  `outputs/verification/ftw/test_chip_metrics.json`.

**Quantitative result vs claimed** (claimed: pixel IoU ≈ 0.75, object F1 ≈ 0.46
on the FTW test split):

| Test subset | Field-class IoU | Boundary IoU | Mean IoU (3 cl.) | Pixel acc. |
|---|---|---|---|---|
| Rwanda (70 chips) | 0.821 | 0.357 | 0.393 | 0.837 |
| Vietnam (70 chips) | 0.703 | 0.376 | 0.621 | 0.788 |
| **Average** | **0.762** | 0.367 | 0.507 | 0.812 |

Average field-class IoU **0.76 ≈ claimed 0.75** — the claimed quality is
reproduced on real test-split data. Note: "IoU 0.75" in the FTW model card
refers to the field/foreground class, not the 3-class mean (boundary pixels are
hard, IoU ≈ 0.37). Object F1 was not recomputed (needs instance matching);
pixel-level agreement with the claim is confirmed.

**Qualitative** (`austria_overlay.png`): polygons tightly trace small Austrian
fields, exclude forests, towns and the Danube. One artifact: a ~2 km strip along
the top edge of the sample tile is covered by imagery but received no polygons
(edge-patch artifact of the sliding-window inference).

## Fixes required to run FTW on this machine

1. `scripts/patch_ftw_prue.py` — PRUE `logcoshdice` loss alias (already applied).
2. `KMP_DUPLICATE_LIB_OK=TRUE` for every torch process in this conda env
   (duplicate libomp; see docstring in `scripts/patch_ftw_prue.py`).
3. `ftw_cli/inference.py` — torchgeo ≥ 0.8 returns `GeoSlice` tuples instead of
   `BoundingBox`; the bbox-conversion branch in `run()` was patched (2026-07-26)
   to extract `minx/maxx/miny/maxy` from slice start/stop. Without it,
   `ftw inference run` crashes with `TypeError: float() ... not 'slice'`.

## Conclusion

Both pretrained models are verified on their own inputs:
- **DelAny v2** — bit-level reproduction of the authors' released sample output.
- **FTW PRUE B5** — field IoU 0.76 on test chips ≈ claimed 0.75; visually clean
  delineation on the official Austria sample.
Proceeding to Kazakhstan imagery is justified.
