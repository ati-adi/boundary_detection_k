# 04 — Kazakhstan assessment (Akmola oblast, new bbox)

Date: 2026-07-26. Pipeline: `scripts/run_experiment.py` on
`source_data/kz_akmola_atbasar.tiff` → `outputs/kz_akmola_atbasar/`.

## AOI

- **New bbox** (replaces the failed first Akmola attempt):
  `[68.05, 51.70, 68.34, 51.88]` — Atbasar district grain belt, Akmola oblast,
  ~20×20 km, UTM 42N. Added as `akmola_atbasar` in `scripts/acquire_kz_imagery.py`.
- The first Akmola bbox `[69.40, 51.44, 69.69, 51.62]` produced a near-empty
  TIFF (mean DN ≈ 0.2 — the selected scene's footprint barely covered the bbox);
  caught by overview QC and replaced.
- Scenes (both 0 % cloud, MGRS 42UVC, via Planetary Computer STAC):
  RGB = `S2B_MSIL2A_20250720` (2025-07-20);
  FTW pair = `S2A_MSIL2A_20240928` (win_a) + `S2B_MSIL2A_20250720` (win_b).

## Results (AOI subset, postprocessed + cropland filter)

| Model | Polygons | Area km² | Median ha | p90 ha | Max ha | Edge align |
|---|---|---|---|---|---|---|
| **DelAny v2** | 1612 | 365.3 | 7.0 | 59 | 464 | 0.279 |
| DelAny v1 | 76 | 141.1 | 193.5 | — | 484 | 0.327 |
| DelAny-S | 435 | 240.7 | 7.9 | — | 491 | 0.297 |
| **FTW PRUE B5** (summer pair) | 160 | 303.6 | 0.85 | 390 | **6734** | 0.312 |
| Ensemble (v2+FTW) | 1643 | 366.2 | 6.9 | 58 | 464 | 0.280 |

(Edge alignment = Canny-overlap sharpness metric from `scripts/visualize.py`.)

## Qualitative verdicts (see `outputs/kz_akmola_atbasar/compare/*.png`)

- **DelAny v2 — best overall for KZ.** Boundaries hug visible field edges with
  sharp corners; the regular northern wheat grid is delineated almost perfectly;
  lakes are excluded (cropland mask). Weaknesses: over-splits some large uniform
  fields; traces eroded gully lines in the badland terrain near the southern
  lakes; a few artifacts around the settlement at the east edge.
- **DelAny v1 — unusable here** (76 mega-polygons, median 193 ha): merges most
  adjacent fields. **DelAny-S** — usable but noisier than v2.
- **FTW PRUE B5 — works on KZ only with a proper bi-temporal pair.** With the
  summer pair it cleanly delineates the big rectangular fields (straightest
  boundaries of all models, best edge alignment 0.312), but merges aggressively
  in low-contrast steppe: 26 polygons > 100 ha, one 67 km² mega-polygon
  swallowing many southern fields, plus small edge fragments (median 0.85 ha).
- **Ensemble** ≈ v2 coverage with minor FTW additions; inherits v2's noise.

## Critical bug found: FTW window auto-selection picks snow scenes

First pipeline run: `find_stac_pair.py` auto-selected **2026-02-25 (winter,
snow-covered) + 2026-07-25** — both "0 % cloud", but snow ≠ cloud. FTW output
degraded to speckle (510 polygons, 18.5 km² total, edge alignment 0.073).
Rerun with explicit summer windows (`--win-a 2024-09-28 --win-b 2025-07-20`)
gave the good result above (303.6 km², 0.312).

**Recommendation**: constrain `find_stac_pair.py` to growing-season months
(May–October) or always pass `--win-a/--win-b` explicitly in this latitude.
Until fixed, `run_experiment.py` on KZ inputs should be run with explicit FTW
windows.

## Answer to "does claimed quality hold on KZ?"

- DelAny v2: yes — zero-shot quality on Kazakhstan large-field steppe is high
  and consistent with the authors' claims (verified reproducibility in doc 03).
- FTW PRUE B5: partially — claimed IoU reproduces on FTW's own test chips
  (doc 03), but on KZ the model needs a careful scene pair and still
  over-merges very large/low-contrast fields. Use as a complement to v2,
  not as the primary.

## Next steps (if pursued)

1. Constrain `find_stac_pair.py` to May–Oct (one-line query change).
2. Hand-digitize reference polygons over 2–3 KZ sub-AOIs →
   `run_experiment.py --ref` for object-F1 (no public KZ ground truth exists,
   see doc 01).
3. Raise `MIN_AREA_HA` / tighten the cropland filter in badland terrain to cut
   v2's gully false positives.
