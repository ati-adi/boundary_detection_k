# 02 — Projects & Pretrained Models for Field Boundary Delineation

Research date: 2026-07-26. Method: web/paper review + live HTTP verification of every
weight URL cited below (HEAD / Range `bytes=0-0` requests; all "verified" URLs returned
HTTP 206 with the stated `Content-Range` total size).

**Bottom line:** only two projects ship downloadable pretrained weights that are
practically usable for Sentinel-2-class field delineation — **Delineate Anything
(v1/v1-S/v2)** and **Fields of The World baselines (FTW v1–v3.1)**. Both weight sets
referenced by this repo's `scripts/download_models.py` / `scripts/config.py` are still
valid and downloadable (details in §4). Everything else surveyed either publishes a
dataset + training code but **no weights** (AI4SmallFarms), an empty model repo
(Revnev HF), or wraps the two projects above (Agribound, Autobounds, TorchGeo).

## Summary table

| Project | Model / weights | Verified size | License (weights) | Input | Claimed metrics |
|---|---|---|---|---|---|
| FTW v3 PRUE | `prue_efnet5_checkpoint.ckpt` (B5, default) | 125.9 MB | Mixed open (incl. CC-BY-NC-SA data) | 8-band bi-temporal S2 @10 m | Pixel IoU 0.75; object F1 0.46 |
| FTW v3.1 PRUE CC-BY | `prue_efnetb5_ccby_checkpoint.ckpt` | 125.9 MB | CC-BY-4.0 | same | Pixel IoU 0.76, P 0.88 / R 0.86; object F1 0.41 |
| Delineate Anything v1 | `DelineateAnything.pt` (YOLOv11x-seg) | 124.7 MB | AGPL-3.0 | single-date RGB GeoTIFF, 0.25–10 m | mAP@0.5 = 0.720, mAP@0.5:0.95 = 0.477 |
| Delineate Anything-S | `DelineateAnything-S.pt` (YOLOv11n-seg) | 17.6 MB | AGPL-3.0 | same | mAP@0.5 = 0.632, mAP@0.5:0.95 = 0.383 |
| Delineate Anything v2 | `DelineateAnythingv2.pt` | 124.7 MB | AGPL-3.0 | same | +0.284 mAP@0.5 over v1 on 100-country benchmark (+103.3% rel.) |

---

## 1. Fields of The World — `ftw-baselines` (FTW PRUE)

- **Repo:** https://github.com/fieldsoftheworld/ftw-baselines (code: **MIT**)
- **Docs/tutorial:** https://fieldsofthe.world/tutorial.html
- **Training data:** the FTW dataset (NeurIPS 2024) — ≈1.6 M field boundary samples
  across 24 countries, paired with bi-temporal Sentinel-2. v3 "PRUE" models were trained
  with the recipe from *PRUE: A Practical Recipe for Field Boundary Segmentation at Scale*
  (CVPR 2026, [arXiv:2603.27101](https://arxiv.org/html/2603.27101v1)).
- **Architecture:** 3-class U-Net (field / field boundary / neither) with EfficientNet
  B3/B5/B7 encoders; logcosh-dice loss, class weights bg:0.05 field:0.2 boundary:0.75,
  channel-shuffle + normalization + resize augmentation.
- **Input spec:** bi-temporal Sentinel-2 L2A — two acquisitions (start & end of growing
  season), 4 bands each (B04, B03, B02, B08) stacked to **8 channels @ 10 m**. The CLI
  auto-selects scenes from a STAC catalog (Microsoft Planetary Computer by default) or
  accepts explicit scene IDs.
- **Checkpoint licenses:** v3 checkpoints = "Mixed Open Licenses" (training data includes
  CC-BY-NC-SA → non-commercial restrictions); v3.1 checkpoints are retrained on CC-BY/CC0
  countries only → **CC-BY-4.0**.
- **Claimed metrics** (full FTW test set, `ftw model test -p3 -t3 --temporal_options stacked`,
  from the [releases page](https://github.com/fieldsoftheworld/ftw-baselines/releases)):

  | Model (v3.1 CC-BY) | Pixel IoU | Pixel P | Pixel R | Object P | Object R | Object F1 |
  |---|---|---|---|---|---|---|
  | EfficientNet-B3 | 0.76 | 0.87 | 0.86 | 0.54 | 0.31 | 0.39 |
  | EfficientNet-B5 | 0.76 | 0.88 | 0.86 | 0.53 | 0.33 | 0.41 |
  | EfficientNet-B7 | 0.77 | 0.88 | 0.86 | 0.58 | 0.35 | 0.44 |

  v3 (original, mixed-license) reference: B5 pixel IoU 0.75 / object F1 0.46; B7 0.76 / 0.47.
- **Weight URLs (all verified 2026-07-26, HTTP 206):**
  - v3 (tag `v3`, published 2025-11-15):
    `https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v3/prue_efnet5_checkpoint.ckpt`
    — **125,916,483 bytes**. Also `prue_efnet3_checkpoint.ckpt` (53.2 MB),
    `prue_efnet7_checkpoint.ckpt` (270.1 MB), plus `*_standard_weight_*` and
    `prue_logcoshdice_only_checkpoint.ckpt` variants.
  - v3.1 CC-BY (tag `v3.1`, published 2025-11-30):
    `.../v3.1/prue_efnetb5_ccby_checkpoint.ckpt` — **125,921,939 bytes** (verified);
    same for `efnetb3`/`efnetb7`.
  - Legacy: v1 (`2_Class/3_Class_{CCBY,FULL}_FTW_Pretrained.ckpt`) and v2
    (multi-window + single-window 4-channel) under tags `v1`/`v2`.
- **Inference:** `pip install ftw-tools`, then model registry CLI (no manual download
  needed): `ftw model list` / `ftw model show FTW_PRUE_EFNET_B5` (default), and
  `ftw inference all <bbox>` (or the step-wise `download → inference → polygonize`)
  producing fiboa-compliant GeoParquet. Registry source:
  `ftw_tools/inference/model_registry.py` — `FTW_PRUE_EFNET_B5` maps exactly to the v3
  URL in our `config.py` and is flagged `default=True`.
- **Note for KZ:** needs two S2 scenes per AOI; output is semantic masks → polygonize.
  This repo's `scripts/run_ftw.py` + `patch_ftw_prue.py` handle that.

## 2. Delineate Anything (DelAny v1 / v1-S / v2) — Lavreniuk et al.

- **Repo:** https://github.com/Lavreniuk/Delineate-Anything — **AGPL-3.0** (code AND
  weights; Ultralytics YOLOv11-based). Project page: https://lavreniuk.github.io/Delineate-Anything/
- **Weights (HuggingFace `MykolaL/DelineateAnything`, license `agpl-3.0`,
  `pipeline_tag: image-segmentation`):** https://huggingface.co/MykolaL/DelineateAnything
  - `DelineateAnything.pt` — **124,746,842 bytes** (verified)
  - `DelineateAnything-S.pt` — **17,635,629 bytes** (verified)
  - `DelineateAnythingv2.pt` — **124,747,297 bytes** (verified)
  - Mirrored for TorchGeo under `hf.co/torchgeo/delineate-anything[-s]` (v1 only);
    DelAny is integrated into TorchGeo and into the `ftw-tools` model registry.
- **Training data:**
  - v1 / v1-S: **FBIS-22M** — 672,909 image patches (0.25 m–10 m, multi-source incl.
    Sentinel-2) with 22.9 M field instance masks ([arXiv:2504.02534](https://arxiv.org/abs/2504.02534),
    ECAI 2025). Dataset published at `huggingface.co/datasets/MykolaL/FBIS-22M`.
  - v2: **FBIS-73M** — 73 M instances, 61 countries, resolution-specific curation to fix
    merged-parcel labels ([arXiv:2607.19069](https://arxiv.org/abs/2607.19069), ECCV 2026 WS).
- **Input spec:** **single-date 3-band RGB GeoTIFF**, resolution-agnostic (0.25–10 m);
  Sentinel-2 10 m true-color works out of the box. Instance segmentation (no
  polygonize step needed; output is already per-field instances).
- **Claimed metrics:**
  - v1 full: **mAP@0.5 = 0.720, mAP@0.5:0.95 = 0.477**, 25.0 ms latency;
    v1-S: 0.632 / 0.383, 16.8 ms (repo model table). Paper: +88.5% mAP@0.5 and +103%
    mAP@0.5:0.95 vs prior SOTA.
  - v2: **+0.284 mAP@0.5 over v1 (+103.3% relative)** on a new manually curated
    100-country zero-shot benchmark; maps Ukraine (603,000 km²) in 5.4 h on one RTX 5070 Ti.
  - DelAnyFlow ([arXiv:2511.13417](https://arxiv.org/abs/2511.13417)): production
    pipeline (inference + merge + vectorize + simplify); claims >100% higher mAP and
    400× faster inference than SAM2; Ukraine 2024 run: 3.75 M fields @5 m / 5.15 M @2.5 m
    vs 2.66 M (Sinergise) / 1.69 M (NASA Harvest) operational products.
- **Inference:** place RGB GeoTIFFs in `data/images/`, run
  `python delineate.py -b batch_sample.yaml` → vectorized boundaries as **GeoPackage** in
  `data/delineated/`. Also: Colab demo; **Autobounds** browser app (2025-09) runs DelAny
  client-side.
- **Note for KZ:** single-date RGB is the cheapest input; v2 is the strongest global
  zero-shot model available. AGPL-3.0 applies.

## 3. Other projects surveyed (none ship usable field-boundary weights)

- **AI4SmallFarms** — https://github.com/jeroengrift/AI4SmallFarms (Persello et al.,
  IEEE GRSL 2023; dataset on DANS). U-Net (`satellite_unet`/`custom_unet`) for S2/Google
  imagery in SE Asia + NL. **Verified via GitHub git-tree API: no weight files
  (`.h5/.pt/.ckpt/...`) in the repo** — `base.py` references `BEST_ASIA_S2` etc. but they
  are not published. Pipeline also requires ImageJ watershed + ArcGIS Pro polygonization.
  → dataset useful, model not reproducible-as-pretrained.
- **Kerner et al. 2023** (*Field boundary delineation with multi-region transfer
  learning*, AAAI-23 AI2SE) — no public repo/weights found (nasaharvest org has no such
  repo).
- **`Revnev/field-boundary-detection`** (HuggingFace) — repo exists but contains only
  `.gitattributes`; **no weights** (verified via HF API).
- **Agribound** — https://github.com/montimaj/agribound (Zenodo DOI 19229666): a Python
  *meta-package* combining seven delineation methods (pretrained segmentation, geospatial
  foundation-model embeddings, SAM-class models). It orchestrates the models above
  rather than providing a new checkpoint; potentially useful as an alternative ensemble
  runner, unverified here.
- **SAM-based field delineation** (e.g. Ha et al. 2026, *Field boundary delineation with
  seasonal Sentinel-2 imagery using SAM*) — uses **vanilla pretrained SAM**, no
  field-specific fine-tuned weights published.
- **Zenodo scan** ("field boundary" + model + sentinel-2): hits are datasets (GloCAB,
  MT4AFE, FieldSeg reference boundaries, AI4SmallFarms S1 extension), not model weights.
- **eCognition/OBIA tools** (e.g. `fkroeber/field_boundary_delineation`) — rule-based,
  no DL weights.

## 4. Verification of URLs in this repo

| Location | URL | Status (2026-07-26) |
|---|---|---|
| `scripts/download_models.py:23` | `hf.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt` | ✅ 206, 124.7 MB |
| `scripts/download_models.py:24` | `.../DelineateAnything-S.pt` | ✅ 206, 17.6 MB |
| `scripts/download_models.py:25` | `.../DelineateAnythingv2.pt` | ✅ 206, 124.7 MB |
| `scripts/config.py:26-29` (`FTW_PRUE_B5_URL`) | `github.com/fieldsoftheworld/ftw-baselines/releases/download/v3/prue_efnet5_checkpoint.ckpt` | ✅ 206, 125,916,483 B; matches `FTW_PRUE_EFNET_B5` default entry in upstream model registry |

Caveats:
- The HF URLs 302-redirect to the Xet CDN; `requests.get(stream=True)` and `curl -L`
  both follow this fine (as used in `download_models.py`).
- `config.py` pins the **v3 (mixed-license, includes CC-BY-NC-SA data)** checkpoint. If a
  fully permissive license matters, switch `FTW_PRUE_B5_URL` to the v3.1 CC-BY asset
  (`.../v3.1/prue_efnetb5_ccby_checkpoint.ckpt`, verified above); metric delta is small
  (object F1 0.46 → 0.41).
- `FTW_MODEL_NAME = "FTW_PRUE_EFNET_B5"` matches the current upstream registry name.

## 5. Recommendation for the KZ experiment

1. **DelAny v2** (single-date S2 RGB @10 m, strongest claimed zero-shot quality,
   instance masks → polygons directly) as primary — matches this repo's README.
2. **FTW PRUE B5 (v3)** as the bi-temporal specialist/ensemble partner (needs 2 scenes;
   semantic mask + polygonize, object F1 0.46 claimed).
3. No third model with downloadable weights worth adding was found; AI4SmallFarms could
   only serve as an extra evaluation dataset.
