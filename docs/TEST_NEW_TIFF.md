# Testing on a New GeoTIFF

Run field boundary detection and compare models on a **new Sentinel-2 RGB GeoTIFF**.

## Before you start

### Input image requirements

| Requirement | Details |
|-------------|---------|
| Format | GeoTIFF (`.tif` / `.tiff`) |
| Bands | 3 (Red, Green, Blue) |
| Resolution | ~10 m/pixel (Sentinel-2 L2A) |
| Georeferencing | Valid CRS and geotransform |
| NoData | Usually `0` for Sentinel-2 |

```bash
gdalinfo source_data/your_image.tiff
```

### Environment

```bash
cd boundary_detector
source .venv/bin/activate

# First time only:
bash scripts/setup_env.sh
python scripts/download_models.py
python scripts/patch_ftw_prue.py
```

---

## Quick start: one command

```bash
python scripts/run_experiment.py --input source_data/your_image.tiff
```

Results land in **`outputs/your_image/`**:

```
outputs/your_image/
├── data/
│   ├── aoi_subset.tif
│   ├── cropland_mask.tif
│   ├── overview.png
│   └── aoi_meta.json
├── delany_v2/boundaries.gpkg      # primary (postprocessed)
├── delany/boundaries.gpkg         # v1 full (unless --skip-v1)
├── delany_s/boundaries.gpkg       # v1-S (unless --skip-v1)
├── ftw_prue/boundaries.gpkg       # FTW (unless --skip-ftw)
├── ensemble/boundaries.gpkg       # DelAny v2 + FTW
├── compare/
│   ├── side_by_side.png           # ← start here
│   ├── metrics.json
│   └── *_overlay.png
└── run_summary.json
```

### Faster AOI-only test

```bash
python scripts/run_experiment.py \
  --input source_data/your_image.tiff \
  --skip-ftw --skip-full-tile --skip-v1
```

---

## Comparing models

### 1. Visual comparison

Open **`outputs/<run>/compare/side_by_side.png`**.

Checklist:

- [ ] Field corners are sharp (not rounded blobs)
- [ ] Adjacent fields are separated (not merged)
- [ ] Small fields (~0.5–2 ha) are preserved
- [ ] Roads, rivers, and settlements are excluded
- [ ] Boundaries follow visible crop/soil edges

### 2. Metrics

See `compare/metrics.json` for `polygon_count`, `mean_area_ha` / `median_area_ha`, and `edge_alignment` (Canny overlap, higher = sharper).

### 3. Choosing a model

| Model | Best when |
|-------|-----------|
| **DelAny v2** | Default primary; strong single-date RGB |
| **Ensemble** | FTW STAC pair available; fuse with DelAny v2 |
| **DelAny-S** | Many small/fragmented fields; fast iteration |
| **FTW** | Good bi-temporal STAC coverage for the AOI |

---

## Step-by-step (manual control)

### Step 1 — Crop a test AOI

```bash
python scripts/crop_aoi.py \
  --input source_data/your_image.tiff \
  --output-dir outputs/your_image/data \
  --size-px 2000
```

### Step 2 — Cropland mask + DelAny v2

```bash
AOI=outputs/your_image/data/aoi_subset.tif

python scripts/cropland_mask.py \
  --reference "$AOI" \
  --output outputs/your_image/data/cropland_mask.tif

python scripts/run_delany.py --model v2 \
  --input "$AOI" \
  --output outputs/your_image/delany_v2/boundaries_raw.gpkg \
  --entry-name your_image_aoi_v2

python scripts/postprocess_boundaries.py \
  --input outputs/your_image/delany_v2/boundaries_raw.gpkg \
  --output outputs/your_image/delany_v2/boundaries.gpkg \
  --cropland-mask outputs/your_image/data/cropland_mask.tif
```

### Step 3 — FTW (optional)

FTW needs two Sentinel-2 scene IDs from [Planetary Computer](https://planetarycomputer.microsoft.com/explore), or let `run_ftw.py` / `find_stac_pair.py` resolve a pair from `aoi_meta.json`.

```bash
python scripts/run_ftw.py \
  --meta outputs/your_image/data/aoi_meta.json \
  --output-dir outputs/your_image/ftw_prue
```

### Step 4 — Compare

```bash
python scripts/visualize.py \
  --input outputs/your_image/data/aoi_subset.tif \
  --output-dir outputs/your_image/compare \
  --delany-v2 outputs/your_image/delany_v2/boundaries.gpkg \
  --ftw outputs/your_image/ftw_prue/boundaries.gpkg
```

### Step 5 — Scale to full tile

```bash
python scripts/run_delany.py --model v2 \
  --input source_data/your_image.tiff \
  --output outputs/your_image/delany_v2/boundaries_full_raw.gpkg \
  --entry-name your_image_full_v2 \
  --confidence 0.08 --batch-size 2

python scripts/visualize_full_tile.py \
  --input source_data/your_image.tiff \
  --boundaries outputs/your_image/delany_v2/boundaries.gpkg \
  --output outputs/your_image/compare/full_tile_overlay.png
```

---

## Troubleshooting

```bash
pip install gdal==$(gdal-config --version)
pip install torchgeo==0.6.2
python scripts/patch_ftw_prue.py
```

---

## Output formats

Boundary outputs are **GeoPackage** (`.gpkg`) with a `fields` layer:

```python
import geopandas as gpd
gdf = gpd.read_file("outputs/your_image/delany_v2/boundaries.gpkg")
print(len(gdf), "fields")
```
