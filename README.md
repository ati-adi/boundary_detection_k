# Boundary Detector

Agricultural field boundary detection from Sentinel-2 RGB GeoTIFF imagery.

## New TIFF? Start here

```bash
source .venv/bin/activate
python scripts/run_experiment.py --input source_data/your_scene.tiff
```

Open `outputs/your_scene/compare/side_by_side.png`.

Full guide: **[docs/TEST_NEW_TIFF.md](docs/TEST_NEW_TIFF.md)**

## Models

| Model | Role | Input |
|-------|------|-------|
| **Delineate Anything v2** (primary) | Instance segmentation | Single-date RGB GeoTIFF @ 10 m |
| **Delineate Anything / -S** | v1 baselines (`--skip-v1` to omit) | Same |
| **FTW PRUE B5** | S2 bi-temporal specialist | Auto STAC pair → 8-band stack |
| **Ensemble** | DelAny v2 + FTW fusion + cropland filter | Post-processed vectors |

Pipeline extras: ESA WorldCover cropland mask, topology/area postprocess, optional object-F1 vs `--ref`.

## Setup

```bash
# Automated (recommended)
bash scripts/setup_env.sh

# Or manual:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install gdal==$(gdal-config --version)
git clone https://github.com/Lavreniuk/Delineate-Anything.git vendor/Delineate-Anything
pip install -r vendor/Delineate-Anything/requirements.txt
python scripts/download_models.py
python scripts/patch_ftw_prue.py
```

Put input GeoTIFFs under `source_data/` (gitignored).

## Quick Start

```bash
source .venv/bin/activate

python scripts/run_experiment.py \
  --input source_data/your_scene.tiff \
  --run-id sample
```

Faster AOI-only (skip FTW + full tile):

```bash
python scripts/run_experiment.py \
  --input source_data/your_scene.tiff \
  --skip-ftw --skip-full-tile --skip-v1
```

## Project Structure

```
boundary_detector/
├── source_data/     # Input GeoTIFFs (gitignored)
├── outputs/         # Per-run results (gitignored)
├── models/          # Weights via download_models.py (gitignored)
├── vendor/          # Delineate-Anything clone (gitignored)
├── docs/            # User documentation
├── notebooks/       # Comparison notebook
└── scripts/         # Pipeline scripts
```

## Hardware

Optimized for Apple Silicon (MPS). FTW: `--mps_mode` when available. DelAny auto-detects MPS/CUDA/CPU. Full-tile uses higher confidence to avoid NMS stalls.

## License Note

Delineate Anything is AGPL-3.0. FTW tools and models have their own licenses — see respective repos.
