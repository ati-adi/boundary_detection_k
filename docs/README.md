# Boundary Detector — Documentation

| Document | Audience | Purpose |
|----------|----------|---------|
| [README.md](../README.md) | Everyone | Project overview and setup |
| [TEST_NEW_TIFF.md](TEST_NEW_TIFF.md) | Users | Run and compare models on a new GeoTIFF |

## Scripts reference

| Script | Description |
|--------|-------------|
| `scripts/run_experiment.py` | **Main entry** — full pipeline on a new TIFF |
| `scripts/crop_aoi.py` | Crop AOI + RGB overview |
| `scripts/cropland_mask.py` | ESA WorldCover cropland mask |
| `scripts/run_delany.py` | Delineate Anything inference (`--model v2\|full\|s`) |
| `scripts/run_ftw.py` | FTW bi-temporal inference + polygonize |
| `scripts/postprocess_boundaries.py` | Topology / area / cropland filter |
| `scripts/ensemble_fields.py` | DelAny v2 + FTW fusion |
| `scripts/visualize.py` | Compare models on AOI (PNG + metrics) |
| `scripts/visualize_full_tile.py` | Full-tile QA overlay |
| `scripts/evaluate.py` | Optional object-F1 vs reference polygons |
| `scripts/find_stac_pair.py` | Find STAC scene pair for FTW |
| `scripts/build_ftw_input.py` | Build 8-band S2 stack for FTW |
| `scripts/download_models.py` | Download DelAny / FTW weights |
| `scripts/patch_ftw_prue.py` | Patch torchgeo for FTW PRUE |
| `scripts/setup_env.sh` | One-time environment setup |

## Typical workflow

```
New TIFF → run_experiment.py → compare/side_by_side.png → pick model → full tile
```

For details, see [TEST_NEW_TIFF.md](TEST_NEW_TIFF.md).
