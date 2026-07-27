"""Shared configuration for the boundary detection pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_TIFF = PROJECT_ROOT / "source_data" / "S2A_MSIL2A_20250619T053241_R105_T44TNS_20250619T073113.tiff"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VENDOR_DIR = PROJECT_ROOT / "vendor"
DELANY_DIR = VENDOR_DIR / "Delineate-Anything"

AOI_SUBSET = DATA_DIR / "aoi_subset.tif"
AOI_OVERVIEW = DATA_DIR / "overview.png"
AOI_META = DATA_DIR / "aoi_meta.json"

# 20 km at 10 m/px = 2000 px
AOI_SIZE_PX = 2000

# DelAny model weights
DELANY_FULL = MODELS_DIR / "DelineateAnything.pt"
DELANY_S = MODELS_DIR / "DelineateAnything-S.pt"
DELANY_V2 = MODELS_DIR / "DelineateAnythingv2.pt"

# FTW PRUE (v3) — EfficientNet-B5, full training corpus
FTW_PRUE_B5 = MODELS_DIR / "prue_efnet5_checkpoint.ckpt"
FTW_PRUE_B5_URL = (
    "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v3/"
    "prue_efnet5_checkpoint.ckpt"
)
FTW_MODEL_NAME = "FTW_PRUE_EFNET_B5"

# Post-process defaults (S2 ~10 m Kazakhstan-scale fields)
MIN_AREA_HA = 0.25
MAX_AREA_HA = 500.0
SIMPLIFY_TOLERANCE_M = 8.0
CROPLAND_OVERLAP_MIN = 0.35

# ESA WorldCover classes treated as agricultural / field-capable
# 40=Cropland, 30=Grassland (fallow/steppe fields often map to 30)
WORLDCOVER_AG_CLASSES = (40, 30)
# Always exclude these from field inventory
WORLDCOVER_EXCLUDE_CLASSES = (10, 50, 80, 90, 95)  # trees, built, water, wetland, mangrove
WORLDCOVER_CROPLAND_CLASS = 40  # kept for backwards compat

# Planetary Computer
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
