#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Setting up boundary_detector environment"

if ! command -v conda &>/dev/null; then
    echo "conda not found; using pip with system/homebrew GDAL"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    if ! conda env list | grep -q "boundary_detector"; then
        conda create -n boundary_detector python=3.11 -y
    fi
    eval "$(conda shell.bash hook)"
    conda activate boundary_detector
    conda install -c conda-forge gdal rasterio geopandas -y
    pip install -r requirements.txt
fi

if [ ! -d "vendor/Delineate-Anything" ]; then
    echo "==> Cloning Delineate-Anything"
    git clone https://github.com/Lavreniuk/Delineate-Anything.git vendor/Delineate-Anything
fi

pip install -r vendor/Delineate-Anything/requirements.txt

python scripts/download_models.py

echo "==> Setup complete. Activate with: conda activate boundary_detector"
