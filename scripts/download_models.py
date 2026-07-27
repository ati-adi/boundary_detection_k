#!/usr/bin/env python3
"""Download pretrained model weights for DelAny (v1 + v2) and FTW PRUE."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DELANY_FULL,
    DELANY_S,
    DELANY_V2,
    FTW_PRUE_B5,
    FTW_PRUE_B5_URL,
    MODELS_DIR,
)

WEIGHTS = {
    DELANY_FULL: "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt",
    DELANY_S: "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything-S.pt",
    DELANY_V2: "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnythingv2.pt",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already exists: {dest}")
        return

    print(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for dest, url in WEIGHTS.items():
        download_file(url, dest)

    if not FTW_PRUE_B5.exists() or FTW_PRUE_B5.stat().st_size < 1_000_000:
        print(f"Downloading FTW PRUE B5 → {FTW_PRUE_B5}")
        subprocess.run(["curl", "-L", "-o", str(FTW_PRUE_B5), FTW_PRUE_B5_URL], check=True)
    else:
        print(f"Already exists: {FTW_PRUE_B5}")

    # Enable PRUE checkpoints on ftw-tools 1.4.3 (Python 3.11)
    patch = Path(__file__).resolve().parent / "patch_ftw_prue.py"
    subprocess.run([sys.executable, str(patch)], check=False)

    print("\nDone. Models ready under models/")


if __name__ == "__main__":
    main()
