"""
download_tile_images.py
從 GitHub 下載 34 class 麻將牌圖片，替換 tile_crops/ 中的圖片。

GitHub repo: ZhengChaoAn0119/Mahjong-Tile-Classification-with-ResNet18-34-Classes-
每個 class 下載所有圖片，存到 tile_crops/{mid}/ (mid = MODEL_NAMES 的 index)
"""

import requests
import shutil
from pathlib import Path

# MODEL_NAMES (alphabetical, 與 tile_codec.py 一致)
MODEL_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

REPO    = "ZhengChaoAn0119/Mahjong-Tile-Classification-with-ResNet18-34-Classes-"
BRANCH  = "main"
SUBDIR  = "dataset/train"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{SUBDIR}"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{SUBDIR}"

CROPS_DIR = Path(__file__).parent / "tile_crops"


def list_files(class_name: str) -> list[str]:
    """Return list of filenames in dataset/train/{class_name}/ via GitHub API."""
    url = f"{API_URL}/{class_name}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return [item["name"] for item in resp.json() if item["type"] == "file"]


def download_file(url: str, dest: Path):
    resp = requests.get(url, timeout=30, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        shutil.copyfileobj(resp.raw, f)


def main():
    import time

    for mid, name in enumerate(MODEL_NAMES):
        folder = CROPS_DIR / str(mid)
        folder.mkdir(parents=True, exist_ok=True)

        # Skip if already has images
        existing = list(folder.iterdir()) if folder.is_dir() else []
        if existing:
            print(f"[{mid:2d}] {name:6s} — skip ({len(existing)} files already)")
            continue

        print(f"[{mid:2d}] {name:6s} — listing files...", end=" ", flush=True)
        try:
            files = list_files(name)
        except Exception as e:
            print(f"ERROR listing: {e}")
            continue

        if not files:
            print("no files found")
            continue

        print(f"{len(files)} files — downloading...", end=" ", flush=True)
        ok = 0
        for fname in files:
            dest = folder / fname
            if dest.exists():
                ok += 1
                continue
            raw = f"{RAW_URL}/{name}/{fname}"
            try:
                download_file(raw, dest)
                ok += 1
            except Exception as e:
                print(f"\n  WARN {fname}: {e}", end="")
        print(f"done ({ok}/{len(files)})")
        time.sleep(2)  # avoid API rate limit

    print("\nAll done. tile_crops/ updated.")


if __name__ == "__main__":
    main()
