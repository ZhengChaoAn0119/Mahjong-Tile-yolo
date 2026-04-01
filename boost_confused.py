"""
boost_confused.py
Targeted boost for confused class pairs.

Strategy:
  1. Read misclass_report.json — get top confused pairs.
  2. Collect images containing EITHER class of each confused pair.
  3. Apply harder per-image augmentation (brightness / contrast / blur)
     so the model learns to distinguish near-identical-looking tiles.
  4. Fine-tune from majsoul_boost2/best.pt for 60 epochs.

Usage:
  python boost_confused.py
"""

import json, shutil, random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT  = Path("E:/project/majsoul_yolo")
REPORT_FILE   = PROJECT_ROOT / "runs/detect/majsoul_boost2/analysis/misclass_report.json"
BASE_MODEL    = PROJECT_ROOT / "runs/detect/majsoul_boost2/weights/best.pt"
MERGED_TR     = PROJECT_ROOT / "dataset_merged/train"
BGSWAP_TR     = PROJECT_ROOT / "dataset_bgswap/train"
BGSWAP_VAL    = PROJECT_ROOT / "dataset_bgswap/val"
OUT_DIR       = PROJECT_ROOT / "dataset_confused_boost"

TOP_N_PAIRS   = 14    # use top N confused pairs from report
BOOST_COPIES  = 8     # duplicate each confused-class image N times
BOOST_EPOCHS  = 60
BOOST_LR0     = 0.00015
SEED          = 42

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]
NAME2ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def hard_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply harder brightness / contrast / blur augmentation."""
    # Random brightness
    beta = rng.uniform(-40, 40)
    img = np.clip(img.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)

    # Random contrast
    alpha = rng.uniform(0.7, 1.3)
    img = np.clip(alpha * img.astype(np.float32), 0, 255).astype(np.uint8)

    # Occasional gaussian blur (simulate motion / low resolution)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Occasional JPEG re-compression artefacts
    if rng.random() < 0.3:
        quality = rng.randint(55, 85)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img


def get_confused_class_ids(report_path: Path, top_n: int) -> set:
    with open(report_path) as f:
        report = json.load(f)
    pairs = report["top_confused_pairs"][:top_n]
    ids = set()
    for p in pairs:
        a, b = p["actual"], p["predicted"]
        if a in NAME2ID:
            ids.add(NAME2ID[a])
        if b in NAME2ID:
            ids.add(NAME2ID[b])
    print(f"Confused classes ({len(ids)}): {sorted([CLASS_NAMES[i] for i in ids])}")
    return ids


def collect_images_with_classes(src_dirs, target_ids: set):
    found = []
    for src in src_dirs:
        lbl_dir = src / "labels"
        img_dir = src / "images"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text().strip().splitlines()
            ids = {int(l.split()[0]) for l in lines if l.strip()}
            if ids & target_ids:
                img = img_dir / (lbl_file.stem + ".jpg")
                if img.exists():
                    found.append((img, lbl_file))
    return found


def build_dataset(confused_ids: set, rng: random.Random):
    dst_imgs = OUT_DIR / "train/images"
    dst_lbls = OUT_DIR / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    # ── Copy full merged train as base ──────────────────────────────────────
    base_count = 0
    for img in (MERGED_TR / "images").glob("*.jpg"):
        shutil.copy2(img, dst_imgs / img.name)
        lbl = MERGED_TR / "labels" / (img.stem + ".txt")
        dst_lbl = dst_lbls / (img.stem + ".txt")
        shutil.copy2(lbl, dst_lbl) if lbl.exists() else dst_lbl.touch()
        base_count += 1

    # Also add bgswap train
    for img in (BGSWAP_TR / "images").glob("*.jpg"):
        shutil.copy2(img, dst_imgs / img.name)
        lbl = BGSWAP_TR / "labels" / (img.stem + ".txt")
        dst_lbl = dst_lbls / (img.stem + ".txt")
        shutil.copy2(lbl, dst_lbl) if lbl.exists() else dst_lbl.touch()
        base_count += 1
    print(f"  Base: {base_count} images")

    # ── Boost confused-class images with hard augmentation ──────────────────
    src_dirs = [MERGED_TR, BGSWAP_TR]
    confused_samples = collect_images_with_classes(src_dirs, confused_ids)
    rng.shuffle(confused_samples)
    added = 0
    for img_path, lbl_path in confused_samples:
        src_img = cv2.imread(str(img_path))
        if src_img is None:
            continue
        for k in range(BOOST_COPIES):
            aug = hard_augment(src_img.copy(), rng)
            out_name = f"conf_{k}_{img_path.name}"
            cv2.imwrite(str(dst_imgs / out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shutil.copy2(lbl_path, dst_lbls / (out_name.replace(".jpg", ".txt")))
            added += 1
    print(f"  Confused-class boost: +{added} hard-augmented images "
          f"({len(confused_samples)} unique × {BOOST_COPIES})")

    # ── data.yaml ────────────────────────────────────────────────────────────
    cfg = {
        "path":  OUT_DIR.as_posix(),
        "train": (OUT_DIR / "train/images").as_posix(),
        "val":   (BGSWAP_VAL / "images").as_posix(),
        "nc":    34,
        "names": CLASS_NAMES,
    }
    data_yaml = OUT_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)
    return data_yaml


def main():
    rng = random.Random(SEED)

    print("="*60)
    print("CONFUSED-CLASS BOOST")
    print("="*60)

    confused_ids = get_confused_class_ids(REPORT_FILE, TOP_N_PAIRS)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    print("\nBuilding dataset...")
    data_yaml = build_dataset(confused_ids, rng)

    print(f"\nFine-tuning from: {BASE_MODEL}")
    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data         = str(data_yaml),
        epochs       = BOOST_EPOCHS,
        imgsz        = 640,
        batch        = 16,
        device       = 0,
        amp          = True,
        optimizer    = "AdamW",
        lr0          = BOOST_LR0,
        lrf          = 0.1,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 2,
        close_mosaic = 5,
        patience     = 15,
        # keep colour augmentation minimal (grayscale-like game)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=3.0, translate=0.05, scale=0.15,
        flipud=0.0, fliplr=0.0,
        mosaic=0.6, mixup=0.0, copy_paste=0.0,
        project  = str(PROJECT_ROOT / "runs/detect"),
        name     = "majsoul_confused_boost",
        exist_ok = True,
    )

    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    best  = PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt"
    print("\n" + "="*60)
    print("CONFUSED BOOST COMPLETE")
    print(f"Model : {best}")
    print(f"mAP50 : {map50}")
    print("="*60)


if __name__ == "__main__":
    main()
