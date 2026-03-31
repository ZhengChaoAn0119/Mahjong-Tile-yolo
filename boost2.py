"""
boost2.py
Second-round targeted boost from majsoul_bgswap/best.pt.
Reads weak_classes.json from bgswap analysis, boosts them,
then fine-tunes for 50 epochs.

Usage:
  python boost2.py
"""

import json, shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
BASE_MODEL   = PROJECT_ROOT / "runs/detect/majsoul_bgswap/weights/best.pt"
MERGED_TR    = PROJECT_ROOT / "dataset_merged/train"
BGSWAP_VAL   = PROJECT_ROOT / "dataset_bgswap/val"
BOOST2_DIR   = PROJECT_ROOT / "dataset_boost2"

WEAK_THRESH  = 0.90
BOOST_COPIES = 5
BOOST_EPOCHS = 50
BOOST_LR0    = 0.0002

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]


def get_weak_ids_from_model():
    """Run quick val to find weak classes directly."""
    model = YOLO(str(BASE_MODEL))
    data_yaml = PROJECT_ROOT / "dataset_merged/data.yaml"
    results = model.val(
        data=str(data_yaml), imgsz=640, batch=16, device=0,
        split="val", verbose=False,
    )
    ap50 = results.box.ap50
    weak = {}
    for i, name in enumerate(CLASS_NAMES):
        score = float(ap50[i]) if i < len(ap50) else 0.0
        if score < WEAK_THRESH:
            weak[name] = {"class_id": i, "mAP50": score}
    return weak


def collect_weak_images(weak_ids: set):
    found = []
    for lbl_file in (MERGED_TR / "labels").glob("*.txt"):
        lines = lbl_file.read_text().strip().splitlines()
        ids = {int(l.split()[0]) for l in lines if l.strip()}
        if ids & weak_ids:
            img = MERGED_TR / "images" / (lbl_file.stem + ".jpg")
            if img.exists():
                found.append((img, lbl_file))
    return found


def build_boost2_dataset(weak_ids: set):
    dst_imgs = BOOST2_DIR / "train/images"
    dst_lbls = BOOST2_DIR / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    # Copy full merged train
    copied = 0
    for img in (MERGED_TR / "images").glob("*.jpg"):
        shutil.copy2(img, dst_imgs / img.name)
        lbl = MERGED_TR / "labels" / (img.stem + ".txt")
        dst_lbl = dst_lbls / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.touch()
        copied += 1
    print(f"  Base: {copied} images")

    # Boost weak-class images
    weak_samples = collect_weak_images(weak_ids)
    added = 0
    for img, lbl in weak_samples:
        for k in range(BOOST_COPIES):
            shutil.copy2(img, dst_imgs / f"b2_{k}_{img.name}")
            shutil.copy2(lbl, dst_lbls / f"b2_{k}_{lbl.name}")
            added += 1
    print(f"  Weak boost: +{added} images ({len(weak_samples)} unique × {BOOST_COPIES})")

    cfg = {
        "path":  BOOST2_DIR.as_posix(),
        "train": (BOOST2_DIR / "train/images").as_posix(),
        "val":   (BGSWAP_VAL / "images").as_posix(),
        "nc":    34,
        "names": CLASS_NAMES,
    }
    data_yaml = BOOST2_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)
    return data_yaml


def main():
    print("="*60)
    print("BOOST ROUND 2 — from majsoul_bgswap")
    print("="*60)

    print("\nDetecting weak classes...")
    weak = get_weak_ids_from_model()
    if not weak:
        print("No weak classes — model already ≥ 0.90 on all classes.")
        return

    weak_ids = {int(m["class_id"]) for m in weak.values()}
    print(f"Weak ({len(weak)}): " + ", ".join(
        f"{n}({m['mAP50']:.3f})" for n, m in sorted(weak.items(), key=lambda x: x[1]["mAP50"])
    ))

    if BOOST2_DIR.exists():
        shutil.rmtree(BOOST2_DIR)

    print("\nBuilding boost2 dataset...")
    data_yaml = build_boost2_dataset(weak_ids)

    print("\nFine-tuning...")
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
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        degrees=3.0, translate=0.05, scale=0.2,
        flipud=0.0, fliplr=0.0, mosaic=0.5,
        mixup=0.0, copy_paste=0.0,
        project      = str(PROJECT_ROOT / "runs/detect"),
        name         = "majsoul_boost2",
        exist_ok     = True,
    )

    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    best  = PROJECT_ROOT / "runs/detect/majsoul_boost2/weights/best.pt"
    print("\n" + "="*60)
    print("BOOST2 COMPLETE")
    print(f"Model : {best}")
    print(f"mAP50 : {map50}")
    print("="*60)


if __name__ == "__main__":
    main()
