"""
boost_weak.py
One round of targeted boosting: copy extra augmented samples of weak classes
into dataset_combined, then fine-tune the Phase 2 model for a short burst.

Usage:
  python boost_weak.py

Reads:
  runs/detect/majsoul_phase2/analysis/weak_classes.json
  dataset_augmented/train/  (source of extra samples)

Output:
  runs/detect/majsoul_phase2_boost/weights/best.pt
"""

import json, shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT  = Path("E:/project/majsoul_yolo")
WEAK_FILE     = PROJECT_ROOT / "runs/detect/majsoul_phase2/analysis/weak_classes.json"
BASE_MODEL    = PROJECT_ROOT / "runs/detect/majsoul_phase2/weights/best.pt"
AUG_TRAIN     = PROJECT_ROOT / "dataset_augmented/train"
SYN_TRAIN     = PROJECT_ROOT / "dataset_synthetic/train"
BOOST_DIR     = PROJECT_ROOT / "dataset_boost"
DATA_YAML     = PROJECT_ROOT / "dataset_combined/data.yaml"

BOOST_COPIES  = 4      # replicate each weak-class image N extra times
BOOST_EPOCHS  = 40
BOOST_LR0     = 0.0003


def collect_weak_class_images(weak_ids: set[int]):
    """Scan label files and return image paths whose labels contain a weak class."""
    found = []
    for src in [AUG_TRAIN, SYN_TRAIN]:
        lbl_dir = src / "labels"
        img_dir = src / "images"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text().strip().splitlines()
            ids = {int(l.split()[0]) for l in lines if l.strip()}
            if ids & weak_ids:
                img = img_dir / (lbl_file.stem + ".jpg")
                if img.exists():
                    found.append((img, lbl_file))
    return found


def build_boost_dataset(weak_ids: set[int]):
    dst_imgs = BOOST_DIR / "train/images"
    dst_lbls = BOOST_DIR / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    # Start with the existing combined dataset
    src_imgs = PROJECT_ROOT / "dataset_combined/train/images"
    src_lbls = PROJECT_ROOT / "dataset_combined/train/labels"
    copied = 0
    for img in src_imgs.glob("*.jpg"):
        shutil.copy2(img, dst_imgs / img.name)
        lbl = src_lbls / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_lbls / lbl.name)
        else:
            (dst_lbls / (img.stem + ".txt")).touch()
        copied += 1
    print(f"  Base dataset: {copied} images copied")

    # Duplicate weak-class images
    weak_samples = collect_weak_class_images(weak_ids)
    added = 0
    for img, lbl in weak_samples:
        for k in range(BOOST_COPIES):
            new_name = f"boost_{k}_{img.name}"
            shutil.copy2(img, dst_imgs / new_name)
            new_lbl = dst_lbls / f"boost_{k}_{lbl.name}"
            shutil.copy2(lbl, new_lbl)
            added += 1
    print(f"  Weak-class boost: {added} extra images added ({len(weak_samples)} unique × {BOOST_COPIES})")

    # Reuse the same data.yaml but point train at boost dir
    import yaml
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["train"] = (BOOST_DIR / "train/images").as_posix()
    boost_yaml = BOOST_DIR / "data.yaml"
    with open(boost_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)
    return boost_yaml


def main():
    print("="*60)
    print("BOOST TRAINING — weak class targeted fine-tune")
    print("="*60)

    if not WEAK_FILE.exists():
        print(f"ERROR: {WEAK_FILE} not found. Run analyze_confusion.py first.")
        return

    with open(WEAK_FILE) as f:
        data = json.load(f)

    weak_classes = data["weak_classes"]
    if not weak_classes:
        print("No weak classes found — model is already strong on all classes.")
        return

    weak_ids = {int(m["class_id"]) for m in weak_classes.values()}
    print(f"Weak classes to boost: {list(weak_classes.keys())}")

    if BOOST_DIR.exists():
        shutil.rmtree(BOOST_DIR)

    print("\nBuilding boost dataset...")
    boost_yaml = build_boost_dataset(weak_ids)

    print("\nFine-tuning from Phase 2 best.pt...")
    model = YOLO(str(BASE_MODEL))

    COMMON_AUG = dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        degrees=3.0, translate=0.05, scale=0.2,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.5, mixup=0.0, copy_paste=0.0,
    )

    results = model.train(
        data         = str(boost_yaml),
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
        project      = str(PROJECT_ROOT / "runs/detect"),
        name         = "majsoul_phase2_boost",
        exist_ok     = True,
        **COMMON_AUG,
    )

    best = PROJECT_ROOT / "runs/detect/majsoul_phase2_boost/weights/best.pt"
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")

    print("\n" + "="*60)
    print("BOOST TRAINING COMPLETE")
    print(f"Model: {best}")
    print(f"mAP50: {map50}")
    print("="*60)


if __name__ == "__main__":
    main()
