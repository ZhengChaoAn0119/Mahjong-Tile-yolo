"""
train_bgswap.py
Fine-tune from phase2_boost best.pt using merged train set:
  dataset_combined/train  (800 images)
  + dataset_bgswap/train  (432 images)
  = ~1232 images total

Validation set: dataset_bgswap/val (108 images) — far larger than original 12.

Usage:
  python train_bgswap.py
"""

import shutil, yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
COMBINED_TR  = PROJECT_ROOT / "dataset_combined/train"
BGSWAP_TR    = PROJECT_ROOT / "dataset_bgswap/train"
BGSWAP_VAL   = PROJECT_ROOT / "dataset_bgswap/val"
MERGED_DIR   = PROJECT_ROOT / "dataset_merged"
BASE_MODEL   = PROJECT_ROOT / "runs/detect/majsoul_phase2_boost/weights/best.pt"

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]


def build_merged():
    dst_imgs = MERGED_DIR / "train/images"
    dst_lbls = MERGED_DIR / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    total = 0
    for src in [COMBINED_TR, BGSWAP_TR]:
        imgs = list((src / "images").glob("*.jpg"))
        for img in imgs:
            shutil.copy2(img, dst_imgs / img.name)
            lbl = src / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbls / lbl.name)
            else:
                (dst_lbls / (img.stem + ".txt")).touch()
        total += len(imgs)
        print(f"  {src.parent.name}/{src.name}: {len(imgs)} images")

    cfg = {
        "path":  MERGED_DIR.as_posix(),
        "train": (MERGED_DIR / "train/images").as_posix(),
        "val":   (BGSWAP_VAL / "images").as_posix(),
        "nc":    34,
        "names": CLASS_NAMES,
    }
    data_yaml = MERGED_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)
    print(f"  Total: {total} training images")
    return data_yaml


def main():
    print("Building merged dataset...")
    data_yaml = build_merged()

    print(f"\nFine-tuning from: {BASE_MODEL}")
    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data         = str(data_yaml),
        epochs       = 100,
        imgsz        = 640,
        batch        = 16,
        device       = 0,
        amp          = True,
        optimizer    = "SGD",
        lr0          = 0.0005,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        close_mosaic = 10,
        patience     = 20,
        hsv_h        = 0.0,
        hsv_s        = 0.0,
        hsv_v        = 0.2,
        degrees      = 3.0,
        translate    = 0.05,
        scale        = 0.2,
        flipud       = 0.0,
        fliplr       = 0.0,
        mosaic       = 0.5,
        mixup        = 0.0,
        copy_paste   = 0.0,
        project      = str(PROJECT_ROOT / "runs/detect"),
        name         = "majsoul_bgswap",
        exist_ok     = True,
    )

    best = PROJECT_ROOT / "runs/detect/majsoul_bgswap/weights/best.pt"
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Model : {best}")
    print(f"mAP50 : {map50}")
    print("="*60)


if __name__ == "__main__":
    main()
