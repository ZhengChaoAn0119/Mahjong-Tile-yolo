"""
train.py
Two-phase transfer learning with yolov8n.pt on the combined augmented+synthetic dataset.

Phase 1: Frozen backbone (layers 0-9), AdamW, 20 epochs — warms up detection head
Phase 2: Full fine-tune, SGD + cosine LR, 150 epochs

Usage:
  python train.py

Output:
  runs/detect/majsoul_phase1/weights/best.pt
  runs/detect/majsoul_phase2/weights/best.pt  ← use this for inference
"""

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
AUG_TRAIN    = PROJECT_ROOT / "dataset_augmented/train"
SYN_TRAIN    = PROJECT_ROOT / "dataset_synthetic/train"
COMBINED_DIR = PROJECT_ROOT / "dataset_combined"
RUNS_DIR     = PROJECT_ROOT / "runs/detect"

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

# ─── Common augmentation settings (game screenshot specific) ─────────────────
COMMON_AUG = dict(
    hsv_h      = 0.0,   # DISABLED: grayscale data, hue shift is no-op
    hsv_s      = 0.0,   # DISABLED: grayscale data
    hsv_v      = 0.2,   # mild brightness variation only
    degrees    = 3.0,   # slight rotation ±3°
    translate  = 0.05,
    scale      = 0.2,
    shear      = 0.0,
    perspective= 0.0,
    flipud     = 0.0,   # DISABLED: tiles have vertical orientation
    fliplr     = 0.0,   # DISABLED: Chinese characters are not mirror-symmetric
    mosaic     = 0.5,   # 50% online mosaic (on top of offline mosaic data)
    mixup      = 0.0,
    copy_paste = 0.0,
)


# ─── Step 1: Build combined dataset ──────────────────────────────────────────

def build_combined_dataset():
    """Merge augmented + synthetic training images into dataset_combined/."""
    print("Building combined dataset...")
    dst_imgs = COMBINED_DIR / "train/images"
    dst_lbls = COMBINED_DIR / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    total = 0
    for src_dir in [AUG_TRAIN, SYN_TRAIN]:
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found, skipping")
            continue
        imgs = list((src_dir / "images").glob("*.jpg"))
        for img in imgs:
            shutil.copy2(img, dst_imgs / img.name)
            lbl = src_dir / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbls / lbl.name)
            else:
                (dst_lbls / (img.stem + ".txt")).touch()  # empty label file
        total += len(imgs)
        print(f"  Copied {len(imgs)} images from {src_dir.parent.name}/")

    # Write data.yaml
    val_path  = (PROJECT_ROOT / "majsoul.v4i.yolov8/valid/images").as_posix()
    test_path = (PROJECT_ROOT / "majsoul.v4i.yolov8/test/images").as_posix()
    data_cfg = {
        "path": COMBINED_DIR.as_posix(),
        "train": "train/images",
        "val": val_path,
        "test": test_path,
        "nc": 34,
        "names": CLASS_NAMES,
    }
    data_yaml_path = COMBINED_DIR / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=None, allow_unicode=True)

    print(f"  Total combined training images: {total}")
    print(f"  data.yaml: {data_yaml_path}")
    return data_yaml_path


# ─── Step 2: Phase 1 — Frozen backbone ───────────────────────────────────────

def train_phase1(data_yaml_path):
    """
    Freeze backbone (layers 0-9), train detection head + neck for 20 epochs.
    Uses AdamW with higher LR to quickly adapt the head to 34 mahjong classes.
    """
    print("\n" + "="*60)
    print("PHASE 1: Frozen backbone warm-up (20 epochs)")
    print("="*60)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data         = str(data_yaml_path),
        epochs       = 20,
        imgsz        = 640,
        batch        = 16,
        device       = 0,
        amp          = True,
        freeze       = 10,          # freeze layers 0-9 (backbone)
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.1,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        close_mosaic = 5,
        patience     = 20,
        max_det      = 300,
        iou          = 0.7,
        conf         = None,        # use default during training
        project      = str(RUNS_DIR),
        name         = "majsoul_phase1",
        exist_ok     = False,
        plots        = True,
        save_period  = -1,          # only save best/last
        seed         = 42,
        verbose      = True,
        **COMMON_AUG,
    )

    best_ckpt = RUNS_DIR / "majsoul_phase1/weights/best.pt"
    if not best_ckpt.exists():
        best_ckpt = RUNS_DIR / "majsoul_phase1/weights/last.pt"

    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"\nPhase 1 complete. mAP50={map50:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    return best_ckpt


# ─── Step 3: Phase 2 — Full fine-tune ────────────────────────────────────────

def train_phase2(phase1_ckpt, data_yaml_path):
    """
    Unfreeze all layers, full fine-tune with SGD + cosine LR decay for 150 epochs.
    Starts from Phase 1 best weights.
    """
    print("\n" + "="*60)
    print("PHASE 2: Full fine-tune (150 epochs)")
    print("="*60)

    model = YOLO(str(phase1_ckpt))

    results = model.train(
        data         = str(data_yaml_path),
        epochs       = 150,
        imgsz        = 640,
        batch        = 16,
        device       = 0,
        amp          = True,
        freeze       = None,        # unfreeze all layers
        optimizer    = "SGD",
        lr0          = 0.005,       # lower than default (backbone already warmed)
        lrf          = 0.01,        # decay to 5e-5 at end
        momentum     = 0.937,
        weight_decay = 0.0005,
        cos_lr       = True,        # cosine annealing
        warmup_epochs= 1,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.05,
        close_mosaic = 10,          # disable mosaic last 10 epochs for stable eval
        patience     = 50,
        max_det      = 300,
        iou          = 0.7,
        project      = str(RUNS_DIR),
        name         = "majsoul_phase2",
        exist_ok     = False,
        plots        = True,
        save_period  = -1,
        seed         = 42,
        verbose      = True,
        **COMMON_AUG,
    )

    best_ckpt = RUNS_DIR / "majsoul_phase2/weights/best.pt"
    if not best_ckpt.exists():
        best_ckpt = RUNS_DIR / "majsoul_phase2/weights/last.pt"

    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"\nPhase 2 complete. mAP50={map50:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    return best_ckpt, map50


# ─── Step 4: Final evaluation ─────────────────────────────────────────────────

def final_evaluation(model_path, data_yaml_path):
    """Run final validation and print per-class summary."""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    model = YOLO(str(model_path))

    # Validate on validation set
    metrics = model.val(
        data    = str(data_yaml_path),
        split   = "val",
        imgsz   = 640,
        batch   = 8,
        device  = 0,
        conf    = 0.001,
        iou     = 0.6,
        max_det = 300,
        plots   = True,
        verbose = True,
    )

    map50_all = metrics.box.map50
    print(f"\nOverall mAP50: {map50_all:.4f}")

    # Per-class results
    print("\nPer-class mAP50:")
    print(f"{'Class':<10} {'mAP50':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 40)
    weak_classes = []
    for i, name in enumerate(CLASS_NAMES):
        try:
            p, r, ap50, ap = metrics.box.class_result(i)
        except Exception:
            continue
        flag = " ← WEAK" if ap50 < 0.95 else ""
        print(f"{name:<10} {ap50:>8.4f} {p:>10.4f} {r:>8.4f}{flag}")
        if ap50 < 0.95:
            weak_classes.append(name)

    print(f"\nWeak classes (mAP50 < 0.95): {weak_classes}")
    if map50_all >= 0.98:
        print("\nTARGET ACHIEVED: mAP50 >= 0.98!")
    else:
        print(f"\nTarget not yet reached (need 0.98, got {map50_all:.4f})")
        print("Next step: run analyze_confusion.py then iterative_improve.py")

    return map50_all, weak_classes


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Check that augmented and synthetic datasets exist
    if not AUG_TRAIN.exists():
        print("ERROR: dataset_augmented/ not found. Run augment_data.py first.")
        return
    if not SYN_TRAIN.exists():
        print("ERROR: dataset_synthetic/ not found. Run synthetic_data.py first.")
        return

    # Build combined dataset
    data_yaml = build_combined_dataset()

    # Phase 1: frozen backbone
    phase1_ckpt = train_phase1(data_yaml)

    # Phase 2: full fine-tune
    phase2_ckpt, map50 = train_phase2(phase1_ckpt, data_yaml)

    # Final evaluation
    # Use the original data.yaml for eval (points to original valid/test splits)
    original_yaml = PROJECT_ROOT / "majsoul.v4i.yolov8/data.yaml"
    final_evaluation(phase2_ckpt, data_yaml)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Final model: {phase2_ckpt}")
    print(f"mAP50: {map50:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
