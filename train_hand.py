"""
train_hand.py
全新手牌專用 YOLOv8 模型（只偵測 13+1 手牌）。

資料來源: dataset_hand/images/train/（合成手牌圖）
起點: yolov8n.pt（COCO pretrained，與全畫面模型無關）
輸出: runs/detect/majsoul_hand_phase2/weights/best.pt

Usage:
    python train_hand.py
"""

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
HAND_DIR     = PROJECT_ROOT / "dataset_hand"
HAND_IMGS    = HAND_DIR / "images/train"
HAND_LBLS    = HAND_DIR / "labels/train"
RUNS_DIR     = PROJECT_ROOT / "runs/detect"

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

# ── Step 1: Build train/val split ──────────────────────────────────────────────

def build_split() -> Path:
    """
    Split dataset_hand/images/train/ 80/20 → dataset_hand/train/ + dataset_hand/val/.
    Clears and rebuilds the split directories each call. Returns path to data.yaml.
    """
    print("Building train/val split...")

    for tag in ("train", "val"):
        for sub in ("images", "labels"):
            d = HAND_DIR / tag / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    tr_imgs = HAND_DIR / "train/images"
    tr_lbls = HAND_DIR / "train/labels"
    va_imgs = HAND_DIR / "val/images"
    va_lbls = HAND_DIR / "val/labels"

    all_imgs = sorted(HAND_IMGS.glob("*.jpg"))
    if not all_imgs:
        raise FileNotFoundError(f"No images in {HAND_IMGS}")

    n_train = int(len(all_imgs) * 0.8)
    split   = {"train": all_imgs[:n_train], "val": all_imgs[n_train:]}

    dst_map = {"train": (tr_imgs, tr_lbls), "val": (va_imgs, va_lbls)}
    for tag, imgs in split.items():
        dst_i, dst_l = dst_map[tag]
        for img in imgs:
            shutil.copy2(img, dst_i / img.name)
            lbl = HAND_LBLS / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_l / lbl.name)
            else:
                (dst_l / (img.stem + ".txt")).touch()
        print(f"  {tag}: {len(imgs)} images")

    # Write data.yaml
    cfg = {
        "path":  HAND_DIR.as_posix(),
        "train": "train/images",
        "val":   "val/images",
        "nc":    34,
        "names": CLASS_NAMES,
    }
    data_yaml = HAND_DIR / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)
    print(f"  data.yaml → {data_yaml}")
    return data_yaml


# ── Step 2: Phase 1 — Frozen backbone ─────────────────────────────────────────

def train_phase1(data_yaml: Path) -> Path:
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen backbone warm-up (20 epochs)")
    print("=" * 60)

    model = YOLO("yolov8n.pt")

    model.train(
        data          = str(data_yaml),
        epochs        = 20,
        imgsz         = 1024,        # 高解析度：每牌 ~73×109px（vs 46×70px@640）
        rect          = True,        # 矩形訓練，避免手牌條狀圖極端 letterbox 浪費
        batch         = 16,          # imgsz=1024 顯存需求較大
        device        = 0,
        amp           = True,
        freeze        = 10,          # 凍結 backbone layers 0-9
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.1,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,
        close_mosaic  = 5,
        patience      = 20,
        max_det       = 20,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=2.0, translate=0.03, scale=0.3,   # 提高尺度變化，應對實際遊戲尺度差異
        flipud=0.0, fliplr=0.0,
        mosaic=0.0,
        mixup=0.0, copy_paste=0.0,
        project      = str(RUNS_DIR),
        name         = "majsoul_hand_phase1",
        exist_ok     = True,
        plots        = True,
        save_period  = -1,
        seed         = 42,
        verbose      = True,
    )

    ckpt = RUNS_DIR / "majsoul_hand_phase1/weights/best.pt"
    if not ckpt.exists():
        ckpt = RUNS_DIR / "majsoul_hand_phase1/weights/last.pt"
    print(f"\nPhase 1 complete. Checkpoint: {ckpt}")
    return ckpt


# ── Step 3: Phase 2 — Full fine-tune ──────────────────────────────────────────

def train_phase2(phase1_ckpt: Path, data_yaml: Path) -> Path:
    print("\n" + "=" * 60)
    print("PHASE 2: Full fine-tune (100 epochs)")
    print("=" * 60)

    model = YOLO(str(phase1_ckpt))

    results = model.train(
        data          = str(data_yaml),
        epochs        = 100,
        imgsz         = 1024,
        rect          = True,
        batch         = 16,
        device        = 0,
        amp           = True,
        freeze        = None,
        optimizer     = "SGD",
        lr0           = 0.005,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        cos_lr        = True,
        warmup_epochs = 1,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.05,
        close_mosaic  = 10,
        patience      = 40,
        max_det       = 20,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=2.0, translate=0.03, scale=0.3,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0,
        mixup=0.0, copy_paste=0.0,
        project      = str(RUNS_DIR),
        name         = "majsoul_hand_phase2",
        exist_ok     = True,
        plots        = True,
        save_period  = -1,
        seed         = 42,
        verbose      = True,
    )

    ckpt = RUNS_DIR / "majsoul_hand_phase2/weights/best.pt"
    if not ckpt.exists():
        ckpt = RUNS_DIR / "majsoul_hand_phase2/weights/last.pt"

    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"\nPhase 2 complete. mAP50={map50:.4f}")
    print(f"Final model: {ckpt}")
    return ckpt


# ── Fine-tune from existing checkpoint ────────────────────────────────────────

def fine_tune(ckpt: Path, data_yaml: Path,
              epochs: int = 60,
              run_name: str = "majsoul_hand_finetune") -> Path:
    """Continue training from an existing checkpoint with a lower LR."""
    print("\n" + "=" * 60)
    print(f"FINE-TUNE: {run_name} ({epochs} epochs from {ckpt.name})")
    print("=" * 60)

    model = YOLO(str(ckpt))

    results = model.train(
        data          = str(data_yaml),
        epochs        = epochs,
        imgsz         = 1024,
        rect          = True,
        batch         = 16,
        device        = 0,
        amp           = True,
        freeze        = None,
        optimizer     = "SGD",
        lr0           = 0.0003,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        cos_lr        = True,
        warmup_epochs = 1,
        close_mosaic  = 5,
        patience      = 30,
        max_det       = 20,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=2.0, translate=0.03, scale=0.3,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        project      = str(RUNS_DIR),
        name         = run_name,
        exist_ok     = True,
        plots        = True,
        save_period  = -1,
        seed         = 42,
        verbose      = True,
    )

    out_ckpt = RUNS_DIR / run_name / "weights/best.pt"
    if not out_ckpt.exists():
        out_ckpt = RUNS_DIR / run_name / "weights/last.pt"

    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"\nFine-tune complete. mAP50={map50:.4f}, model: {out_ckpt}")
    return out_ckpt


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not HAND_IMGS.exists():
        print(f"ERROR: {HAND_IMGS} not found. Run hand_synth.py first.")
        return

    data_yaml    = build_split()
    phase1_ckpt  = train_phase1(data_yaml)
    final_ckpt   = train_phase2(phase1_ckpt, data_yaml)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model: {final_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
