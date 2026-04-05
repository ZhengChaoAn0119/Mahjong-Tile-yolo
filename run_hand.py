"""
run_hand.py
自動化手牌模型訓練 Pipeline。

流程：
  1. 確認初始資料 (≥500 張)，不足則自動補齊
  2. Phase 1 + Phase 2 訓練
  3. 評估 mAP50 與 per-class AP50
  4. 若未達標 → 弱類別加權補圖 → fine-tune（最多 MAX_ITER 輪）
  5. 印出最終結果

Usage:
    python run_hand.py
"""

from pathlib import Path
from ultralytics import YOLO

# ── Import from sibling scripts ────────────────────────────────────────────────
from hand_synth import generate
from train_hand import (
    build_split, train_phase1, train_phase2, fine_tune,
    HAND_DIR, HAND_IMGS, RUNS_DIR, CLASS_NAMES,
)

# ── Pipeline parameters ────────────────────────────────────────────────────────
INIT_IMGS   = 1000   # 初始圖數（提高到1000，預先補足難類別）
MAP_TARGET  = 0.90   # mAP50 達標閾值
AP_WEAK_THR = 0.80   # 個別 class AP50 低於此 → 弱類別
BOOST_MULT  = 3.0    # 弱類別生成機率倍數
INCR_IMGS   = 500    # 每輪補充張數
MAX_ITER    = 3      # 最大擴增輪數

# 初始資料生成就預先 boost 視覺相似的難類別
INITIAL_BOOST: dict[int, float] = {
    17: 4.0,  # 6s — 外觀和 8s 極相似（矩形竹節，只差數量）
    20: 4.0,  # 7s — 常辨識失敗
    23: 4.0,  # 8s — 常辨識失敗
    26: 4.0,  # 9s — 常辨識失敗
    16: 2.0,  # 6p — 筒子高段也相似
    19: 2.0,  # 7p
    22: 2.0,  # 8p
    25: 2.0,  # 9p
}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_path: Path, data_yaml: Path) -> tuple[float, dict[int, float]]:
    """
    Run validation and return (mAP50, {class_id: ap50}).
    """
    print(f"\n[eval] Validating {model_path.name} ...")
    model   = YOLO(str(model_path))
    metrics = model.val(
        data    = str(data_yaml),
        imgsz   = 1024,
        rect    = True,
        batch   = 16,
        conf    = 0.001,
        iou     = 0.6,
        max_det = 20,
        verbose = False,
    )

    map50 = float(metrics.box.map50)
    per_class: dict[int, float] = {}
    for cls_id in range(len(CLASS_NAMES)):
        try:
            _, _, ap50, _ = metrics.box.class_result(cls_id)
            per_class[cls_id] = float(ap50)
        except Exception:
            per_class[cls_id] = 0.0

    # Print summary
    print(f"  Overall mAP50 = {map50:.4f}  (target: {MAP_TARGET})")
    weak = {c: v for c, v in per_class.items() if v < AP_WEAK_THR}
    if weak:
        print(f"  Weak classes  = {[CLASS_NAMES[c] for c in sorted(weak)]}")
    else:
        print("  All classes >= AP threshold")

    return map50, per_class


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HAND MODEL PIPELINE")
    print("=" * 60)

    # ── Step 0: ensure initial data ───────────────────────────────────────────
    n_existing = len(list(HAND_IMGS.glob("*.jpg")))
    if n_existing < INIT_IMGS:
        need = INIT_IMGS - n_existing
        print(f"\n[data] Only {n_existing} images found, generating {need} more "
              f"(with hard-class boost)...")
        generate(need, class_weights=INITIAL_BOOST)
        n_existing = len(list(HAND_IMGS.glob("*.jpg")))
    print(f"\n[data] Total images available: {n_existing}")

    # ── Step 1: initial split + training ─────────────────────────────────────
    data_yaml  = build_split()
    p1_ckpt    = train_phase1(data_yaml)
    best_ckpt  = train_phase2(p1_ckpt, data_yaml)

    # ── Step 2: iterative improvement ────────────────────────────────────────
    final_map50 = 0.0
    for iteration in range(1, MAX_ITER + 1):
        map50, per_class = evaluate(best_ckpt, data_yaml)
        final_map50 = map50

        if map50 >= MAP_TARGET:
            print(f"\n[pipeline] Target achieved! mAP50={map50:.4f} >= {MAP_TARGET}")
            break

        if iteration == MAX_ITER:
            print(f"\n[pipeline] Reached max iterations ({MAX_ITER}). "
                  f"Final mAP50={map50:.4f}")
            break

        # Find weak classes and build boosted weights
        weak = {c: v for c, v in per_class.items() if v < AP_WEAK_THR}
        class_weights = {c: (BOOST_MULT if c in weak else 1.0) for c in range(34)}

        n_before = len(list(HAND_IMGS.glob("*.jpg")))
        print(f"\n[data] iter {iteration}: generating {INCR_IMGS} more images "
              f"(boost: {[CLASS_NAMES[c] for c in sorted(weak)]})")
        generate(INCR_IMGS, class_weights=class_weights)
        n_after = len(list(HAND_IMGS.glob("*.jpg")))
        print(f"[data] Dataset: {n_before} → {n_after} images")

        # Rebuild split with all new images
        data_yaml = build_split()

        # Fine-tune from current best
        run_name  = f"majsoul_hand_ft{iteration}"
        best_ckpt = fine_tune(best_ckpt, data_yaml, epochs=60, run_name=run_name)

    # ── Final report ──────────────────────────────────────────────────────────
    map50, per_class = evaluate(best_ckpt, data_yaml)
    final_map50 = map50

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Final model : {best_ckpt}")
    print(f"Final mAP50 : {final_map50:.4f}")
    print(f"Total images: {len(list(HAND_IMGS.glob('*.jpg')))}")
    print()
    print(f"{'Class':<10} {'AP50':>6}")
    print("-" * 18)
    for cls_id, ap50 in sorted(per_class.items()):
        flag = " ←" if ap50 < AP_WEAK_THR else ""
        print(f"{CLASS_NAMES[cls_id]:<10} {ap50:>6.3f}{flag}")
    print("=" * 60)


if __name__ == "__main__":
    main()
