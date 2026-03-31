"""
analyze_confusion.py
Run validation on the trained model, extract per-class P/R/mAP50,
and identify weak classes (mAP50 < threshold).

Usage:
  python analyze_confusion.py

Output:
  runs/detect/majsoul_phase2/analysis/per_class_metrics.json
  runs/detect/majsoul_phase2/analysis/weak_classes.json   ← used by boost_weak.py
"""

import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
MODEL_PATH   = PROJECT_ROOT / "runs/detect/majsoul_phase2/weights/best.pt"
DATA_YAML    = PROJECT_ROOT / "dataset_combined/data.yaml"
OUT_DIR      = PROJECT_ROOT / "runs/detect/majsoul_phase2/analysis"
WEAK_THRESH  = 0.90   # classes below this mAP50 are considered weak

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Running validation...")
    results = model.val(
        data   = str(DATA_YAML),
        imgsz  = 640,
        batch  = 16,
        device = 0,
        split  = "val",
        verbose= False,
    )

    # Extract per-class metrics from results
    metrics = results.box  # BoxMetrics object
    ap50_per_class = metrics.ap50            # shape: (nc,)
    p_per_class    = metrics.p              # precision per class
    r_per_class    = metrics.r              # recall per class

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "class_id": i,
            "precision": float(p_per_class[i]) if i < len(p_per_class) else 0.0,
            "recall":    float(r_per_class[i])  if i < len(r_per_class)  else 0.0,
            "mAP50":     float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0,
        }

    # Sort by mAP50 ascending
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["mAP50"])

    print("\n" + "="*60)
    print("PER-CLASS RESULTS (sorted by mAP50)")
    print("="*60)
    print(f"{'Class':<10} {'P':>7} {'R':>7} {'mAP50':>8}")
    print("-"*36)
    for name, m in sorted_classes:
        flag = " ← WEAK" if m["mAP50"] < WEAK_THRESH else ""
        print(f"{name:<10} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['mAP50']:>8.3f}{flag}")

    overall_map50 = float(metrics.map50)
    print(f"\nOverall mAP50: {overall_map50:.4f}")

    # Identify weak classes
    weak = {name: m for name, m in per_class.items() if m["mAP50"] < WEAK_THRESH}
    print(f"\nWeak classes (mAP50 < {WEAK_THRESH}): {list(weak.keys())}")

    # Save outputs
    metrics_file = OUT_DIR / "per_class_metrics.json"
    weak_file    = OUT_DIR / "weak_classes.json"

    with open(metrics_file, "w") as f:
        json.dump({"overall_mAP50": overall_map50, "classes": per_class}, f, indent=2)

    with open(weak_file, "w") as f:
        json.dump({"weak_threshold": WEAK_THRESH, "weak_classes": weak}, f, indent=2)

    print(f"\nSaved: {metrics_file}")
    print(f"Saved: {weak_file}")

    return weak


if __name__ == "__main__":
    main()
