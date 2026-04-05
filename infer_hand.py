"""
infer_hand.py
手牌條狀圖推論腳本。

針對 998×109 手牌條狀圖設計，使用低信度閾值確保 14 張全偵測。
輸出按 x 座標從左到右排序的 14 張牌（13手牌 + 1摸牌）。

Usage:
    python infer_hand.py --img <hand_strip.jpg>
    python infer_hand.py --img <hand_strip.jpg> --model runs/detect/majsoul_hand_phase2/weights/best.pt
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

DEFAULT_MODEL    = Path("runs/detect/majsoul_hand_phase2/weights/best.pt")
HAND_CONF_LEVELS = [0.20, 0.10, 0.05, 0.02]   # adaptive conf for 13 hand tiles
DRAWN_CONF_THRESH = 0.35                         # drawn tile must exceed this conf
HAND_IOU          = 0.30
HAND_IMGSZ        = 1024


def detect_hand(model: YOLO, hand_img: np.ndarray) -> list[dict]:
    """
    Detect 13+1 hand tiles.

    Phase A — 13 hand tiles: adaptive conf ensures all 13 are found.
    Phase B — drawn tile (14th): only accepted when conf >= DRAWN_CONF_THRESH
              AND positioned to the right of all 13 hand tiles.
              Returns 13 dicts when no drawn tile is held (桌布/empty slot).
    """
    # Single inference at lowest threshold
    r_all      = model(hand_img, imgsz=HAND_IMGSZ, conf=HAND_CONF_LEVELS[-1],
                       iou=HAND_IOU, max_det=20, verbose=False)[0]
    all_sorted = sorted(r_all.boxes, key=lambda b: float(b.xywh[0][0]))

    def box_to_dict(b) -> dict:
        cls_id = int(b.cls[0])
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        return {
            "cls_id": cls_id,
            "name":   MODEL_NAMES[cls_id],
            "conf":   float(b.conf[0]),
            "bbox":   (int(x1), int(y1), int(x2), int(y2)),
            "cx":     (x1 + x2) / 2,
        }

    # Phase A: 13 hand tiles with adaptive conf
    hand_boxes = []
    for conf_thr in HAND_CONF_LEVELS:
        candidates = [b for b in all_sorted if float(b.conf[0]) >= conf_thr]
        if len(candidates) >= 13:
            hand_boxes = candidates[:13]
            break
    if not hand_boxes:
        hand_boxes = all_sorted[:min(13, len(all_sorted))]

    # Phase B: drawn tile validation
    drawn_box = None
    if len(hand_boxes) == 13:
        rightmost_cx = float(hand_boxes[-1].xywh[0][0])
        drawn_candidates = [
            b for b in all_sorted
            if b not in hand_boxes
            and float(b.xywh[0][0]) > rightmost_cx
            and float(b.conf[0]) >= DRAWN_CONF_THRESH
        ]
        if drawn_candidates:
            drawn_box = drawn_candidates[0]

    final_boxes = hand_boxes + ([drawn_box] if drawn_box else [])
    detections  = [box_to_dict(b) for b in final_boxes]

    n_hand  = len(hand_boxes)
    n_drawn = 1 if drawn_box else 0
    print(f"[infer] hand={n_hand}  drawn={'yes' if drawn_box else 'no (empty/桌布)'}")
    return detections


def draw_result(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    vis = img.copy()
    H, W = vis.shape[:2]
    scale = max(1, W // 400)

    colors = {
        "m": (0, 0, 220),    # red  → man
        "p": (220, 60, 0),   # blue → pin
        "s": (0, 180, 0),    # green → sou
    }
    honor_color = (150, 0, 200)

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        name = d["name"]

        if name[-1] in colors:
            color = colors[name[-1]]
        else:
            color = honor_color

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, scale)
        label = f"{name} {d['conf']:.2f}"
        font_scale = 0.35 * scale
        cv2.putText(vis, label, (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, scale)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",   required=True, help="Hand strip image path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model path")
    parser.add_argument("--out",   default=None,  help="Output image path (optional)")
    args = parser.parse_args()

    img_path   = Path(args.img)
    model_path = Path(args.model)

    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        return
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return

    img   = cv2.imread(str(img_path))
    model = YOLO(str(model_path))

    detections = detect_hand(model, img)

    print(f"\nHand ({len(detections)} tiles):")
    for i, d in enumerate(detections):
        slot = "drawn" if i == 13 else f"{i+1:2d}"
        print(f"  [{slot}] {d['name']:<6}  conf={d['conf']:.3f}")

    if not detections:
        print("No tiles detected.")
        return

    # Draw and save/show result
    vis = draw_result(img, detections)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = img_path.parent / (img_path.stem + "_result.jpg")
    cv2.imwrite(str(out_path), vis)
    print(f"\nResult saved: {out_path}")


if __name__ == "__main__":
    main()
