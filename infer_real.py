"""
infer_real.py
Run inference on real game screenshots in images/,
draw bounding boxes + labels, save to runs/infer_real/.

Usage:
  python infer_real.py [model_path]

Default model: runs/detect/majsoul_confused_boost/weights/best.pt
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
IMG_DIR      = PROJECT_ROOT / "images"
OUT_DIR      = PROJECT_ROOT / "runs/infer_real"

CONF_THRESH  = 0.25
IOU_THRESH   = 0.45

# colour per suit (BGR)
COLOURS = {
    'm': (50,  180, 255),   # orange  — man
    'p': (80,  220,  80),   # green   — pin
    's': (255, 120,  60),   # blue    — sou
    '_': (200, 200,  60),   # cyan    — honours (east/west/…)
}

def tile_colour(name: str):
    if name[-1] in ('m', 'p', 's'):
        return COLOURS[name[-1]]
    return COLOURS['_']


def draw(img: np.ndarray, boxes, names) -> np.ndarray:
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = names[cls]
        col   = tile_colour(label)

        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), col, -1)
        cv2.putText(out, txt, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                 PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    names = model.names

    imgs = sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpg"))
    print(f"Model : {model_path.name}")
    print(f"Images: {len(imgs)}")
    print(f"Output: {OUT_DIR}\n")

    total_det = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results = model(gray3, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
        n = len(results.boxes)
        total_det += n

        out_img = draw(img, results.boxes, names)
        out_path = OUT_DIR / img_path.name
        cv2.imwrite(str(out_path), out_img)
        print(f"  {img_path.name}: {n} detections")

    print(f"\nTotal detections: {total_det} across {len(imgs)} images")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
