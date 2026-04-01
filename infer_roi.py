"""
infer_roi.py  v3
ROI-based Mahjong Soul tile inference — NO colour filtering.

Key principle: within correctly-defined ROIs, there are NO tile backs.
  - Hand strip   → always face-up (player's own tiles)
  - Center area  → always face-up (discards)
  - Meld strips  → always face-up (open melds)
  Tile backs (牌山) live OUTSIDE these ROIs by design.

Auto-detects 3-player vs 4-player layout by checking the top-left label area.

Usage:
  python infer_roi.py [model_path]
Output: runs/infer_roi/
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
IMG_DIR      = PROJECT_ROOT / "images"
OUT_DIR      = PROJECT_ROOT / "runs/infer_roi"
CONF_THRESH  = 0.28
IOU_THRESH   = 0.35

W, H = 1418, 837

# ── Colour palette for drawing ───────────────────────────────────────────────
TILE_COLOURS = {
    'm': (30,  190, 255),   # orange — man
    'p': (50,  210,  50),   # green  — pin
    's': (255, 110,  40),   # blue   — sou
    '_': (210, 170,  50),   # cyan   — honours
}
ROI_COLOURS = {
    "hand":   (100, 230, 100),
    "center": (100, 100, 230),
    "meld":   (50,  210, 220),
}

def tile_colour(name):
    return TILE_COLOURS.get(name[-1], TILE_COLOURS['_'])


# ── ROI definitions ───────────────────────────────────────────────────────────
# All coords for 1418×837 canvas.
# Tile walls (excluded by ROI boundary, not by colour):
#   Left wall:  x ≈  80–140  (outside MELD_LEFT right edge)
#   Right wall: x ≈ 1280–1345 (outside MELD_RIGHT left edge)

ROI_4P = {
    # Own hand — full bottom strip, no filter needed
    "hand": (90, 693, 1330, 837),

    # Center discard diamond — polygon (excludes corner avatars and score board)
    "center_poly": np.array([
        [490, 165], [930, 165],
        [1105, 385], [1105, 635],
        [930,  695], [490,  695],
        [310,  635], [310,  385],
    ], dtype=np.int32),
    "center_excl": (575, 325, 845, 520),  # score board

    # Open melds — inset from tile walls
    "meld_left":  (5,   340,  80,  690),
    "meld_right": (1338, 340, 1415, 690),
    "meld_top":   (295,  22, 1120,  138),
}

ROI_3P = {
    # 3-player: narrower table, hand is still at bottom but slightly different
    "hand": (90, 680, 1330, 837),

    "center_poly": np.array([
        [410, 150], [1010, 150],
        [1120, 380], [1120, 620],
        [1010, 700], [410,  700],
        [295,  620], [295,  380],
    ], dtype=np.int32),
    "center_excl": (560, 310, 860, 530),

    "meld_left":  (5,   300,  78,  670),
    "meld_right": (1340, 300, 1415, 670),
    "meld_top":   (295,  22, 1120,  138),
}


def detect_game_mode(img_bgr: np.ndarray) -> str:
    """
    Detect 3P vs 4P by reading top-left info box.
    '三人' appears in 3-player mode.
    Falls back to '4p' if uncertain.
    """
    # Crop the top-left label area where '金之間・三人麻' or '四人南' appears
    label_crop = img_bgr[40:80, 55:250]
    gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
    # Simple heuristic: 3P games have more greenish tones (different table colour)
    # More reliable: check if the top tile area (y=0-25) is empty or has tiles
    top_strip = img_bgr[8:28, 350:1070]
    # In 3P, the top wall is absent (only 3 sides); in 4P, top strip has tile backs
    top_gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY)
    # Tile backs are high-contrast rectangular objects; blank = lower std
    std = float(top_gray.std())
    return "3p" if std < 28 else "4p"


def grayscale(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def infer_rect(model, img_bgr, roi_xywh, conf, iou):
    """Crop rect ROI, infer, remap coords."""
    x1, y1, x2, y2 = roi_xywh
    crop = grayscale(img_bgr[y1:y2, x1:x2])
    if crop.size == 0:
        return []
    r = model(crop, conf=conf, iou=iou, verbose=False)[0]
    return [{"xyxy": (int(b.xyxy[0][0])+x1, int(b.xyxy[0][1])+y1,
                      int(b.xyxy[0][2])+x1, int(b.xyxy[0][3])+y1),
             "conf": float(b.conf[0]), "cls": int(b.cls[0])}
            for b in r.boxes]


def infer_poly(model, img_bgr, poly, excl, conf, iou):
    """Crop bounding box of polygon, mask outside, infer, remap coords."""
    x1 = int(poly[:, 0].min()); y1 = int(poly[:, 1].min())
    x2 = int(poly[:, 0].max()); y2 = int(poly[:, 1].max())
    crop = img_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return []

    # Polygon mask
    pmask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(pmask, [poly - np.array([x1, y1])], 255)
    # Exclusion rect (score board)
    ex1,ey1,ex2,ey2 = excl
    pmask[max(0,ey1-y1):ey2-y1, max(0,ex1-x1):ex2-x1] = 0

    inp = grayscale(crop)
    inp[pmask == 0] = 0

    r = model(inp, conf=conf, iou=iou, verbose=False)[0]
    return [{"xyxy": (int(b.xyxy[0][0])+x1, int(b.xyxy[0][1])+y1,
                      int(b.xyxy[0][2])+x1, int(b.xyxy[0][3])+y1),
             "conf": float(b.conf[0]), "cls": int(b.cls[0])}
            for b in r.boxes]


def draw_rois(img, rois, mode):
    out = img.copy()
    labels = {
        "hand": ("HAND", ROI_COLOURS["hand"]),
        "meld_left":  ("MELD-L", ROI_COLOURS["meld"]),
        "meld_right": ("MELD-R", ROI_COLOURS["meld"]),
        "meld_top":   ("MELD-T", ROI_COLOURS["meld"]),
    }
    for key, (lbl, col) in labels.items():
        x1,y1,x2,y2 = rois[key]
        cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
        cv2.putText(out,lbl,(x1+3,y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)

    # Center polygon
    cv2.polylines(out,[rois["center_poly"]],True,ROI_COLOURS["center"],2)
    ex1,ey1,ex2,ey2 = rois["center_excl"]
    cv2.rectangle(out,(ex1,ey1),(ex2,ey2),(80,80,180),1)

    cv2.putText(out,f"MODE:{mode}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    return out


def draw_boxes(img, boxes, names):
    out = img.copy()
    for b in boxes:
        x1,y1,x2,y2 = b["xyxy"]
        label = names[b["cls"]]
        col = tile_colour(label)
        cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
        txt = f"{label} {b['conf']:.2f}"
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        cv2.rectangle(out,(x1,y1-th-4),(x1+tw+2,y1),col,-1)
        cv2.putText(out,txt,(x1+1,y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1,cv2.LINE_AA)
    return out


def run_inference(model, img, rois, conf, iou):
    all_boxes = []
    counts = {}

    hand = infer_rect(model, img, rois["hand"], conf, iou)
    all_boxes.extend(hand); counts["hand"] = len(hand)

    center = infer_poly(model, img, rois["center_poly"], rois["center_excl"], conf, iou)
    all_boxes.extend(center); counts["center"] = len(center)

    melds = 0
    for key in ["meld_left","meld_right","meld_top"]:
        m = infer_rect(model, img, rois[key], conf, iou)
        all_boxes.extend(m); melds += len(m)
    counts["melds"] = melds

    return all_boxes, counts


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                 PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "debug").mkdir(exist_ok=True)

    model = YOLO(str(model_path))
    names = model.names

    imgs = sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpg"))
    print(f"Model : {model_path.name}  conf={CONF_THRESH}")
    print(f"Images: {len(imgs)}\n")

    total = 0
    mode_counts = {"3p": 0, "4p": 0}

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None: continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))

        mode = detect_game_mode(img)
        rois = ROI_3P if mode == "3p" else ROI_4P
        mode_counts[mode] += 1

        all_boxes, counts = run_inference(model, img, rois, CONF_THRESH, IOU_THRESH)

        out = draw_rois(img, rois, mode)
        out = draw_boxes(out, all_boxes, names)
        cv2.imwrite(str(OUT_DIR / img_path.name), out)
        total += len(all_boxes)

        print(f"  {img_path.name} [{mode}]: {len(all_boxes)} det "
              f"(hand={counts['hand']} center={counts['center']} melds={counts['melds']})")

    print(f"\nTotal: {total} detections  |  4P:{mode_counts['4p']}  3P:{mode_counts['3p']}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
