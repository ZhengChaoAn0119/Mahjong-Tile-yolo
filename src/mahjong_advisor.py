"""
mahjong_advisor.py
Detection layer: YOLO inference helpers + ROI definitions + MahjongAdvisor container.
All CLI/draw/overlay code has been removed — use windows_app.py for the GUI.
"""

from pathlib import Path
from typing import List
import cv2
import numpy as np

from ultralytics import YOLO

from .tile_codec     import MODEL_TO_TILE
from .frame_smoother import FrameSmoother, RawDetection
from .game_state     import GameState

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
MODEL_PATH   = PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt"

# ── Detection parameters ──────────────────────────────────────────────────────
W, H        = 1418, 837
CONF_THRESH = 0.28
IOU_THRESH  = 0.35
CONF_WARN   = 0.45

# ── ROI definitions ───────────────────────────────────────────────────────────
ROI_4P = dict(
    hand         = (90, 693, 1330, 837),
    center_poly  = np.array([[490,165],[930,165],[1105,385],[1105,635],
                              [930,695],[490,695],[310,635],[310,385]], np.int32),
    center_excl  = (575, 325, 845, 520),
    meld_left    = (5,   340,  80,  690),
    meld_right   = (1338, 340, 1415, 690),
    meld_top     = (295,  22, 1120,  138),
)
ROI_3P = dict(
    hand         = (90, 680, 1330, 837),
    center_poly  = np.array([[410,150],[1010,150],[1120,380],[1120,620],
                              [1010,700],[410,700],[295,620],[295,380]], np.int32),
    center_excl  = (560, 310, 860, 530),
    meld_left    = (5,   300,  78,  670),
    meld_right   = (1340, 300, 1415, 670),
    meld_top     = (295,  22, 1120,  138),
)


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers (same as infer_roi.py but returns RawDetection objects)
# ─────────────────────────────────────────────────────────────────────────────

def _gray(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _infer_rect(model, img, roi_xywh, zone, conf, iou) -> List[RawDetection]:
    x1, y1, x2, y2 = roi_xywh
    crop = _gray(img[y1:y2, x1:x2])
    if crop.size == 0:
        return []
    r = model(crop, conf=conf, iou=iou, verbose=False)[0]
    dets = []
    for b in r.boxes:
        mid  = int(b.cls[0])
        tid  = int(MODEL_TO_TILE[mid])
        xyxy = (int(b.xyxy[0][0])+x1, int(b.xyxy[0][1])+y1,
                int(b.xyxy[0][2])+x1, int(b.xyxy[0][3])+y1)
        dets.append(RawDetection(tile_id=tid, model_cid=mid,
                                  conf=float(b.conf[0]), xyxy=xyxy, zone=zone))
    return dets


def _infer_poly(model, img, poly, excl, zone, conf, iou) -> List[RawDetection]:
    x1 = int(poly[:,0].min()); y1 = int(poly[:,1].min())
    x2 = int(poly[:,0].max()); y2 = int(poly[:,1].max())
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return []
    pmask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(pmask, [poly - np.array([x1,y1])], 255)
    ex1,ey1,ex2,ey2 = excl
    pmask[max(0,ey1-y1):ey2-y1, max(0,ex1-x1):ex2-x1] = 0
    inp = _gray(crop)
    inp[pmask == 0] = 0
    r = model(inp, conf=conf, iou=iou, verbose=False)[0]
    dets = []
    for b in r.boxes:
        mid  = int(b.cls[0])
        tid  = int(MODEL_TO_TILE[mid])
        xyxy = (int(b.xyxy[0][0])+x1, int(b.xyxy[0][1])+y1,
                int(b.xyxy[0][2])+x1, int(b.xyxy[0][3])+y1)
        dets.append(RawDetection(tile_id=tid, model_cid=mid,
                                  conf=float(b.conf[0]), xyxy=xyxy, zone=zone))
    return dets


def detect_game_mode(img) -> str:
    top = img[8:28, 350:1070]
    return "3p" if cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).std() < 28 else "4p"


# Hand-strip model dimensions (ROI_4P hand: x1=90,y1=693,x2=1330,y2=837)
HAND_MODEL_W = 1240
HAND_MODEL_H = 144


def run_hand_detection(model, hand_bgr,
                        screen_x: int, screen_y: int,
                        screen_w: int, screen_h: int):
    """
    Detect hand tiles from a hand-strip BGR crop (any screen size).
    Resizes the crop to HAND_MODEL_W × HAND_MODEL_H, runs inference,
    then maps bounding-box coords back to screen space.
    Returns (List[RawDetection], "4p").
    """
    resized  = cv2.resize(hand_bgr, (HAND_MODEL_W, HAND_MODEL_H))
    gray     = _gray(resized)
    r        = model(gray, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
    scale_x  = screen_w / HAND_MODEL_W
    scale_y  = screen_h / HAND_MODEL_H
    dets: List[RawDetection] = []
    for b in r.boxes:
        mid  = int(b.cls[0])
        tid  = int(MODEL_TO_TILE[mid])
        mx1  = int(b.xyxy[0][0]); my1 = int(b.xyxy[0][1])
        mx2  = int(b.xyxy[0][2]); my2 = int(b.xyxy[0][3])
        xyxy = (screen_x + int(mx1 * scale_x),
                screen_y + int(my1 * scale_y),
                screen_x + int(mx2 * scale_x),
                screen_y + int(my2 * scale_y))
        dets.append(RawDetection(tile_id=tid, model_cid=mid,
                                  conf=float(b.conf[0]),
                                  xyxy=xyxy, zone="hand"))
    return dets, "4p"


def run_detection(model, img) -> List[RawDetection]:
    mode = detect_game_mode(img)
    rois = ROI_3P if mode == "3p" else ROI_4P
    dets: List[RawDetection] = []
    dets += _infer_rect(model, img, rois["hand"],       "hand",      CONF_THRESH, IOU_THRESH)
    dets += _infer_poly(model, img, rois["center_poly"], rois["center_excl"], "center", CONF_THRESH, IOU_THRESH)
    for key in ("meld_left","meld_right","meld_top"):
        dets += _infer_rect(model, img, rois[key], key, CONF_THRESH, IOU_THRESH)
    return dets, mode


# ─────────────────────────────────────────────────────────────────────────────
# MahjongAdvisor  — model + smoother + state container (used by controller)
# ─────────────────────────────────────────────────────────────────────────────

class MahjongAdvisor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model    = YOLO(str(model_path))
        self.smoother = FrameSmoother(window=6, min_hits=3)
        self.state    = GameState(seat_wind=0, round_wind=0)
