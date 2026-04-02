"""
live_test.py — Live tile detection test (tkinter display)
==========================================================
Captures the game window continuously, runs YOLO on the hand ROI,
and shows an annotated frame in a tkinter window.

Controls:
  Q / Esc  — quit
  S        — save current frame to runs/advisor/live_save.png
  +  / =   — raise confidence threshold (+0.05)
  -        — lower confidence threshold (-0.05)
  Space    — pause / resume
  H        — toggle hand-strip only ↔ full frame
  M        — cycle models (confused_boost ↔ boost2)

Usage:
  python live_test.py
  python live_test.py --model runs/detect/majsoul_boost2/weights/best.pt
  python live_test.py --full
  python live_test.py --crop 100 70   # absolute screen offset X Y
  python live_test.py --conf 0.20
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

PROJECT_ROOT = Path("E:/project/majsoul_yolo")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.mahjong_advisor import ROI_4P, ROI_3P, _gray, CONF_THRESH, IOU_THRESH
from src.screen_capture  import ScreenCapture
from src.tile_codec      import MODEL_TO_TILE, TILE_NAMES

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS = [
    PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt",
    PROJECT_ROOT / "runs/detect/majsoul_boost2/weights/best.pt",
]

DISPLAY_W = 1418
DISPLAY_H = 200   # hand-strip mode height (ROI area + margin)
SCALE     = 0.65  # display scale factor

# ── Tile colours (BGR → RGB for PIL) ─────────────────────────────────────────
SUIT_RGB = {
    "m": (255, 120,  40),   # orange — man
    "p": ( 80, 200,  80),   # green  — pin
    "s": ( 80, 160, 240),   # blue   — sou
}
HON_RGB = (220, 200,  60)   # yellow — honours

def _tile_rgb(name: str):
    return SUIT_RGB.get(name[-1], HON_RGB) if len(name) >= 2 else HON_RGB


# ─────────────────────────────────────────────────────────────────────────────

def detect_hand(model, img: np.ndarray, conf: float, rois: dict):
    """Return list of (name, conf, (x1,y1,x2,y2)) sorted left→right."""
    x1, y1, x2, y2 = rois["hand"]
    crop = _gray(img[y1:y2, x1:x2])
    if crop.size == 0:
        return []
    res = model(crop, conf=conf, iou=IOU_THRESH, verbose=False)[0]
    dets = []
    for b in res.boxes:
        mid  = int(b.cls[0])
        name = TILE_NAMES[int(MODEL_TO_TILE[mid])]
        c    = float(b.conf[0])
        bx1  = int(b.xyxy[0][0]) + x1
        by1  = int(b.xyxy[0][1]) + y1
        bx2  = int(b.xyxy[0][2]) + x1
        by2  = int(b.xyxy[0][3]) + y1
        dets.append((name, c, (bx1, by1, bx2, by2)))
    return sorted(dets, key=lambda d: d[2][0])


def annotate_pil(img_bgr: np.ndarray, dets, conf: float,
                 rois: dict, model_name: str, fps: float,
                 paused: bool, show_full: bool) -> Image.Image:
    """Draw detections on the frame and return a PIL Image."""
    # Crop to hand strip + margin if not full mode
    if not show_full:
        _, y1, _, y2 = rois["hand"]
        margin = 80
        y_top = max(0, y1 - margin)
        frame = img_bgr[y_top:, :]
        offset_y = y_top
    else:
        frame = img_bgr
        offset_y = 0

    # BGR → RGB for PIL
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(pil)

    # Hand ROI outline
    _, ry1, _, ry2 = rois["hand"]
    rx1, rx2 = rois["hand"][0], rois["hand"][2]
    d.rectangle([rx1, ry1 - offset_y, rx2, ry2 - offset_y],
                outline=(80, 80, 80), width=1)

    # Detection boxes
    try:
        font = ImageFont.truetype("consola.ttf", 13)
    except Exception:
        font = None

    for name, c, (bx1, by1, bx2, by2) in dets:
        col = _tile_rgb(name)
        by1 -= offset_y; by2 -= offset_y
        d.rectangle([bx1, by1, bx2, by2], outline=col, width=2)
        label = f"{name} {c:.2f}"
        bbox  = d.textbbox((bx1, by1 - 16), label, font=font)
        d.rectangle([bbox[0]-1, bbox[1]-1, bbox[2]+1, bbox[3]+1],
                    fill=(0, 0, 0))
        d.text((bx1, by1 - 16), label, fill=col, font=font)

    # Status bar at top
    bar_h = 40
    bar_img = Image.new("RGB", (pil.width, bar_h), (20, 20, 28))
    bd = ImageDraw.Draw(bar_img)
    try:
        bfont = ImageFont.truetype("consola.ttf", 12)
    except Exception:
        bfont = None

    status = (f" Model: {model_name}  |  Conf: {conf:.2f}  |"
              f"  Tiles: {len(dets)}  |  FPS: {fps:.1f}"
              + ("  [PAUSED]" if paused else ""))
    bd.text((4, 4),  status,  fill=(180, 220, 255), font=bfont)
    tiles_str = "  ".join(n for n, _, _ in dets) or "—"
    bd.text((4, 22), tiles_str, fill=(0, 220, 100), font=bfont)

    # Hint bar at bottom
    hint_h = 18
    hint_img = Image.new("RGB", (pil.width, hint_h), (15, 15, 20))
    hd = ImageDraw.Draw(hint_img)
    try:
        hfont = ImageFont.truetype("consola.ttf", 10)
    except Exception:
        hfont = None
    hd.text((4, 3),
            "Q=quit  S=save  +/-=conf  Space=pause  H=full/strip  M=model",
            fill=(80, 80, 90), font=hfont)

    # Stack: status bar | frame | hint
    total_h = bar_h + pil.height + hint_h
    combined = Image.new("RGB", (pil.width, total_h))
    combined.paste(bar_img,  (0, 0))
    combined.paste(pil,      (0, bar_h))
    combined.paste(hint_img, (0, bar_h + pil.height))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Tkinter app
# ─────────────────────────────────────────────────────────────────────────────

class LiveTestApp:
    POLL_MS = 80   # ~12 fps target

    def __init__(self, args):
        # ── State ─────────────────────────────────────────────────────────────
        self.conf       = args.conf
        self.paused     = False
        self.show_full  = args.full
        self.ox, self.oy = args.crop
        self.save_dir   = PROJECT_ROOT / "runs" / "advisor"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.rois       = ROI_4P
        self.last_bgr   = None
        self.last_dets  = []
        self.fps        = 0.0
        self._t_history = []

        # ── Models ────────────────────────────────────────────────────────────
        if args.model:
            self.model_list = [Path(args.model)]
        else:
            self.model_list = [p for p in MODELS if p.exists()]
        if not self.model_list:
            print("ERROR: no model found.")
            sys.exit(1)
        self.model_idx = 0
        print(f"Loading: {self.model_list[0].parent.parent.name} …")
        self.model = YOLO(str(self.model_list[0]))
        print("Model loaded.")

        # ── Capture ───────────────────────────────────────────────────────────
        self.capture = ScreenCapture()
        print(f"Capture: {self.capture.get_status()}")

        # ── Tkinter window ────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Majsoul Live Test")
        self.root.configure(bg="#12121f")
        self.root.resizable(True, True)

        self._img_label = tk.Label(self.root, bg="#12121f")
        self._img_label.pack(fill="both", expand=True)
        self._photo = None   # keep reference

        # Keyboard bindings
        self.root.bind("<q>",      lambda e: self._quit())
        self.root.bind("<Q>",      lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("<s>",      lambda e: self._save())
        self.root.bind("<S>",      lambda e: self._save())
        self.root.bind("<plus>",   lambda e: self._adj_conf(+0.05))
        self.root.bind("<equal>",  lambda e: self._adj_conf(+0.05))
        self.root.bind("<minus>",  lambda e: self._adj_conf(-0.05))
        self.root.bind("<space>",  lambda e: self._toggle_pause())
        self.root.bind("<h>",      lambda e: self._toggle_full())
        self.root.bind("<H>",      lambda e: self._toggle_full())
        self.root.bind("<m>",      lambda e: self._cycle_model())
        self.root.bind("<M>",      lambda e: self._cycle_model())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.root.after(100, self._loop)

    # ── Controls ──────────────────────────────────────────────────────────────

    def _quit(self):
        self.root.destroy()

    def _save(self):
        if self.last_bgr is not None:
            path = self.save_dir / "live_save.png"
            cv2.imwrite(str(path), self.last_bgr)
            print(f"Saved: {path}")

    def _adj_conf(self, delta: float):
        self.conf = round(max(0.05, min(0.95, self.conf + delta)), 2)
        print(f"Conf: {self.conf:.2f}")

    def _toggle_pause(self):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")

    def _toggle_full(self):
        self.show_full = not self.show_full
        print("Full frame" if self.show_full else "Hand strip")

    def _cycle_model(self):
        if len(self.model_list) < 2:
            return
        self.model_idx = (self.model_idx + 1) % len(self.model_list)
        name = self.model_list[self.model_idx].parent.parent.name
        print(f"Loading: {name} …")
        self.model = YOLO(str(self.model_list[self.model_idx]))
        print("Done.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        t0 = time.time()

        if not self.paused:
            try:
                if self.ox > 0 or self.oy > 0:
                    raw = self.capture.capture_fullscreen()
                else:
                    raw = self.capture.capture()

                h_r, w_r = raw.shape[:2]
                if self.ox > 0 or self.oy > 0:
                    raw = raw[self.oy : min(self.oy + 837, h_r),
                              self.ox : min(self.ox + 1418, w_r)]

                if raw.shape[:2] != (837, 1418):
                    raw = cv2.resize(raw, (1418, 837))

                # 3P/4P auto-detect
                top = raw[8:28, 350:1070]
                self.rois = (ROI_3P
                             if cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).std() < 28
                             else ROI_4P)

                self.last_dets = detect_hand(
                    self.model, raw, self.conf, self.rois)
                self.last_bgr  = raw

            except Exception as e:
                print(f"Capture error: {e}")

        # FPS
        elapsed = time.time() - t0
        self._t_history.append(elapsed)
        if len(self._t_history) > 10:
            self._t_history.pop(0)
        avg = sum(self._t_history) / len(self._t_history)
        self.fps = 1.0 / avg if avg > 0 else 0.0

        # Draw + display
        if self.last_bgr is not None:
            model_name = self.model_list[self.model_idx].parent.parent.name
            pil_img = annotate_pil(
                self.last_bgr, self.last_dets, self.conf,
                self.rois, model_name, self.fps,
                self.paused, self.show_full)

            # Scale for display
            w = int(pil_img.width  * SCALE)
            h = int(pil_img.height * SCALE)
            pil_img = pil_img.resize((w, h), Image.BILINEAR)

            self._photo = ImageTk.PhotoImage(pil_img)
            self._img_label.config(image=self._photo)
            self.root.geometry(f"{w}x{h}")

        self.root.after(self.POLL_MS, self._loop)

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live tile detection test")
    parser.add_argument("--model", type=str,   default=None)
    parser.add_argument("--conf",  type=float, default=CONF_THRESH)
    parser.add_argument("--full",  action="store_true")
    parser.add_argument("--crop",  nargs=2, type=int, default=[0, 0],
                        metavar=("X", "Y"))
    args = parser.parse_args()
    LiveTestApp(args).run()


if __name__ == "__main__":
    main()
