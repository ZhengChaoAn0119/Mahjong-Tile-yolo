"""
windows_app.py
Mahjong Soul advisor companion window — View layer (MVC).

Layout (top → bottom):
  LiveViewPanel   — live capture canvas with detection overlay
  HandPanel       — 13+1 tile strip, click to correct
  EVPanel         — #1 large card + #2/#3 compact; effective tiles as chips
  DoraPanel       — dora input + chip list with remove
  MeldPanel       — collapsible; meld list + add via popup
  DiscardPanel    — collapsible; 34×4 visual grid (drag to mark)
  ControlPanel    — Analyze [F9] + Auto; ⚙ drawer

Press F9 to analyze. Analyze button pulses to confirm trigger.
Skeleton screen during computation reduces visual flicker.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable

import tkinter as tk
from tkinter import ttk

import numpy as np

# ── Project path ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
sys.path.insert(0, str(PROJECT_ROOT))

from src.tile_codec         import N_TILES, TILE_NAMES, tile_name
from src.frame_smoother     import ConfirmedTile
from src.advisor_controller import AdvisorController, Phase1Result, Phase2Result

# ── Window ────────────────────────────────────────────────────────────────────
WINDOW_W = 560
WINDOW_H = 920

# ── Color palette ─────────────────────────────────────────────────────────────
BG       = "#12121f"
PANEL    = "#1a1f35"
CARD     = "#1f2640"
BORDER   = "#2a3050"
ACCENT   = "#0d3a5c"
SAFE     = "#26c6da"
SAFE_DIM = "#0d7a86"
WARN     = "#ff9800"
WARN_DIM = "#7a4800"
DANGER   = "#ef5350"
BEST     = "#00e676"
GOLD     = "#ffd700"
TEXT     = "#dce3f0"
MUTED    = "#6b7394"
SKEL     = "#252a42"
SKEL2    = "#2e3450"

SUIT_COLOR = {"m": "#ffa040", "p": "#60c060", "s": "#5ab0f0"}

def _suit_fg(name: str) -> str:
    return SUIT_COLOR.get(name[-1], TEXT) if name and len(name) >= 2 else TEXT


# ─────────────────────────────────────────────────────────────────────────────
# TileImageCache
# ─────────────────────────────────────────────────────────────────────────────

class TileImageCache:
    W, H   = 28, 40
    CW, CH = 20, 28

    def __init__(self, crops_dir: Path):
        self._photos:      Dict[int, object] = {}
        self._chip_photos: Dict[int, object] = {}
        self._crops_dir = crops_dir
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk
            self._Image     = Image
            self._ImageDraw = ImageDraw
            self._ImageFont = ImageFont
            self._ImageTk   = ImageTk
            self._has_pil   = True
            self._preload_all()
        except ImportError:
            self._has_pil = False

    def _make_fallback(self, tid: int, w: int, h: int):
        name = tile_name(tid)
        suit = name[-1] if len(name) >= 2 else "?"
        bg_map = {"m": (80, 40, 10), "p": (20, 60, 20), "s": (15, 45, 80)}
        bg  = bg_map.get(suit, (40, 30, 70))
        img = self._Image.new("RGB", (w, h), bg)
        d   = self._ImageDraw.Draw(img)
        try:
            font = self._ImageFont.truetype("consola.ttf", max(8, h // 5))
        except Exception:
            font = None
        lines = [name[:-1], name[-1]] if len(name) > 2 else [name]
        y = h // 4
        for line in lines:
            bbox = d.textbbox((0, 0), line, font=font) if font else (0, 0, w, 10)
            tw = bbox[2] - bbox[0]
            d.text(((w - tw) // 2, y), line, fill=(220, 220, 220), font=font)
            y += h // 3
        return img

    def _preload_all(self):
        # tile_crops/{mid}/ is indexed by MODEL class ID (alphabetical), not tile ID.
        # Must map: model_class_id → tile_id via MODEL_TO_TILE before storing.
        from src.tile_codec import MODEL_TO_TILE as _M2T
        # Initialise fallbacks for every tile ID first
        for tid in range(N_TILES):
            fb = self._make_fallback(tid, self.W, self.H)
            self._photos[tid]      = self._ImageTk.PhotoImage(
                fb.resize((self.W, self.H), self._Image.LANCZOS))
            self._chip_photos[tid] = self._ImageTk.PhotoImage(
                fb.resize((self.CW, self.CH), self._Image.LANCZOS))
        # Overwrite with real crops where available (folder = model class id)
        from src.tile_codec import MODEL_NAMES as _MNAMES
        for mid in range(len(_M2T)):
            tid    = int(_M2T[mid])
            folder = self._crops_dir / str(mid)
            if not folder.is_dir():
                continue
            # Prefer the clean reference image {name}.png if present
            preferred = folder / (_MNAMES[mid] + ".png")
            if preferred.exists():
                pick = preferred
            else:
                files = sorted(folder.iterdir())
                if not files:
                    continue
                pick = files[len(files) // 2]
            try:
                pil_img = self._Image.open(pick).convert("RGB")
                self._photos[tid]      = self._ImageTk.PhotoImage(
                    pil_img.resize((self.W, self.H), self._Image.LANCZOS))
                self._chip_photos[tid] = self._ImageTk.PhotoImage(
                    pil_img.resize((self.CW, self.CH), self._Image.LANCZOS))
            except Exception:
                pass

    def get(self, tid: int):
        return self._photos.get(tid)

    def get_chip(self, tid: int):
        return self._chip_photos.get(tid)


# ─────────────────────────────────────────────────────────────────────────────
# PreviewThread  — 5 fps lightweight screen capture (no YOLO)
# ─────────────────────────────────────────────────────────────────────────────

class PreviewThread(threading.Thread):
    def __init__(self, region: Tuple[int, int, int, int],
                 preview_q: queue.Queue):
        super().__init__(daemon=True, name="PreviewThread")
        self._region   = region   # (x, y, w, h) absolute screen coords
        self._q        = preview_q
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            import mss
            sct = mss.mss()
            x, y, w, h = self._region
            monitor = {"left": x, "top": y, "width": w, "height": h}
            while not self._stop_evt.is_set():
                try:
                    raw = sct.grab(monitor)
                    import numpy as _np
                    from PIL import Image as _Img
                    img = _Img.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                    # Drop old frame, put new one
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put(img)
                except Exception:
                    pass
                self._stop_evt.wait(timeout=0.2)  # 5 fps
        except ImportError:
            # fallback: PIL ImageGrab
            from PIL import ImageGrab
            x, y, w, h = self._region
            while not self._stop_evt.is_set():
                try:
                    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put(img)
                except Exception:
                    pass
                self._stop_evt.wait(timeout=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# LiveViewPanel  — canvas showing captured region + detection overlay
# ─────────────────────────────────────────────────────────────────────────────

LIVE_W    = WINDOW_W - 8
LIVE_H_DEFAULT = 80   # shown before region is selected
LIVE_H_MIN     = 40
LIVE_H_MAX     = 160

class LiveViewPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._live_h = LIVE_H_DEFAULT
        self._canvas = tk.Canvas(self, width=LIVE_W, height=self._live_h,
                                 bg=SKEL, highlightthickness=0)
        self._canvas.pack()
        self._canvas.create_text(
            LIVE_W // 2, self._live_h // 2,
            text="Click  ⊞ Select Region  to start live view",
            fill=MUTED, font=("Consolas", 9), tags="placeholder")

        self._preview_q:  queue.Queue  = queue.Queue(maxsize=1)
        self._preview_thread: Optional[PreviewThread] = None
        self._region:     Optional[Tuple] = None
        self._tk_img      = None   # keep reference
        self._det_boxes:  List     = []   # [(x1,y1,x2,y2, label), ...]
        self._scale_x:    float    = 1.0
        self._scale_y:    float    = 1.0
        self._poll_id     = None

    def _set_canvas_height(self, h: int):
        """Resize canvas widget to h pixels."""
        self._live_h = h
        self._canvas.config(height=h)

    def set_region(self, x: int, y: int, w: int, h: int):
        """Called after user drags a region; starts live capture."""
        if self._preview_thread:
            self._preview_thread.stop()
        self._region = (x, y, w, h)
        # Compute canvas height that preserves aspect ratio (no stretching)
        aspect   = w / max(h, 1)
        canvas_h = max(LIVE_H_MIN, min(LIVE_H_MAX, int(LIVE_W / aspect)))
        self._set_canvas_height(canvas_h)
        self._scale_x = LIVE_W / w
        self._scale_y = canvas_h / h
        self._preview_q = queue.Queue(maxsize=1)
        self._preview_thread = PreviewThread((x, y, w, h), self._preview_q)
        self._preview_thread.start()
        self._canvas.delete("placeholder")
        self._start_poll()

    def _start_poll(self):
        self._poll_id = self._canvas.after(50, self._poll_frame)

    def _poll_frame(self):
        try:
            pil_img = self._preview_q.get_nowait()
            pil_img = pil_img.resize((LIVE_W, self._live_h))
            try:
                from PIL import ImageTk
                self._tk_img = ImageTk.PhotoImage(pil_img)
            except Exception:
                self._tk_img = None
            if self._tk_img:
                self._canvas.delete("frame")
                self._canvas.create_image(0, 0, anchor="nw",
                                          image=self._tk_img, tags="frame")
                self._canvas.tag_lower("frame")
                self._draw_boxes()
        except queue.Empty:
            pass
        self._poll_id = self._canvas.after(50, self._poll_frame)

    def update_detections(self, hand_tiles: List[ConfirmedTile],
                          region: Tuple[int, int, int, int]):
        """Overlay detection boxes from latest Phase1 result."""
        if region is None:
            return
        rx, ry, rw, rh = region
        sx = LIVE_W / rw
        sy = self._live_h / rh
        self._det_boxes = []
        for ct in hand_tiles:
            x1, y1, x2, y2 = ct.xyxy
            # Offset: xyxy is in full-image coords; subtract region origin
            bx1 = int((x1 - rx) * sx)
            by1 = int((y1 - ry) * sy)
            bx2 = int((x2 - rx) * sx)
            by2 = int((y2 - ry) * sy)
            self._det_boxes.append((bx1, by1, bx2, by2,
                                    tile_name(ct.tile_id),
                                    WARN if ct.low_conf else BEST))
        self._draw_boxes()

    def _draw_boxes(self):
        self._canvas.delete("box")
        for bx1, by1, bx2, by2, label, color in self._det_boxes:
            self._canvas.create_rectangle(bx1, by1, bx2, by2,
                                          outline=color, width=1, tags="box")
            self._canvas.create_text(bx1 + 2, by1 + 2, anchor="nw",
                                     text=label, fill=color,
                                     font=("Consolas", 6), tags="box")

    def stop(self):
        if self._poll_id:
            try:
                self._canvas.after_cancel(self._poll_id)
            except Exception:
                pass
        if self._preview_thread:
            self._preview_thread.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RegionSelector  — full-screen drag-to-select
# ─────────────────────────────────────────────────────────────────────────────

class RegionSelector(tk.Toplevel):
    MIN_SIZE = 80

    def __init__(self, root: tk.Tk,
                 on_select: Callable[[int, int, int, int], None],
                 on_cancel: Callable):
        super().__init__(root)
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._sx = self._sy = 0
        self._rect_id = self._size_id = None

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.40)
        self.configure(bg="#000018")

        self._cv = tk.Canvas(self, bg="#000018", cursor="crosshair",
                             highlightthickness=0)
        self._cv.pack(fill="both", expand=True)

        cx, cy = sw // 2, sh // 2
        self._cv.create_text(cx, cy - 60,
                             text="拖曳框選手牌區域（14張牌的橫條）",
                             fill="white", font=("Consolas", 18, "bold"))
        self._cv.create_text(cx, cy - 30,
                             text="Drag to select the hand tiles strip",
                             fill="#aaaaaa", font=("Consolas", 12))
        self._cv.create_text(cx, cy,
                             text="[Esc] 取消", fill=MUTED,
                             font=("Consolas", 10))

        self._cv.bind("<ButtonPress-1>",   self._press)
        self._cv.bind("<B1-Motion>",       self._drag)
        self._cv.bind("<ButtonRelease-1>", self._release)
        self.bind("<Escape>", lambda e: self._cancel())
        self.focus_set()

    def _press(self, event):
        self._sx, self._sy = event.x, event.y
        for rid in (self._rect_id, self._size_id):
            if rid:
                self._cv.delete(rid)
        self._rect_id = self._size_id = None

    def _drag(self, event):
        if self._rect_id:
            self._cv.delete(self._rect_id)
        if self._size_id:
            self._cv.delete(self._size_id)
        self._rect_id = self._cv.create_rectangle(
            self._sx, self._sy, event.x, event.y,
            outline=SAFE, width=2, dash=(6, 3))
        w = abs(event.x - self._sx)
        h = abs(event.y - self._sy)
        self._size_id = self._cv.create_text(
            event.x + 4, event.y + 12,
            text=f"{w}×{h}", fill=SAFE, anchor="nw",
            font=("Consolas", 9))

    def _release(self, event):
        x = min(self._sx, event.x)
        y = min(self._sy, event.y)
        w = abs(event.x - self._sx)
        h = abs(event.y - self._sy)
        self.destroy()
        if w >= self.MIN_SIZE and h >= self.MIN_SIZE:
            self._on_select(x, y, w, h)
        else:
            self._on_cancel()

    def _cancel(self):
        self.destroy()
        self._on_cancel()


# ─────────────────────────────────────────────────────────────────────────────
# TilePopup  — tile correction
# ─────────────────────────────────────────────────────────────────────────────

class TilePopup(tk.Toplevel):
    def __init__(self, parent_root, ct: ConfirmedTile,
                 on_apply: Callable[[int, str], None]):
        super().__init__(parent_root)
        self._ct       = ct
        self._on_apply = on_apply
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=CARD)

        border = tk.Frame(self, bg=WARN if ct.low_conf else BORDER,
                          padx=1, pady=1)
        border.pack()
        inner = tk.Frame(border, bg=CARD, padx=8, pady=6)
        inner.pack()

        tk.Label(inner, text=f"Tile at x={ct.cx}",
                 bg=CARD, fg=MUTED, font=("Consolas", 7)).pack(anchor="w")
        self._var = tk.StringVar(value=tile_name(ct.tile_id))
        cb = ttk.Combobox(inner, textvariable=self._var,
                          values=list(TILE_NAMES), width=7, state="readonly")
        cb.pack(pady=3)

        btn_row = tk.Frame(inner, bg=CARD)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="Apply", bg=SAFE_DIM, fg=TEXT,
                  font=("Consolas", 8), relief="flat",
                  command=self._apply).pack(side="left", padx=(0, 4))
        tk.Button(btn_row, text="✕", bg=CARD, fg=MUTED,
                  font=("Consolas", 8), relief="flat",
                  command=self.destroy).pack(side="left")

        self.bind("<FocusOut>", lambda e: self.after(100, self._check_focus))

    def _check_focus(self):
        try:
            focused = self.focus_get()
            if focused is None or str(focused) == ".":
                self.destroy()
        except Exception:
            self.destroy()

    def _apply(self):
        self._on_apply(self._ct.cx, self._var.get())
        self.destroy()

    def place_near(self, x: int, y: int):
        self.update_idletasks()
        w  = self.winfo_reqwidth()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        px = min(x, sw - w - 4)
        py = min(y - 10, sh - self.winfo_reqheight() - 4)
        self.geometry(f"+{max(0,px)}+{max(0,py)}")
        self.focus_set()


# ─────────────────────────────────────────────────────────────────────────────
# HandPanel  — 13 + 1 tile strip
# ─────────────────────────────────────────────────────────────────────────────

class HandPanel(tk.Frame):
    SKEL_COUNT = 13

    def __init__(self, parent, cache: TileImageCache,
                 on_correction: Callable, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._cache         = cache
        self._on_correction = on_correction
        self._skel_job      = None
        self._skel_phase    = 0
        self._skel_cells    = []

        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=6, pady=(5, 2))
        self._hand_lbl = tk.Label(hdr, text="HAND", bg=PANEL, fg=MUTED,
                                  font=("Consolas", 8, "bold"))
        self._hand_lbl.pack(side="left")
        self._shan_lbl = tk.Label(hdr, text="", bg=PANEL, fg=TEXT,
                                  font=("Consolas", 10, "bold"))
        self._shan_lbl.pack(side="right")

        self._strip = tk.Frame(self, bg=PANEL)
        self._strip.pack(fill="x", padx=4, pady=(0, 5))
        self._show_skeleton()

    def _show_skeleton(self):
        self._clear_strip()
        self._hand_lbl.config(text="HAND")
        self._shan_lbl.config(text="")
        self._skel_cells = []
        for i in range(self.SKEL_COUNT):
            f = tk.Frame(self._strip, bg=SKEL, width=30, height=44)
            f.pack_propagate(False)
            f.pack(side="left", padx=1)
            self._skel_cells.append(f)
        # +1 drawn tile skeleton
        tk.Frame(self._strip, bg=BG, width=6, height=44).pack(side="left")
        draw_f = tk.Frame(self._strip, bg=SKEL, width=30, height=44,
                          highlightbackground=MUTED, highlightthickness=1)
        draw_f.pack_propagate(False)
        draw_f.pack(side="left", padx=1)
        self._skel_cells.append(draw_f)
        self._start_skel_shimmer()

    def _start_skel_shimmer(self):
        self._skel_phase = 0
        self._shimmer_step()

    def _shimmer_step(self):
        if not self._skel_cells:
            return
        try:
            col = SKEL if self._skel_phase % 2 == 0 else SKEL2
            for f in self._skel_cells:
                f.config(bg=col)
            self._skel_phase += 1
            self._skel_job = self.after(550, self._shimmer_step)
        except tk.TclError:
            pass

    def _stop_shimmer(self):
        if self._skel_job:
            self.after_cancel(self._skel_job)
            self._skel_job = None
        self._skel_cells = []

    def _clear_strip(self):
        self._stop_shimmer()
        for w in self._strip.winfo_children():
            w.destroy()

    def update(self, hand_tiles: List[ConfirmedTile], s: int, best_tid: int = -1):
        self._clear_strip()
        sorted_tiles = sorted(hand_tiles, key=lambda t: t.cx)
        count = len(sorted_tiles)
        self._hand_lbl.config(text=f"HAND  ({count})", fg=MUTED)

        if s == -1:
            self._shan_lbl.config(
                text="和牌!", fg=GOLD, font=("Consolas", 10, "bold"))
        elif s == 0:
            self._shan_lbl.config(
                text="聽牌", fg=SAFE, font=("Consolas", 10, "bold"))
        elif s == 1:
            self._shan_lbl.config(text=f"向聴 {s}", fg=WARN,
                                  font=("Consolas", 10, "bold"))
        else:
            self._shan_lbl.config(text=f"向聴 {s}", fg=TEXT,
                                  font=("Consolas", 10))

        # Split: first 13 = hand, 14th = drawn tile
        hand_part  = sorted_tiles[:13]
        drawn_part = sorted_tiles[13:]

        def _add_tile(ct, is_drawn=False):
            tid     = ct.tile_id
            name    = tile_name(tid)
            is_best = (tid == best_tid)
            is_warn = ct.low_conf

            if is_drawn:
                border_color = BEST if is_best else (WARN if is_warn else SAFE_DIM)
            else:
                border_color = BEST if is_best else (WARN if is_warn else BORDER)

            cell = tk.Frame(self._strip, bg=border_color, padx=1, pady=1,
                            cursor="hand2")
            cell.pack(side="left", padx=1)
            inner = tk.Frame(cell, bg=CARD)
            inner.pack()

            photo = self._cache.get(tid)
            if photo:
                lbl = tk.Label(inner, image=photo, bg=CARD, cursor="hand2")
                lbl.image = photo
            else:
                lbl = tk.Label(inner, text=name, bg=CARD, fg=_suit_fg(name),
                               font=("Consolas", 7, "bold"),
                               width=3, height=2, cursor="hand2")
            lbl.pack()
            tk.Label(inner, text=name, bg=CARD,
                     fg=BEST if is_best else (WARN if is_warn else MUTED),
                     font=("Consolas", 6)).pack()

            for widget in (cell, inner, lbl):
                widget.bind("<Button-1>",
                            self._make_click_handler(ct))

        for ct in hand_part:
            _add_tile(ct)

        # Separator gap before drawn tile
        if drawn_part:
            tk.Frame(self._strip, bg=BG, width=6, height=44).pack(side="left")
            for ct in drawn_part:
                _add_tile(ct, is_drawn=True)

    def _make_click_handler(self, ct: ConfirmedTile):
        def handler(event):
            popup = TilePopup(self.winfo_toplevel(), ct, self._on_correction)
            popup.place_near(event.x_root, event.y_root)
        return handler

    def set_skeleton(self):
        self._show_skeleton()
        self._shan_lbl.config(text="")

    def clear(self):
        self._clear_strip()
        self._hand_lbl.config(text="HAND", fg=MUTED)
        self._shan_lbl.config(text="")


# ─────────────────────────────────────────────────────────────────────────────
# EVPanel  — simplified (no win_rate)
# ─────────────────────────────────────────────────────────────────────────────

class EVPanel(tk.Frame):
    def __init__(self, parent, cache: TileImageCache, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._cache      = cache
        self._skel_job   = None
        self._skel_cells = []

        # ── #1 large card ─────────────────────────────────────────────────────
        self._card1 = tk.Frame(self, bg=CARD, padx=8, pady=6)
        self._card1.pack(fill="x", padx=6, pady=(5, 2))

        card1_top = tk.Frame(self._card1, bg=CARD)
        card1_top.pack(fill="x")
        self._r1_rank  = tk.Label(card1_top, text="#1", bg=CARD, fg=BEST,
                                  font=("Consolas", 9, "bold"), width=3)
        self._r1_rank.pack(side="left")
        self._r1_name  = tk.Label(card1_top, text="—", bg=CARD, fg=TEXT,
                                  font=("Consolas", 14, "bold"), width=5)
        self._r1_name.pack(side="left")
        self._r1_ev    = tk.Label(card1_top, text="", bg=CARD, fg=SAFE,
                                  font=("Consolas", 10))
        self._r1_ev.pack(side="left", padx=6)
        self._r1_shan  = tk.Label(card1_top, text="", bg=CARD, fg=MUTED,
                                  font=("Consolas", 8))
        self._r1_shan.pack(side="right")

        self._eff_frame = tk.Frame(self._card1, bg=CARD)
        self._eff_frame.pack(fill="x", pady=(3, 0))
        self._eff_count_lbl = tk.Label(self._eff_frame, text="", bg=CARD,
                                       fg=MUTED, font=("Consolas", 7))
        self._eff_count_lbl.pack(side="right")

        self._r1_yaku  = tk.Label(self._card1, text="", bg=CARD, fg=MUTED,
                                  font=("Consolas", 7),
                                  wraplength=370, justify="left")
        self._r1_yaku.pack(anchor="w", pady=(2, 0))

        # ── #2/#3 compact row ─────────────────────────────────────────────────
        compact_row = tk.Frame(self, bg=PANEL)
        compact_row.pack(fill="x", padx=6, pady=(0, 2))
        self._card2 = self._make_compact_card(compact_row, "#2", SAFE_DIM)
        self._card2.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self._card3 = self._make_compact_card(compact_row, "#3", MUTED)
        self._card3.pack(side="left", fill="x", expand=True)

        # ── Status ────────────────────────────────────────────────────────────
        status_row = tk.Frame(self, bg=PANEL)
        status_row.pack(fill="x", padx=6, pady=(2, 4))
        self._status_var = tk.StringVar(value="Press F9 to analyze")
        tk.Label(status_row, textvariable=self._status_var,
                 bg=PANEL, fg=MUTED, font=("Consolas", 7)).pack(side="left")
        self._progress = ttk.Progressbar(status_row,
                                         style="EV.Horizontal.TProgressbar",
                                         mode="determinate", length=160)
        self._progress.pack(side="right")

    def _make_compact_card(self, parent, rank_text: str,
                           rank_fg: str) -> tk.Frame:
        f = tk.Frame(parent, bg=CARD, padx=6, pady=4)
        widgets = {}
        top = tk.Frame(f, bg=CARD)
        top.pack(fill="x")
        widgets["rank"] = tk.Label(top, text=rank_text, bg=CARD, fg=rank_fg,
                                   font=("Consolas", 8, "bold"), width=3)
        widgets["rank"].pack(side="left")
        widgets["name"] = tk.Label(top, text="—", bg=CARD, fg=TEXT,
                                   font=("Consolas", 11, "bold"), width=5)
        widgets["name"].pack(side="left")
        bot = tk.Frame(f, bg=CARD)
        bot.pack(fill="x")
        widgets["ev"]  = tk.Label(bot, text="", bg=CARD, fg=MUTED,
                                  font=("Consolas", 8))
        widgets["ev"].pack(side="left")
        widgets["eff"] = tk.Label(bot, text="", bg=CARD, fg=MUTED,
                                  font=("Consolas", 8))
        widgets["eff"].pack(side="left", padx=4)
        widgets["shan"] = tk.Label(bot, text="", bg=CARD, fg=MUTED,
                                   font=("Consolas", 8))
        widgets["shan"].pack(side="right")
        f._widgets = widgets
        return f

    def set_computing(self):
        self._stop_skel_shimmer()
        self._r1_name.config(text="…", fg=SKEL2)
        self._r1_ev.config(text="")
        self._r1_shan.config(text="")
        self._r1_yaku.config(text="")
        self._eff_count_lbl.config(text="")
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()
        for card in (self._card2, self._card3):
            card._widgets["name"].config(text="…", fg=SKEL2)
            card._widgets["ev"].config(text="")
            card._widgets["eff"].config(text="")
            card._widgets["shan"].config(text="")
        self._status_var.set("Computing…")
        self._progress.config(mode="indeterminate")
        self._progress.start(8)
        self._skel_cells = [self._card1, self._card2, self._card3]
        self._skel_phase = 0
        self._shimmer_step()

    def _shimmer_step(self):
        if not self._skel_cells:
            return
        try:
            col = SKEL if self._skel_phase % 2 == 0 else SKEL2
            for c in self._skel_cells:
                c.config(bg=col)
            self._skel_phase += 1
            self._skel_job = self.after(600, self._shimmer_step)
        except tk.TclError:
            pass

    def _stop_skel_shimmer(self):
        if self._skel_job:
            self.after_cancel(self._skel_job)
            self._skel_job = None
        self._skel_cells = []
        for c in (self._card1, self._card2, self._card3):
            try:
                c.config(bg=CARD)
            except Exception:
                pass

    def update_phase1(self, effs: List[Tuple[int, int]], s: int):
        self._update_eff_chips(effs)

    def update_agari(self, agari_info: Dict):
        """Display agari (winning hand) result instead of EV discard advice."""
        self._stop_skel_shimmer()
        self._progress.stop()
        self._progress.config(mode="determinate", value=100)
        self._status_var.set("和牌!")

        han   = agari_info.get("han", 0)
        fu    = agari_info.get("fu", 0)
        score = agari_info.get("score", 0)
        yaku  = agari_info.get("yaku", [])

        self._r1_name.config(text="和牌!", fg=GOLD)
        self._r1_ev.config(text=f"≈{score}", fg=GOLD)
        self._r1_shan.config(text=f"{han}han {fu}fu", fg=MUTED)

        yaku_str = "  ".join(f"{n}({h})" for n, h in yaku
                             if isinstance(h, int) and h > 0)
        self._r1_yaku.config(text=yaku_str if yaku_str else "役なし(立直?)")

        # Clear eff chips and compact cards — not relevant for agari
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()
        self._eff_count_lbl.config(text="")
        for card in (self._card2, self._card3):
            for k in ("name", "ev", "eff", "shan"):
                card._widgets[k].config(text="—", fg=MUTED)

    def update(self, ev_results: List[Dict], is_mc: bool, dt: float):
        self._stop_skel_shimmer()
        self._progress.stop()
        self._progress.config(mode="determinate", value=100)
        self._status_var.set(f"SimpleEV  {dt*1000:.0f}ms")

        def _fill_r1(r: Dict):
            self._r1_name.config(text=r["discard_name"], fg=BEST)
            self._r1_ev.config(text=f"EV {r['ev']:.0f}")
            self._r1_shan.config(text=f"向聴{r['shanten']}")
            yaku_str  = "  ".join(f"{n}({h})" for n, h in r.get("yaku", [])
                                  if isinstance(h, int) and h > 0)
            score_str = f"≈{r['est_score']}  {r['han']}han {r['fu']}fu"
            self._r1_yaku.config(text=f"{score_str}  {yaku_str}")
            self._update_eff_chips(r.get("eff_tiles", []))

        def _fill_compact(card: tk.Frame, r: Dict):
            card._widgets["name"].config(text=r["discard_name"], fg=TEXT)
            card._widgets["ev"].config(text=f"EV{r['ev']:.0f}", fg=SAFE_DIM)
            card._widgets["eff"].config(
                text=f"Eff{r['eff_count']}", fg=MUTED)
            card._widgets["shan"].config(
                text=f"向{r['shanten']}", fg=MUTED)

        def _clear_compact(card: tk.Frame):
            for k in ("name", "ev", "eff", "shan"):
                card._widgets[k].config(text="—", fg=MUTED)

        if ev_results:
            _fill_r1(ev_results[0])
        if len(ev_results) > 1:
            _fill_compact(self._card2, ev_results[1])
        else:
            _clear_compact(self._card2)
        if len(ev_results) > 2:
            _fill_compact(self._card3, ev_results[2])
        else:
            _clear_compact(self._card3)

    def _update_eff_chips(self, effs: List[Tuple[int, int]]):
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()
        total = sum(c for _, c in effs)
        self._eff_count_lbl.config(text=f"{total} tiles" if total else "")
        for tid, cnt in effs[:10]:
            chip_photo = self._cache.get_chip(tid)
            if chip_photo:
                lbl = tk.Label(self._eff_frame, image=chip_photo, bg=CARD,
                               cursor="hand2")
                lbl.image = chip_photo
            else:
                name = tile_name(tid)
                lbl  = tk.Label(self._eff_frame, text=name, bg=CARD,
                                fg=_suit_fg(name), font=("Consolas", 6),
                                width=3)
            lbl.pack(side="left", padx=1)
            tk.Label(self._eff_frame, text=f"×{cnt}", bg=CARD, fg=MUTED,
                     font=("Consolas", 6)).pack(side="left")

    def clear(self):
        self._stop_skel_shimmer()
        self._progress.stop()
        self._progress.config(mode="determinate", value=0)
        self._status_var.set("—")
        self._r1_name.config(text="—", fg=MUTED)
        self._r1_ev.config(text="")
        self._r1_shan.config(text="")
        self._r1_yaku.config(text="")
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()
        self._eff_count_lbl.config(text="")
        for card in (self._card2, self._card3):
            for k in ("name", "ev", "eff", "shan"):
                card._widgets[k].config(text="—", fg=MUTED)


# ─────────────────────────────────────────────────────────────────────────────
# DoraPanel  — add + chip list with remove
# ─────────────────────────────────────────────────────────────────────────────

class DoraPanel(tk.Frame):
    def __init__(self, parent,
                 on_dora: Callable,
                 on_remove_dora: Callable,
                 on_reset: Callable, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._on_remove = on_remove_dora

        row = tk.Frame(self, bg=PANEL)
        row.pack(fill="x", padx=6, pady=(3, 0))
        tk.Label(row, text="Dora:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")
        self._dora_var = tk.StringVar(value=TILE_NAMES[0])
        ttk.Combobox(row, textvariable=self._dora_var,
                     values=list(TILE_NAMES), width=6,
                     state="readonly").pack(side="left", padx=4)
        tk.Button(row, text="Add", bg=ACCENT, fg=TEXT,
                  font=("Consolas", 8), relief="flat",
                  command=lambda: on_dora(self._dora_var.get())
                  ).pack(side="left")
        tk.Button(row, text="Reset State", bg="#2a1010", fg=DANGER,
                  font=("Consolas", 8), relief="flat",
                  command=on_reset).pack(side="right", padx=(0, 2))

        # Chip row for added doras
        self._chip_row = tk.Frame(self, bg=PANEL)
        self._chip_row.pack(fill="x", padx=6, pady=(2, 3))

    def refresh(self, dora_indicators: List[int]):
        """Rebuild chip row from current dora_indicators list."""
        for w in self._chip_row.winfo_children():
            w.destroy()
        for idx, tid in enumerate(dora_indicators):
            name   = tile_name(tid)
            chip   = tk.Frame(self._chip_row, bg=ACCENT, padx=3, pady=1)
            chip.pack(side="left", padx=2)
            tk.Label(chip, text=name, bg=ACCENT, fg=SAFE,
                     font=("Consolas", 8, "bold")).pack(side="left")
            _idx = idx  # capture
            tk.Button(chip, text="×", bg=ACCENT, fg=MUTED,
                      font=("Consolas", 7), relief="flat", padx=0, pady=0,
                      command=lambda i=_idx: self._on_remove(i)
                      ).pack(side="left")


# ─────────────────────────────────────────────────────────────────────────────
# MeldInputDialog  — popup to select kind + tiles for a new meld
# ─────────────────────────────────────────────────────────────────────────────

class MeldInputDialog(tk.Toplevel):
    def __init__(self, parent_root,
                 on_confirm: Callable[[str, List[str]], None]):
        super().__init__(parent_root)
        self._on_confirm = on_confirm
        self.title("Add Meld")
        self.configure(bg=CARD)
        self.resizable(False, False)
        self.attributes("-topmost", True)

        # Kind selection
        kind_row = tk.Frame(self, bg=CARD)
        kind_row.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(kind_row, text="Kind:", bg=CARD, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")
        self._kind_var = tk.StringVar(value="pon")
        for k in ("pon", "chi", "open_kan", "closed_kan"):
            tk.Radiobutton(kind_row, text=k, variable=self._kind_var,
                           value=k, bg=CARD, fg=TEXT, selectcolor=ACCENT,
                           activebackground=CARD,
                           font=("Consolas", 8),
                           command=self._on_kind_change
                           ).pack(side="left", padx=4)

        # Tile selectors
        tile_row = tk.Frame(self, bg=CARD)
        tile_row.pack(fill="x", padx=10, pady=4)
        self._tile_vars = []
        self._tile_cbs  = []
        for i in range(4):
            v  = tk.StringVar(value=TILE_NAMES[0])
            cb = ttk.Combobox(tile_row, textvariable=v,
                              values=list(TILE_NAMES), width=5,
                              state="readonly")
            cb.pack(side="left", padx=2)
            self._tile_vars.append(v)
            self._tile_cbs.append(cb)

        self._on_kind_change()  # set initial visibility

        # Confirm / Cancel
        btn_row = tk.Frame(self, bg=CARD)
        btn_row.pack(fill="x", padx=10, pady=(4, 8))
        tk.Button(btn_row, text="Confirm", bg=SAFE_DIM, fg=TEXT,
                  font=("Consolas", 9), relief="flat",
                  command=self._confirm).pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="Cancel", bg=CARD, fg=MUTED,
                  font=("Consolas", 9), relief="flat",
                  command=self.destroy).pack(side="left")

        self.grab_set()

    def _on_kind_change(self):
        k = self._kind_var.get()
        n = 4 if "kan" in k else 3
        for i, cb in enumerate(self._tile_cbs):
            if i < n:
                cb.config(state="readonly")
            else:
                cb.config(state="disabled")

    def _confirm(self):
        k = self._kind_var.get()
        n = 4 if "kan" in k else 3
        tiles = [self._tile_vars[i].get() for i in range(n)]
        self.destroy()
        self._on_confirm(k, tiles)

    def center_on(self, x: int, y: int):
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"+{x - w//2}+{y - h//2}")


# ─────────────────────────────────────────────────────────────────────────────
# MeldPanel  — collapsible list + add button
# ─────────────────────────────────────────────────────────────────────────────

class MeldPanel(tk.Frame):
    def __init__(self, parent,
                 on_add_meld: Callable[[str, List[str]], None],
                 on_remove_meld: Callable[[int], None], **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._on_add    = on_add_meld
        self._on_remove = on_remove_meld
        self._open      = False

        # Header row (collapsible toggle)
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=6, pady=2)
        self._toggle_btn = tk.Button(hdr, text="MELDS ▶", bg=PANEL, fg=MUTED,
                                     font=("Consolas", 8, "bold"), relief="flat",
                                     command=self._toggle)
        self._toggle_btn.pack(side="left")
        tk.Button(hdr, text="+ Add", bg=ACCENT, fg=SAFE,
                  font=("Consolas", 8), relief="flat",
                  command=self._open_add_dialog).pack(side="right")

        # Collapsible body
        self._body = tk.Frame(self, bg=PANEL)
        # NOT packed yet

        self._melds_data: List[Tuple[str, List[int]]] = []

    def _toggle(self):
        self._open = not self._open
        if self._open:
            self._body.pack(fill="x", padx=6, pady=(0, 4))
            self._toggle_btn.config(text="MELDS ▼")
        else:
            self._body.pack_forget()
            self._toggle_btn.config(text="MELDS ▶")

    def _open_add_dialog(self):
        dlg = MeldInputDialog(self.winfo_toplevel(), self._on_add)
        dlg.center_on(self.winfo_rootx() + 100,
                      self.winfo_rooty() + 20)

    def refresh(self, melds: List[Tuple[str, List[int]]]):
        """Rebuild list from (kind, tiles) tuples."""
        self._melds_data = melds
        for w in self._body.winfo_children():
            w.destroy()
        if not melds:
            tk.Label(self._body, text="(none)", bg=PANEL, fg=MUTED,
                     font=("Consolas", 7)).pack(anchor="w")
            return
        for idx, (kind, tiles) in enumerate(melds):
            row = tk.Frame(self._body, bg=PANEL)
            row.pack(fill="x", pady=1)
            kind_short = {"pon": "PON", "chi": "CHI",
                          "open_kan": "KAN", "closed_kan": "kKAN"}.get(kind, kind)
            tk.Label(row, text=kind_short, bg=PANEL, fg=SAFE_DIM,
                     font=("Consolas", 8, "bold"), width=5).pack(side="left")
            tile_str = " ".join(tile_name(t) for t in tiles)
            tk.Label(row, text=tile_str, bg=PANEL, fg=TEXT,
                     font=("Consolas", 8)).pack(side="left", padx=4)
            _i = idx
            tk.Button(row, text="×", bg=PANEL, fg=DANGER,
                      font=("Consolas", 8), relief="flat",
                      command=lambda i=_i: self._on_remove(i)
                      ).pack(side="right")


# ─────────────────────────────────────────────────────────────────────────────
# DiscardPanel  — collapsible 34×4 visual grid
# ─────────────────────────────────────────────────────────────────────────────

# Tile layout: Man(0-8), Pin(9-17), Sou(18-26), Winds(27-30), Dragons(31-33)
_CELL_W   = 15   # px per tile column
_CELL_H   = 11   # px per copy row
_BAND_H   = 11   # coloured suit-name band height
_NUM_H    = 13   # tile number / honour label row height
_HDR_H    = _BAND_H + _NUM_H
_GAP      = 6    # px gap between suit groups
_HON_GAP  = 4    # extra gap between winds and dragons

# (first_tid, count, display_label, band_bg, text_fg)
_SUIT_GROUPS = [
    ( 0, 9, "萬", "#3d1800", "#ffa040"),
    ( 9, 9, "筒", "#0d2d0d", "#60c060"),
    (18, 9, "索", "#0a1e30", "#5ab0f0"),
    (27, 4, "風", "#1a1a38", "#9898ff"),
    (31, 3, "龍", "#2d0f0f", "#ff8080"),
]

_HON_LABEL = {27:"東", 28:"南", 29:"西", 30:"北",
              31:"白", 32:"發", 33:"中"}
_HON_COLOR = {27:"#9898ff", 28:"#9898ff", 29:"#9898ff", 30:"#9898ff",
              31:"#e0e0e0", 32:"#60c060",  33:"#ef5350"}

def _tid_to_canvas_x(tid: int) -> int:
    """Return left-edge x for tile tid in the discard canvas."""
    if tid < 9:    return tid * _CELL_W
    elif tid < 18: return 9  * _CELL_W + _GAP     + (tid - 9)  * _CELL_W
    elif tid < 27: return 18 * _CELL_W + 2*_GAP   + (tid - 18) * _CELL_W
    elif tid < 31: return 27 * _CELL_W + 3*_GAP   + (tid - 27) * _CELL_W
    else:          return 31 * _CELL_W + 3*_GAP + _HON_GAP + (tid - 31) * _CELL_W

_CANVAS_W  = _tid_to_canvas_x(33) + _CELL_W
_CANVAS_H  = _HDR_H + 4 * _CELL_H + 2

_DISC_FILL  = "#3a5a7a"   # filled (discarded)
_DISC_EMPTY = "#1e2a3a"   # unfilled
_DISC_OVER  = "#ef5350"   # over-limit highlight


class DiscardPanel(tk.Frame):
    def __init__(self, parent,
                 on_change: Callable[[int, int], None], **kw):
        """on_change(tid, new_count) called when user edits a tile's discard count."""
        super().__init__(parent, bg=PANEL, **kw)
        self._on_change = on_change
        self._counts    = np.zeros(N_TILES, dtype=np.int32)
        self._open      = False
        self._drag_count: Optional[int] = None   # count being set during drag

        # Header
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=6, pady=2)
        self._toggle_btn = tk.Button(hdr, text="DISCARDS ▶", bg=PANEL, fg=MUTED,
                                     font=("Consolas", 8, "bold"), relief="flat",
                                     command=self._toggle)
        self._toggle_btn.pack(side="left")
        tk.Button(hdr, text="Clear", bg=PANEL, fg=DANGER,
                  font=("Consolas", 8), relief="flat",
                  command=self._clear_all).pack(side="right")

        # Canvas
        self._body = tk.Frame(self, bg=PANEL)
        self._cv = tk.Canvas(self._body, width=_CANVAS_W, height=_CANVAS_H,
                             bg=BG, highlightthickness=0)
        self._cv.pack(padx=4, pady=2)
        self._draw_grid()

        self._cv.bind("<ButtonPress-1>",   self._on_press)
        self._cv.bind("<B1-Motion>",       self._on_drag)
        self._cv.bind("<ButtonRelease-1>", self._on_release)

    def _toggle(self):
        self._open = not self._open
        if self._open:
            self._body.pack(fill="x")
            self._toggle_btn.config(text="DISCARDS ▼")
        else:
            self._body.pack_forget()
            self._toggle_btn.config(text="DISCARDS ▶")

    def _clear_all(self):
        self._counts[:] = 0
        self._draw_grid()
        for tid in range(N_TILES):
            self._on_change(tid, 0)

    # ── Grid drawing ──────────────────────────────────────────────────────────

    def _draw_grid(self):
        self._cv.delete("all")
        # ── Suit band row ─────────────────────────────────────────────────────
        for (start, count, label, band_bg, band_fg) in _SUIT_GROUPS:
            x1 = _tid_to_canvas_x(start)
            x2 = _tid_to_canvas_x(start + count - 1) + _CELL_W
            self._cv.create_rectangle(x1, 0, x2, _BAND_H - 1,
                                      fill=band_bg, outline="")
            self._cv.create_text((x1 + x2) // 2, _BAND_H // 2,
                                 text=label, fill=band_fg,
                                 font=("Consolas", 7, "bold"), anchor="center")
        # ── Tile number / honour label row ────────────────────────────────────
        for tid in range(N_TILES):
            cx = _tid_to_canvas_x(tid)
            if tid >= 27:
                text  = _HON_LABEL[tid]
                color = _HON_COLOR[tid]
                font  = ("Consolas", 8)
            else:
                text  = str(tid % 9 + 1)
                color = _suit_fg(tile_name(tid))
                font  = ("Consolas", 8, "bold")
            self._cv.create_text(cx + _CELL_W // 2,
                                 _BAND_H + _NUM_H // 2,
                                 text=text, fill=color, font=font,
                                 anchor="center")
        # ── Count cells ───────────────────────────────────────────────────────
        for tid in range(N_TILES):
            cx    = _tid_to_canvas_x(tid)
            count = int(self._counts[tid])
            for row in range(4):
                y1   = _HDR_H + row * _CELL_H
                y2   = y1 + _CELL_H - 1
                over = count > 4
                fill = (DANGER if over else
                        _DISC_FILL if row < count else _DISC_EMPTY)
                self._cv.create_rectangle(
                    cx + 1, y1 + 1, cx + _CELL_W - 1, y2,
                    fill=fill, outline="", tags=f"cell_{tid}_{row}")

    def _canvas_to_tid_row(self, x: int, y: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert canvas (x,y) to (tile_id, row_0..3) or (None, None)."""
        if y < _HDR_H:
            return None, None
        row = (y - _HDR_H) // _CELL_H
        if row < 0 or row >= 4:
            return None, None
        # Determine tid by x
        for tid in range(N_TILES):
            cx = _tid_to_canvas_x(tid)
            if cx <= x < cx + _CELL_W:
                return tid, row
        return None, None

    def _on_press(self, event):
        tid, row = self._canvas_to_tid_row(event.x, event.y)
        if tid is None:
            return
        new_count = row + 1
        if new_count == self._counts[tid]:
            new_count = row   # click same row → decrement
        new_count = max(0, min(4, new_count))
        self._drag_count = new_count
        self._set_tile_count(tid, new_count)

    def _on_drag(self, event):
        if self._drag_count is None:
            return
        tid, row = self._canvas_to_tid_row(event.x, event.y)
        if tid is None:
            return
        self._set_tile_count(tid, self._drag_count)

    def _on_release(self, event):
        self._drag_count = None

    def _set_tile_count(self, tid: int, count: int):
        if self._counts[tid] == count:
            return
        self._counts[tid] = count
        self._redraw_tile(tid)
        self._on_change(tid, count)

    def _redraw_tile(self, tid: int):
        """Redraw only the 4 rows for a single tile (faster than full redraw)."""
        cx    = _tid_to_canvas_x(tid)
        count = int(self._counts[tid])
        for row in range(4):
            self._cv.delete(f"cell_{tid}_{row}")
            y1     = _HDR_H + row * _CELL_H
            y2     = y1 + _CELL_H - 1
            filled = row < count
            over   = count > 4
            if over:
                fill = DANGER
            elif filled:
                fill = _DISC_FILL
            else:
                fill = _DISC_EMPTY
            self._cv.create_rectangle(
                cx + 1, y1 + 1, cx + _CELL_W - 1, y2,
                fill=fill, outline="", tags=f"cell_{tid}_{row}")

    def refresh(self, discards_arr: np.ndarray):
        """Refresh grid from external counts array (e.g. after YOLO update)."""
        self._counts = discards_arr.copy()
        self._draw_grid()


# ─────────────────────────────────────────────────────────────────────────────
# ControlPanel  — Analyze + Auto; ⚙ drawer (no X/Y fields)
# ─────────────────────────────────────────────────────────────────────────────

class ControlPanel(tk.Frame):
    def __init__(self, parent, on_analyze: Callable,
                 on_auto_toggle: Callable,
                 on_select_region: Callable,
                 on_model_change: Callable,
                 model_list: List[str],
                 current_model: str = "", **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.auto_var     = tk.BooleanVar(value=False)
        self.interval_var = tk.IntVar(value=5)
        self.sims_var     = tk.IntVar(value=0)
        self._model_var   = tk.StringVar(value=current_model)

        main_row = tk.Frame(self, bg=PANEL)
        main_row.pack(fill="x", padx=6, pady=4)

        self._analyze_btn = tk.Button(
            main_row, text="  Analyze  [F9]  ",
            bg=ACCENT, fg=TEXT,
            font=("Consolas", 11, "bold"),
            relief="flat", padx=10, pady=6,
            command=on_analyze)
        self._analyze_btn.pack(side="left", padx=(0, 8))

        tk.Checkbutton(main_row, text="Auto", variable=self.auto_var,
                       bg=PANEL, fg=TEXT, selectcolor=ACCENT,
                       activebackground=PANEL, font=("Consolas", 9),
                       command=on_auto_toggle).pack(side="left")

        self._dot = tk.Label(main_row, text="●", bg=PANEL, fg=MUTED,
                             font=("Consolas", 16))
        self._dot.pack(side="right")
        self._status_lbl = tk.Label(main_row, text="Ready", bg=PANEL,
                                    fg=MUTED, font=("Consolas", 8))
        self._status_lbl.pack(side="right", padx=4)

        self._drawer_open = False
        gear_btn = tk.Button(main_row, text="⚙", bg=PANEL, fg=MUTED,
                             font=("Consolas", 10), relief="flat",
                             command=self._toggle_drawer)
        gear_btn.pack(side="right", padx=4)

        # ── Drawer ────────────────────────────────────────────────────────────
        self._drawer = tk.Frame(self, bg=PANEL)

        d = self._drawer
        tk.Frame(d, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)

        r1 = tk.Frame(d, bg=PANEL)
        r1.pack(fill="x", padx=6, pady=2)
        tk.Button(r1, text="⊞ Select Region", bg=ACCENT, fg=SAFE,
                  font=("Consolas", 9, "bold"), relief="flat", padx=6, pady=3,
                  command=on_select_region).pack(side="left")
        self._region_lbl = tk.Label(r1, text="(not set)", bg=PANEL, fg=MUTED,
                                    font=("Consolas", 7))
        self._region_lbl.pack(side="left", padx=6)

        r2 = tk.Frame(d, bg=PANEL)
        r2.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(r2, text="Interval:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")
        ttk.Combobox(r2, textvariable=self.interval_var,
                     values=[3, 5, 10, 15, 30], width=3,
                     state="readonly").pack(side="left", padx=2)
        tk.Label(r2, text="s", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")

        r3 = tk.Frame(d, bg=PANEL)
        r3.pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(r3, text="Model:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")
        self._model_cb = ttk.Combobox(r3, textvariable=self._model_var,
                                      values=model_list, width=26,
                                      state="readonly")
        self._model_cb.pack(side="left", padx=(4, 0))
        self._model_cb.bind("<<ComboboxSelected>>",
                            lambda e: on_model_change(self._model_var.get()))

    def _toggle_drawer(self):
        if self._drawer_open:
            self._drawer.pack_forget()
        else:
            self._drawer.pack(fill="x")
        self._drawer_open = not self._drawer_open

    def pulse(self, step: int = 0):
        FRAMES = [(SAFE_DIM, 12), (SAFE_DIM, 12), (ACCENT, 11), (ACCENT, 11)]
        if step < len(FRAMES):
            bg, sz = FRAMES[step]
            try:
                self._analyze_btn.config(bg=bg, font=("Consolas", sz, "bold"))
                self.after(70, lambda: self.pulse(step + 1))
            except tk.TclError:
                pass
        else:
            try:
                self._analyze_btn.config(bg=ACCENT,
                                         font=("Consolas", 11, "bold"))
            except tk.TclError:
                pass

    def set_status(self, text: str, color: str = MUTED):
        self._status_lbl.config(text=text, fg=color)
        self._dot.config(fg=color)

    def get_interval(self) -> int:
        return self.interval_var.get()

    def set_region_label(self, text: str):
        try:
            self._region_lbl.config(text=text)
        except tk.TclError:
            pass

    def set_model(self, name: str):
        try:
            self._model_var.set(name)
        except tk.TclError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MainWindow  — View orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow:
    MODEL_PATH = PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt"
    CROPS_DIR  = PROJECT_ROOT / "tile_crops"
    DEBUG_DIR  = PROJECT_ROOT / "runs" / "advisor"

    def __init__(self, model_path=None):
        if model_path:
            self.MODEL_PATH = model_path

        self.root = tk.Tk()
        self._setup_window()
        self._setup_style()

        self._cache    = TileImageCache(self.CROPS_DIR)
        self._last_p1: Optional[Phase1Result] = None
        self._crop:    Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h)

        # Scan available models (best.pt only, named by experiment folder)
        self._model_map: Dict[str, Path] = {}
        for pt in sorted((PROJECT_ROOT / "runs" / "detect").glob("*/weights/best.pt")):
            self._model_map[pt.parent.parent.name] = pt
        self._model_list = list(self._model_map.keys())
        self._current_model_name = self.MODEL_PATH.parent.parent.name \
            if self.MODEL_PATH.parent.name == "weights" else self.MODEL_PATH.stem

        self._build_panels()

        self._advisor = AdvisorController(
            root       = self.root,
            model_path = self.MODEL_PATH,
            on_phase1  = self._handle_phase1,
            on_phase2  = self._handle_phase2,
            on_hotkey  = self._trigger_analysis,
        )
        self._advisor.start()
        self.root.after(2000, self._update_cap_status)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("MJ Advisor")
        sw = self.root.winfo_screenwidth()
        x  = max(0, sw - WINDOW_W - 10)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+30")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)

    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame",    background=BG)
        style.configure("TLabel",    background=BG, foreground=TEXT)
        style.configure("TCombobox", fieldbackground=CARD, foreground=TEXT,
                        background=CARD, selectbackground=ACCENT)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)])
        style.configure("EV.Horizontal.TProgressbar",
                        troughcolor=CARD, background=SAFE, thickness=5)

    def _build_panels(self):
        # Scrollable main area
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=4, pady=4)

        self._cap_lbl = tk.Label(main, text="Initializing…", bg=BG, fg=MUTED,
                                 font=("Consolas", 7), anchor="w")
        self._cap_lbl.pack(fill="x", pady=(0, 2))

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        # Live view (hidden until region selected)
        self._live = LiveViewPanel(main)
        self._live.pack(fill="x", pady=(2, 0))
        self._live_visible = True

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._hand = HandPanel(main, self._cache, self._on_correction)
        self._hand.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._ev = EVPanel(main, self._cache)
        self._ev.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._dora = DoraPanel(main,
                               on_dora=self._on_add_dora,
                               on_remove_dora=self._on_remove_dora,
                               on_reset=self._on_reset_state)
        self._dora.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        self._meld = MeldPanel(main,
                               on_add_meld=self._on_add_meld,
                               on_remove_meld=self._on_remove_meld)
        self._meld.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        self._discard = DiscardPanel(main,
                                     on_change=self._on_discard_change)
        self._discard.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._ctrl = ControlPanel(main,
                                  on_analyze=self._trigger_analysis,
                                  on_auto_toggle=self._toggle_auto,
                                  on_select_region=self._on_select_region,
                                  on_model_change=self._on_model_change,
                                  model_list=self._model_list,
                                  current_model=self._current_model_name)
        self._ctrl.pack(fill="x")

    # ── User actions → controller ─────────────────────────────────────────────

    def _trigger_analysis(self):
        crop = self._crop if self._crop else (0, 0, 0, 0)
        if self._advisor.trigger(crop=crop):
            self._ctrl.set_status("Detecting…", SAFE)
            self._ctrl.pulse()
            self._hand.set_skeleton()
            self._ev.set_computing()

    def _toggle_auto(self):
        enabled = self._ctrl.auto_var.get()
        crop    = self._crop if self._crop else (0, 0, 0, 0)
        ms      = self._ctrl.get_interval() * 1000
        self._advisor.set_auto(enabled, interval_ms=ms, crop=crop)
        self._ctrl.set_status("Auto ON" if enabled else "Auto OFF",
                              SAFE if enabled else MUTED)

    def _on_correction(self, cx: int, tile_name_str: str):
        if self._advisor.apply_correction(cx, tile_name_str):
            self._trigger_analysis()

    def _on_add_dora(self, tile_name_str: str):
        self._advisor.add_dora(tile_name_str)
        self._refresh_state_panels()

    def _on_remove_dora(self, idx: int):
        self._advisor.remove_dora(idx)
        self._refresh_state_panels()

    def _on_add_meld(self, kind: str, tile_names: List[str]):
        ok = self._advisor.add_meld(kind, tile_names)
        if not ok:
            self._ctrl.set_status("Meld rejected (tile limit)", DANGER)
        self._refresh_state_panels()

    def _on_remove_meld(self, idx: int):
        self._advisor.remove_meld(idx)
        self._refresh_state_panels()

    def _on_discard_change(self, tid: int, count: int):
        self._advisor.set_discard_count_by_id(tid, count)

    def _on_reset_state(self):
        self._advisor.reset_state()
        self._hand.clear()
        self._ev.clear()
        self._discard.refresh(np.zeros(N_TILES, dtype=np.int32))
        self._meld.refresh([])
        self._dora.refresh([])
        self._ctrl.set_status("Reset", SAFE)

    def _refresh_state_panels(self):
        snap = self._advisor.get_state_snapshot()
        self._dora.refresh(snap["doras"])
        self._meld.refresh(snap["melds"])
        if "discards_arr" in snap:
            self._discard.refresh(snap["discards_arr"])

    # ── Controller callbacks → panels ─────────────────────────────────────────

    def _handle_phase1(self, r1: Phase1Result):
        self._last_p1 = r1
        if not r1.capture_ok:
            self._hand.clear()
            self._ev.clear()
            self._ctrl.set_status(f"Err: {r1.error_msg[:28]}", DANGER)
        else:
            n = len(r1.hand_tiles)
            self._hand.update(r1.hand_tiles, r1.shanten)
            if r1.shanten == -1 and r1.agari_info:
                self._ev.update_agari(r1.agari_info)
                self._ctrl.set_status(f"和牌! [{r1.game_mode}]", GOLD)
            elif n == 14:
                self._ev.update_phase1(r1.effective_tiles, r1.shanten)
                self._ctrl.set_status(f"EV… [{r1.game_mode}]", WARN)
            else:
                self._ev.clear()
                self._ctrl.set_status(f"手牌 {n}/14 [{r1.game_mode}]", MUTED)
            # Update live view detection boxes
            if self._crop:
                self._live.update_detections(r1.hand_tiles, self._crop)
            # Sync discard panel from YOLO state
            snap = self._advisor.get_state_snapshot()
            if "discards_arr" in snap:
                self._discard.refresh(snap["discards_arr"])

    def _handle_phase2(self, r2: Phase2Result):
        # If last Phase1 was an agari, skip updating EV panel (agari display takes priority)
        if self._last_p1 and self._last_p1.shanten == -1 and self._last_p1.agari_info:
            return
        self._ev.update(r2.ev_results, r2.is_mc, r2.compute_time)
        if r2.ev_results and self._last_p1 and self._last_p1.capture_ok:
            best_tid = r2.ev_results[0]["discard_tid"]
            self._hand.update(self._last_p1.hand_tiles,
                              self._last_p1.shanten, best_tid)
        mode = self._last_p1.game_mode if self._last_p1 else "?"
        self._ctrl.set_status(f"Done [{mode}]", SAFE)

    # ── Capture / region / debug ──────────────────────────────────────────────

    def _update_cap_status(self):
        self._cap_lbl.config(
            text=f"Capture: {self._advisor.get_capture_status()}")
        self.root.after(5000, self._update_cap_status)

    def _on_select_region(self):
        self.root.withdraw()

        def on_select(x: int, y: int, w: int, h: int):
            self.root.deiconify()
            self._crop = (x, y, w, h)
            self._live.set_region(x, y, w, h)
            self._ctrl.set_region_label(f"{w}×{h} @ ({x},{y})")
            self._ctrl.set_status(f"Region set", SAFE)
            if not self._ctrl._drawer_open:
                self._ctrl._toggle_drawer()

        def on_cancel():
            self.root.deiconify()

        RegionSelector(self.root, on_select, on_cancel)

    def _on_model_change(self, model_name: str):
        pt = self._model_map.get(model_name)
        if pt is None:
            return
        self._ctrl.set_status(f"Loading {model_name}…", WARN)
        self.root.update_idletasks()
        self._advisor.switch_model(pt)
        self._current_model_name = model_name
        self._ctrl.set_status(f"Model: {model_name}", SAFE)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._live.stop()
        self._advisor.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Majsoul Windows Advisor")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    app  = MainWindow(model_path=Path(args.model) if args.model else None)
    app.run()
