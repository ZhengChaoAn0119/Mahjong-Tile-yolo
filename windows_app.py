"""
windows_app.py
Mahjong Soul advisor companion window — View layer (MVC).

Layout:
  HandPanel       — tile strip, click any tile to correct it
  EVPanel         — #1 large card + #2/#3 compact; effective tiles as chips
  DoraPanel       — compact dora input + state reset
  ControlPanel    — big Analyze button + Auto; ⚙ drawer for advanced settings

Press F9 to analyze. Analyze button pulses to confirm trigger.
Skeleton screen during computation reduces visual flicker.
"""
from __future__ import annotations

import sys, argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable

import tkinter as tk
from tkinter import ttk

import numpy as np

# ── Project path ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
sys.path.insert(0, str(PROJECT_ROOT))

from tile_codec          import N_TILES, TILE_NAMES, tile_name
from frame_smoother      import ConfirmedTile
from advisor_controller  import AdvisorController, Phase1Result, Phase2Result

# ── Window ────────────────────────────────────────────────────────────────────
WINDOW_W = 420
WINDOW_H = 740

# ── Color palette  (safe=teal, warn=orange, danger=red) ───────────────────────
BG       = "#12121f"   # near-black navy
PANEL    = "#1a1f35"   # dark panel
CARD     = "#1f2640"   # slightly lighter card background
BORDER   = "#2a3050"   # subtle border
ACCENT   = "#0d3a5c"   # button / active bg

SAFE     = "#26c6da"   # teal  — best / tenpai
SAFE_DIM = "#0d7a86"   # dimmer teal
WARN     = "#ff9800"   # orange — low-confidence
WARN_DIM = "#7a4800"
DANGER   = "#ef5350"   # red   — error / danger
BEST     = "#00e676"   # green — #1 rank accent

TEXT     = "#dce3f0"   # primary text
MUTED    = "#6b7394"   # secondary / labels
SKEL     = "#252a42"   # skeleton placeholder color
SKEL2    = "#2e3450"   # skeleton shimmer

SUIT_COLOR = {"m": "#ffa040", "p": "#60c060", "s": "#5ab0f0"}

# ── Tile suit helper ──────────────────────────────────────────────────────────
def _suit_fg(name: str) -> str:
    return SUIT_COLOR.get(name[-1], TEXT) if name and len(name) >= 2 else TEXT


# ─────────────────────────────────────────────────────────────────────────────
# TileImageCache  (force-generate colored tiles if crops missing)
# ─────────────────────────────────────────────────────────────────────────────

class TileImageCache:
    """
    Loads crop images from tile_crops/<tid>/.
    If directory is missing or empty, generates colored label images via PIL
    so the UI always has a visual for every tile.
    """
    W, H = 28, 40    # display size (tile strip)
    CW, CH = 20, 28  # chip size (effective tiles)

    def __init__(self, crops_dir: Path):
        self._photos:      Dict[int, object] = {}  # full size
        self._chip_photos: Dict[int, object] = {}  # chip size
        self._crops_dir = crops_dir
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk
            self._Image    = Image
            self._ImageDraw = ImageDraw
            self._ImageTk  = ImageTk
            self._has_pil  = True
            self._preload_all()
        except ImportError:
            self._has_pil = False

    def _make_fallback(self, tid: int, w: int, h: int):
        """Generate a colored rectangle with tile name as PIL Image."""
        name = tile_name(tid)
        suit = name[-1] if len(name) >= 2 else "?"
        bg_map = {"m": (80, 40, 10), "p": (20, 60, 20), "s": (15, 45, 80)}
        bg = bg_map.get(suit, (40, 30, 70))
        img = self._Image.new("RGB", (w, h), bg)
        d = self._ImageDraw.Draw(img)
        # tile name — small font
        try:
            font = self._ImageFont.truetype("consola.ttf", max(8, h // 5))
        except Exception:
            font = None
        text_color = (220, 220, 220)
        # draw lines for name
        lines = [name[:-1], name[-1]] if len(name) > 2 else [name]
        y = h // 4
        for line in lines:
            bbox = d.textbbox((0, 0), line, font=font) if font else (0, 0, w, 10)
            tw = bbox[2] - bbox[0]
            d.text(((w - tw) // 2, y), line, fill=text_color, font=font)
            y += (h // 3)
        return img

    def _preload_all(self):
        # Lazy-import ImageFont inside the same try block
        try:
            from PIL import ImageFont
            self._ImageFont = ImageFont
        except ImportError:
            self._ImageFont = None

        for tid in range(N_TILES):
            pil_img = None
            folder = self._crops_dir / str(tid)
            files  = sorted(folder.iterdir()) if folder.is_dir() else []
            if files:
                mid = files[len(files) // 2]
                try:
                    pil_img = self._Image.open(mid).convert("RGB")
                except Exception:
                    pass
            if pil_img is None:
                pil_img = self._make_fallback(tid, self.W, self.H)

            # Full size
            full = pil_img.resize((self.W, self.H), self._Image.LANCZOS)
            self._photos[tid] = self._ImageTk.PhotoImage(full)

            # Chip size
            chip = pil_img.resize((self.CW, self.CH), self._Image.LANCZOS)
            self._chip_photos[tid] = self._ImageTk.PhotoImage(chip)

    def get(self, tid: int):
        return self._photos.get(tid)

    def get_chip(self, tid: int):
        return self._chip_photos.get(tid)


# ─────────────────────────────────────────────────────────────────────────────
# RegionSelector  — full-screen drag-to-select game viewport
# ─────────────────────────────────────────────────────────────────────────────

class RegionSelector(tk.Toplevel):
    """
    Semi-transparent full-screen overlay.
    User drags to mark the game viewport; releases to confirm.
    Calls on_select(x, y, w, h) with absolute screen coordinates.
    Press Esc to cancel.
    """
    MIN_SIZE = 80   # minimum px in each axis to accept selection

    def __init__(self, root: tk.Tk,
                 on_select: Callable[[int, int, int, int], None],
                 on_cancel: Callable):
        super().__init__(root)
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._sx = self._sy = 0
        self._rect_id   = None
        self._size_id   = None

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
                             text="拖曳框選遊戲畫面區域",
                             fill="white", font=("Consolas", 18, "bold"))
        self._cv.create_text(cx, cy - 30,
                             text="Drag to select the game viewport",
                             fill="#aaaaaa", font=("Consolas", 12))
        self._cv.create_text(cx, cy,
                             text="[Esc] 取消 / Cancel",
                             fill=MUTED, font=("Consolas", 10))

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
# TilePopup  — micro correction popup on tile click
# ─────────────────────────────────────────────────────────────────────────────

class TilePopup(tk.Toplevel):
    """
    Small floating window that appears when user clicks a tile.
    Shows current tile name + combobox to correct it.
    Auto-dismisses on focus loss.
    """
    def __init__(self, parent_root, ct: ConfirmedTile,
                 on_apply: Callable[[int, str], None]):
        super().__init__(parent_root)
        self._ct       = ct
        self._on_apply = on_apply

        self.overrideredirect(True)   # no title bar
        self.attributes("-topmost", True)
        self.configure(bg=CARD)

        # Border frame
        border = tk.Frame(self, bg=WARN if ct.low_conf else BORDER, padx=1, pady=1)
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
        w = self.winfo_reqwidth()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        # Clamp to screen
        px = min(x, sw - w - 4)
        py = min(y - 10, sh - self.winfo_reqheight() - 4)
        self.geometry(f"+{max(0,px)}+{max(0,py)}")
        self.focus_set()


# ─────────────────────────────────────────────────────────────────────────────
# HandPanel
# ─────────────────────────────────────────────────────────────────────────────

class HandPanel(tk.Frame):
    """
    Horizontal strip of detected hand tiles.
    - Skeleton placeholders while loading
    - Click any tile → TilePopup for correction
    - Best discard: green border; low-conf: orange border
    """
    SKEL_COUNT = 13

    def __init__(self, parent, cache: TileImageCache,
                 on_correction: Callable, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._cache         = cache
        self._on_correction = on_correction
        self._skel_job      = None
        self._skel_phase    = 0
        self._skel_cells    = []

        # Header row
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=6, pady=(5, 2))
        self._hand_lbl = tk.Label(hdr, text="HAND", bg=PANEL, fg=MUTED,
                                  font=("Consolas", 8, "bold"))
        self._hand_lbl.pack(side="left")
        self._shan_lbl = tk.Label(hdr, text="", bg=PANEL, fg=TEXT,
                                  font=("Consolas", 10, "bold"))
        self._shan_lbl.pack(side="right")

        # Tile strip
        self._strip = tk.Frame(self, bg=PANEL)
        self._strip.pack(fill="x", padx=4, pady=(0, 5))

        self._show_skeleton()

    # ── Skeleton ──────────────────────────────────────────────────────────────

    def _show_skeleton(self):
        self._clear_strip()
        self._hand_lbl.config(text="HAND")
        self._shan_lbl.config(text="")
        self._skel_cells = []
        for _ in range(self.SKEL_COUNT):
            f = tk.Frame(self._strip, bg=SKEL, width=30, height=44)
            f.pack_propagate(False)
            f.pack(side="left", padx=1)
            self._skel_cells.append(f)
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

    # ── Update with real data ─────────────────────────────────────────────────

    def update(self, hand_tiles: List, s: int, best_tid: int = -1):
        self._clear_strip()
        sorted_tiles = sorted(hand_tiles, key=lambda t: t.cx)
        count = len(sorted_tiles)
        self._hand_lbl.config(text=f"HAND  ({count})", fg=MUTED)

        if s <= -1 or s == 0:
            self._shan_lbl.config(
                text="TENPAI" if s == 0 else "TENPAI!",
                fg=SAFE, font=("Consolas", 10, "bold"))
        elif s == 1:
            self._shan_lbl.config(text=f"向聴 {s}", fg=WARN,
                                  font=("Consolas", 10, "bold"))
        else:
            self._shan_lbl.config(text=f"向聴 {s}", fg=TEXT,
                                  font=("Consolas", 10))

        for ct in sorted_tiles:
            tid     = ct.tile_id
            name    = tile_name(tid)
            is_best = (tid == best_tid)
            is_warn = ct.low_conf

            border = BEST if is_best else (WARN if is_warn else BORDER)
            cell   = tk.Frame(self._strip, bg=border, padx=1, pady=1,
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

            # Click → correction popup
            for widget in (cell, inner, lbl):
                widget.bind("<Button-1>",
                            self._make_click_handler(ct))

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
# EVPanel   — #1 large card + #2/#3 compact side by side
# ─────────────────────────────────────────────────────────────────────────────

class EVPanel(tk.Frame):
    """
    Top recommendation displayed as a large card.
    Ranks #2 and #3 as compact side-by-side cards below.
    Effective tiles shown as small image chips.
    Skeleton state while computing.
    """

    def __init__(self, parent, cache: TileImageCache, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._cache    = cache
        self._skel_job = None
        self._skel_cells = []

        # ── #1 large card ─────────────────────────────────────────────────────
        self._card1 = tk.Frame(self, bg=CARD, padx=8, pady=6)
        self._card1.pack(fill="x", padx=6, pady=(5, 2))

        card1_top = tk.Frame(self._card1, bg=CARD)
        card1_top.pack(fill="x")
        self._r1_rank = tk.Label(card1_top, text="#1", bg=CARD, fg=BEST,
                                 font=("Consolas", 9, "bold"), width=3)
        self._r1_rank.pack(side="left")
        self._r1_name = tk.Label(card1_top, text="—", bg=CARD, fg=TEXT,
                                 font=("Consolas", 14, "bold"), width=5)
        self._r1_name.pack(side="left")
        self._r1_ev   = tk.Label(card1_top, text="", bg=CARD, fg=SAFE,
                                 font=("Consolas", 10))
        self._r1_ev.pack(side="left", padx=6)
        self._r1_win  = tk.Label(card1_top, text="", bg=CARD, fg=TEXT,
                                 font=("Consolas", 9))
        self._r1_win.pack(side="left")
        self._r1_shan = tk.Label(card1_top, text="", bg=CARD, fg=MUTED,
                                 font=("Consolas", 8))
        self._r1_shan.pack(side="right")

        # Effective tiles chip strip
        self._eff_frame = tk.Frame(self._card1, bg=CARD)
        self._eff_frame.pack(fill="x", pady=(3, 0))
        self._eff_count_lbl = tk.Label(self._eff_frame, text="", bg=CARD,
                                       fg=MUTED, font=("Consolas", 7))
        self._eff_count_lbl.pack(side="right")

        # Yaku
        self._r1_yaku = tk.Label(self._card1, text="", bg=CARD, fg=MUTED,
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

        # ── Status + progress ─────────────────────────────────────────────────
        status_row = tk.Frame(self, bg=PANEL)
        status_row.pack(fill="x", padx=6, pady=(2, 4))
        self._status_var = tk.StringVar(value="Press F9 to analyze")
        tk.Label(status_row, textvariable=self._status_var,
                 bg=PANEL, fg=MUTED, font=("Consolas", 7)).pack(side="left")
        self._progress = ttk.Progressbar(status_row,
                                         style="EV.Horizontal.TProgressbar",
                                         mode="determinate", length=160)
        self._progress.pack(side="right")

    def _make_compact_card(self, parent, rank_text: str, rank_fg: str) -> tk.Frame:
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
        widgets["win"] = tk.Label(bot, text="", bg=CARD, fg=MUTED,
                                  font=("Consolas", 8))
        widgets["win"].pack(side="left", padx=4)
        widgets["eff"] = tk.Label(bot, text="", bg=CARD, fg=MUTED,
                                  font=("Consolas", 8))
        widgets["eff"].pack(side="right")
        f._widgets = widgets
        return f

    # ── Skeleton ──────────────────────────────────────────────────────────────

    def set_computing(self):
        self._stop_skel_shimmer()
        # Gray out card1
        self._r1_name.config(text="…", fg=SKEL2)
        self._r1_ev.config(text="")
        self._r1_win.config(text="")
        self._r1_shan.config(text="")
        self._r1_yaku.config(text="")
        self._eff_count_lbl.config(text="")
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()
        for card in (self._card2, self._card3):
            card._widgets["name"].config(text="…", fg=SKEL2)
            card._widgets["ev"].config(text="")
            card._widgets["win"].config(text="")
            card._widgets["eff"].config(text="")
        self._status_var.set("Computing…")
        self._progress.config(mode="indeterminate")
        self._progress.start(8)
        # Start shimmer on card1 background
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

    # ── Update with real data ─────────────────────────────────────────────────

    def update_phase1(self, effs: List[Tuple[int, int]], s: int):
        """Show effective tiles immediately after detection (before EV)."""
        self._update_eff_chips(effs)

    def update(self, ev_results: List[Dict], is_mc: bool, dt: float):
        self._stop_skel_shimmer()
        self._progress.stop()
        self._progress.config(mode="determinate", value=100)
        mode = "MC" if is_mc else "Analytical"
        self._status_var.set(f"{mode}  {dt*1000:.0f}ms")

        def _fill_r1(r: Dict):
            self._r1_name.config(text=r["discard_name"], fg=BEST)
            self._r1_ev.config(text=f"EV {r['ev']:.0f}")
            self._r1_win.config(text=f"Win {r['win_rate']*100:.1f}%")
            self._r1_shan.config(text=f"向聴{r['shanten']}")
            yaku_str = "  ".join(f"{n}({h})" for n, h in r.get("yaku", [])
                                 if isinstance(h, int) and h > 0)
            score_str = f"≈{r['est_score']}  {r['han']}han {r['fu']}fu"
            self._r1_yaku.config(text=f"{score_str}  {yaku_str}")
            self._update_eff_chips(r.get("eff_tiles", []))

        def _fill_compact(card: tk.Frame, r: Dict):
            card._widgets["name"].config(text=r["discard_name"], fg=TEXT)
            card._widgets["ev"].config(text=f"EV{r['ev']:.0f}", fg=SAFE_DIM)
            card._widgets["win"].config(
                text=f"W{r['win_rate']*100:.0f}%", fg=MUTED)
            card._widgets["eff"].config(
                text=f"Eff{r['eff_count']}", fg=MUTED)

        def _clear_compact(card: tk.Frame):
            for k in ("name", "ev", "win", "eff"):
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
        """Render effective tiles as small image chips in card1."""
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()

        total = sum(c for _, c in effs)
        self._eff_count_lbl.config(
            text=f"{total} tiles" if total else "")

        for tid, cnt in effs[:10]:
            chip_photo = self._cache.get_chip(tid)
            if chip_photo:
                lbl = tk.Label(self._eff_frame, image=chip_photo, bg=CARD,
                               cursor="hand2")
                lbl.image = chip_photo
            else:
                name = tile_name(tid)
                lbl = tk.Label(self._eff_frame, text=name, bg=CARD,
                               fg=_suit_fg(name), font=("Consolas", 6),
                               width=3)
            lbl.pack(side="left", padx=1)
            # count badge
            tk.Label(self._eff_frame, text=f"×{cnt}", bg=CARD, fg=MUTED,
                     font=("Consolas", 6)).pack(side="left")

    def clear(self):
        self._stop_skel_shimmer()
        self._progress.stop()
        self._progress.config(mode="determinate", value=0)
        self._status_var.set("—")
        self._r1_name.config(text="—", fg=MUTED)
        self._r1_ev.config(text="")
        self._r1_win.config(text="")
        self._r1_shan.config(text="")
        self._r1_yaku.config(text="")
        for w in self._eff_frame.winfo_children():
            if w is not self._eff_count_lbl:
                w.destroy()
        self._eff_count_lbl.config(text="")
        for card in (self._card2, self._card3):
            for k in ("name", "ev", "win", "eff"):
                card._widgets[k].config(text="—", fg=MUTED)


# ─────────────────────────────────────────────────────────────────────────────
# DoraPanel  — compact dora input + reset (replaces bulky WarningPanel)
# ─────────────────────────────────────────────────────────────────────────────

class DoraPanel(tk.Frame):
    def __init__(self, parent, on_dora: Callable, on_reset: Callable, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        row = tk.Frame(self, bg=PANEL)
        row.pack(fill="x", padx=6, pady=3)
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


# ─────────────────────────────────────────────────────────────────────────────
# ControlPanel  — big button + auto; ⚙ drawer for advanced settings
# ─────────────────────────────────────────────────────────────────────────────

class ControlPanel(tk.Frame):
    def __init__(self, parent, on_analyze: Callable,
                 on_auto_toggle: Callable, on_save_frame: Callable,
                 on_select_region: Callable, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.auto_var     = tk.BooleanVar(value=False)
        # Advanced (drawer) vars
        self.interval_var = tk.IntVar(value=5)
        self.sims_var     = tk.IntVar(value=0)
        self.crop_x_var   = tk.IntVar(value=0)
        self.crop_y_var   = tk.IntVar(value=0)

        # ── Main row ──────────────────────────────────────────────────────────
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

        # Status dot
        self._dot = tk.Label(main_row, text="●", bg=PANEL, fg=MUTED,
                             font=("Consolas", 16))
        self._dot.pack(side="right")
        self._status_lbl = tk.Label(main_row, text="Ready", bg=PANEL,
                                    fg=MUTED, font=("Consolas", 8))
        self._status_lbl.pack(side="right", padx=4)

        # Gear button toggles drawer
        self._drawer_open = False
        gear_btn = tk.Button(main_row, text="⚙", bg=PANEL, fg=MUTED,
                             font=("Consolas", 10), relief="flat",
                             command=self._toggle_drawer)
        gear_btn.pack(side="right", padx=4)

        # ── Drawer (hidden by default) ─────────────────────────────────────
        self._drawer = tk.Frame(self, bg=PANEL)
        # NOT packed yet

        d = self._drawer
        tk.Frame(d, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)

        r1 = tk.Frame(d, bg=PANEL)
        r1.pack(fill="x", padx=6, pady=2)
        tk.Button(r1, text="⊞ Select Region", bg=ACCENT, fg=SAFE,
                  font=("Consolas", 9, "bold"), relief="flat", padx=6, pady=3,
                  command=on_select_region).pack(side="left")
        tk.Button(r1, text="Save Frame", bg="#152a1a", fg=SAFE,
                  font=("Consolas", 8), relief="flat",
                  command=on_save_frame).pack(side="left", padx=(8, 0))

        r1b = tk.Frame(d, bg=PANEL)
        r1b.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(r1b, text="X:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")
        tk.Entry(r1b, textvariable=self.crop_x_var, width=5,
                 bg=CARD, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 8), relief="flat").pack(side="left", padx=2)
        tk.Label(r1b, text="Y:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left", padx=(6, 0))
        tk.Entry(r1b, textvariable=self.crop_y_var, width=5,
                 bg=CARD, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 8), relief="flat").pack(side="left", padx=2)
        self._region_lbl = tk.Label(r1b, text="(not set)", bg=PANEL, fg=MUTED,
                                    font=("Consolas", 7))
        self._region_lbl.pack(side="left", padx=6)

        r2 = tk.Frame(d, bg=PANEL)
        r2.pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(r2, text="MC sims:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")
        ttk.Combobox(r2, textvariable=self.sims_var,
                     values=[0, 200, 500, 2000], width=5,
                     state="readonly").pack(side="left", padx=4)
        tk.Label(r2, text="Interval:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0))
        ttk.Combobox(r2, textvariable=self.interval_var,
                     values=[3, 5, 10, 15, 30], width=3,
                     state="readonly").pack(side="left", padx=2)
        tk.Label(r2, text="s", bg=PANEL, fg=MUTED,
                 font=("Consolas", 8)).pack(side="left")

    def _toggle_drawer(self):
        if self._drawer_open:
            self._drawer.pack_forget()
        else:
            self._drawer.pack(fill="x")
        self._drawer_open = not self._drawer_open

    # ── Pulse animation on trigger ────────────────────────────────────────────

    def pulse(self, step: int = 0):
        """Subtle expand-then-contract animation on Analyze button."""
        FRAMES = [
            (SAFE_DIM, 12), (SAFE_DIM, 12), (ACCENT, 11), (ACCENT, 11)
        ]
        if step < len(FRAMES):
            bg, sz = FRAMES[step]
            try:
                self._analyze_btn.config(
                    bg=bg, font=("Consolas", sz, "bold"))
                self.after(70, lambda: self.pulse(step + 1))
            except tk.TclError:
                pass
        else:
            try:
                self._analyze_btn.config(
                    bg=ACCENT, font=("Consolas", 11, "bold"))
            except tk.TclError:
                pass

    def set_status(self, text: str, color: str = MUTED):
        self._status_lbl.config(text=text, fg=color)
        self._dot.config(fg=color)

    def get_sims(self) -> int:
        return self.sims_var.get()

    def get_interval(self) -> int:
        return self.interval_var.get()

    def get_crop(self) -> Tuple[int, int]:
        try:
            return (self.crop_x_var.get(), self.crop_y_var.get())
        except Exception:
            return (0, 0)

    def set_region_label(self, text: str):
        try:
            self._region_lbl.config(text=text)
        except tk.TclError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MainWindow  — View orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow:
    MODEL_PATH = PROJECT_ROOT / "runs/detect/majsoul_confused_boost/weights/best.pt"
    CROPS_DIR  = PROJECT_ROOT / "tile_crops"
    DEBUG_DIR  = PROJECT_ROOT / "runs" / "advisor"

    def __init__(self, model_path: Optional[Path] = None):
        if model_path:
            self.MODEL_PATH = model_path

        self.root = tk.Tk()
        self._setup_window()
        self._setup_style()

        self._cache   = TileImageCache(self.CROPS_DIR)
        self._last_p1: Optional[Phase1Result] = None

        self._build_panels()

        # Controller wired after panels (callbacks reference panels)
        self._advisor = AdvisorController(
            root       = self.root,
            model_path = self.MODEL_PATH,
            on_phase1  = self._handle_phase1,
            on_phase2  = self._handle_phase2,
        )
        self._advisor.start()

        self.root.after(2000, self._update_cap_status)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Window / style setup ──────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("Majsoul Advisor")
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
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=4, pady=4)

        self._cap_lbl = tk.Label(main, text="Initializing…", bg=BG, fg=MUTED,
                                 font=("Consolas", 7), anchor="w")
        self._cap_lbl.pack(fill="x", pady=(0, 2))

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        self._hand = HandPanel(main, self._cache, self._on_correction)
        self._hand.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._ev = EVPanel(main, self._cache)
        self._ev.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._dora = DoraPanel(main,
                               on_dora=self._on_add_dora,
                               on_reset=self._on_reset_state)
        self._dora.pack(fill="x")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", pady=2)

        self._ctrl = ControlPanel(main,
                                  on_analyze=self._trigger_analysis,
                                  on_auto_toggle=self._toggle_auto,
                                  on_save_frame=self._save_debug_frame,
                                  on_select_region=self._on_select_region)
        self._ctrl.pack(fill="x")

    # ── User actions → controller ─────────────────────────────────────────────

    def _trigger_analysis(self):
        ox, oy = self._ctrl.get_crop()
        n_sims = self._ctrl.get_sims()
        if self._advisor.trigger(crop=(ox, oy), n_sims=n_sims):
            self._ctrl.set_status("Detecting…", SAFE)
            self._ctrl.pulse()
            self._hand.set_skeleton()
            self._ev.set_computing()

    def _toggle_auto(self):
        enabled = self._ctrl.auto_var.get()
        ox, oy  = self._ctrl.get_crop()
        ms      = self._ctrl.get_interval() * 1000
        n_sims  = self._ctrl.get_sims()
        self._advisor.set_auto(enabled, interval_ms=ms,
                               crop=(ox, oy), n_sims=n_sims)
        self._ctrl.set_status("Auto ON" if enabled else "Auto OFF",
                              SAFE if enabled else MUTED)

    def _on_correction(self, cx: int, tile_name_str: str):
        if self._advisor.apply_correction(cx, tile_name_str):
            self._trigger_analysis()

    def _on_add_dora(self, tile_name_str: str):
        self._advisor.add_dora(tile_name_str)

    def _on_reset_state(self):
        self._advisor.reset_state()
        self._hand.clear()
        self._ev.clear()
        self._ctrl.set_status("Reset", SAFE)

    # ── Controller callbacks → panels ─────────────────────────────────────────

    def _handle_phase1(self, r1: Phase1Result):
        self._last_p1 = r1
        if not r1.capture_ok:
            self._hand.clear()
            self._ev.clear()
            self._ctrl.set_status(f"Err: {r1.error_msg[:28]}", DANGER)
        else:
            self._hand.update(r1.hand_tiles, r1.shanten)
            self._ev.update_phase1(r1.effective_tiles, r1.shanten)
            self._ctrl.set_status(f"EV… [{r1.game_mode}]", WARN)

    def _handle_phase2(self, r2: Phase2Result):
        self._ev.update(r2.ev_results, r2.is_mc, r2.compute_time)
        if r2.ev_results and self._last_p1 and self._last_p1.capture_ok:
            best_tid = r2.ev_results[0]["discard_tid"]
            self._hand.update(self._last_p1.hand_tiles,
                              self._last_p1.shanten, best_tid)
        mode = self._last_p1.game_mode if self._last_p1 else "?"
        self._ctrl.set_status(f"Done [{mode}]", SAFE)

    # ── Capture status + debug frame ──────────────────────────────────────────

    def _update_cap_status(self):
        self._cap_lbl.config(text=f"Capture: {self._advisor.get_capture_status()}")
        self.root.after(5000, self._update_cap_status)

    def _save_debug_frame(self):
        ox, oy = self._ctrl.get_crop()
        if self._advisor.has_last_img:
            # Use cached frame from last analysis — no hide needed
            self._finish_debug_save(ox, oy)
        else:
            # No analysis yet: hide advisor so it doesn't appear in screenshot
            self.root.withdraw()
            self.root.update()   # flush geometry event before mss grab
            self.root.after(150, lambda: self._finish_debug_save(ox, oy))

    def _finish_debug_save(self, ox: int, oy: int):
        try:
            path = self._advisor.save_debug_frame(self.DEBUG_DIR, crop=(ox, oy))
            self._ctrl.set_status(f"Saved {path.name}", SAFE)
        except Exception as e:
            self._ctrl.set_status(f"Cap err: {e}", DANGER)
        finally:
            self.root.deiconify()

    # ── Region selector ───────────────────────────────────────────────────────

    def _on_select_region(self):
        self.root.withdraw()

        def on_select(x: int, y: int, w: int, h: int):
            self.root.deiconify()
            self._ctrl.crop_x_var.set(x)
            self._ctrl.crop_y_var.set(y)
            self._ctrl.set_region_label(f"{w}×{h} px")
            self._ctrl.set_status(f"Region ({x},{y})", SAFE)
            if not self._ctrl._drawer_open:
                self._ctrl._toggle_drawer()

        def on_cancel():
            self.root.deiconify()

        RegionSelector(self.root, on_select, on_cancel)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
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
    app = MainWindow(model_path=Path(args.model) if args.model else None)
    app.run()
