"""
screen_capture.py
Capture the Mahjong Soul game window and return a numpy BGR array at 1418×837.

Priority chain:
  1. mss  — DXGI Desktop Duplication, captures WebGL/hardware-accelerated correctly (~15ms)
  2. win32 BitBlt — may return black for WebGL canvas, but available without extra deps
  3. PIL ImageGrab — slowest, same WebGL limitation as win32, always available
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import cv2

TARGET_W, TARGET_H = 1418, 837

# Substrings to match in window title (case-insensitive)
WINDOW_TITLES = ["mahjong soul", "雀魂", "majsoul"]


class CaptureFailed(Exception):
    pass


class ScreenCapture:
    """Captures the game window and returns BGR uint8 array at TARGET_W×TARGET_H."""

    def __init__(self, prefer_window: bool = True):
        self._prefer_window = prefer_window
        self._hwnd: Optional[int] = None
        self._sct = None          # mss.mss() instance
        self._mss_ok = False
        self._mode = "pil"        # last-used mode: "mss" | "win32" | "pil"
        self._init_mss()
        # Set initial mode to reflect what will actually be used
        if self._mss_ok:
            self._mode = "mss"

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_mss(self):
        try:
            import mss as _mss
            self._sct = _mss.mss()
            self._mss_ok = True
        except ImportError:
            self._mss_ok = False

    # ── Window discovery ──────────────────────────────────────────────────────

    def _find_game_window(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Search for a visible window whose title contains one of WINDOW_TITLES.
        Returns (x, y, w, h) of the window bounding rect, or None.
        Also updates self._hwnd for win32 capture.
        """
        try:
            import win32gui
            found: list = []

            def _enum(hwnd, _):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd).lower()
                if any(t in title for t in WINDOW_TITLES):
                    x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)
                    w, h = x2 - x1, y2 - y1
                    if w > 100 and h > 100:
                        found.append((hwnd, x1, y1, w, h))

            win32gui.EnumWindows(_enum, None)
            if found:
                hwnd, x, y, w, h = found[0]
                self._hwnd = hwnd
                return (x, y, w, h)
        except Exception:
            pass
        return None

    # ── Main capture entry point ──────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        """
        Capture and return BGR uint8 array resized to TARGET_W×TARGET_H.
        Tries mss first, then win32, then PIL.
        """
        rect = self._find_game_window() if self._prefer_window else None

        if self._mss_ok:
            return self._capture_mss(rect)

        if rect is not None:
            try:
                return self._capture_win32(rect)
            except Exception:
                pass

        return self._capture_pil(rect)

    # ── mss path ──────────────────────────────────────────────────────────────

    def _capture_mss(self, rect: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if rect is not None:
            x, y, w, h = rect
            monitor = {"top": y, "left": x, "width": w, "height": h}
        else:
            monitor = self._sct.monitors[1]   # primary monitor

        bgra = np.array(self._sct.grab(monitor), dtype=np.uint8)
        img = bgra[:, :, :3]   # BGRA → BGR (drop alpha)
        if img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
            img = cv2.resize(img, (TARGET_W, TARGET_H))
        self._mode = "mss"
        return img

    # ── win32 BitBlt path ─────────────────────────────────────────────────────

    def _capture_win32(self, rect: Tuple[int, int, int, int]) -> np.ndarray:
        import win32gui, win32ui, win32con
        x, y, w, h = rect
        hwnd = self._hwnd if self._hwnd else win32gui.GetDesktopWindow()

        hwnd_dc  = win32gui.GetWindowDC(hwnd)
        mfc_dc   = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc  = mfc_dc.CreateCompatibleDC()
        bmp      = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        save_dc.BitBlt((0, 0), (w, h), mfc_dc, (0, 0), win32con.SRCCOPY)

        bmp_info = bmp.GetInfo()
        bmp_str  = bmp.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype=np.uint8)
        img = img.reshape((bmp_info["bmHeight"], bmp_info["bmWidth"], 4))
        img = img[:, :, :3].copy()   # BGRA → BGR

        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        bmp.DeleteObject()

        if img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
            img = cv2.resize(img, (TARGET_W, TARGET_H))
        self._mode = "win32"
        return img

    # ── PIL fallback ──────────────────────────────────────────────────────────

    def _capture_pil(self, rect: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        from PIL import ImageGrab
        if rect is not None:
            x, y, w, h = rect
            pil_img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            pil_img = ImageGrab.grab()
        img = np.array(pil_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
            img = cv2.resize(img, (TARGET_W, TARGET_H))
        self._mode = "pil"
        return img

    def capture_fullscreen(self) -> np.ndarray:
        """
        Always capture the full primary monitor (for absolute crop offsets).
        Used when the user has set a manual crop region via region-selector.
        """
        if self._mss_ok:
            return self._capture_mss(None)   # None → monitors[1] = primary
        return self._capture_pil(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def is_valid_frame(img: np.ndarray) -> bool:
        """Returns False if the frame is all-black (WebGL capture may fail with BitBlt)."""
        return float(img.mean()) > 5.0

    def get_status(self) -> str:
        """Human-readable capture mode string for the status bar."""
        window_str = "window" if self._hwnd else "fullscreen"
        return f"{self._mode} | {window_str}"
