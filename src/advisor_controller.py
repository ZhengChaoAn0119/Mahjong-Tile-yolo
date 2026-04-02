"""
advisor_controller.py
MVC Controller layer for the Majsoul advisor.

Owns: AnalysisWorker, HotkeyManager, ScreenCapture, result queues, auto-timer.
Exposes a clean API to MainWindow (View); communicates results via injected callbacks
called on the main thread through root.after() queue polling.

No tkinter import — root is typed as object and used only for root.after().
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ev_engine      import compute_simple_ev
from .frame_smoother import ConfirmedTile
from .game_state     import GameState
from .mahjong_advisor import MahjongAdvisor, run_detection, ROI_4P
from .mahjong_engine  import effective_tiles, shanten
from .screen_capture  import ScreenCapture
from .tile_codec      import name_to_tile

PROJECT_ROOT = Path("E:/project/majsoul_yolo")


# ─────────────────────────────────────────────────────────────────────────────
# Inter-thread result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Phase1Result:
    hand_tiles:      List[ConfirmedTile]
    shanten:         int
    effective_tiles: List[Tuple[int, int]]
    warnings:        List[ConfirmedTile]
    game_mode:       str
    timestamp:       float
    capture_ok:      bool
    error_msg:       str = ""


@dataclass
class Phase2Result:
    ev_results:   List[Dict]
    is_mc:        bool
    compute_time: float


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisWorker  — background detection + EV thread
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisWorker(threading.Thread):
    def __init__(self, trigger_q: queue.Queue,
                 phase1_q: queue.Queue, phase2_q: queue.Queue,
                 capture: ScreenCapture, model_path: Path):
        super().__init__(daemon=True, name="AnalysisWorker")
        self._trigger_q  = trigger_q
        self._phase1_q   = phase1_q
        self._phase2_q   = phase2_q
        self._capture    = capture
        self._model_path = model_path
        self._advisor: Optional[MahjongAdvisor] = None
        self._last_img:  Optional[np.ndarray]   = None

    @property
    def last_img(self) -> Optional[np.ndarray]:
        return self._last_img

    @property
    def state(self) -> Optional[GameState]:
        return self._advisor.state if self._advisor else None

    def reset_state(self):
        if self._advisor:
            self._advisor.state   = GameState(seat_wind=0, round_wind=0)
            self._advisor.smoother.reset()

    def run(self):
        self._advisor = MahjongAdvisor(model_path=self._model_path)
        while True:
            signal = self._trigger_q.get()
            if signal == "STOP":
                break
            if isinstance(signal, tuple) and signal[0] == "ANALYZE":
                self._do_analysis(signal[1], signal[2], signal[3])

    def _do_analysis(self, ox: int, oy: int, n_sims: int):
        try:
            # Absolute-coord crop → always capture full primary monitor.
            if ox > 0 or oy > 0:
                raw = self._capture.capture_fullscreen()
            else:
                raw = self._capture.capture()

            if not ScreenCapture.is_valid_frame(raw):
                self._phase1_q.put(Phase1Result(
                    hand_tiles=[], shanten=99, effective_tiles=[],
                    warnings=[], game_mode="4p",
                    timestamp=time.time(), capture_ok=False,
                    error_msg="Black frame — open ⚙ and click Save Frame to debug"))
                return

            h_raw, w_raw = raw.shape[:2]
            if ox > 0 or oy > 0:
                img = raw[oy : min(oy + 837, h_raw), ox : min(ox + 1418, w_raw)]
            else:
                img = raw

            self._last_img = img.copy()

            if img.shape[:2] != (837, 1418):
                img = cv2.resize(img, (1418, 837))

            dets, mode = run_detection(self._advisor.model, img)
            smoother   = self._advisor.smoother
            for _ in range(smoother.min_hits):
                smoother.update(dets)

            hand_tiles   = smoother.hand_tiles_sorted()
            center_tiles = smoother.confirmed_tiles("center")
            meld_tiles   = (smoother.confirmed_tiles("meld_left") +
                            smoother.confirmed_tiles("meld_right") +
                            smoother.confirmed_tiles("meld_top"))
            warnings     = smoother.low_conf_warnings()
            smoother.reset()

            state = self._advisor.state
            state.update_from_detection(hand_tiles, meld_tiles, center_tiles)

            hand34    = state.hand34()
            remaining = state.remaining_tiles()
            s         = int(shanten(hand34))
            effs      = effective_tiles(hand34, remaining)

            self._phase1_q.put(Phase1Result(
                hand_tiles=hand_tiles, shanten=s,
                effective_tiles=effs, warnings=warnings,
                game_mode=mode, timestamp=time.time(), capture_ok=True))

            if len(state._hand) >= 13:
                doras = state.dora_tiles()
                t0 = time.time()
                ev_results = compute_simple_ev(
                    hand34, remaining, state.melds,
                    state.seat_wind, state.round_wind, doras)
                dt = time.time() - t0
            else:
                ev_results, dt = [], 0.0

            self._phase2_q.put(Phase2Result(
                ev_results=ev_results, is_mc=False, compute_time=dt))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[AnalysisWorker] ERROR:\n{tb}")
            self._phase1_q.put(Phase1Result(
                hand_tiles=[], shanten=99, effective_tiles=[],
                warnings=[], game_mode="4p",
                timestamp=time.time(), capture_ok=False,
                error_msg=f"{type(e).__name__}: {e}"))


# ─────────────────────────────────────────────────────────────────────────────
# HotkeyManager  — Win32 F9 global hotkey in a dedicated message-pump thread
# ─────────────────────────────────────────────────────────────────────────────

class HotkeyManager:
    HOTKEY_ID = 1
    VK_F9     = 0x78
    WM_HOTKEY = 0x0312
    PM_REMOVE = 0x0001

    def __init__(self, root, callback: Callable):
        # root typed as object — used only for root.after(); no tkinter import needed
        self._root     = root
        self._callback = callback
        self._stop     = threading.Event()

    def start(self):
        threading.Thread(target=self._pump, daemon=True,
                         name="HotkeyPump").start()

    def _pump(self):
        user32       = ctypes.windll.user32
        MOD_NOREPEAT = 0x4000
        hwnd = user32.CreateWindowExW(
            0, "STATIC", "", 0, 0, 0, 0, 0, None, None, None, None)
        user32.RegisterHotKey(hwnd, self.HOTKEY_ID, MOD_NOREPEAT, self.VK_F9)
        msg = ctypes.wintypes.MSG()
        while not self._stop.is_set():
            if user32.PeekMessageW(ctypes.byref(msg), hwnd, 0, 0, self.PM_REMOVE):
                if msg.message == self.WM_HOTKEY:
                    self._root.after(0, self._callback)
            else:
                self._stop.wait(timeout=0.05)
        user32.UnregisterHotKey(hwnd, self.HOTKEY_ID)

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# AdvisorController  — public facade for MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class AdvisorController:
    """
    Owns threading infrastructure and game-state access.
    View (MainWindow) calls public methods; results arrive via on_phase1/on_phase2
    callbacks, which are always invoked on the main thread through root.after() polling.
    """

    def __init__(self, root,
                 model_path: Path,
                 on_phase1: Callable[[Phase1Result], None],
                 on_phase2: Callable[[Phase2Result], None],
                 on_hotkey: Optional[Callable] = None,
                 poll_ms: int = 100):
        self._root      = root
        self._on_phase1 = on_phase1
        self._on_phase2 = on_phase2
        self._poll_ms   = poll_ms

        self._capture   = ScreenCapture()
        self._phase1_q: queue.Queue = queue.Queue()
        self._phase2_q: queue.Queue = queue.Queue()
        self._trigger_q: queue.Queue = queue.Queue(maxsize=1)

        self._auto_after_id  = None
        self._auto_interval  = 5000
        self._auto_crop:  Tuple[int, int] = (0, 0)
        self._auto_n_sims: int = 0

        self._worker = AnalysisWorker(
            self._trigger_q, self._phase1_q, self._phase2_q,
            self._capture, model_path)
        # on_hotkey lets the View inject its own trigger (with UI feedback).
        # Falls back to bare controller.trigger if not provided.
        self._hotkey = HotkeyManager(root, on_hotkey if on_hotkey else self.trigger)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start worker thread, hotkey pump, and result polling."""
        self._worker.start()
        self._hotkey.start()
        self._root.after(self._poll_ms, self._poll)

    def stop(self) -> None:
        """Clean shutdown."""
        self.set_auto(False)
        self._hotkey.stop()
        try:
            self._trigger_q.put_nowait("STOP")
        except queue.Full:
            pass

    # ── Analysis ──────────────────────────────────────────────────────────────

    def trigger(self, crop: Tuple[int, int] = (0, 0), n_sims: int = 0) -> bool:
        """
        Enqueue an analysis request.
        Returns True if accepted, False if worker is still busy with previous request.
        """
        ox, oy = crop
        try:
            self._trigger_q.put_nowait(("ANALYZE", ox, oy, n_sims))
            return True
        except queue.Full:
            return False

    # ── Auto mode ─────────────────────────────────────────────────────────────

    def set_auto(self, enabled: bool, interval_ms: int = 5000,
                 crop: Tuple[int, int] = (0, 0), n_sims: int = 0) -> None:
        if self._auto_after_id:
            self._root.after_cancel(self._auto_after_id)
            self._auto_after_id = None
        if enabled:
            self._auto_interval = interval_ms
            self._auto_crop     = crop
            self._auto_n_sims   = n_sims
            self._schedule_auto()

    def _schedule_auto(self) -> None:
        self._auto_after_id = self._root.after(
            self._auto_interval, self._auto_fire)

    def _auto_fire(self) -> None:
        self.trigger(self._auto_crop, self._auto_n_sims)
        self._schedule_auto()

    # ── Game state ────────────────────────────────────────────────────────────

    def apply_correction(self, cx: int, tile_name_str: str) -> bool:
        state = self._worker.state
        return state.apply_manual_correction(cx, tile_name_str) if state else False

    def add_dora(self, tile_name_str: str) -> None:
        state = self._worker.state
        if state is None:
            return
        try:
            state.dora_indicators.append(name_to_tile(tile_name_str))
        except KeyError:
            pass

    def reset_state(self) -> None:
        self._worker.reset_state()

    def remove_dora(self, idx: int) -> None:
        state = self._worker.state
        if state:
            state.remove_dora(idx)

    def add_meld(self, kind: str, tile_names: List[str]) -> bool:
        state = self._worker.state
        if state is None:
            return False
        try:
            tiles = [name_to_tile(n) for n in tile_names]
            return state.add_meld(kind, tiles)
        except KeyError:
            return False

    def remove_meld(self, idx: int) -> None:
        state = self._worker.state
        if state:
            state.remove_meld(idx)

    def set_discard_count(self, tile_name_str: str, count: int) -> None:
        state = self._worker.state
        if state is None:
            return
        try:
            tid = name_to_tile(tile_name_str)
            state.set_discard_count(tid, count)
        except KeyError:
            pass

    def set_discard_count_by_id(self, tid: int, count: int) -> None:
        state = self._worker.state
        if state:
            state.set_discard_count(tid, count)

    def get_state_snapshot(self) -> dict:
        """Return a snapshot of current game state for UI display (main thread safe)."""
        state = self._worker.state
        if state is None:
            return {"doras": [], "melds": [], "discards": {}}
        from .tile_codec import tile_name as _tn
        return {
            "doras":    list(state.dora_indicators),
            "melds":    [(m.kind, list(m.tiles)) for m in state.melds],
            "discards": {tid: int(state.discards_seen[tid])
                         for tid in range(34) if state.discards_seen[tid] > 0},
            "discards_arr": state.discards_seen.copy(),
            "can_add":  lambda tid: state.can_add_tile(tid),
        }

    # ── Capture / debug ───────────────────────────────────────────────────────

    @property
    def has_last_img(self) -> bool:
        return self._worker.last_img is not None

    def get_capture_status(self) -> str:
        return self._capture.get_status()

    def save_debug_frame(self, out_dir: Path,
                         crop: Tuple[int, int] = (0, 0)) -> Path:
        """
        Save the last captured frame (or fresh fullscreen capture) with HAND ROI overlay.
        When last_img is None the caller (MainWindow) must withdraw() the advisor
        window ~150 ms before calling this method so the compositor omits it.
        Returns path of saved file.
        """
        img = self._worker.last_img

        if img is None:
            # capture_fullscreen() uses DXGI with no window-title heuristic —
            # caller must have withdrawn the advisor window before this point.
            ox, oy = crop
            raw    = self._capture.capture_fullscreen()
            h_raw, w_raw = raw.shape[:2]
            if ox > 0 or oy > 0:
                img = raw[oy : min(oy + 837, h_raw), ox : min(ox + 1418, w_raw)]
            else:
                img = raw

        disp = (cv2.resize(img, (1418, 837))
                if img.shape[:2] != (837, 1418) else img.copy())
        x1, y1, x2, y2 = ROI_4P["hand"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(disp, "HAND ROI", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "debug_capture.png"
        cv2.imwrite(str(out_path), disp)
        return out_path

    # ── Internal polling ──────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                r1: Phase1Result = self._phase1_q.get_nowait()
                self._on_phase1(r1)
        except queue.Empty:
            pass
        try:
            while True:
                r2: Phase2Result = self._phase2_q.get_nowait()
                self._on_phase2(r2)
        except queue.Empty:
            pass
        self._root.after(self._poll_ms, self._poll)
