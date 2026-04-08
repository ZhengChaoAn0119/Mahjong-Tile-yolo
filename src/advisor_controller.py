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
from .mahjong_advisor import MahjongAdvisor, run_detection, run_hand_detection, ROI_4P
from .mahjong_engine  import (effective_tiles, shanten,
                               detect_yaku, calculate_fu, estimate_score)
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
    agari_info:      Optional[Dict] = None   # set when shanten == -1


@dataclass
class Phase2Result:
    ev_results:   List[Dict]
    is_mc:        bool
    compute_time: float


# ─────────────────────────────────────────────────────────────────────────────
# EV cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ev_cache_key(hand34, remaining34, melds, seat_wind, round_wind, doras):
    melds_key = tuple(sorted((m.kind, tuple(sorted(m.tiles))) for m in melds))
    return (tuple(hand34), tuple(remaining34), melds_key,
            tuple(sorted(doras)), seat_wind, round_wind)


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
        self._ev_cache:      dict           = {}
        self._cache_lock:    threading.Lock = threading.Lock()
        self._precomp_q:     queue.Queue    = queue.Queue(maxsize=1)
        self._precomp_busy:  threading.Event = threading.Event()  # set while batch is running
        self._last_precomp_hand: Optional[tuple] = None           # hand34 of last submitted job

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
        with self._cache_lock:
            self._ev_cache.clear()
        self._last_precomp_hand = None

    def run(self):
        self._advisor = MahjongAdvisor(model_path=self._model_path)
        threading.Thread(target=self._precompute_loop,
                         daemon=True, name="PrecomputeWorker").start()
        while True:
            signal = self._trigger_q.get()
            if signal == "STOP":
                break
            if isinstance(signal, tuple) and signal[0] == "ANALYZE":
                self._do_analysis(signal[1], signal[2], signal[3], signal[4], signal[5])

    def _do_analysis(self, x: int, y: int, w: int, h: int, n_sims: int):
        try:
            if w > 0 and h > 0:
                # Hand-strip mode: user selected the hand tile region on screen.
                # Capture full screen at native resolution (no resize!) then crop.
                raw = self._capture.capture_fullscreen_raw()
                if not ScreenCapture.is_valid_frame(raw):
                    self._phase1_q.put(Phase1Result(
                        hand_tiles=[], shanten=99, effective_tiles=[],
                        warnings=[], game_mode="4p",
                        timestamp=time.time(), capture_ok=False,
                        error_msg="Black frame — check capture permissions"))
                    return
                h_raw, w_raw = raw.shape[:2]
                # If tkinter logical coords differ from physical (DPI scaling),
                # scale coordinates to match the captured frame dimensions.
                sx = w_raw / self._capture._screen_w
                sy = h_raw / self._capture._screen_h
                px  = int(x * sx);  py  = int(y * sy)
                pw  = int(w * sx);  ph  = int(h * sy)
                hand_crop = raw[py : min(py + ph, h_raw), px : min(px + pw, w_raw)]
                if hand_crop.size == 0:
                    self._phase1_q.put(Phase1Result(
                        hand_tiles=[], shanten=99, effective_tiles=[],
                        warnings=[], game_mode="4p",
                        timestamp=time.time(), capture_ok=False,
                        error_msg="Empty crop — region out of screen bounds"))
                    return
                self._last_img = hand_crop.copy()
                dets, mode = run_hand_detection(
                    self._advisor.hand_model, hand_crop, x, y, w, h)
            else:
                # Full-screen mode: no region selected, capture and detect normally.
                raw = self._capture.capture()
                if not ScreenCapture.is_valid_frame(raw):
                    self._phase1_q.put(Phase1Result(
                        hand_tiles=[], shanten=99, effective_tiles=[],
                        warnings=[], game_mode="4p",
                        timestamp=time.time(), capture_ok=False,
                        error_msg="Black frame — check capture permissions"))
                    return
                self._last_img = raw.copy()
                if raw.shape[:2] != (837, 1418):
                    raw = cv2.resize(raw, (1418, 837))
                dets, mode = run_detection(self._advisor.model, raw)

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

            agari_info = None
            if s == -1 and len(hand_tiles) == 14:
                # Hand is complete — compute yaku/score instead of EV
                doras = state.dora_tiles()
                # Drawn tile is the 14th tile (highest cx after sort)
                sorted_by_cx = sorted(hand_tiles, key=lambda t: t.cx)
                winning_tile = sorted_by_cx[-1].tile_id if sorted_by_cx else 0
                yaku = detect_yaku(
                    hand34, state.melds, is_tsumo=True,
                    seat_wind=state.seat_wind, round_wind=state.round_wind,
                    dora_tiles=doras)
                han  = sum(h for _, h in yaku if isinstance(h, int))
                fu   = calculate_fu(
                    hand34, state.melds, winning_tile,
                    is_tsumo=True, is_open=bool(state.melds),
                    seat_wind=state.seat_wind, round_wind=state.round_wind)
                score = estimate_score(han, fu)
                agari_info = {"yaku": yaku, "han": han, "fu": fu,
                              "score": score, "winning_tile": winning_tile}

            self._phase1_q.put(Phase1Result(
                hand_tiles=hand_tiles, shanten=s,
                effective_tiles=effs, warnings=warnings,
                game_mode=mode, timestamp=time.time(), capture_ok=True,
                agari_info=agari_info))

            # While waiting for a draw (13 tiles), kick off background
            # pre-computation for all possible drawn tiles — but only once
            # per unique 13-tile hand to avoid continuous CPU load.
            if s != -1 and len(hand_tiles) == 13:
                hand_key = tuple(hand34)
                if (hand_key != self._last_precomp_hand
                        and not self._precomp_busy.is_set()):
                    self._last_precomp_hand = hand_key
                    snap = (
                        hand34.copy(), remaining.copy(),
                        list(state.melds),
                        state.seat_wind, state.round_wind,
                        state.dora_tiles(),
                    )
                    try:
                        self._precomp_q.put_nowait(snap)
                    except queue.Full:
                        pass

            if s != -1 and len(hand_tiles) == 14:
                doras = state.dora_tiles()
                key   = _ev_cache_key(hand34, remaining, state.melds,
                                      state.seat_wind, state.round_wind, doras)
                with self._cache_lock:
                    cached = self._ev_cache.get(key)
                if cached is not None:
                    ev_results, dt = cached, 0.0
                else:
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

    def _precompute_loop(self) -> None:
        """Pre-compute EV for all possible drawn tiles while waiting for a draw.

        Runs in a dedicated daemon thread. Only one batch runs at a time
        (_precomp_busy flag). A 100 ms sleep between iterations yields the GIL
        so the UI thread and AnalysisWorker are never starved.
        """
        while True:
            try:
                hand34, remaining, melds, seat_wind, round_wind, doras = \
                    self._precomp_q.get()
            except Exception:
                continue

            self._precomp_busy.set()
            try:
                results: dict = {}
                for draw_tid in range(34):
                    if remaining[draw_tid] <= 0:
                        continue
                    h14 = hand34.copy();    h14[draw_tid] += 1
                    r14 = remaining.copy(); r14[draw_tid] -= 1
                    if shanten(h14) == -1:
                        continue  # Would be agari — no EV advice needed
                    try:
                        ev = compute_simple_ev(h14, r14, melds,
                                               seat_wind, round_wind, doras)
                    except Exception:
                        continue
                    key = _ev_cache_key(h14, r14, melds, seat_wind, round_wind, doras)
                    results[key] = ev

                    # 100 ms sleep between iterations: yields GIL so UI stays responsive.
                    # Total batch time ≈ 34 × (compute + 100 ms) ≈ 3–5 s, well within
                    # the time other players take to discard.
                    time.sleep(0.1)

                if results:
                    with self._cache_lock:
                        self._ev_cache.update(results)
            finally:
                self._precomp_busy.clear()


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
        self._auto_crop:  Tuple[int, int, int, int] = (0, 0, 0, 0)
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

    def trigger(self, crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
                n_sims: int = 0) -> bool:
        """
        Enqueue an analysis request.
        crop = (x, y, w, h) of the hand region in screen coords; (0,0,0,0) = full screen.
        Returns True if accepted, False if worker is still busy with previous request.
        """
        x, y, w, h = crop
        try:
            self._trigger_q.put_nowait(("ANALYZE", x, y, w, h, n_sims))
            return True
        except queue.Full:
            return False

    # ── Auto mode ─────────────────────────────────────────────────────────────

    def set_auto(self, enabled: bool, interval_ms: int = 5000,
                 crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 n_sims: int = 0) -> None:
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

    def switch_model(self, model_path: Path) -> None:
        """Hot-swap the YOLO model: stop current worker, start a new one."""
        # Drain any pending trigger and stop the current worker
        try:
            self._trigger_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._trigger_q.put_nowait("STOP")
        except queue.Full:
            pass
        # New queue + worker with the new model
        self._trigger_q = queue.Queue(maxsize=1)
        self._worker = AnalysisWorker(
            self._trigger_q, self._phase1_q, self._phase2_q,
            self._capture, model_path)
        self._worker.start()

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
