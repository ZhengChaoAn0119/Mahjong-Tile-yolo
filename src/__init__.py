"""
src/__init__.py
Public API for the majsoul_yolo backend package.
"""
from .tile_codec import (
    N_TILES, TILE_NAMES, TOTAL_TILES, MODEL_TO_TILE, TILE_TO_MODEL,
    tile_name, name_to_tile, model_to_tile,
    get_suit, get_num, is_honour, is_terminal, is_yakuhai, hand_str,
)
from .frame_smoother import FrameSmoother, RawDetection, ConfirmedTile
from .game_state import GameState, Meld, MELD_CHI, MELD_PON, MELD_KAN_O, MELD_KAN_C
from .mahjong_engine import (
    shanten, effective_tiles, detect_yaku, calculate_fu, estimate_score,
)
from .ev_engine import compute_discard_ev
from .mahjong_advisor import (
    MahjongAdvisor, run_detection,
    ROI_4P, ROI_3P, CONF_THRESH, IOU_THRESH, _gray,
)
from .screen_capture import ScreenCapture
from .advisor_controller import (
    AdvisorController, AnalysisWorker, HotkeyManager,
    Phase1Result, Phase2Result,
)

__all__ = [
    # tile_codec
    "N_TILES", "TILE_NAMES", "TOTAL_TILES", "MODEL_TO_TILE", "TILE_TO_MODEL",
    "tile_name", "name_to_tile", "model_to_tile",
    "get_suit", "get_num", "is_honour", "is_terminal", "is_yakuhai", "hand_str",
    # frame_smoother
    "FrameSmoother", "RawDetection", "ConfirmedTile",
    # game_state
    "GameState", "Meld", "MELD_CHI", "MELD_PON", "MELD_KAN_O", "MELD_KAN_C",
    # mahjong_engine
    "shanten", "effective_tiles", "detect_yaku", "calculate_fu", "estimate_score",
    # ev_engine
    "compute_discard_ev",
    # mahjong_advisor
    "MahjongAdvisor", "run_detection", "ROI_4P", "ROI_3P",
    "CONF_THRESH", "IOU_THRESH", "_gray",
    # screen_capture
    "ScreenCapture",
    # advisor_controller
    "AdvisorController", "AnalysisWorker", "HotkeyManager",
    "Phase1Result", "Phase2Result",
]
