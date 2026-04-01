"""
game_state.py
Tracks game state: hand tiles, open melds, discards, dora.
Provides manual correction interface with confidence warnings.

Remaining tiles = 4×each - hand - melds - all_known_discards - dora_indicators
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from tile_codec import (
    N_TILES, TILE_NAMES, TOTAL_TILES, tile_name,
    name_to_tile, model_to_tile, hand_str
)
from frame_smoother import ConfirmedTile


# ── Meld representation ───────────────────────────────────────────────────────
MELD_CHI   = "chi"
MELD_PON   = "pon"
MELD_KAN_O = "open_kan"
MELD_KAN_C = "closed_kan"

@dataclass
class Meld:
    kind:  str              # chi / pon / open_kan / closed_kan
    tiles: List[int]        # tile IDs (3 for chi/pon, 4 for kan)
    called_from: int = -1   # player position who the tile was called from (0=self)


# ── Game state ────────────────────────────────────────────────────────────────
class GameState:
    """
    Tracks all visible game information.
    Hand is stored as a sorted list of tile IDs (14 tiles including draw).
    """

    def __init__(self, seat_wind: int = 0, round_wind: int = 0):
        self.seat_wind  = seat_wind   # 0=east 1=south 2=west 3=north
        self.round_wind = round_wind

        # Own hand (13 or 14 tile IDs, sorted)
        self._hand: List[int] = []

        # Open melds
        self.melds: List[Meld] = []

        # All visible discards (own + opponents, as tile ID counts)
        self.discards_seen = np.zeros(N_TILES, dtype=np.int32)

        # Dora indicator tile IDs → actual dora = (indicator+1) wrapping within suit
        self.dora_indicators: List[int] = []

        # Manual override registry: tile_id → corrected tile_id per hand slot
        self._manual_overrides: Dict[int, int] = {}   # cx → corrected tile_id

        # Confidence warnings (from latest update)
        self.low_conf_tiles: List[ConfirmedTile] = []

    # ── Update from detector ──────────────────────────────────────────────────
    def update_from_detection(self,
                              hand_tiles:  List[ConfirmedTile],
                              meld_tiles:  List[ConfirmedTile],
                              center_tiles: List[ConfirmedTile]):
        """Called each frame after smoothing."""
        # Build hand (apply manual overrides)
        raw_hand = [t.tile_id for t in hand_tiles]
        for i, ct in enumerate(hand_tiles):
            if ct.cx in self._manual_overrides:
                raw_hand[i] = self._manual_overrides[ct.cx]
        self._hand = sorted(raw_hand)

        # Build melds from meld zone (simplified: treat as pon/open kan)
        # Each group of 3 same tiles = pon, 4 = kan
        meld_counts = np.zeros(N_TILES, dtype=np.int32)
        for t in meld_tiles:
            meld_counts[t.tile_id] += 1
        self.melds = []
        for tid in range(N_TILES):
            c = meld_counts[tid]
            if c == 3:
                self.melds.append(Meld(MELD_PON, [tid]*3))
            elif c == 4:
                self.melds.append(Meld(MELD_KAN_O, [tid]*4))
            elif c >= 2:
                self.melds.append(Meld(MELD_PON, [tid]*3))  # best guess

        # Update known discards from center
        for t in center_tiles:
            self.discards_seen[t.tile_id] = min(
                self.discards_seen[t.tile_id] + 1, 4
            )

        # Collect low-confidence warnings
        self.low_conf_tiles = [
            t for t in (hand_tiles + meld_tiles)
            if t.low_conf
        ]

    # ── Remaining tiles ───────────────────────────────────────────────────────
    def remaining_tiles(self) -> np.ndarray:
        """
        Estimate of tiles still in the wall (unknown to us).
        remaining = 4 each - hand - melds - seen_discards - dora_indicators
        """
        used = np.zeros(N_TILES, dtype=np.int32)

        # Own hand
        for tid in self._hand:
            used[tid] += 1

        # Open melds
        for m in self.melds:
            for tid in m.tiles:
                used[tid] += 1

        # Seen discards
        used += self.discards_seen

        # Dora indicators (the indicator itself is out of the wall)
        for ind in self.dora_indicators:
            used[ind] = min(used[ind] + 1, 4)

        remaining = np.maximum(0, TOTAL_TILES - used)
        return remaining

    def hand34(self) -> np.ndarray:
        """Hand as 34-element count array."""
        arr = np.zeros(N_TILES, dtype=np.int32)
        for tid in self._hand:
            arr[tid] += 1
        return arr

    def melds34(self) -> np.ndarray:
        """Open melds as 34-element count array."""
        arr = np.zeros(N_TILES, dtype=np.int32)
        for m in self.melds:
            for tid in m.tiles:
                arr[tid] += 1
        return arr

    def meld_count(self) -> int:
        return len(self.melds)

    def is_open(self) -> bool:
        return len(self.melds) > 0

    # ── Dora calculation ──────────────────────────────────────────────────────
    def dora_tiles(self) -> List[int]:
        """Convert dora indicators to actual dora tile IDs."""
        doras = []
        for ind in self.dora_indicators:
            if ind >= 27:   # honours wrap within honours
                offsets = {27:28, 28:29, 29:30, 30:27, 31:32, 32:33, 33:31}
                doras.append(offsets.get(ind, ind))
            else:
                suit_start = (ind // 9) * 9
                num = ind % 9
                doras.append(suit_start + (num + 1) % 9)
        return doras

    # ── Manual correction ─────────────────────────────────────────────────────
    def apply_manual_correction(self, cx: int, correct_name: str) -> bool:
        """
        Correct the tile at approximate x-position cx to correct_name.
        Returns True if successful.
        """
        try:
            tid = name_to_tile(correct_name.strip().lower())
            self._manual_overrides[cx] = tid
            # Re-sort hand
            self._hand = sorted([
                self._manual_overrides.get(0, t) for t in self._hand
            ])
            return True
        except KeyError:
            return False

    def manual_add_discard(self, tile_name: str):
        """Manually register a discard (e.g., opponent's discard)."""
        try:
            tid = name_to_tile(tile_name.strip().lower())
            self.discards_seen[tid] = min(self.discards_seen[tid] + 1, 4)
        except KeyError:
            pass

    def clear_overrides(self):
        self._manual_overrides.clear()

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = []
        lines.append(f"Hand ({len(self._hand)}): {' '.join(tile_name(t) for t in self._hand)}")
        for i, m in enumerate(self.melds):
            lines.append(f"Meld {i+1} [{m.kind}]: {' '.join(tile_name(t) for t in m.tiles)}")
        rem = self.remaining_tiles()
        lines.append(f"Remaining tiles total: {int(rem.sum())}")
        if self.low_conf_tiles:
            lines.append(f"⚠ Low-confidence: {[tile_name(t.tile_id) for t in self.low_conf_tiles]}")
        doras = self.dora_tiles()
        if doras:
            lines.append(f"Dora: {[tile_name(d) for d in doras]}")
        return "\n".join(lines)
