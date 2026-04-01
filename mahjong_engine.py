"""
mahjong_engine.py
Mahjong logic: shanten, effective tiles, improvement tiles, yaku, fu.
Core calculation functions are Numba-accelerated.

Reference: standard shanten algorithm (recursive block counting).
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except ImportError:
    # Fallback: plain Python
    def njit(*args, **kwargs):
        def decorator(fn): return fn
        return decorator if args and callable(args[0]) else decorator
    NUMBA_OK = False

from tile_codec import N_TILES, TILE_NAMES, get_suit, is_honour, is_terminal

# ─────────────────────────────────────────────────────────────────────────────
# Shanten calculation (Numba-accelerated)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _dfs(hand: np.ndarray, pos: int, mentsu: int, taatsu: int, jantai: int) -> int:
    """
    DFS over tiles to find maximum useful block count.
    Returns 2*mentsu + taatsu + jantai (capped for overflow).
    """
    while pos < 34 and hand[pos] == 0:
        pos += 1
    if pos >= 34:
        eff_taatsu = min(taatsu, 4 - mentsu)
        return 2 * mentsu + eff_taatsu + jantai

    best = _dfs(hand, pos + 1, mentsu, taatsu, jantai)  # ignore tile at pos

    # ── Triplet ───────────────────────────────────────────────────────────────
    if hand[pos] >= 3:
        hand[pos] -= 3
        v = _dfs(hand, pos, mentsu + 1, taatsu, jantai)
        best = max(best, v)
        hand[pos] += 3

    # ── Sequence (suited only) ────────────────────────────────────────────────
    suit = pos // 9
    if suit < 3:
        r = pos % 9
        if r <= 6 and hand[pos + 1] >= 1 and hand[pos + 2] >= 1:
            hand[pos] -= 1; hand[pos + 1] -= 1; hand[pos + 2] -= 1
            v = _dfs(hand, pos, mentsu + 1, taatsu, jantai)
            best = max(best, v)
            hand[pos] += 1; hand[pos + 1] += 1; hand[pos + 2] += 1

    # ── Pair as head (jantai) ─────────────────────────────────────────────────
    if jantai == 0 and hand[pos] >= 2:
        hand[pos] -= 2
        v = _dfs(hand, pos, mentsu, taatsu, 1)
        best = max(best, v)
        hand[pos] += 2

    # ── Pair as taatsu ────────────────────────────────────────────────────────
    if hand[pos] >= 2:
        hand[pos] -= 2
        v = _dfs(hand, pos, mentsu, taatsu + 1, jantai)
        best = max(best, v)
        hand[pos] += 2

    # ── Sequential taatsu (ryanmen / kanchan) ────────────────────────────────
    if suit < 3:
        r = pos % 9
        if r <= 7 and hand[pos + 1] >= 1:
            hand[pos] -= 1; hand[pos + 1] -= 1
            v = _dfs(hand, pos, mentsu, taatsu + 1, jantai)
            best = max(best, v)
            hand[pos] += 1; hand[pos + 1] += 1
        if r <= 6 and hand[pos + 2] >= 1:
            hand[pos] -= 1; hand[pos + 2] -= 1
            v = _dfs(hand, pos, mentsu, taatsu + 1, jantai)
            best = max(best, v)
            hand[pos] += 1; hand[pos + 2] += 1

    return best


@njit(cache=True)
def shanten_regular(hand34: np.ndarray) -> int:
    """Shanten for regular 4-mentsu+1-jantai hand. -1 = complete."""
    h = hand34.copy()
    score = _dfs(h, 0, 0, 0, 0)
    return 8 - score


@njit(cache=True)
def shanten_chiitoitsu(hand34: np.ndarray) -> int:
    """Shanten for 7-pairs hand."""
    pairs = 0
    unique = 0
    for i in range(34):
        if hand34[i] >= 2:
            pairs += 1
        if hand34[i] >= 1:
            unique += 1
    return 6 - pairs


@njit(cache=True)
def shanten_kokushi(hand34: np.ndarray) -> int:
    """Shanten for kokushi (13 orphans)."""
    terminals = np.array([0,8,9,17,18,26,27,28,29,30,31,32,33], dtype=np.int64)
    unique = 0
    has_pair = 0
    for t in terminals:
        if hand34[t] >= 1:
            unique += 1
        if hand34[t] >= 2:
            has_pair = 1
    return 13 - unique - has_pair


@njit(cache=True)
def shanten(hand34: np.ndarray) -> int:
    """Minimum shanten across all hand types."""
    s = shanten_regular(hand34)
    c = shanten_chiitoitsu(hand34)
    k = shanten_kokushi(hand34)
    if c < s: s = c
    if k < s: s = k
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Effective tiles (有效牌) and improvement tiles (改良牌)
# ─────────────────────────────────────────────────────────────────────────────

def effective_tiles(hand34: np.ndarray,
                    remaining34: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns list of (tile_id, count_remaining) for tiles that reduce shanten.
    Sorted by count descending.
    """
    current = shanten(hand34)
    effs = []
    for tid in range(N_TILES):
        if remaining34[tid] <= 0:
            continue
        h = hand34.copy()
        h[tid] += 1
        if shanten(h) < current:
            effs.append((tid, int(remaining34[tid])))
    effs.sort(key=lambda x: -x[1])
    return effs


def improvement_tiles(hand34: np.ndarray,
                      remaining34: np.ndarray) -> Dict[int, List[Tuple[int,int]]]:
    """
    For each possible discard, returns the effective tiles of the resulting hand.
    improvement_tiles[discard_id] = [(tile_id, remaining_count), ...]
    Only includes discards that keep shanten equal to current or better.
    """
    current_shan = shanten(hand34)
    result: Dict[int, List[Tuple[int,int]]] = {}

    for discard_tid in range(N_TILES):
        if hand34[discard_tid] <= 0:
            continue
        h_after = hand34.copy()
        h_after[discard_tid] -= 1
        if shanten(h_after) > current_shan:
            continue   # this discard worsens shanten
        effs = effective_tiles(h_after, remaining34)
        result[discard_tid] = effs

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Winning tile detection (tenpai only)
# ─────────────────────────────────────────────────────────────────────────────

def winning_tiles(hand34: np.ndarray,
                  remaining34: np.ndarray) -> List[Tuple[int, int]]:
    """For a tenpai hand, return tiles that complete it."""
    assert shanten(hand34) == 0, "Hand must be tenpai (shanten=0)"
    wins = []
    for tid in range(N_TILES):
        if remaining34[tid] <= 0:
            continue
        h = hand34.copy()
        h[tid] += 1
        if shanten(h) == -1:
            wins.append((tid, int(remaining34[tid])))
    wins.sort(key=lambda x: -x[1])
    return wins


# ─────────────────────────────────────────────────────────────────────────────
# Basic yaku detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_yaku(hand34: np.ndarray,
                melds,               # List[Meld]
                is_tsumo: bool,
                seat_wind: int,
                round_wind: int,
                dora_tiles: List[int]) -> List[Tuple[str, int]]:
    """
    Returns list of (yaku_name, han_count).
    han_count uses closed/open values.
    """
    yaku = []
    is_open = len(melds) > 0
    total14 = hand34.copy()
    for m in melds:
        for t in m.tiles:
            total14[t] += 1

    # ── Chiitoi (7 pairs, closed only) ───────────────────────────────────────
    if not is_open and shanten_chiitoitsu(hand34) == -1:
        yaku.append(("七対子", 2))
        return yaku   # chiitoi has its own set; return immediately

    # ── Kokushi (closed only) ────────────────────────────────────────────────
    if not is_open and shanten_kokushi(hand34) == -1:
        yaku.append(("国士無双", 13))
        return yaku

    # ── Tanyao ───────────────────────────────────────────────────────────────
    has_terminal = any(is_terminal(i) for i in range(N_TILES) if total14[i] > 0)
    if not has_terminal:
        yaku.append(("断么九", 1))

    # ── Yakuhai (honour triplets) ─────────────────────────────────────────────
    YAKUHAI_TILES = {31: "白", 32: "發", 33: "中"}
    WIND_TILES = {27: "東", 28: "南", 29: "西", 30: "北"}
    for tid, name in YAKUHAI_TILES.items():
        count = total14[tid]
        if count >= 3:
            yaku.append((f"役牌:{name}", 1))
    for tid, name in WIND_TILES.items():
        count = total14[tid]
        if count >= 3:
            offset = tid - 27
            is_seat  = (offset == seat_wind)
            is_round = (offset == round_wind)
            if is_seat and is_round:
                yaku.append((f"ダブ{name}", 2))
            elif is_seat or is_round:
                yaku.append((f"役牌:{name}", 1))

    # ── Toitoi (all triplets, requires open or closed) ────────────────────────
    all_triplets = all(total14[i] in (0, 3, 4) for i in range(N_TILES))
    if all_triplets:
        yaku.append(("対々和", 2))

    # ── Honitsu / Chinitsu ────────────────────────────────────────────────────
    suits_used = set()
    for i in range(N_TILES):
        if total14[i] > 0:
            s = i // 9 if i < 27 else 3
            suits_used.add(s)
    if len(suits_used) == 1 and 3 not in suits_used:
        yaku.append(("清一色", 6 if not is_open else 5))
    elif len(suits_used) == 2 and 3 in suits_used:
        yaku.append(("混一色", 3 if not is_open else 2))

    # ── Tsumo bonus ───────────────────────────────────────────────────────────
    if is_tsumo and not is_open and yaku:
        yaku.append(("門前清自摸和", 1))

    # ── Dora count ────────────────────────────────────────────────────────────
    dora_count = sum(int(total14[d]) for d in dora_tiles)
    if dora_count > 0:
        yaku.append((f"ドラ", dora_count))

    # Fallback: riichi not detectable from screen → note
    if not yaku:
        yaku.append(("役なし(立直?)", 0))

    return yaku


# ─────────────────────────────────────────────────────────────────────────────
# Fu calculation (simplified)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_fu(hand34: np.ndarray,
                 melds,
                 winning_tile: int,
                 is_tsumo: bool,
                 is_open: bool,
                 seat_wind: int,
                 round_wind: int) -> int:
    """
    Simplified fu calculation.
    Returns fu rounded up to nearest 10.
    """
    fu = 30  # base fu for closed ron; 20 for open/tsumo

    # Chiitoi: fixed 25 fu
    if shanten_chiitoitsu(hand34) == -1:
        return 25

    if is_tsumo:
        fu = 20
    elif is_open:
        fu = 30

    # Tsumo bonus
    if is_tsumo:
        fu += 2

    # Meld fu
    for m in melds:
        if not m.tiles:
            continue
        tid = m.tiles[0]
        terminal = is_terminal(tid)
        if m.kind == MELD_PON:
            base = 4 if terminal else 2
            fu += base * 2   # open pon
        elif m.kind in (MELD_KAN_O, MELD_KAN_C):
            base = 16 if terminal else 8
            fu += base * (2 if m.kind == MELD_KAN_O else 4)

    # Pair fu (jantai)
    for tid in range(N_TILES):
        if hand34[tid] >= 2:
            if tid == 31 or tid == 32 or tid == 33:
                fu += 4
            elif tid - 27 == seat_wind or tid - 27 == round_wind:
                fu += 2

    # Round up to nearest 10
    return int(np.ceil(fu / 10) * 10)


# ─────────────────────────────────────────────────────────────────────────────
# Basic score calculation
# ─────────────────────────────────────────────────────────────────────────────

HAN_SCORE_TABLE = {
    # (han, fu) → dealer_tsumo_each / non_dealer_tsumo_dealer / non_dealer_tsumo_each / non_dealer_ron
    # Simplified: just return approximate score for non-dealer ron
}

def estimate_score(han: int, fu: int, is_dealer: bool = False) -> int:
    """
    Estimate base score for non-dealer ron.
    Simplified: base_points = fu × 2^(han+2), rounded up to 100.
    """
    if han >= 13:
        return 32000 if is_dealer else 24000
    if han >= 11:
        return 24000 if is_dealer else 16000
    if han >= 8:
        return 24000 if is_dealer else 16000
    if han >= 5:
        return 12000 if is_dealer else 8000
    if han == 4 and fu >= 30:
        return 12000 if is_dealer else 8000

    base = fu * (2 ** (han + 2))
    base = int(np.ceil(base / 100) * 100)
    score = base * (6 if is_dealer else 4)
    return min(score, 8000 if not is_dealer else 12000)

