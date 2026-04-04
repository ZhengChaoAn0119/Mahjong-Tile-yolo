"""
ev_engine.py
EV engine with two modes:
  1. FAST analytical: O(34) per discard — used for real-time overlay
  2. Monte Carlo:     NumPy vectorized — deeper analysis (background)

Analytical EV formula:
  win_prob ≈ 1 − (1 − eff_tiles/remaining)^max_draws
  EV = win_prob × estimated_score

Monte Carlo uses NumPy random for speed (no Numba dependency).
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

from .tile_codec   import N_TILES, TILE_NAMES, tile_name
from .mahjong_engine import (
    shanten, shanten_regular, shanten_chiitoitsu,
    effective_tiles, detect_yaku, calculate_fu, estimate_score
)

N_SIMS_DEFAULT  = 500
MAX_DRAWS       = 18
URGENCY_K       = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# Fast analytical EV (no MC)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_ev(h_after: np.ndarray,
                  remaining: np.ndarray,
                  score: int,
                  max_draws: int = MAX_DRAWS) -> float:
    """
    Estimate win probability using effective tile analysis.
    P(win) ≈ 1 - prod_{draw=1}^{max_draws} (1 - eff_avail_draw / rem_draw)
    Simplified: assume effective tiles drawn optimally.
    """
    effs = effective_tiles(h_after, remaining)
    if not effs:
        return 0.0
    eff_count = sum(c for _, c in effs)
    rem_total = int(remaining.sum())
    if rem_total == 0:
        return 0.0

    # Simplified geometric: probability of NOT drawing eff tile in n draws
    p_miss_one  = max(0.0, 1.0 - eff_count / rem_total)
    p_win       = 1.0 - p_miss_one ** max_draws
    urgency     = URGENCY_K ** (max_draws * (1 - p_win))
    return p_win * score * urgency


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation (NumPy vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_discard(hand34: np.ndarray) -> int:
    """Find discard that minimises shanten (greedy)."""
    best_tid  = -1
    best_shan = 99
    for tid in range(N_TILES):
        if hand34[tid] == 0:
            continue
        hand34[tid] -= 1
        s = int(shanten(hand34))
        hand34[tid] += 1
        if s < best_shan:
            best_shan = s
            best_tid  = tid
    return best_tid


def _simulate_batch(hand34: np.ndarray,
                    remaining34: np.ndarray,
                    n_sims: int,
                    max_draws: int) -> Tuple[float, float]:
    """
    Vectorized MC: simulate n_sims games, return (win_rate, avg_draws_to_win).
    Each sim plays greedily (discard to minimise shanten).
    """
    wins        = 0
    total_draws = 0

    # Build weighted tile array for fast sampling
    tile_ids = np.repeat(np.arange(N_TILES, dtype=np.int32), remaining34)

    for _ in range(n_sims):
        hand   = hand34.copy()
        rem    = remaining34.copy()
        pool   = tile_ids.copy()
        rem_n  = int(rem.sum())
        won    = False

        np.random.shuffle(pool)
        ptr = 0

        for draw_n in range(max_draws):
            if ptr >= rem_n:
                break

            drawn = pool[ptr]; ptr += 1
            hand[drawn] += 1

            s = int(shanten(hand))
            if s == -1:
                wins        += 1
                total_draws += draw_n + 1
                won = True
                break

            d = _greedy_discard(hand)
            if d >= 0:
                hand[d] -= 1

    win_rate  = wins / n_sims
    avg_draws = (total_draws / wins) if wins > 0 else float(max_draws)
    return win_rate, avg_draws


# ─────────────────────────────────────────────────────────────────────────────
# Main EV computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_discard_ev(
    hand34:      np.ndarray,
    remaining34: np.ndarray,
    melds,
    seat_wind:   int,
    round_wind:  int,
    dora_tiles:  List[int],
    n_sims:      int  = 0,        # 0 = analytical only (fast)
    max_draws:   int  = MAX_DRAWS,
) -> List[Dict]:
    """
    For each unique discard, compute EV.
    n_sims=0 → analytical only (fast, ~0.1s)
    n_sims>0 → Monte Carlo for win_rate (slower but more accurate)
    """
    results = []
    is_open = len(melds) > 0

    # Deduplicate discards (same tile type counts once)
    seen_types = set()
    for discard_tid in range(N_TILES):
        if hand34[discard_tid] == 0:
            continue
        if discard_tid in seen_types:
            continue
        seen_types.add(discard_tid)

        h_after = hand34.copy()
        h_after[discard_tid] -= 1

        new_shan = int(shanten(h_after))

        # Effective tiles of resulting hand
        effs      = effective_tiles(h_after, remaining34)
        eff_count = sum(c for _, c in effs)
        rem_total = int(remaining34.sum())

        # Score estimate
        yaku  = detect_yaku(h_after, melds, False, seat_wind, round_wind, dora_tiles)
        han   = max(1, sum(h for _, h in yaku if h > 0))
        fu    = calculate_fu(h_after, melds, -1, False, is_open, seat_wind, round_wind)
        score = estimate_score(han, fu)

        # EV
        if n_sims > 0:
            win_rate, avg_draws = _simulate_batch(
                h_after, remaining34, n_sims, max_draws
            )
            urgency  = URGENCY_K ** avg_draws
            ev       = win_rate * score * urgency
        else:
            ev        = analytical_ev(h_after, remaining34, score, max_draws)
            win_rate  = ev / score if score > 0 else 0.0
            avg_draws = float(max_draws)

        results.append({
            "discard_tid":  discard_tid,
            "discard_name": TILE_NAMES[discard_tid],
            "shanten":      new_shan,
            "eff_tiles":    effs,
            "eff_count":    eff_count,
            "rem_total":    rem_total,
            "win_rate":     win_rate,
            "avg_draws":    avg_draws,
            "est_score":    score,
            "ev":           ev,
            "yaku":         yaku,
            "han":          han,
            "fu":           fu,
        })

    results.sort(key=lambda x: -x["ev"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Simple EV  (Shanten × winning points — fast, no MC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_simple_ev(
    hand34:      np.ndarray,
    remaining34: np.ndarray,
    melds,
    seat_wind:   int,
    round_wind:  int,
    dora_tiles:  List[int],
) -> List[Dict]:
    """
    Rank each possible discard by:
        simple_ev = est_score * eff_count / (shanten_after + 1)

    Much faster than Monte Carlo; suitable for real-time overlay.
    win_rate is set to 0.0 (not computed).
    """
    results  = []
    is_open  = len(melds) > 0
    rem_total = int(remaining34.sum())

    seen_types: set = set()
    for discard_tid in range(N_TILES):
        if hand34[discard_tid] == 0:
            continue
        if discard_tid in seen_types:
            continue
        seen_types.add(discard_tid)

        h_after = hand34.copy()
        h_after[discard_tid] -= 1

        new_shan  = int(shanten(h_after))
        effs      = effective_tiles(h_after, remaining34)
        eff_count = sum(c for _, c in effs)

        yaku  = detect_yaku(h_after, melds, False, seat_wind, round_wind, dora_tiles)
        han   = max(1, sum(h for _, h in yaku if h > 0))
        fu    = calculate_fu(h_after, melds, -1, False, is_open, seat_wind, round_wind)
        score = estimate_score(han, fu)

        simple_ev = score * eff_count / max(new_shan + 1, 1)

        results.append({
            "discard_tid":  discard_tid,
            "discard_name": TILE_NAMES[discard_tid],
            "shanten":      new_shan,
            "eff_tiles":    effs,
            "eff_count":    eff_count,
            "rem_total":    rem_total,
            "win_rate":     0.0,
            "avg_draws":    0.0,
            "est_score":    score,
            "ev":           simple_ev,
            "yaku":         yaku,
            "han":          han,
            "fu":           fu,
        })

    results.sort(key=lambda x: (-x["ev"], -x["eff_count"]))
    return results
