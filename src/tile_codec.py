"""
tile_codec.py
Tile encoding: convert between YOLO model class IDs and standard mahjong tile IDs.

Standard tile ID layout (0-33):
  Man  : 1m=0  … 9m=8
  Pin  : 1p=9  … 9p=17
  Sou  : 1s=18 … 9s=26
  Hon  : east=27 south=28 west=29 north=30 white=31 green=32 red=33
"""
import numpy as np

# ── YOLO model class names (alphabetical, fixed by training) ─────────────────
MODEL_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

# ── Standard mahjong tile names ───────────────────────────────────────────────
TILE_NAMES = [
    '1m','2m','3m','4m','5m','6m','7m','8m','9m',   # 0-8   man
    '1p','2p','3p','4p','5p','6p','7p','8p','9p',   # 9-17  pin
    '1s','2s','3s','4s','5s','6s','7s','8s','9s',   # 18-26 sou
    'east','south','west','north','white','green','red',  # 27-33 honours
]
N_TILES     = 34
TOTAL_TILES = np.array([4] * 34, dtype=np.int32)

# ── Mapping arrays ────────────────────────────────────────────────────────────
_t2id  = {n: i for i, n in enumerate(TILE_NAMES)}
_m2id  = {n: i for i, n in enumerate(MODEL_NAMES)}

# model_class_id → standard tile_id
MODEL_TO_TILE = np.array([_t2id[n] for n in MODEL_NAMES], dtype=np.int32)
# standard tile_id → model_class_id
TILE_TO_MODEL = np.array([_m2id[n] for n in TILE_NAMES], dtype=np.int32)

# ── Tile property helpers ─────────────────────────────────────────────────────
def tile_name(tid: int) -> str:
    return TILE_NAMES[tid]

def model_to_tile(mid: int) -> int:
    return int(MODEL_TO_TILE[mid])

def name_to_tile(name: str) -> int:
    return _t2id[name]

def get_suit(tid: int) -> int:
    """0=man 1=pin 2=sou 3=honour"""
    return 3 if tid >= 27 else tid // 9

def get_num(tid: int) -> int:
    """1-9 for suited tiles, 0 for honours."""
    return 0 if tid >= 27 else tid % 9 + 1

def is_honour(tid: int) -> bool:
    return tid >= 27

def is_terminal(tid: int) -> bool:
    return tid >= 27 or tid % 9 in (0, 8)

def is_yakuhai(tid: int, seat_wind: int, round_wind: int) -> bool:
    """seat_wind / round_wind: 0=east 1=south 2=west 3=north"""
    if tid == 31: return True   # haku (white)
    if tid == 32: return True   # hatsu (green)
    if tid == 33: return True   # chun (red)
    if tid == 27 + seat_wind:  return True
    if tid == 27 + round_wind: return True
    return False

def hand_str(counts34: np.ndarray) -> str:
    """Human-readable string for a 34-count hand."""
    parts = []
    for i, c in enumerate(counts34):
        parts.extend([TILE_NAMES[i]] * int(c))
    return ' '.join(parts)
