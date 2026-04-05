"""
hand_synth.py
Generate synthetic 13+1 hand images by compositing tile_images onto the template.

Usage:
    python hand_synth.py            # generate N_IMAGES (default 10)
    python hand_synth.py --n 200    # generate 200 images
"""

import argparse
import cv2
import json
import numpy as np
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
N_IMAGES   = 10
TEMPLATE   = Path("dataset_hand/13+1.png")
TILE_DIR   = Path("dataset_hand/tile_images")
OUT_IMGS   = Path("dataset_hand/images/train")
OUT_LBLS   = Path("dataset_hand/labels/train")
SLOTS_JSON = Path("dataset_hand/slots.json")

MODEL_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]
N_CLASSES = len(MODEL_NAMES)  # 34
N_SLOTS   = 14                # 13 hand tiles + 1 drawn tile


# ── Slot detection ────────────────────────────────────────────────────────────

def _find_segments(presence: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Find contiguous x-ranges where presence >= threshold."""
    in_seg = False
    segs = []
    x0 = 0
    for x, v in enumerate(presence):
        if v >= threshold:
            if not in_seg:
                x0, in_seg = x, True
        else:
            if in_seg:
                segs.append((x0, x - 1))
                in_seg = False
    if in_seg:
        segs.append((x0, len(presence) - 1))
    return segs


def _merge_close_segs(segs: list[tuple[int, int]], min_gap: int = 3) -> list[tuple[int, int]]:
    """Merge segments that are separated by fewer than min_gap pixels."""
    if not segs:
        return segs
    merged = [segs[0]]
    for s in segs[1:]:
        if s[0] - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def detect_slots(template_rgba: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Return 14 tile bounding boxes [(x1,y1,x2,y2), ...] from the RGBA template.
    Uses vertical edge detection to find tile boundaries, then falls back to
    background-color contrast, then uniform split.
    """
    H, W = template_rgba.shape[:2]
    img_gray = cv2.cvtColor(template_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)

    # ── Method 1: vertical edge detection ────────────────────────────────────
    # Tiles have strong vertical edges at their borders.
    # Compute column-wise Sobel magnitude; gaps between tiles appear as peaks.
    sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    col_edge = np.abs(sobel_x).mean(axis=0)           # average edge strength per column
    # Columns that are "gap-like": strong edges nearby
    # Smooth to find local maxima regions
    smoothed = np.convolve(col_edge, np.ones(3) / 3, mode='same')
    # Find valleys (low edge regions are within tiles; peaks are at tile borders)
    # Instead, find all columns with edge > threshold and group the non-edge columns
    threshold = np.percentile(smoothed, 60)  # top 40% by edge strength are borders
    non_border = smoothed < threshold
    segs = _find_segments(non_border.astype(float), threshold=0.5)
    segs = _merge_close_segs(segs, min_gap=1)
    # Filter out very narrow segments (noise)
    segs = [(a, b) for a, b in segs if (b - a) >= 20]

    if len(segs) == N_SLOTS:
        y1, y2 = 1, H - 1
        print(f"[slots] edge method → {len(segs)} segments ✓")
        return [(xs, y1, xe, y2) for xs, xe in segs]

    # ── Method 2: background color contrast ──────────────────────────────────
    img_rgb = template_rgba[:, :, :3].astype(float)
    # Sample background from left/right edges and top/bottom rows
    bg_samples = np.vstack([
        img_rgb[:, :2, :].reshape(-1, 3),
        img_rgb[:, -2:, :].reshape(-1, 3),
        img_rgb[:2, :, :].reshape(-1, 3),
        img_rgb[-2:, :, :].reshape(-1, 3),
    ])
    bg = np.median(bg_samples, axis=0)
    diff = np.abs(img_rgb - bg).sum(axis=2)             # (H, W)
    col_diff = (diff > 20).mean(axis=0)
    segs2 = _find_segments(col_diff, threshold=0.3)
    segs2 = _merge_close_segs(segs2, min_gap=2)

    if len(segs2) == N_SLOTS:
        row_diff = (diff > 20).mean(axis=1)
        y_rows   = np.where(row_diff > 0.05)[0]
        y1 = int(y_rows[0])  if len(y_rows) else 1
        y2 = int(y_rows[-1]) if len(y_rows) else H - 1
        print(f"[slots] contrast method → {len(segs2)} segments ✓")
        return [(xs, y1, xe, y2) for xs, xe in segs2]

    # ── Method 3: uniform fallback ────────────────────────────────────────────
    print(f"[slots] WARNING: auto-detect got {len(segs)} (edge) / {len(segs2)} (contrast) segs, "
          f"falling back to uniform split.")
    y1, y2 = 1, H - 1
    # 13 tiles + gap + 1 drawn tile; estimate tile width from image
    # Rough: tile_w = (W - left_margin - right_margin - drawn_gap) / 14
    left_margin  = 3
    right_margin = 3
    drawn_gap    = 8  # extra gap before drawn tile
    inner_gap    = 1  # gap between consecutive hand tiles
    usable = W - left_margin - right_margin - drawn_gap - inner_gap * 12
    tile_w = usable // N_SLOTS
    slots_out = []
    x = left_margin
    for i in range(N_SLOTS):
        x2 = min(x + tile_w, W - 1)
        slots_out.append((x, y1, x2, y2))
        if i == 12:
            x = x2 + inner_gap + drawn_gap
        else:
            x = x2 + inner_gap
    return slots_out


# ── Tile image loader ─────────────────────────────────────────────────────────

def load_tile_images() -> dict[int, np.ndarray]:
    """Load all 34 tile images (RGBA) from tile_images/{cls_id}/*.png."""
    tile_imgs: dict[int, np.ndarray] = {}
    for cls_id in range(N_CLASSES):
        d = TILE_DIR / str(cls_id)
        pngs = sorted(d.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"No PNG in {d}")
        img = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read {pngs[0]}")
        # Ensure 4 channels
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        tile_imgs[cls_id] = img
    return tile_imgs


# ── Alpha-composite a tile onto canvas ───────────────────────────────────────

def paste_tile(canvas_rgb: np.ndarray,
               tile_rgba: np.ndarray,
               x1: int, y1: int, x2: int, y2: int) -> None:
    """Resize tile_rgba to fit the canvas region [y1:y2, x1:x2] and alpha-composite in-place."""
    # Use actual slice dimensions (handles any boundary clamping)
    bg_patch = canvas_rgb[y1:y2, x1:x2]
    slot_h, slot_w = bg_patch.shape[:2]
    if slot_h == 0 or slot_w == 0:
        return
    resized  = cv2.resize(tile_rgba, (slot_w, slot_h), interpolation=cv2.INTER_AREA)
    alpha    = resized[:, :, 3:4].astype(float) / 255.0
    tile_bgr = resized[:, :, :3].astype(float)
    canvas_rgb[y1:y2, x1:x2] = (tile_bgr * alpha + bg_patch.astype(float) * (1.0 - alpha)).astype(np.uint8)


# ── Main generation ───────────────────────────────────────────────────────────

def generate(n: int, class_weights: dict | None = None) -> None:
    """
    Generate n synthetic hand images.

    Args:
        n:             Number of images to generate.
        class_weights: Optional {class_id: weight} dict.
                       Classes with higher weight appear more often.
                       Defaults to uniform (all weights = 1.0).
    """
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LBLS.mkdir(parents=True, exist_ok=True)

    # Load template
    template_rgba = cv2.imread(str(TEMPLATE), cv2.IMREAD_UNCHANGED)
    if template_rgba is None:
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")
    if template_rgba.shape[2] == 3:
        template_rgba = cv2.cvtColor(template_rgba, cv2.COLOR_BGR2BGRA)
    H, W = template_rgba.shape[:2]
    print(f"[template] {W}x{H} RGBA")

    # Detect or load slots
    if SLOTS_JSON.exists():
        with open(SLOTS_JSON) as f:
            slots = [tuple(s) for s in json.load(f)]
        print(f"[slots] loaded from {SLOTS_JSON} ({len(slots)} slots)")
    else:
        slots = detect_slots(template_rgba)
        with open(SLOTS_JSON, "w") as f:
            json.dump(slots, f, indent=2)
        print(f"[slots] saved to {SLOTS_JSON}")

    if len(slots) != N_SLOTS:
        print(f"[WARNING] expected {N_SLOTS} slots, got {len(slots)} — check {SLOTS_JSON}")

    # Load tile images
    tile_imgs = load_tile_images()
    print(f"[tiles] loaded {len(tile_imgs)} tile images")

    # Continue numbering from highest existing file
    existing = list(OUT_IMGS.glob("hand_synth_*.jpg"))
    start_idx = max(
        (int(f.stem.rsplit("_", 1)[-1]) for f in existing), default=0
    ) + 1

    # Build sampling weights (uniform if not specified)
    weights = [class_weights.get(c, 1.0) if class_weights else 1.0
               for c in range(N_CLASSES)]

    # Generate images
    template_rgb = cv2.cvtColor(template_rgba, cv2.COLOR_BGRA2BGR)

    for i in range(n):
        idx    = start_idx + i
        canvas = template_rgb.copy()

        classes = random.choices(range(N_CLASSES), weights=weights, k=len(slots))
        labels  = []

        for (x1, y1, x2, y2), cls_id in zip(slots, classes):
            paste_tile(canvas, tile_imgs[cls_id], x1, y1, x2, y2)
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w  = (x2 - x1) / W
            h  = (y2 - y1) / H
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        name = f"hand_synth_{idx:04d}"
        cv2.imwrite(str(OUT_IMGS / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT_LBLS / f"{name}.txt").write_text("\n".join(labels) + "\n")

        if (i + 1) % 100 == 0 or n <= 20:
            tile_names = [MODEL_NAMES[c] for c in classes]
            print(f"  [{idx:04d}] {' '.join(tile_names)}")

    print(f"\nDone — {n} new images in {OUT_IMGS}  (total: {start_idx + n - 1})")
    print(f"Labels in {OUT_LBLS}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_IMAGES, help="Number of images to generate")
    args = parser.parse_args()
    generate(args.n)
