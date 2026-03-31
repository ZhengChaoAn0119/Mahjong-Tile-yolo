"""
synthetic_data.py
Generate synthetic training images by compositing tile crops onto backgrounds.

Steps:
  1. Extract tile crops from original training images
  2. Load background screenshots from images/
  3. Generate 200 random-layout composites
  4. Generate 120 weak-class-boosted composites

Usage:
  python synthetic_data.py

Output:
  E:/project/majsoul_yolo/tile_crops/   (individual tile PNG crops)
  E:/project/majsoul_yolo/dataset_synthetic/
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path("E:/project/majsoul_yolo")
SRC_TRAIN      = PROJECT_ROOT / "majsoul.v4i.yolov8/train"
BG_DIR         = PROJECT_ROOT / "images"
TILE_CROPS_DIR = PROJECT_ROOT / "tile_crops"
DST_ROOT       = PROJECT_ROOT / "dataset_synthetic"
RANDOM_SEED    = 42

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

# Weak classes (low instance count in original dataset)
WEAK_CLASSES = ['5m', '4m', '6m', '4s', '4p', '7p']

RANDOM_LAYOUT_COUNT    = 200
WEAK_BOOST_COUNT       = 120  # total across all weak classes (~20 per class)
MIN_TILE_PX            = 12
CROP_PADDING_PX        = 2
IMG_SIZE               = 640

# ─── Label utilities ──────────────────────────────────────────────────────────

def poly_to_bbox_line(line: str) -> str:
    parts = line.strip().split()
    if len(parts) == 5:
        return line.strip()
    if len(parts) < 5:
        return None
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    xs = coords[0::2]
    ys = coords[1::2]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def load_labels(label_path: Path):
    lines = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            converted = poly_to_bbox_line(line)
            if converted:
                parts = converted.split()
                lines.append([int(parts[0])] + [float(x) for x in parts[1:]])
    if not lines:
        return np.zeros((0, 5))
    return np.array(lines, dtype=np.float32)


def save_labels(labels, label_path: Path):
    with open(label_path, 'w') as f:
        for row in labels:
            cls = int(row[0])
            f.write(f"{cls} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


# ─── IoU helper ───────────────────────────────────────────────────────────────

def compute_iou_pixel(box1, box2):
    """box: [x1, y1, x2, y2] in pixels."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter)


# ─── Step 1: Extract tile crops ───────────────────────────────────────────────

def extract_tile_crops():
    """Extract individual tile crops from annotated training images."""
    print("Step 1: Extracting tile crops...")
    TILE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(34):
        (TILE_CROPS_DIR / str(i)).mkdir(exist_ok=True)

    src_imgs = sorted((SRC_TRAIN / "images").glob("*.jpg"))
    counts = [0] * 34

    for img_path in src_imgs:
        lbl_path = SRC_TRAIN / "labels" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        lbls = load_labels(lbl_path)

        for row in lbls:
            cls = int(row[0])
            bcx, bcy, bw, bh = row[1], row[2], row[3], row[4]
            x1 = int((bcx - bw / 2) * W) - CROP_PADDING_PX
            y1 = int((bcy - bh / 2) * H) - CROP_PADDING_PX
            x2 = int((bcx + bw / 2) * W) + CROP_PADDING_PX
            y2 = int((bcy + bh / 2) * H) + CROP_PADDING_PX
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            cw, ch = x2 - x1, y2 - y1
            if cw < MIN_TILE_PX or ch < MIN_TILE_PX:
                continue
            crop = img[y1:y2, x1:x2]
            save_path = TILE_CROPS_DIR / str(cls) / f"{img_path.stem}_{counts[cls]:04d}.png"
            cv2.imwrite(str(save_path), crop)
            counts[cls] += 1

    total = sum(counts)
    print(f"  Extracted {total} crops across {sum(1 for c in counts if c > 0)} classes")
    for i, (name, cnt) in enumerate(zip(CLASS_NAMES, counts)):
        if cnt < 50:
            print(f"  WARNING: class '{name}' (idx={i}) has only {cnt} crops")
    return counts


def load_tile_crops():
    """Load all tile crops into a dict: {class_idx: [np.ndarray, ...]}"""
    crops = {}
    for i in range(34):
        class_dir = TILE_CROPS_DIR / str(i)
        if not class_dir.exists():
            crops[i] = []
            continue
        imgs = []
        for p in class_dir.glob("*.png"):
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)
        crops[i] = imgs
    return crops


# ─── Step 2: Load backgrounds ─────────────────────────────────────────────────

def load_backgrounds():
    """Load and prepare background images from images/ directory."""
    print("Step 2: Loading backgrounds...")
    backgrounds = []
    for p in BG_DIR.glob("*.png"):
        img = cv2.imread(str(p))
        if img is None:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.shape[2] == 4:
                img = img[:, :, :3]  # drop alpha
        # Resize to 640x640
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        backgrounds.append(img)

    if not backgrounds:
        # Fallback: gray background
        backgrounds = [np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)]

    print(f"  Loaded {len(backgrounds)} backgrounds")
    return backgrounds


# ─── Step 3: Composite image generation ──────────────────────────────────────

def generate_composite(crops_by_class, backgrounds, target_classes=None,
                        min_tiles=20, max_tiles=60, min_required_class_count=5):
    """
    Generate one synthetic 640x640 composite image.

    Args:
        crops_by_class: dict {cls_idx: [crop images]}
        backgrounds: list of background images
        target_classes: if set, these classes must appear >= min_required_class_count times
        min_tiles, max_tiles: total tile count range

    Returns:
        (image, labels_list) where labels_list is [[cls, cx, cy, w, h], ...]
    """
    canvas = random.choice(backgrounds).copy()
    H = W = IMG_SIZE

    # Available class indices (only those with at least 1 crop)
    available = [i for i in range(34) if crops_by_class[i]]
    if not available:
        return canvas, []

    labels = []
    placed_boxes = []  # [x1, y1, x2, y2] pixel coords
    target_count = random.randint(min_tiles, max_tiles)

    # Choose layout mode
    layout = random.choice(['scattered', 'hand_row', 'grid'])

    def get_positions_for_layout(n):
        """Pre-generate candidate positions based on layout mode."""
        positions = []
        if layout == 'scattered':
            for _ in range(n * 3):
                positions.append((random.randint(5, W - 45), random.randint(5, H - 60)))
        elif layout == 'hand_row':
            # Bottom row: y ~ 0.85*H to 0.93*H
            y_base = random.randint(int(H * 0.84), int(H * 0.88))
            for k in range(n * 3):
                x = random.randint(5, W - 45)
                y = y_base + random.randint(-5, 5)
                positions.append((x, y))
        elif layout == 'grid':
            cols = random.randint(5, 8)
            rows = random.randint(3, 6)
            cell_w = W // cols
            cell_h = H // rows
            for r in range(rows):
                for c in range(cols):
                    x = c * cell_w + random.randint(2, max(3, cell_w - 30))
                    y = r * cell_h + random.randint(2, max(3, cell_h - 40))
                    positions.append((x, y))
            random.shuffle(positions)
            positions = positions * 3
        return positions

    positions = get_positions_for_layout(target_count)

    # First pass: ensure target_classes appear if specified
    classes_to_place = []
    if target_classes:
        for tc in target_classes:
            for _ in range(min_required_class_count):
                if crops_by_class[tc]:
                    classes_to_place.append(tc)

    # Fill rest randomly
    remaining = max(0, target_count - len(classes_to_place))
    for _ in range(remaining):
        classes_to_place.append(random.choice(available))
    random.shuffle(classes_to_place)

    pos_idx = 0
    for cls in classes_to_place:
        if not crops_by_class[cls]:
            continue
        if pos_idx >= len(positions):
            break

        crop = random.choice(crops_by_class[cls])
        scale = random.uniform(0.8, 1.4)
        ch, cw = crop.shape[:2]
        new_w = max(MIN_TILE_PX, int(cw * scale))
        new_h = max(MIN_TILE_PX, int(ch * scale))
        resized = cv2.resize(crop, (new_w, new_h))

        # Try positions until no significant overlap
        placed = False
        for attempt in range(8):
            if pos_idx + attempt >= len(positions):
                break
            px, py = positions[pos_idx + attempt]
            # Ensure within canvas
            px = min(px, W - new_w - 2)
            py = min(py, H - new_h - 2)
            px = max(2, px)
            py = max(2, py)
            x2 = px + new_w
            y2 = py + new_h
            if x2 > W or y2 > H:
                continue
            box = [px, py, x2, y2]
            # Check overlap
            overlap = any(compute_iou_pixel(box, pb) > 0.15 for pb in placed_boxes)
            if not overlap:
                canvas[py:y2, px:x2] = resized
                placed_boxes.append(box)
                # YOLO normalized label
                cx_n = (px + x2) / 2 / W
                cy_n = (py + y2) / 2 / H
                w_n  = new_w / W
                h_n  = new_h / H
                labels.append([cls, cx_n, cy_n, w_n, h_n])
                placed = True
                pos_idx += attempt + 1
                break

        if not placed:
            pos_idx += 1

    return canvas, labels


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dst_imgs = DST_ROOT / "train/images"
    dst_lbls = DST_ROOT / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    # Extract crops
    extract_tile_crops()

    # Load crops and backgrounds
    crops_by_class = load_tile_crops()
    backgrounds = load_backgrounds()

    total_crops = sum(len(v) for v in crops_by_class.values())
    print(f"  Total crops loaded: {total_crops}")

    # Step 3: Generate random-layout composites
    print(f"Step 3: Generating {RANDOM_LAYOUT_COUNT} random composite images...")
    for k in range(RANDOM_LAYOUT_COUNT):
        img, labels = generate_composite(
            crops_by_class, backgrounds,
            target_classes=None,
            min_tiles=20, max_tiles=55
        )
        name = f"synth_{k:04d}"
        cv2.imwrite(str(dst_imgs / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_labels(np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5)),
                    dst_lbls / f"{name}.txt")

    print(f"  Generated {RANDOM_LAYOUT_COUNT} random synthetic images")

    # Step 4: Generate weak-class-boosted composites
    print(f"Step 4: Generating {WEAK_BOOST_COUNT} weak-class-boosted images...")
    weak_indices = [CLASS_NAMES.index(c) for c in WEAK_CLASSES if c in CLASS_NAMES]
    per_class_boost = WEAK_BOOST_COUNT // max(1, len(weak_indices))

    boost_count = 0
    for cls_idx in weak_indices:
        cls_name = CLASS_NAMES[cls_idx]
        if not crops_by_class[cls_idx]:
            print(f"  WARNING: no crops for '{cls_name}', skipping boost")
            continue
        for k in range(per_class_boost):
            img, labels = generate_composite(
                crops_by_class, backgrounds,
                target_classes=[cls_idx],
                min_tiles=15, max_tiles=45,
                min_required_class_count=5
            )
            name = f"boost_{cls_name}_{k:04d}"
            cv2.imwrite(str(dst_imgs / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            save_labels(np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5)),
                        dst_lbls / f"{name}.txt")
            boost_count += 1

    print(f"  Generated {boost_count} weak-class-boosted images")

    # Step 5: Write data.yaml
    total = len(list(dst_imgs.glob("*.jpg")))
    yaml_content = f"""path: {DST_ROOT.as_posix()}
train: train/images
val: {(PROJECT_ROOT / "majsoul.v4i.yolov8/valid/images").as_posix()}
test: {(PROJECT_ROOT / "majsoul.v4i.yolov8/test/images").as_posix()}

nc: 34
names: {CLASS_NAMES}
"""
    with open(DST_ROOT / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nDone! Total synthetic training images: {total}")
    print(f"Output: {DST_ROOT}")


if __name__ == "__main__":
    main()
