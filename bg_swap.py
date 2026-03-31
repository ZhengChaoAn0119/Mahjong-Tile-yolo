"""
bg_swap.py
Background-swapping augmentation:
  1. From each source image, extract all tile crops (via YOLO bbox labels).
  2. Randomly paste those crops onto backgrounds taken from OTHER game screenshots.
  3. Output new images + YOLO labels, then split 80/20 into train/val.

Output layout:
  dataset_bgswap/
    train/images/ & train/labels/
    val/images/   & val/labels/
    data.yaml

Usage:
  python bg_swap.py
"""

import random, shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
SRC_DIR      = PROJECT_ROOT / "majsoul.v4i.yolov8/train"   # 42 real images
VAL_SRC_DIR  = PROJECT_ROOT / "majsoul.v4i.yolov8/valid"   # 12 real images
OUT_DIR      = PROJECT_ROOT / "dataset_bgswap"

COPIES_PER_SRC  = 10    # how many bg-swap variants per source image
VAL_SPLIT       = 0.20  # 20% → val
TILE_SCALE_JITTER = 0.15  # ±15% random scale of pasted tiles
MAX_TILES_PER_IMG = 20   # cap pasted tiles to avoid clutter
MIN_TILE_VIS      = 0.75  # require ≥75% tile area visible (inside canvas)
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_labels(lbl_path: Path):
    """Return list of (class_id, cx, cy, w, h) normalised."""
    if not lbl_path.exists():
        return []
    lines = lbl_path.read_text().strip().splitlines()
    boxes = []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *[float(x) for x in parts[1:]]))
    return boxes


def crop_tile(img: np.ndarray, cx, cy, w, h):
    """Crop a tile from normalised YOLO coords; return the crop."""
    H, W = img.shape[:2]
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def paste_tile(canvas: np.ndarray, tile: np.ndarray, cx_norm, cy_norm, scale):
    """
    Paste tile onto canvas at a position derived from cx_norm, cy_norm
    with a random scale jitter.  Returns (new_cx, new_cy, new_w, new_h)
    in normalised coords, or None if visibility check fails.
    """
    H, W = canvas.shape[:2]
    th, tw = tile.shape[:2]
    new_tw = int(tw * scale)
    new_th = int(th * scale)
    if new_tw < 4 or new_th < 4:
        return None

    tile_resized = cv2.resize(tile, (new_tw, new_th))

    # Target paste position
    px = int(cx_norm * W - new_tw / 2)
    py = int(cy_norm * H - new_th / 2)

    # Clamp to canvas
    src_x1 = max(0, -px)
    src_y1 = max(0, -py)
    dst_x1 = max(0, px)
    dst_y1 = max(0, py)
    src_x2 = min(new_tw, W - px)
    src_y2 = min(new_th, H - py)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return None

    visible_area = (src_x2 - src_x1) * (src_y2 - src_y1)
    total_area   = new_tw * new_th
    if visible_area / total_area < MIN_TILE_VIS:
        return None

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = tile_resized[src_y1:src_y2, src_x1:src_x2]

    # Normalised coords of the VISIBLE bounding box centre
    vis_cx = (dst_x1 + dst_x2) / 2 / W
    vis_cy = (dst_y1 + dst_y2) / 2 / H
    vis_w  = (dst_x2 - dst_x1) / W
    vis_h  = (dst_y2 - dst_y1) / H
    return vis_cx, vis_cy, vis_w, vis_h


def build_sample_pool(src_dirs):
    """Return list of (image_path, label_path) pairs."""
    pool = []
    for d in src_dirs:
        img_dir = d / "images"
        lbl_dir = d / "labels"
        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            pool.append((img_path, lbl_path))
    return pool


def generate(pool, out_dir: Path, copies_per_src: int, rng: random.Random):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    generated = 0
    for img_path, lbl_path in pool:
        src_img  = cv2.imread(str(img_path))
        if src_img is None:
            continue
        src_boxes = load_labels(lbl_path)
        if not src_boxes:
            continue

        for copy_idx in range(copies_per_src):
            # Pick a different image as background
            bg_path, _ = rng.choice(pool)
            while bg_path == img_path and len(pool) > 1:
                bg_path, _ = rng.choice(pool)
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            bg = cv2.resize(bg, (src_img.shape[1], src_img.shape[0]))
            canvas = bg.copy()

            # Randomly jitter tile positions & scale
            tiles_to_paste = src_boxes[:MAX_TILES_PER_IMG]
            rng.shuffle(tiles_to_paste)
            new_labels = []

            for cls_id, cx, cy, w, h in tiles_to_paste:
                tile_crop = crop_tile(src_img, cx, cy, w, h)
                if tile_crop is None:
                    continue

                # Position jitter ±10%
                jx = rng.uniform(-0.05, 0.05)
                jy = rng.uniform(-0.05, 0.05)
                scale = rng.uniform(1 - TILE_SCALE_JITTER, 1 + TILE_SCALE_JITTER)

                result = paste_tile(canvas, tile_crop, cx + jx, cy + jy, scale)
                if result:
                    vis_cx, vis_cy, vis_w, vis_h = result
                    new_labels.append(f"{cls_id} {vis_cx:.6f} {vis_cy:.6f} {vis_w:.6f} {vis_h:.6f}")

            if not new_labels:
                continue

            name = f"bgswap_{img_path.stem}_c{copy_idx:03d}.jpg"
            cv2.imwrite(str(img_out / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_out / name.replace(".jpg", ".txt")).write_text("\n".join(new_labels))
            generated += 1

    return generated


def main():
    rng = random.Random(SEED)

    src_dirs = [SRC_DIR, VAL_SRC_DIR]
    pool = build_sample_pool(src_dirs)
    print(f"Source pool: {len(pool)} images")

    # ── Generate all swapped images into a staging dir ──────────────────────
    stage_dir = OUT_DIR / "_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    n = generate(pool, stage_dir, COPIES_PER_SRC, rng)
    print(f"Generated {n} bg-swapped images")

    # ── Train / val split ───────────────────────────────────────────────────
    all_imgs = sorted((stage_dir / "images").glob("*.jpg"))
    rng.shuffle(all_imgs)
    split = int(len(all_imgs) * (1 - VAL_SPLIT))
    splits = {"train": all_imgs[:split], "val": all_imgs[split:]}

    for split_name, img_list in splits.items():
        img_dst = OUT_DIR / split_name / "images"
        lbl_dst = OUT_DIR / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_p in img_list:
            shutil.move(str(img_p), img_dst / img_p.name)
            lbl_src = stage_dir / "labels" / img_p.with_suffix(".txt").name
            if lbl_src.exists():
                shutil.move(str(lbl_src), lbl_dst / lbl_src.name)
        print(f"  {split_name}: {len(img_list)} images")

    shutil.rmtree(stage_dir)

    # ── data.yaml ────────────────────────────────────────────────────────────
    orig_yaml = PROJECT_ROOT / "majsoul.v4i.yolov8/data.yaml"
    with open(orig_yaml) as f:
        orig_cfg = yaml.safe_load(f)

    cfg = {
        "path":  OUT_DIR.as_posix(),
        "train": (OUT_DIR / "train/images").as_posix(),
        "val":   (OUT_DIR / "val/images").as_posix(),
        "nc":    orig_cfg["nc"],
        "names": orig_cfg["names"],
    }
    with open(OUT_DIR / "data.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)

    print(f"\ndata.yaml saved to {OUT_DIR / 'data.yaml'}")
    print("Done.")


if __name__ == "__main__":
    main()
