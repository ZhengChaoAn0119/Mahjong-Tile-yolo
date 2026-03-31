"""
augment_data.py
Expand 42 training images to ~480 via offline data augmentation.

Steps:
  1. Copy original train split to dataset_augmented/train/ and fix polygon labels
  2. Generate 9 augmented variants per original image
  3. Generate 60 mosaic images (4-image composites)

Usage:
  python augment_data.py

Output: E:/project/majsoul_yolo/dataset_augmented/
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("E:/project/majsoul_yolo")
SRC_TRAIN    = PROJECT_ROOT / "majsoul.v4i.yolov8/train"
DST_ROOT     = PROJECT_ROOT / "dataset_augmented"
RANDOM_SEED  = 42
AUG_PER_IMG  = 9   # augmented variants per original image
MOSAIC_COUNT = 60  # total mosaic images to generate

CLASS_NAMES = [
    '1m','1p','1s','2m','2p','2s','3m','3p','3s',
    '4m','4p','4s','5m','5p','5s','6m','6p','6s',
    '7m','7p','7s','8m','8p','8s','9m','9p','9s',
    'east','green','north','red','south','west','white'
]

# ─── Label utilities ──────────────────────────────────────────────────────────

def poly_to_bbox_line(line: str) -> str:
    """Convert a YOLO polygon annotation line to a 5-column bbox line."""
    parts = line.strip().split()
    if len(parts) == 5:
        return line.strip()
    if len(parts) < 5:
        return None
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    if len(coords) % 2 != 0:
        coords = coords[:-1]  # drop trailing odd value
    xs = coords[0::2]
    ys = coords[1::2]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def load_labels(label_path: Path):
    """Load YOLO labels as numpy array [N, 5]: cls cx cy w h."""
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


def save_labels(labels: np.ndarray, label_path: Path):
    """Save labels array to YOLO txt format."""
    with open(label_path, 'w') as f:
        for row in labels:
            cls = int(row[0])
            f.write(f"{cls} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


# ─── Bounding box transforms ──────────────────────────────────────────────────

def clip_bboxes(bboxes: np.ndarray, min_wh: float = 0.003) -> np.ndarray:
    """Clip cx,cy,w,h to valid range and remove tiny boxes."""
    if len(bboxes) == 0:
        return bboxes
    # Clip center and size
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, 1)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, 1)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, 1)
    bboxes[:, 4] = np.clip(bboxes[:, 4], 0, 1)
    # Keep only boxes larger than min_wh
    mask = (bboxes[:, 3] >= min_wh) & (bboxes[:, 4] >= min_wh)
    return bboxes[mask]


def rotate_image_bboxes(img, bboxes, angle_deg):
    """Rotate image and transform bboxes. Returns (img, bboxes)."""
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=128)
    if len(bboxes) == 0:
        return rotated, bboxes

    new_bboxes = []
    for row in bboxes:
        cls, bcx, bcy, bw, bh = row
        # 4 corners in pixel space
        x1 = (bcx - bw / 2) * W
        y1 = (bcy - bh / 2) * H
        x2 = (bcx + bw / 2) * W
        y2 = (bcy + bh / 2) * H
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        # Apply rotation to corners
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated_corners = (M @ corners_h.T).T  # shape (4,2)
        rx_min, rx_max = rotated_corners[:, 0].min(), rotated_corners[:, 0].max()
        ry_min, ry_max = rotated_corners[:, 1].min(), rotated_corners[:, 1].max()
        new_cx = (rx_min + rx_max) / 2 / W
        new_cy = (ry_min + ry_max) / 2 / H
        new_w  = (rx_max - rx_min) / W
        new_h  = (ry_max - ry_min) / H
        new_bboxes.append([cls, new_cx, new_cy, new_w, new_h])

    new_bboxes = np.array(new_bboxes, dtype=np.float32)
    return rotated, clip_bboxes(new_bboxes)


def random_crop_image_bboxes(img, bboxes, crop_ratio):
    """Randomly crop image to (crop_ratio * original) and resize back to 640."""
    H, W = img.shape[:2]
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    x_start = random.randint(0, W - crop_w)
    y_start = random.randint(0, H - crop_h)
    cropped = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
    resized = cv2.resize(cropped, (W, H))

    if len(bboxes) == 0:
        return resized, bboxes

    # Normalized crop boundaries
    x1n = x_start / W
    y1n = y_start / H
    new_bboxes = []
    for row in bboxes:
        cls, bcx, bcy, bw, bh = row
        # Transform to crop-relative coords
        new_cx = (bcx - x1n) / crop_ratio
        new_cy = (bcy - y1n) / crop_ratio
        new_w  = bw / crop_ratio
        new_h  = bh / crop_ratio
        # Keep only if center is inside crop
        if 0 < new_cx < 1 and 0 < new_cy < 1:
            new_bboxes.append([cls, new_cx, new_cy, new_w, new_h])

    if not new_bboxes:
        return resized, np.zeros((0, 5), dtype=np.float32)
    return resized, clip_bboxes(np.array(new_bboxes, dtype=np.float32))


def zoom_image_bboxes(img, bboxes, scale):
    """Zoom in (scale>1) or out (scale<1) keeping center. Resize back to 640."""
    H, W = img.shape[:2]
    if scale > 1.0:
        # Zoom in: crop center region then resize up
        crop_w = int(W / scale)
        crop_h = int(H / scale)
        x_start = (W - crop_w) // 2
        y_start = (H - crop_h) // 2
        cropped = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
        result = cv2.resize(cropped, (W, H))
        x1n = x_start / W
        y1n = y_start / H
        crop_ratio_w = crop_w / W
        crop_ratio_h = crop_h / H
    else:
        # Zoom out: shrink image and pad with gray
        new_w = int(W * scale)
        new_h = int(H * scale)
        small = cv2.resize(img, (new_w, new_h))
        result = np.full((H, W, img.shape[2] if img.ndim == 3 else 1), 128, dtype=np.uint8)
        if img.ndim == 3:
            result = np.full((H, W, img.shape[2]), 128, dtype=np.uint8)
        x_start = (W - new_w) // 2
        y_start = (H - new_h) // 2
        result[y_start:y_start+new_h, x_start:x_start+new_w] = small
        # Bbox transform: shrink and shift
        if len(bboxes) == 0:
            return result, bboxes
        new_bboxes = bboxes.copy()
        new_bboxes[:, 1] = bboxes[:, 1] * scale + x_start / W
        new_bboxes[:, 2] = bboxes[:, 2] * scale + y_start / H
        new_bboxes[:, 3] = bboxes[:, 3] * scale
        new_bboxes[:, 4] = bboxes[:, 4] * scale
        return result, clip_bboxes(new_bboxes)

    if len(bboxes) == 0:
        return result, bboxes
    new_bboxes = []
    for row in bboxes:
        cls, bcx, bcy, bw, bh = row
        new_cx = (bcx - x1n) / crop_ratio_w
        new_cy = (bcy - y1n) / crop_ratio_h
        new_w_  = bw / crop_ratio_w
        new_h_  = bh / crop_ratio_h
        if 0 < new_cx < 1 and 0 < new_cy < 1:
            new_bboxes.append([cls, new_cx, new_cy, new_w_, new_h_])
    if not new_bboxes:
        return result, np.zeros((0, 5), dtype=np.float32)
    return result, clip_bboxes(np.array(new_bboxes, dtype=np.float32))


# ─── Pixel-level augmentations ────────────────────────────────────────────────

def aug_brightness_contrast(img, alpha, beta):
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def aug_gaussian_blur(img, ksize=3, sigma=1.0):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def aug_gaussian_noise(img, std=8.0):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def aug_jpeg_compression(img, quality=75):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def aug_sharpen(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)


# ─── Mosaic ───────────────────────────────────────────────────────────────────

def make_mosaic(images, labels_list, out_size=640):
    """
    Combine 4 random images into a single mosaic.
    Returns (mosaic_image, mosaic_labels).
    """
    H = W = out_size
    # Random cut point (35%-65% range)
    cut_x = random.randint(int(W * 0.35), int(W * 0.65))
    cut_y = random.randint(int(H * 0.35), int(H * 0.65))

    # Quadrant sizes: TL, TR, BL, BR
    quads = [
        (0,      0,      cut_x,   cut_y),   # TL: x from 0..cut_x, y from 0..cut_y
        (cut_x,  0,      W,       cut_y),   # TR
        (0,      cut_y,  cut_x,   H),       # BL
        (cut_x,  cut_y,  W,       H),       # BR
    ]

    mosaic = np.full((H, W, 3), 128, dtype=np.uint8)
    all_bboxes = []

    for i, (qx1, qy1, qx2, qy2) in enumerate(quads):
        idx = random.randint(0, len(images) - 1)
        img = images[idx]
        lbls = labels_list[idx].copy()

        qw = qx2 - qx1
        qh = qy2 - qy1
        resized = cv2.resize(img, (qw, qh))
        mosaic[qy1:qy2, qx1:qx2] = resized

        if len(lbls) == 0:
            continue

        # Transform bboxes from original normalized to mosaic normalized
        new_lbls = lbls.copy()
        new_lbls[:, 1] = lbls[:, 1] * (qw / W) + qx1 / W   # cx
        new_lbls[:, 2] = lbls[:, 2] * (qh / H) + qy1 / H   # cy
        new_lbls[:, 3] = lbls[:, 3] * (qw / W)              # w
        new_lbls[:, 4] = lbls[:, 4] * (qh / H)              # h
        all_bboxes.append(new_lbls)

    if all_bboxes:
        all_bboxes = np.vstack(all_bboxes)
        all_bboxes = clip_bboxes(all_bboxes)
    else:
        all_bboxes = np.zeros((0, 5), dtype=np.float32)

    # Apply one pixel augmentation on top
    mosaic = aug_brightness_contrast(mosaic,
                                     alpha=random.uniform(0.85, 1.2),
                                     beta=random.uniform(-15, 15))
    return mosaic, all_bboxes


# ─── Augmentation pipeline ────────────────────────────────────────────────────

def get_augmentation_pipeline():
    """
    Returns list of (transform_fn) callables.
    Each takes (img, bboxes) and returns (img, bboxes).
    9 variants total.
    """
    def aug1(img, bb):  # rotate +3 + mild brightness
        img, bb = rotate_image_bboxes(img, bb, 3.0)
        img = aug_brightness_contrast(img, 1.1, 10)
        return img, bb

    def aug2(img, bb):  # rotate -4 + noise
        img, bb = rotate_image_bboxes(img, bb, -4.0)
        img = aug_gaussian_noise(img, std=8.0)
        return img, bb

    def aug3(img, bb):  # crop 0.85 + brightness
        img, bb = random_crop_image_bboxes(img, bb, 0.85)
        img = aug_brightness_contrast(img, 1.05, -10)
        return img, bb

    def aug4(img, bb):  # crop 0.90 + jpeg compression
        img, bb = random_crop_image_bboxes(img, bb, 0.90)
        img = aug_jpeg_compression(img, quality=random.randint(60, 80))
        return img, bb

    def aug5(img, bb):  # zoom in 1.15 + blur
        img, bb = zoom_image_bboxes(img, bb, 1.15)
        img = aug_gaussian_blur(img, ksize=3, sigma=random.uniform(0.5, 1.2))
        return img, bb

    def aug6(img, bb):  # zoom out 0.88 + noise
        img, bb = zoom_image_bboxes(img, bb, 0.88)
        img = aug_gaussian_noise(img, std=6.0)
        return img, bb

    def aug7(img, bb):  # dark + noise
        img = aug_brightness_contrast(img, 0.70, -20)
        img = aug_gaussian_noise(img, std=10.0)
        return img, bb

    def aug8(img, bb):  # bright + jpeg
        img = aug_brightness_contrast(img, 1.30, 20)
        img = aug_jpeg_compression(img, quality=random.randint(65, 85))
        return img, bb

    def aug9(img, bb):  # sharpen + slight rotation
        img = aug_sharpen(img)
        img, bb = rotate_image_bboxes(img, bb, 2.0)
        return img, bb

    return [aug1, aug2, aug3, aug4, aug5, aug6, aug7, aug8, aug9]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dst_imgs = DST_ROOT / "train/images"
    dst_lbls = DST_ROOT / "train/labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    src_imgs = sorted((SRC_TRAIN / "images").glob("*.jpg"))
    src_lbls = [SRC_TRAIN / "labels" / (p.stem + ".txt") for p in src_imgs]

    print(f"Found {len(src_imgs)} original training images")

    # Step 1: Copy originals (with polygon→bbox conversion)
    print("Step 1: Copying originals and fixing polygon labels...")
    all_images = []
    all_labels = []
    for img_path, lbl_path in zip(src_imgs, src_lbls):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: cannot read {img_path}")
            continue
        lbls = load_labels(lbl_path) if lbl_path.exists() else np.zeros((0, 5))

        # Save original to dst
        dst_name = img_path.stem
        cv2.imwrite(str(dst_imgs / f"{dst_name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_labels(lbls, dst_lbls / f"{dst_name}.txt")

        all_images.append(img)
        all_labels.append(lbls)

    print(f"  Copied {len(all_images)} images")

    # Step 2: Generate 9 augmented variants per image
    print(f"Step 2: Generating {AUG_PER_IMG} augmented variants per image...")
    aug_fns = get_augmentation_pipeline()
    aug_count = 0
    for i, (img, lbls) in enumerate(zip(all_images, all_labels)):
        for j, aug_fn in enumerate(aug_fns):
            aug_img, aug_lbls = aug_fn(img.copy(), lbls.copy() if len(lbls) > 0 else lbls)
            if len(aug_lbls) < 5:  # skip if too few boxes survived
                # Fall back: use brightness-only variant
                aug_img = aug_brightness_contrast(img.copy(),
                                                  alpha=random.uniform(0.85, 1.15),
                                                  beta=random.uniform(-10, 10))
                aug_lbls = lbls.copy()

            stem = src_imgs[i].stem
            name = f"{stem}_aug{j+1}"
            cv2.imwrite(str(dst_imgs / f"{name}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            save_labels(aug_lbls, dst_lbls / f"{name}.txt")
            aug_count += 1

    print(f"  Generated {aug_count} augmented images")

    # Step 3: Generate mosaic images
    print(f"Step 3: Generating {MOSAIC_COUNT} mosaic images...")
    mosaic_count = 0
    for k in range(MOSAIC_COUNT):
        mosaic_img, mosaic_lbls = make_mosaic(all_images, all_labels)
        name = f"mosaic_{k:04d}"
        cv2.imwrite(str(dst_imgs / f"{name}.jpg"), mosaic_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_labels(mosaic_lbls, dst_lbls / f"{name}.txt")
        mosaic_count += 1

    print(f"  Generated {mosaic_count} mosaic images")

    # Step 4: Write data.yaml
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

    print(f"\nDone! Total training images: {total}")
    print(f"Output: {DST_ROOT}")
    print(f"data.yaml written to: {DST_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
