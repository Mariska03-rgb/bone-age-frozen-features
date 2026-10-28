"""Preprocessing used in the paper.

Four variants are produced from each hand radiograph:
  raw      : polarity-corrected image, letterboxed to 224 x 224 (baseline)
  enh      : enhancement package (min-max stretch + 3x3 median filter + CLAHE)
  roi      : ROI crop (Otsu + morphology + largest component + 5% margin)
  enh_roi  : enhancement package with the same ROI crop
"""
import cv2
import numpy as np

IMG_SIZE = 224    # input size of the CNN backbones
WORK_SIDE = 512   # longer side after the initial downscaling


def fix_inversion(img):
    """Reverse intensities when the median of a 5-pixel border frame exceeds 127."""
    border = np.concatenate([img[:5].ravel(), img[-5:].ravel(), img[:, :5].ravel(), img[:, -5:].ravel()])
    if np.median(border) > 127:
        return 255 - img, 1
    return img, 0


def letterbox(img, size=IMG_SIZE):
    """Resize while preserving the aspect ratio and pad the remainder with black."""
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((size, size), np.uint8)
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def enhance(img):
    """Enhancement package: min-max normalization, 3x3 median filter, CLAHE (clip 2.0, tile 8x8)."""
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.medianBlur(img, 3)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def roi_box(img, margin=0.05, min_area=0.05):
    """Otsu -> closing/opening (15x15 ellipse) -> largest component -> bounding box + margin.
    Returns None when no component covers at least `min_area` of the image."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    j = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, yy, w, h, a = stats[j]
    if a < min_area * img.shape[0] * img.shape[1]:
        return None
    mx, my = int(margin * w), int(margin * h)
    x0, y0 = max(0, x - mx), max(0, yy - my)
    x1, y1 = min(img.shape[1], x + w + mx), min(img.shape[0], yy + h + my)
    return x0, y0, x1, y1


def process_one(path):
    """Return the four variants, the inversion flag, and the ROI fallback flag for one image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    s = WORK_SIDE / max(img.shape)
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    img, inv = fix_inversion(img)
    enh = enhance(img)
    box = roi_box(img)          # the ROI is computed on the non-enhanced image
    fb = 0
    if box is None:
        box, fb = (0, 0, img.shape[1], img.shape[0]), 1
    x0, y0, x1, y1 = box
    out = {"raw": letterbox(img), "enh": letterbox(enh),
           "roi": letterbox(img[y0:y1, x0:x1]), "enh_roi": letterbox(enh[y0:y1, x0:x1])}
    return out, inv, fb
