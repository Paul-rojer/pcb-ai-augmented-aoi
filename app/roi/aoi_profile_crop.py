"""
aoi_profile_crop.py
-------------------
ROI cropper for *AOI machine* PCB images.

Why the old roi_extractor.py failed
-----------------------------------
AOI captures have a teal/blue cast over the WHOLE frame (board AND background),
so HSV colour thresholding selects ~the entire image. The "largest contour"
bounding box then spans ~0.98 of the frame -> the rails are never removed.

This approach instead uses a brightness PROJECTION PROFILE:
the PCB sits in the bright central region; the conveyor rails / fixtures are
dark bands top & bottom. Otsu on the row/column brightness profile finds the
board's extent. On the current AOI set this crops 97/98 images to ~0.60 of the
frame with the rails removed; the rare failure is flagged for manual review
instead of being silently saved.

Usage
-----
    python roi/aoi_profile_crop.py \
        --input  datasets/raw/aoi/good \
        --output datasets/good_dataset_aoi/all/good

Outliers (frac < MIN_FRAC or > MAX_FRAC) go to <output>/_review/ for you to
crop by hand.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

MIN_FRAC = 0.40   # below this, the crop is suspiciously small -> review
MAX_FRAC = 0.85   # above this, it probably failed to remove rails -> review


def needs_crop(bgr, thresh=0.78):
    """True if the image still has conveyor rails (raw AOI) and should be cropped.

    Raw AOI captures have dark rail bands at top/bottom, so the outer border is
    much darker than the centre (ratio ~0.6). Already-cropped boards fill the
    frame (ratio ~0.9-1.1). A 0.78 threshold separates them cleanly.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    b = max(1, int(0.08 * h))
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel()])
    center = g[h // 3:2 * h // 3, w // 3:2 * w // 3]
    return float(border.mean() / (center.mean() + 1e-6)) < thresh


def _extent(prof):
    """Otsu threshold on a 1-D brightness profile -> (start, end) indices."""
    rng = float(prof.max() - prof.min())
    p = ((prof - prof.min()) / (rng + 1e-6) * 255).astype("uint8")
    t, _ = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    idx = np.where(p >= t)[0]
    return int(idx.min()), int(idx.max())


def profile_crop_box(img, proc=700):
    """Return (x, y, w, h) board bounding box via brightness projection.

    For speed the profile is computed on a `proc`x`proc` downscaled copy, then
    the box is scaled back to the original resolution (crop stays full-res)."""
    H, W = img.shape[:2]
    small = cv2.resize(img, (proc, proc))
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    ys, ye = _extent(g.mean(axis=1))
    xs, xe = _extent(g.mean(axis=0))
    x0 = int(xs / proc * W); x1 = int(xe / proc * W)
    y0 = int(ys / proc * H); y1 = int(ye / proc * H)
    return x0, y0, x1 - x0, y1 - y0


def crop_image(img):
    x, y, w, h = profile_crop_box(img)
    frac = (w * h) / (img.shape[0] * img.shape[1])
    return img[y:y + h, x:x + w], frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    review = out / "_review"
    out.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)

    saved = flagged = unreadable = 0
    for f in sorted(Path(args.input).glob("*.*")):
        img = cv2.imread(str(f))
        if img is None:
            unreadable += 1
            continue
        crop, frac = crop_image(img)
        if frac < MIN_FRAC or frac > MAX_FRAC or crop.size == 0:
            # Save the ORIGINAL to _review so you can crop it by hand.
            cv2.imwrite(str(review / f.name), img)
            flagged += 1
        else:
            cv2.imwrite(str(out / f.name), crop)
            saved += 1

    print("=" * 44)
    print(f"Input        : {args.input}")
    print(f"Cropped OK   : {saved}")
    print(f"Flagged      : {flagged}  -> {review}  (crop these by hand)")
    print(f"Unreadable   : {unreadable}")
    print("=" * 44)


if __name__ == "__main__":
    main()
