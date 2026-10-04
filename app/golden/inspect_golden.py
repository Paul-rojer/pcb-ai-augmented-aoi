"""
inspect_golden.py
-----------------
Inspect one raw AOI image: PASS / REVIEW / FAIL + anomaly heatmap.

    python golden/inspect_golden.py path/to/board.jpg --overlay out.png
"""
import argparse
import os

import cv2

from golden_inspector import GoldenInspector, heatmap_overlay, REF_SIZE
from aoi_profile_crop import crop_image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--overlay", default=None)
    args = ap.parse_args()

    insp = GoldenInspector(golden_path=os.path.join(_ROOT, "golden", "golden.npz"))
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit("cannot read image")

    # auto-detect: crop only if the image still has conveyor rails
    from aoi_profile_crop import needs_crop
    will_crop = needs_crop(bgr)
    res, zmap = insp.inspect(bgr, do_crop="auto")
    print({**res, "cropped": will_crop})

    if args.overlay and zmap is not None:
        base = crop_image(bgr)[0] if will_crop else bgr
        cv2.imwrite(args.overlay, heatmap_overlay(base, zmap))
        print(f"overlay -> {args.overlay}")


if __name__ == "__main__":
    main()
