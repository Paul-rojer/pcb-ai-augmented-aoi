"""
calibrate_gallery.py
--------------------
Set the per-type PASS/FAIL thresholds using BOTH good and defect boards, so:
  - PASS  zone covers only clearly-good boards (few defects slip in),
  - FAIL  zone covers only clearly-defective boards,
  - the uncertain overlap -> REVIEW (human).

build_gallery.py sets tau_pass from good boards only (too lenient -> defects pass).
This script fixes that with your defect data.

    python golden/calibrate_gallery.py                # default: <=15% defects may PASS
    python golden/calibrate_gallery.py --max-miss 0.05  # stricter (catch more defects)

Higher --max-miss  = more boards auto-PASS, more missed defects.
Lower  --max-miss  = fewer missed defects, more boards sent to REVIEW.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from multi_golden import MultiGolden

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect(mg, folder):
    """Route each image to its type and score it. Returns {type_name: [scores]}."""
    by_type = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        res, _ = mg.inspect(bgr, do_crop="auto")
        if res.get("score") is None:
            continue
        by_type.setdefault(res["type"], []).append(res["score"])
    return by_type


def merge(a, b):
    for k, v in b.items():
        a.setdefault(k, []).extend(v)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_miss", type=float, default=0.15,
                    help="max fraction of a type's defects allowed into PASS")
    args = ap.parse_args()

    gallery = os.path.join(_ROOT, "golden", "gallery")
    mg = MultiGolden(gallery_dir=gallery)

    good, defect = {}, {}
    for split in ("val", "test"):
        merge(good, collect(mg, os.path.join(_ROOT, "datasets", "good_dataset_aoi", split, "good")))
        merge(defect, collect(mg, os.path.join(_ROOT, "datasets", "defect_dataset_aoi", split, "defect")))

    for t in mg.templates:
        g = sorted(good.get(t.name, []))
        d = sorted(defect.get(t.name, []))
        if not d:
            print(f"{t.name}: no defect samples routed here -> keeping tau_pass={t.tau_pass:.2f}")
            continue
        # tau_pass so only the lowest `max_miss` fraction of defects fall in PASS
        tau_pass = float(np.percentile(d, args.max_miss * 100))
        # tau_fail: where good boards rarely reach (90th pct of good), above tau_pass
        tau_fail = float(np.percentile(g, 90)) if g else tau_pass * 1.15
        if tau_fail <= tau_pass:
            tau_fail = tau_pass * 1.10
        t.tau_pass, t.tau_fail = tau_pass, tau_fail

        def buckets(scores):
            c = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
            for s in scores:
                c[t.decide(s)] += 1
            return c
        gc, dc = buckets(g), buckets(d)
        print(f"{t.name}: good n={len(g)} {gc} | defect n={len(d)} {dc} "
              f"| tau_pass={tau_pass:.2f} tau_fail={tau_fail:.2f}")

    mg.save(gallery)
    print("\nSaved updated thresholds to", gallery)


if __name__ == "__main__":
    main()
