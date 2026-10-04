"""
evaluate_gallery.py
-------------------
Honest final evaluation of the multi-template golden pipeline.

  - Thresholds are set on the VALIDATION set (good+defect, per type).
  - Performance is reported on the HELD-OUT TEST set only.

This avoids the optimistic bias of tuning and testing on the same images.

    python golden/evaluate_gallery.py                 # max_miss=0.05 (strict on defects)
    python golden/evaluate_gallery.py --max_miss 0.15

Prints, per board type and overall:
  GOOD  : PASS / REVIEW / FAIL
  DEFECT: PASS / REVIEW / FAIL
and the headline metrics: good auto-pass %, defect catch %, missed-defect %,
false-reject %, and human-review workload.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from multi_golden import MultiGolden

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_folder(mg, folder):
    """Route + score every image. Returns list of (type_name_or_None, score_or_None)."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        res, _ = mg.inspect(bgr, do_crop="auto")
        out.append((res.get("type"), res.get("score")))
    return out


def gp(_ROOT, kind, split, cls):
    return os.path.join(_ROOT, "datasets", f"{kind}_dataset_aoi", split, cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_miss", type=float, default=0.05)
    args = ap.parse_args()
    mg = MultiGolden(gallery_dir=os.path.join(_ROOT, "golden", "gallery"))

    # --- 1. set thresholds on VAL ---
    vg = score_folder(mg, gp(_ROOT, "good", "val", "good"))
    vd = score_folder(mg, gp(_ROOT, "defect", "val", "defect"))
    for t in mg.templates:
        d = sorted(s for ty, s in vd if ty == t.name and s is not None)
        g = sorted(s for ty, s in vg if ty == t.name and s is not None)
        if d:
            t.tau_pass = float(np.percentile(d, args.max_miss * 100))
            t.tau_fail = float(np.percentile(g, 90)) if g else t.tau_pass * 1.15
            if t.tau_fail <= t.tau_pass:
                t.tau_fail = t.tau_pass * 1.10

    # --- 2. evaluate on TEST ---
    tg = score_folder(mg, gp(_ROOT, "good", "test", "good"))
    td = score_folder(mg, gp(_ROOT, "defect", "test", "defect"))

    def decide(ty, s):
        if s is None:
            return "REVIEW"  # registration failed / unknown type
        for t in mg.templates:
            if t.name == ty:
                return t.decide(s)
        return "REVIEW"

    def confusion(rows):
        c = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        for ty, s in rows:
            c[decide(ty, s)] += 1
        return c

    print(f"=== TEST-set evaluation (thresholds from val, max_miss={args.max_miss}) ===\n")
    for t in mg.templates:
        g = [r for r in tg if r[0] == t.name]
        d = [r for r in td if r[0] == t.name]
        print(f"{t.name}: good n={len(g)} {confusion(g)} | defect n={len(d)} {confusion(d)}")

    G, D = confusion(tg), confusion(td)
    ng, nd = sum(G.values()), sum(D.values())
    print(f"\nOVERALL good   (n={ng}): {G}")
    print(f"OVERALL defect (n={nd}): {D}")
    if ng and nd:
        print("\n--- headline metrics ---")
        print(f"Good auto-PASS      : {100*G['PASS']/ng:.0f}%  (work saved on good boards)")
        print(f"Good wrongly FAILED : {100*G['FAIL']/ng:.0f}%  (false rejects)")
        print(f"Defect caught (FAIL): {100*D['FAIL']/nd:.0f}%")
        print(f"Defect MISSED (PASS): {100*D['PASS']/nd:.0f}%  (escaped defects)")
        auto = G['PASS'] + G['FAIL'] + D['PASS'] + D['FAIL']
        print(f"Auto-decided        : {100*auto/(ng+nd):.0f}%  (rest -> human review)")


if __name__ == "__main__":
    main()
