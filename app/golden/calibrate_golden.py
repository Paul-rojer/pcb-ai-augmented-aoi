"""
calibrate_golden.py
-------------------
Score the validation boards, report separation AUC + a confusion matrix, and
choose two thresholds for the decision-support 3-way output:

    score <= tau_pass  -> PASS   (confidently good)
    score >= tau_fail  -> FAIL   (confidently defective)
    in between         -> REVIEW (human)

Registration failures are reported separately (they route to REVIEW at runtime).

    python golden/calibrate_golden.py
"""
import glob
import os

import cv2
import numpy as np

from golden_inspector import GoldenInspector

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scores(insp, folder):
    vals, reg_fail = [], 0
    for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        res, _ = insp.inspect(bgr, do_crop=False)   # val already cropped
        if res["score"] is None:
            reg_fail += 1
        else:
            vals.append(res["score"])
    return vals, reg_fail


def auc(a, b):
    if not a or not b:
        return float("nan")
    return sum((1.0 if y > x else 0.5 if y == x else 0.0)
               for x in a for y in b) / (len(a) * len(b))


def main():
    insp = GoldenInspector(golden_path=os.path.join(_ROOT, "golden", "golden.npz"))
    # Use good val + test for a more stable threshold (good boards are scarce).
    g, gfail = scores(insp, os.path.join(_ROOT, "datasets", "good_dataset_aoi", "val", "good"))
    gt, gtf = scores(insp, os.path.join(_ROOT, "datasets", "good_dataset_aoi", "test", "good"))
    g += gt; gfail += gtf
    d, dfail = scores(insp, os.path.join(_ROOT, "datasets", "defect_dataset_aoi", "val", "defect"))
    print(f"registered: good={len(g)} defect={len(d)} | reg-fail->REVIEW: good={gfail} defect={dfail}")
    a = auc(g, d)
    print(f"separation AUC (registered) = {a:.3f}")

    # tau_pass: let ~90% of good boards PASS (robust to the small sample).
    # tau_fail: above tau_pass, where defects are confidently flagged.
    import numpy as np
    tau_pass = float(np.percentile(g, 90)) if g else None
    tau_fail = float(np.percentile(d, 75)) if d else None
    if tau_fail is not None and tau_pass is not None and tau_fail <= tau_pass:
        tau_fail = tau_pass * 1.15
    insp.tau_pass, insp.tau_fail = tau_pass, tau_fail
    insp.save(os.path.join(_ROOT, "golden", "golden.npz"))

    # report 3-way confusion at these thresholds
    def bucket(s):
        return insp._decide(s)
    for name, vals in (("GOOD", g), ("DEFECT", d)):
        c = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        for s in vals:
            c[bucket(s)] += 1
        print(f"  {name:6}: PASS={c['PASS']} REVIEW={c['REVIEW']} FAIL={c['FAIL']}")
    print(f"\ntau_pass={tau_pass:.3f}  tau_fail={tau_fail:.3f}  (saved)")


if __name__ == "__main__":
    main()
