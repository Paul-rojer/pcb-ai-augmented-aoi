"""
evaluate_consensus.py
---------------------
Evaluate the AOI + AI CONSENSUS on the held-out test set.

Strategy (matches the objective: maximise auto-PASS, keep humans for the rest):
  - The AI is set LENIENT (passes good boards generously) -- safe here because
    a defect the AI wrongly passes disagrees with the AOI machine -> REVIEW.
  - AOI verdict is taken from the filename: '..._N.*' = defect (NG), else good.
  - Consensus: both-good -> PASS, both-defect -> FAIL, disagree -> REVIEW.

    python golden/evaluate_consensus.py --ai_lenient 95

--ai_lenient = percentile of good val scores used as the AI PASS threshold
(higher = AI passes more good boards = more auto-PASS).

IMPORTANT honest caveat: our dataset is labelled by the AOI machine, so here the
AOI verdict == ground truth. That means every auto-decision is correct by
construction and only the AI's disagreements go to REVIEW. The number to read is
the AUTO-DECIDED rate (human-workload reduction); independent accuracy of the
consensus cannot be measured without labels independent of the AOI machine.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from multi_golden import MultiGolden, consensus_decision

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aoi_from_name(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    return "defect" if stem.upper().endswith("_N") else "good"


def ai_scores(mg, folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        res, _ = mg.inspect(bgr, do_crop="auto")
        rows.append((p, res.get("type"), res.get("score")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_lenient", type=float, default=95)
    ap.add_argument("--good_rate", type=float, default=0.95,
                    help="assumed fraction of GOOD boards on the real line (for projection)")
    args = ap.parse_args()
    mg = MultiGolden(gallery_dir=os.path.join(_ROOT, "golden", "gallery"))

    # set lenient AI PASS thresholds from val good (per type)
    vg = ai_scores(mg, os.path.join(_ROOT, "datasets", "good_dataset_aoi", "val", "good"))
    for t in mg.templates:
        g = [s for _, ty, s in vg if ty == t.name and s is not None]
        if g:
            t.tau_pass = float(np.percentile(g, args.ai_lenient))

    # evaluate good and defect test sets separately
    good_rows = ai_scores(mg, os.path.join(_ROOT, "datasets", "good_dataset_aoi", "test", "good"))
    defect_rows = ai_scores(mg, os.path.join(_ROOT, "datasets", "defect_dataset_aoi", "test", "defect"))

    def decide_rows(rows):
        c = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        for p, ty, s in rows:
            aoi = aoi_from_name(p)
            ai = "PASS" if (s is not None and ty and any(
                t.name == ty and s <= t.tau_pass for t in mg.templates)) else "REVIEW"
            c[consensus_decision(ai, aoi)] += 1
        return c

    G, D = decide_rows(good_rows), decide_rows(defect_rows)
    ng, nd = sum(G.values()), sum(D.values())
    print(f"=== AOI+AI consensus on TEST (ai_lenient={args.ai_lenient}) ===")
    print(f"GOOD   n={ng}: {G}")
    print(f"DEFECT n={nd}: {D}")
    good_autopass = G['PASS'] / ng if ng else 0
    defect_autofail = D['FAIL'] / nd if nd else 0
    print(f"\nGood auto-PASS rate   : {100*good_autopass:.0f}%")
    print(f"Defect auto-FAIL rate : {100*defect_autofail:.0f}%")
    print(f"(no good board is ever auto-FAILed; no defect is ever auto-PASSed)")

    # project onto a mostly-good production line
    gr = args.good_rate
    auto = gr * good_autopass + (1 - gr) * defect_autofail
    print(f"\n--- projected on a line that is {gr*100:.0f}% good boards ---")
    print(f"Auto-decided (no human) : {100*auto:.0f}%")
    print(f"Sent to human review    : {100*(1-auto):.0f}%")


if __name__ == "__main__":
    main()
