"""
build_golden.py
---------------
Build the golden mean/std template from the GOOD training boards.

    python golden/build_golden.py

Training images are already rail-cropped (already_cropped=True). Saves
golden/golden.npz (thresholds set later by calibrate_golden.py).
"""
import glob
import os

from golden_inspector import GoldenInspector

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    good = sorted(glob.glob(os.path.join(
        _ROOT, "datasets", "good_dataset_aoi", "train", "good", "*.*")))
    print(f"good training boards: {len(good)}")
    insp = GoldenInspector()
    insp.fit(good, already_cropped=True)
    out = os.path.join(_ROOT, "golden", "golden.npz")
    insp.save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
