"""
build_gallery.py
----------------
Build a multi-product golden gallery (one template per board type) from ALL good
boards in good_dataset_aoi (train+val+test). Green types are scarce, so all good
samples are used.

    python golden/build_gallery.py
"""
import glob
import os

from multi_golden import MultiGolden

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    good = []
    for split in ("train", "val", "test"):
        good += sorted(glob.glob(os.path.join(
            _ROOT, "datasets", "good_dataset_aoi", split, "good", "*.*")))
    print(f"good boards: {len(good)}")
    mg = MultiGolden()
    mg.build(good)
    out = os.path.join(_ROOT, "golden", "gallery")
    mg.save(out)
    print(f"saved {len(mg.templates)} templates -> {out}")


if __name__ == "__main__":
    main()
