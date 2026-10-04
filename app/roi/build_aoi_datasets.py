"""
build_aoi_datasets.py
---------------------
One command to rebuild clean, AOI-only, rail-cropped datasets for both experts.

It does, for each class:
    1. gather AOI-only source images
    2. ROI-crop them (brightness profile, removes conveyor rails)
    3. split into train / val / test
    4. write the folder layout the AOI loader expects

Sources (auto-detected in this project):
    GOOD   : datasets/raw/aoi/good                      (98 imgs, already AOI-only)
    DEFECT : datasets/defect_dataset/{train,val,test}/defect, files with "_N." in
             the name (real AOI captures; the rest are public images)

Output:
    datasets/good_dataset_aoi/{train,val,test}/good
    datasets/defect_dataset_aoi/{train,val,test}/defect
    plus a _review/ folder per class for images the cropper wasn't confident on.

Run:
    python roi/build_aoi_datasets.py
"""

import os
import random
import shutil
from pathlib import Path

import cv2

from aoi_profile_crop import crop_image, MIN_FRAC, MAX_FRAC  # same folder

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def gather_good():
    src = DS / "raw" / "aoi" / "good"
    return sorted(src.glob("*.*"))


def gather_defect():
    files = []
    for sp in ("train", "val", "test"):
        d = DS / "defect_dataset" / sp / "defect"
        files += [p for p in d.glob("*.*") if "_N." in p.name]
    # de-dup by filename (same image may appear once per split already)
    seen, uniq = set(), []
    for p in sorted(files):
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq


def split_files(files):
    files = list(files)
    random.shuffle(files)
    n = len(files)
    n_tr = int(n * SPLITS["train"])
    n_va = int(n * SPLITS["val"])
    return {"train": files[:n_tr],
            "val": files[n_tr:n_tr + n_va],
            "test": files[n_tr + n_va:]}


def build(class_name, out_root, files):
    review = out_root / "_review"
    review.mkdir(parents=True, exist_ok=True)
    parts = split_files(files)
    ok = flagged = bad = 0
    for split, items in parts.items():
        dst = out_root / split / class_name
        dst.mkdir(parents=True, exist_ok=True)
        for f in items:
            img = cv2.imread(str(f))
            if img is None:
                bad += 1
                continue
            crop, frac = crop_image(img)
            if frac < MIN_FRAC or frac > MAX_FRAC or crop.size == 0:
                cv2.imwrite(str(review / f.name), img)  # original, crop by hand
                flagged += 1
            else:
                cv2.imwrite(str(dst / f.name), crop)
                ok += 1
    print(f"  [{class_name}] cropped OK={ok}  flagged(_review)={flagged}  unreadable={bad}")
    for split in SPLITS:
        n = len(list((out_root / split / class_name).glob('*.*')))
        print(f"      {split}: {n}")


def main():
    print("GOOD  -> datasets/good_dataset_aoi")
    good_files = gather_good()
    print(f"  source images: {len(good_files)}")
    build("good", DS / "good_dataset_aoi", good_files)

    print("DEFECT -> datasets/defect_dataset_aoi")
    defect_files = gather_defect()
    print(f"  source images: {len(defect_files)}")
    build("defect", DS / "defect_dataset_aoi", defect_files)

    print("\nDone. Retrain with:")
    print("  good   data_path = datasets/good_dataset_aoi")
    print("  defect data_path = datasets/defect_dataset_aoi")
    print("Then hand-crop anything left in each _review/ folder.")


if __name__ == "__main__":
    main()
