import cv2
import numpy as np
from pathlib import Path

# ====================================================
# CHANGE THESE
# ====================================================

INPUT_DIR = r"D:\PCB_AI_Augmented_AOI\datasets\raw\aoi\good"
OUTPUT_DIR = r"D:\PCB_AI_Augmented_AOI\datasets\roi\good\train"

# ====================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def crop_roi(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---------- BLUE PCB ----------
    lower_blue = np.array([80, 40, 20])
    upper_blue = np.array([140, 255, 255])

    # ---------- GREEN PCB ----------
    lower_green = np.array([35, 40, 20])
    upper_green = np.array([90, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_or(mask_blue, mask_green)

    kernel = np.ones((15,15),np.uint8)

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours)==0:
        return None

    best=None
    best_area=0

    for c in contours:

        area=cv2.contourArea(c)

        if area<30000:
            continue

        x,y,w,h=cv2.boundingRect(c)

        ratio=w/h

        if ratio<0.45 or ratio>2.5:
            continue

        if area>best_area:
            best_area=area
            best=(x,y,w,h)

    if best is None:
        return None

    x,y,w,h=best

    pad=20

    x=max(0,x-pad)
    y=max(0,y-pad)

    w=min(img.shape[1]-x,w+2*pad)
    h=min(img.shape[0]-y,h+2*pad)

    return img[y:y+h,x:x+w]


count=0
skip=0

for img_path in Path(INPUT_DIR).glob("*.*"):

    img=cv2.imread(str(img_path))

    if img is None:
        continue

    roi=crop_roi(img)

    if roi is None:
        skip+=1
        continue

    cv2.imwrite(
        str(Path(OUTPUT_DIR)/img_path.name),
        roi
    )

    count+=1

print("="*40)
print("Saved :",count)
print("Skipped:",skip)
print("="*40)