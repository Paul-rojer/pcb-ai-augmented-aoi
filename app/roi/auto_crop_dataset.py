import cv2
import os
import glob
import numpy as np

# ==========================
# CHANGE THESE TWO PATHS
# ==========================

INPUT_FOLDER = r"D:\PCB_AI_Augmented_AOI\datasets\raw\aoi\good"
OUTPUT_FOLDER = r"D:\PCB_AI_Augmented_AOI\datasets\roi\good\train"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------------

templates = []

# Blue PCB
templates.append({
    "name":"blue",
    "wmin":850,
    "wmax":1100,
    "hmin":850,
    "hmax":1100
})

# Panel PCB
templates.append({
    "name":"panel",
    "wmin":1800,
    "wmax":2300,
    "hmin":1300,
    "hmax":1700
})

# Relay PCB
templates.append({
    "name":"relay",
    "wmin":1800,
    "wmax":2200,
    "hmin":1800,
    "hmax":2200
})

# Small PCB
templates.append({
    "name":"small",
    "wmin":1400,
    "wmax":1700,
    "hmin":1600,
    "hmax":1900
})

for file in glob.glob(os.path.join(INPUT_FOLDER,"*.jpg")):

    img=cv2.imread(file)

    if img is None:
        continue

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower=np.array([25,25,25])
    upper=np.array([100,255,255])

    mask=cv2.inRange(hsv,lower,upper)

    kernel=np.ones((9,9),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts)==0:
        continue

    c=max(cnts,key=cv2.contourArea)

    x,y,w,h=cv2.boundingRect(c)

    pad=20

    x=max(0,x-pad)
    y=max(0,y-pad)

    w=min(img.shape[1]-x,w+2*pad)
    h=min(img.shape[0]-y,h+2*pad)

    crop=img[y:y+h,x:x+w]

    out=os.path.join(OUTPUT_FOLDER,os.path.basename(file))

    cv2.imwrite(out,crop)

print("Finished.")