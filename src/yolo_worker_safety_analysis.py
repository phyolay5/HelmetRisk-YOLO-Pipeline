# =========================================================
# File: yoloe_worker_safety_analysis.py
# Description:
# YOLOE only image inference + worker helmet safety analysis
# Outputs: image + CSV + JSON + summary
# =========================================================

import cv2
import csv
import json
import os
from statistics import mean
from ultralytics import YOLOE

# =========================================================
# 1) CONFIGURATION
# =========================================================
MODEL_PATH = "yoloe-26m-seg.pt"
IMAGE_PATH = r"D:\yoloedetection\test.jpg"
OUTPUT_DIR = r"D:\yoloedetection\output_yoloe_only"

CONF_THRES = 0.35

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "result.jpg")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "results.json")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "summary.json")


# =========================================================
# 2) LOAD MODEL (YOLOE ONLY)
# =========================================================
model = YOLOE(MODEL_PATH)
model.set_classes(["person", "helmet"])


# =========================================================
# 3) HELPER FUNCTIONS
# =========================================================
def helmet_matches_person(person_box, helmet_box):
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box

    hc_x = (hx1 + hx2) / 2
    hc_y = (hy1 + hy2) / 2

    person_h = py2 - py1
    top_region = py1 + 0.35 * person_h

    return (px1 <= hc_x <= px2) and (py1 <= hc_y <= top_region)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


# =========================================================
# 4) READ IMAGE
# =========================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

h, w = img.shape[:2]


# =========================================================
# 5) INFERENCE
# =========================================================
results = model.predict(IMAGE_PATH, conf=CONF_THRES, verbose=False)

boxes = results[0].boxes
names = results[0].names


# =========================================================
# 6) PARSE DETECTIONS
# =========================================================
persons = []
helmets = []

for i, box in enumerate(boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    det = {
        "id": i,
        "class": label,
        "conf": conf,
        "bbox": [x1, y1, x2, y2],
        "area": box_area((x1, y1, x2, y2))
    }

    if label == "person":
        persons.append(det)
    elif label == "helmet":
        helmets.append(det)


# =========================================================
# 7) MATCH HELMET TO PERSON
# =========================================================
used_helmets = set()
results_data = []

for p_id, person in enumerate(persons):
    best_match = None
    best_conf = -1
    best_idx = -1

    for h_id, helmet in enumerate(helmets):
        if h_id in used_helmets:
            continue

        if helmet_matches_person(person["bbox"], helmet["bbox"]):
            if helmet["conf"] > best_conf:
                best_conf = helmet["conf"]
                best_match = helmet
                best_idx = h_id

    if best_match:
        used_helmets.add(best_idx)
        status = "with_helmet"
        color = (0, 255, 0)
    else:
        status = "no_helmet"
        color = (0, 0, 255)

    results_data.append({
        "person_id": p_id,
        "status": status,
        "person_conf": person["conf"],
        "person_bbox": person["bbox"],
        "helmet": best_match
    })


# =========================================================
# 8) DRAW OUTPUT IMAGE
# =========================================================
for item in results_data:
    x1, y1, x2, y2 = item["person_bbox"]
    status = item["status"]
    conf = item["person_conf"]

    color = (0, 255, 0) if status == "with_helmet" else (0, 0, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"{status} {conf:.2f}"
    cv2.putText(img, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite(OUTPUT_IMAGE, img)


# =========================================================
# 9) SAVE CSV
# =========================================================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "person_id", "status",
        "person_conf", "px1", "py1", "px2", "py2",
        "helmet_conf", "hx1", "hy1", "hx2", "hy2"
    ])

    for item in results_data:
        px1, py1, px2, py2 = item["person_bbox"]

        if item["helmet"]:
            h = item["helmet"]
            hx1, hy1, hx2, hy2 = h["bbox"]
            hconf = h["conf"]
        else:
            hx1 = hy1 = hx2 = hy2 = ""
            hconf = ""

        writer.writerow([
            item["person_id"],
            item["status"],
            item["person_conf"],
            px1, py1, px2, py2,
            hconf, hx1, hy1, hx2, hy2
        ])


# =========================================================
# 10) SAVE JSON
# =========================================================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results_data, f, indent=4)


# =========================================================
# 11) SUMMARY
# =========================================================
with_helmet = sum(1 for x in results_data if x["status"] == "with_helmet")
no_helmet = sum(1 for x in results_data if x["status"] == "no_helmet")

summary = {
    "total_persons": len(persons),
    "total_helmets": len(helmets),
    "with_helmet": with_helmet,
    "no_helmet": no_helmet,
    "avg_person_conf": mean([x["person_conf"] for x in results_data]) if results_data else 0
}

with open(OUTPUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=4)


# =========================================================
# 12) PRINT
# =========================================================
print("\n===== RESULT =====")
print(summary)
print("Saved image:", OUTPUT_IMAGE)
print("Saved CSV:", OUTPUT_CSV)
print("Saved JSON:", OUTPUT_JSON)