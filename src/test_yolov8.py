# =========================================================
# File: worker_helmet_status_two_models.py
# Description:
# Person detector + helmet detector -> final worker status
# Outputs: result.jpg, results.csv, results.json, summary.json
# =========================================================

import os
import csv
import json
import cv2
from statistics import mean
from ultralytics import YOLO

# =========================================================
# 1) CONFIG
# =========================================================
PERSON_MODEL_PATH = r"D:\yoloedetection\yolov8n.pt"   # COCO pretrained model
HELMET_MODEL_PATH = r"D:\yoloedetection\best.pt"      # your helmet model
IMAGE_PATH = r"D:\yoloedetection\test.jpg"
OUTPUT_DIR = r"D:\yoloedetection\output_status"

PERSON_CONF = 0.25
HELMET_CONF = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "result.jpg")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "results.json")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "summary.json")

# =========================================================
# 2) HELPERS
# =========================================================
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def helmet_matches_person(person_box, helmet_box):
    """
    Helmet center must lie inside the upper 35% region of person box.
    """
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box

    hc_x = (hx1 + hx2) / 2
    hc_y = (hy1 + hy2) / 2

    person_h = py2 - py1
    top_region = py1 + 0.35 * person_h

    return (px1 <= hc_x <= px2) and (py1 <= hc_y <= top_region)

# =========================================================
# 3) LOAD MODELS
# =========================================================
person_model = YOLO(PERSON_MODEL_PATH)
helmet_model = YOLO(HELMET_MODEL_PATH)

# =========================================================
# 4) READ IMAGE
# =========================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_h, img_w = img.shape[:2]

# =========================================================
# 5) PERSON DETECTION (from COCO model)
# =========================================================
person_results = person_model.predict(
    source=IMAGE_PATH,
    conf=PERSON_CONF,
    verbose=False
)

pr = person_results[0]
person_boxes = pr.boxes
person_names = pr.names

persons = []
for i, box in enumerate(person_boxes):
    cls_id = int(box.cls[0])
    class_name = person_names[cls_id].lower().strip()

    # COCO class 0 = person
    if class_name != "person":
        continue

    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    persons.append({
        "id": i,
        "class_name": "person",
        "confidence": conf,
        "bbox": [x1, y1, x2, y2],
        "area": box_area((x1, y1, x2, y2))
    })

# =========================================================
# 6) HELMET DETECTION (from your best.pt)
# =========================================================
helmet_results = helmet_model.predict(
    source=IMAGE_PATH,
    conf=HELMET_CONF,
    verbose=False
)

hr = helmet_results[0]
helmet_boxes = hr.boxes
helmet_names = hr.names

helmets = []
for i, box in enumerate(helmet_boxes):
    cls_id = int(box.cls[0])
    class_name = helmet_names[cls_id].lower().strip()

    # keep only helmet class
    if class_name != "helmet":
        continue

    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    helmets.append({
        "id": i,
        "class_name": "helmet",
        "confidence": conf,
        "bbox": [x1, y1, x2, y2],
        "area": box_area((x1, y1, x2, y2))
    })

# =========================================================
# 7) MATCH HELMETS TO PERSONS
# =========================================================
used_helmets = set()
results_data = []

for p_idx, person in enumerate(persons):
    best_match = None
    best_match_idx = -1
    best_conf = -1

    for h_idx, helmet in enumerate(helmets):
        if h_idx in used_helmets:
            continue

        if helmet_matches_person(person["bbox"], helmet["bbox"]):
            if helmet["confidence"] > best_conf:
                best_conf = helmet["confidence"]
                best_match = helmet
                best_match_idx = h_idx

    if best_match is not None:
        used_helmets.add(best_match_idx)
        status = "with_helmet"
        color = (0, 255, 0)
    else:
        status = "no_helmet"
        color = (0, 0, 255)

    results_data.append({
        "person_id": p_idx,
        "status": status,
        "person_confidence": round(person["confidence"], 6),
        "person_bbox": person["bbox"],
        "helmet_confidence": round(best_match["confidence"], 6) if best_match else None,
        "helmet_bbox": best_match["bbox"] if best_match else None
    })

# =========================================================
# 8) DRAW FINAL IMAGE (ONLY PERSON BOXES)
# =========================================================
for item in results_data:
    x1, y1, x2, y2 = item["person_bbox"]
    status = item["status"]

    color = (0, 255, 0) if status == "with_helmet" else (0, 0, 255)
    label = f"{status} {item['person_confidence']:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

cv2.imwrite(OUTPUT_IMAGE, img)

# =========================================================
# 9) SAVE CSV
# =========================================================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "person_id",
        "status",
        "person_confidence",
        "px1", "py1", "px2", "py2",
        "helmet_confidence",
        "hx1", "hy1", "hx2", "hy2"
    ])

    for item in results_data:
        px1, py1, px2, py2 = item["person_bbox"]

        if item["helmet_bbox"] is not None:
            hx1, hy1, hx2, hy2 = item["helmet_bbox"]
            hconf = item["helmet_confidence"]
        else:
            hx1 = hy1 = hx2 = hy2 = ""
            hconf = ""

        writer.writerow([
            item["person_id"],
            item["status"],
            item["person_confidence"],
            px1, py1, px2, py2,
            hconf, hx1, hy1, hx2, hy2
        ])

# =========================================================
# 10) SAVE JSON
# =========================================================
json_output = {
    "image_path": IMAGE_PATH,
    "image_width": img_w,
    "image_height": img_h,
    "person_model_path": PERSON_MODEL_PATH,
    "helmet_model_path": HELMET_MODEL_PATH,
    "person_confidence_threshold": PERSON_CONF,
    "helmet_confidence_threshold": HELMET_CONF,
    "total_persons_detected": len(persons),
    "total_helmets_detected": len(helmets),
    "results": results_data
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

# =========================================================
# 11) SAVE SUMMARY
# =========================================================
with_helmet = sum(1 for x in results_data if x["status"] == "with_helmet")
no_helmet = sum(1 for x in results_data if x["status"] == "no_helmet")

person_conf_list = [x["person_confidence"] for x in results_data]

summary = {
    "person_model_path": PERSON_MODEL_PATH,
    "helmet_model_path": HELMET_MODEL_PATH,
    "image_path": IMAGE_PATH,
    "image_width": img_w,
    "image_height": img_h,
    "total_persons_detected": len(persons),
    "total_helmets_detected": len(helmets),
    "workers_with_helmet": with_helmet,
    "workers_with_no_helmet": no_helmet,
    "average_person_confidence": round(mean(person_conf_list), 6) if person_conf_list else 0,
    "output_image": OUTPUT_IMAGE,
    "output_csv": OUTPUT_CSV,
    "output_json": OUTPUT_JSON
}

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# =========================================================
# 12) PRINT
# =========================================================
print("\n===== DONE =====")
print(f"Persons detected       : {len(persons)}")
print(f"Helmets detected       : {len(helmets)}")
print(f"Workers with helmet    : {with_helmet}")
print(f"Workers with no helmet : {no_helmet}")
print(f"Saved image            : {OUTPUT_IMAGE}")
print(f"Saved CSV              : {OUTPUT_CSV}")
print(f"Saved JSON             : {OUTPUT_JSON}")
print(f"Saved summary          : {OUTPUT_SUMMARY}")