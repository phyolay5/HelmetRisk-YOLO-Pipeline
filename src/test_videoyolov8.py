# =========================================================
# File: video_worker_helmet_status.py
# Description:
# Video inference using:
#   1) person detector (COCO YOLOv8)
#   2) helmet detector (your best.pt)
# Outputs:
#   - result_video.mp4
#   - results.csv
#   - results.json
#   - summary.json
# =========================================================

import os
import csv
import json
import cv2
from ultralytics import YOLO

# =========================================================
# 1) CONFIG
# =========================================================
PERSON_MODEL_PATH = r"D:\yoloedetection\yolov8n.pt"   # person detector
HELMET_MODEL_PATH = r"D:\yoloedetection\best.pt"      # helmet detector
VIDEO_PATH = r"D:\yoloedetection\test_video.mp4"      # input video
OUTPUT_DIR = r"D:\yoloedetection\output_video_status"

PERSON_CONF = 0.25
HELMET_CONF = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "result_video.mp4")
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
    Helmet center must lie inside upper 35% of person box.
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
# 4) OPEN VIDEO
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found or cannot open: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# =========================================================
# 5) STORAGE
# =========================================================
all_results = []
total_persons_all_frames = 0
total_helmets_all_frames = 0
total_with_helmet_all_frames = 0
total_no_helmet_all_frames = 0

# =========================================================
# 6) PROCESS VIDEO FRAME BY FRAME
# =========================================================
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------
    # A) person detection
    # -----------------------------------------
    person_results = person_model.predict(
        source=frame,
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

    # -----------------------------------------
    # B) helmet detection
    # -----------------------------------------
    helmet_results = helmet_model.predict(
        source=frame,
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

    # -----------------------------------------
    # C) match helmets to persons
    # -----------------------------------------
    used_helmets = set()
    frame_results = []

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

        result_item = {
            "frame_id": frame_idx,
            "person_id": p_idx,
            "status": status,
            "person_confidence": round(person["confidence"], 6),
            "person_bbox": person["bbox"],
            "helmet_confidence": round(best_match["confidence"], 6) if best_match else None,
            "helmet_bbox": best_match["bbox"] if best_match else None
        }

        frame_results.append(result_item)

        # draw on frame
        x1, y1, x2, y2 = person["bbox"]
        label = f"{status} {person['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # -----------------------------------------
    # D) write frame summary on screen
    # -----------------------------------------
    frame_with_helmet = sum(1 for x in frame_results if x["status"] == "with_helmet")
    frame_no_helmet = sum(1 for x in frame_results if x["status"] == "no_helmet")

    overlay_text = f"Frame: {frame_idx} | Persons: {len(persons)} | Helmets: {len(helmets)} | With helmet: {frame_with_helmet} | No helmet: {frame_no_helmet}"
    cv2.putText(
        frame,
        overlay_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    # -----------------------------------------
    # E) save structured frame results
    # -----------------------------------------
    all_results.append({
        "frame_id": frame_idx,
        "persons_detected": len(persons),
        "helmets_detected": len(helmets),
        "with_helmet_count": frame_with_helmet,
        "no_helmet_count": frame_no_helmet,
        "results": frame_results
    })

    total_persons_all_frames += len(persons)
    total_helmets_all_frames += len(helmets)
    total_with_helmet_all_frames += frame_with_helmet
    total_no_helmet_all_frames += frame_no_helmet

    frame_idx += 1

# =========================================================
# 7) RELEASE VIDEO
# =========================================================
cap.release()
writer.release()

# =========================================================
# 8) SAVE CSV
# =========================================================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([
        "frame_id",
        "person_id",
        "status",
        "person_confidence",
        "px1", "py1", "px2", "py2",
        "helmet_confidence",
        "hx1", "hy1", "hx2", "hy2"
    ])

    for frame_data in all_results:
        for item in frame_data["results"]:
            px1, py1, px2, py2 = item["person_bbox"]

            if item["helmet_bbox"] is not None:
                hx1, hy1, hx2, hy2 = item["helmet_bbox"]
                hconf = item["helmet_confidence"]
            else:
                hx1 = hy1 = hx2 = hy2 = ""
                hconf = ""

            csv_writer.writerow([
                item["frame_id"],
                item["person_id"],
                item["status"],
                item["person_confidence"],
                px1, py1, px2, py2,
                hconf, hx1, hy1, hx2, hy2
            ])

# =========================================================
# 9) SAVE JSON
# =========================================================
json_output = {
    "video_path": VIDEO_PATH,
    "person_model_path": PERSON_MODEL_PATH,
    "helmet_model_path": HELMET_MODEL_PATH,
    "fps": fps,
    "frame_width": width,
    "frame_height": height,
    "total_frames": total_frames,
    "processed_frames": frame_idx,
    "person_confidence_threshold": PERSON_CONF,
    "helmet_confidence_threshold": HELMET_CONF,
    "frame_results": all_results
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

# =========================================================
# 10) SAVE SUMMARY
# =========================================================
summary = {
    "video_path": VIDEO_PATH,
    "person_model_path": PERSON_MODEL_PATH,
    "helmet_model_path": HELMET_MODEL_PATH,
    "fps": fps,
    "frame_width": width,
    "frame_height": height,
    "total_frames_in_video": total_frames,
    "processed_frames": frame_idx,
    "total_person_detections_all_frames": total_persons_all_frames,
    "total_helmet_detections_all_frames": total_helmets_all_frames,
    "total_with_helmet_all_frames": total_with_helmet_all_frames,
    "total_no_helmet_all_frames": total_no_helmet_all_frames,
    "output_video": OUTPUT_VIDEO,
    "output_csv": OUTPUT_CSV,
    "output_json": OUTPUT_JSON
}

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# =========================================================
# 11) PRINT
# =========================================================
print("\n===== DONE =====")
print(f"Processed frames              : {frame_idx}")
print(f"Total person detections       : {total_persons_all_frames}")
print(f"Total helmet detections       : {total_helmets_all_frames}")
print(f"Total with helmet (all frames): {total_with_helmet_all_frames}")
print(f"Total no helmet (all frames)  : {total_no_helmet_all_frames}")
print(f"Saved video                   : {OUTPUT_VIDEO}")
print(f"Saved CSV                     : {OUTPUT_CSV}")
print(f"Saved JSON                    : {OUTPUT_JSON}")
print(f"Saved summary                 : {OUTPUT_SUMMARY}")