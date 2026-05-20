# =========================================================
# File: yoloe_video_safety_analysis.py
# Description:
# YOLOE only video inference + worker helmet safety analysis
# Outputs:
#   1) annotated output video
#   2) frame-by-frame CSV
#   3) summary JSON
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
VIDEO_PATH = r"D:\yoloedetection\test_video.mp4"
OUTPUT_DIR = r"D:\yoloedetection\output_yoloe_video"

CONF_THRES = 0.35
DISPLAY_WINDOW = False   # True ဆိုရင် video window ပြမယ်

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "result_video.mp4")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "frame_results.csv")
OUTPUT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")


# =========================================================
# 2) LOAD MODEL (YOLOE ONLY)
# =========================================================
model = YOLOE(MODEL_PATH)
model.set_classes(["person", "helmet"])


# =========================================================
# 3) HELPER FUNCTIONS
# =========================================================
def helmet_matches_person(person_box, helmet_box):
    """
    Decide whether a detected helmet belongs to a detected person.
    Rule:
    - helmet center should be inside person's x-range
    - helmet center should lie in upper 35% region of person box
    """
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box

    hc_x = (hx1 + hx2) / 2
    hc_y = (hy1 + hy2) / 2

    person_h = py2 - py1
    top_region_y2 = py1 + 0.35 * person_h

    inside_x = px1 <= hc_x <= px2
    inside_head_region = py1 <= hc_y <= top_region_y2

    return inside_x and inside_head_region


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


# =========================================================
# 4) OPEN VIDEO
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found or cannot be opened: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))


# =========================================================
# 5) CSV SETUP
# =========================================================
csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame_id",
    "timestamp_sec",
    "person_id",
    "status",
    "person_confidence",
    "person_x1",
    "person_y1",
    "person_x2",
    "person_y2",
    "person_area",
    "matched_helmet_confidence",
    "helmet_x1",
    "helmet_y1",
    "helmet_x2",
    "helmet_y2",
    "helmet_area"
])


# =========================================================
# 6) SUMMARY VARIABLES
# =========================================================
frame_id = 0
total_person_detections = 0
total_helmet_detections = 0
total_with_helmet = 0
total_no_helmet = 0

person_conf_all = []
helmet_conf_all = []

frames_with_person = 0
frames_with_no_helmet = 0


# =========================================================
# 7) PROCESS VIDEO FRAME BY FRAME
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp_sec = frame_id / fps if fps > 0 else 0

    # -----------------------------------------
    # 7A) Run YOLOE inference on current frame
    # -----------------------------------------
    results = model.predict(frame, conf=CONF_THRES, verbose=False)
    result = results[0]

    boxes = result.boxes
    names = result.names

    persons = []
    helmets = []

    # -----------------------------------------
    # 7B) Parse detections
    # -----------------------------------------
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        det = {
            "det_id": idx,
            "class_id": cls_id,
            "class_name": label,
            "confidence": round(conf, 6),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_area": box_area((x1, y1, x2, y2))
        }

        if label == "person":
            persons.append(det)
        elif label == "helmet":
            helmets.append(det)

    if persons:
        frames_with_person += 1

    # -----------------------------------------
    # 7C) Associate helmets to persons
    # -----------------------------------------
    used_helmets = set()
    results_person_status = []

    for p_idx, person in enumerate(persons):
        pbox = person["bbox_xyxy"]

        matched_helmet = None
        matched_helmet_idx = -1
        best_conf = -1.0

        for h_idx, helmet in enumerate(helmets):
            if h_idx in used_helmets:
                continue

            hbox = helmet["bbox_xyxy"]

            if helmet_matches_person(pbox, hbox):
                if helmet["confidence"] > best_conf:
                    best_conf = helmet["confidence"]
                    matched_helmet = helmet
                    matched_helmet_idx = h_idx

        if matched_helmet is not None:
            used_helmets.add(matched_helmet_idx)
            status = "with_helmet"
            color = (0, 255, 0)
        else:
            status = "no_helmet"
            color = (0, 0, 255)

        results_person_status.append({
            "person_id": p_idx,
            "status": status,
            "person_confidence": person["confidence"],
            "person_bbox_xyxy": person["bbox_xyxy"],
            "person_area": person["bbox_area"],
            "matched_helmet": matched_helmet,
            "color": color
        })

    if any(x["status"] == "no_helmet" for x in results_person_status):
        frames_with_no_helmet += 1

    # -----------------------------------------
    # 7D) Draw frame
    # -----------------------------------------
    for item in results_person_status:
        x1, y1, x2, y2 = item["person_bbox_xyxy"]
        status = item["status"]
        conf = item["person_confidence"]
        color = item["color"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{status} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        text_y1 = max(0, y1 - th - 8)
        text_y2 = y1

        cv2.rectangle(frame, (x1, text_y1), (x1 + tw + 6, text_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 3, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # Add frame info on top-left
    info_text = f"Frame: {frame_id}  Time: {timestamp_sec:.2f}s"
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -----------------------------------------
    # 7E) Save CSV rows
    # -----------------------------------------
    for item in results_person_status:
        px1, py1, px2, py2 = item["person_bbox_xyxy"]

        if item["matched_helmet"] is not None:
            helmet_conf = item["matched_helmet"]["confidence"]
            hx1, hy1, hx2, hy2 = item["matched_helmet"]["bbox_xyxy"]
            h_area = item["matched_helmet"]["bbox_area"]
            helmet_conf_all.append(helmet_conf)
        else:
            helmet_conf = ""
            hx1 = hy1 = hx2 = hy2 = ""
            h_area = ""

        csv_writer.writerow([
            frame_id,
            round(timestamp_sec, 4),
            item["person_id"],
            item["status"],
            item["person_confidence"],
            px1, py1, px2, py2,
            item["person_area"],
            helmet_conf,
            hx1, hy1, hx2, hy2,
            h_area
        ])

        person_conf_all.append(item["person_confidence"])

    # -----------------------------------------
    # 7F) Update summary counters
    # -----------------------------------------
    total_person_detections += len(persons)
    total_helmet_detections += len(helmets)
    total_with_helmet += sum(1 for x in results_person_status if x["status"] == "with_helmet")
    total_no_helmet += sum(1 for x in results_person_status if x["status"] == "no_helmet")

    # -----------------------------------------
    # 7G) Write output video
    # -----------------------------------------
    writer.write(frame)

    if DISPLAY_WINDOW:
        cv2.imshow("YOLOE Video Safety Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_id += 1


# =========================================================
# 8) CLEANUP
# =========================================================
cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()


# =========================================================
# 9) SAVE SUMMARY JSON
# =========================================================
summary = {
    "video_name": os.path.basename(VIDEO_PATH),
    "video_path": VIDEO_PATH,
    "model": MODEL_PATH,
    "classes_prompted": ["person", "helmet"],
    "confidence_threshold": CONF_THRES,
    "video_info": {
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "total_frames": total_frames,
        "processed_frames": frame_id,
        "duration_sec": round(frame_id / fps, 4) if fps > 0 else 0
    },
    "detection_summary": {
        "total_person_detections": total_person_detections,
        "total_helmet_detections": total_helmet_detections,
        "total_with_helmet": total_with_helmet,
        "total_no_helmet": total_no_helmet,
        "frames_with_person": frames_with_person,
        "frames_with_no_helmet": frames_with_no_helmet
    },
    "confidence_summary": {
        "avg_person_confidence": round(mean(person_conf_all), 4) if person_conf_all else 0,
        "avg_matched_helmet_confidence": round(mean(helmet_conf_all), 4) if helmet_conf_all else 0
    },
    "output_files": {
        "output_video": OUTPUT_VIDEO,
        "output_csv": OUTPUT_CSV
    }
}

with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)


# =========================================================
# 10) PRINT SUMMARY
# =========================================================
print("\n===== YOLOE VIDEO SAFETY ANALYSIS =====")
print(f"Video name                 : {os.path.basename(VIDEO_PATH)}")
print(f"Processed frames           : {frame_id}")
print(f"FPS                        : {fps}")
print(f"Frame size                 : {frame_width} x {frame_height}")
print(f"Total person detections    : {total_person_detections}")
print(f"Total helmet detections    : {total_helmet_detections}")
print(f"Total with helmet          : {total_with_helmet}")
print(f"Total no helmet            : {total_no_helmet}")
print(f"Frames with person         : {frames_with_person}")
print(f"Frames with no helmet      : {frames_with_no_helmet}")
print(f"Avg person confidence      : {summary['confidence_summary']['avg_person_confidence']}")
print(f"Avg helmet confidence      : {summary['confidence_summary']['avg_matched_helmet_confidence']}")
print(f"Saved video                : {OUTPUT_VIDEO}")
print(f"Saved CSV                  : {OUTPUT_CSV}")
print(f"Saved summary JSON         : {OUTPUT_SUMMARY_JSON}")