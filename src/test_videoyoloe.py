import cv2
from ultralytics import YOLOE, YOLO

# -----------------------------
# 1) Load models
# -----------------------------
model_yoloe = YOLOE("yoloe-26s-seg.pt")

# Better zero-shot prompts for helmet
model_yoloe.set_classes([
    "person",
    "hard hat",
    "safety helmet",
    "construction helmet"
])

# Helper trained model
model_best = YOLO("best.pt")

# -----------------------------
# 2) Video path
# -----------------------------
video_path = r"D:\yoloedetection\test_video.mp4"
output_path = r"D:\yoloedetection\result_gpu_worker_helmet_status_better.mp4"

# -----------------------------
# 3) Speed / accuracy settings
# -----------------------------
FRAME_SKIP = 1         # process every frame
IMGSZ = 960            # better small-object detection
SHOW_WINDOW = True
USE_HELPER = True
DEVICE = 0             # GPU 0

CONF_YOLOE = 0.20
CONF_HELPER = 0.25

# -----------------------------
# 4) Helper function
# -----------------------------
def helmet_matches_person(person_box, helmet_box):
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box

    hc_x = (hx1 + hx2) / 2
    hc_y = (hy1 + hy2) / 2

    person_h = py2 - py1

    # head region a little bigger than before
    top_region_y2 = py1 + 0.45 * person_h

    inside_x = px1 <= hc_x <= px2
    inside_head_region = py1 <= hc_y <= top_region_y2

    return inside_x and inside_head_region

# -----------------------------
# 5) Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found or cannot open: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
processed_count = 0
last_annotated_frame = None

# -----------------------------
# 6) Process video
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Skip frames
    if frame_idx % FRAME_SKIP != 0:
        if last_annotated_frame is not None:
            out.write(last_annotated_frame)
            if SHOW_WINDOW:
                cv2.imshow("Worker Helmet Status Video", last_annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        else:
            out.write(frame)
        continue

    processed_count += 1
    annotated = frame.copy()

    # -----------------------------
    # 7) YOLOE prediction
    # -----------------------------
    results_yoloe = model_yoloe.predict(
        annotated,
        conf=CONF_YOLOE,
        imgsz=IMGSZ,
        device=DEVICE,
        agnostic_nms=False,
        verbose=False
    )

    # -----------------------------
    # 8) best.pt helper prediction
    # -----------------------------
    if USE_HELPER:
        results_best = model_best.predict(
            annotated,
            conf=CONF_HELPER,
            imgsz=IMGSZ,
            device=DEVICE,
            agnostic_nms=False,
            verbose=False
        )
    else:
        results_best = None

    # -----------------------------
    # 9) Collect YOLOE detections
    # -----------------------------
    boxes_yoloe = results_yoloe[0].boxes
    names_yoloe = results_yoloe[0].names

    persons = []
    yoloe_helmets = []

    # accept multiple helmet-like class names
    helmet_names = {"helmet", "hard hat", "safety helmet", "construction helmet"}

    for box in boxes_yoloe:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names_yoloe[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        det = {
            "label": label,
            "conf": conf,
            "box": (x1, y1, x2, y2)
        }

        if label == "person":
            persons.append(det)
        elif label in helmet_names:
            yoloe_helmets.append(det)

    # -----------------------------
    # 10) Collect helper helmets
    # -----------------------------
    helper_helmets = []
    if USE_HELPER and results_best is not None:
        boxes_best = results_best[0].boxes
        names_best = results_best[0].names

        for box in boxes_best:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = names_best[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if label.lower() == "helmet":
                helper_helmets.append({
                    "label": label,
                    "conf": conf,
                    "box": (x1, y1, x2, y2)
                })

    # -----------------------------
    # 11) Match helmets to persons
    # -----------------------------
    results_person_status = []
    used_yoloe_helmets = set()
    used_helper_helmets = set()

    for person in persons:
        pbox = person["box"]
        matched_helmet = None
        best_conf = -1
        best_idx_yoloe = -1

        # First: try YOLOE helmets
        for i, helmet in enumerate(yoloe_helmets):
            if i in used_yoloe_helmets:
                continue

            if helmet_matches_person(pbox, helmet["box"]):
                if helmet["conf"] > best_conf:
                    best_conf = helmet["conf"]
                    matched_helmet = helmet
                    best_idx_yoloe = i

        if matched_helmet is not None:
            used_yoloe_helmets.add(best_idx_yoloe)
            status = "worker with helmet"
            color = (0, 255, 0)

        else:
            # Second: try helper model
            matched_helper = None
            best_conf_helper = -1
            best_idx_helper = -1

            for j, helmet in enumerate(helper_helmets):
                if j in used_helper_helmets:
                    continue

                if helmet_matches_person(pbox, helmet["box"]):
                    if helmet["conf"] > best_conf_helper:
                        best_conf_helper = helmet["conf"]
                        matched_helper = helmet
                        best_idx_helper = j

            if matched_helper is not None:
                used_helper_helmets.add(best_idx_helper)
                status = "worker with helmet"
                color = (0, 255, 0)
            else:
                status = "worker with no helmet"
                color = (0, 0, 255)

        results_person_status.append({
            "person_box": pbox,
            "status": status,
            "color": color,
            "person_conf": person["conf"]
        })

    # -----------------------------
    # 12) Draw result
    # -----------------------------
    for item in results_person_status:
        x1, y1, x2, y2 = item["person_box"]
        status = item["status"]
        color = item["color"]
        conf = item["person_conf"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{status} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        text_y1 = max(y1 - th - 10, 0)
        text_y2 = y1

        cv2.rectangle(annotated, (x1, text_y1), (x1 + tw, text_y2), color, -1)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.putText(
        annotated,
        f"Frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    last_annotated_frame = annotated.copy()
    out.write(annotated)

    if SHOW_WINDOW:
        cv2.imshow("Worker Helmet Status Video", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("Stopped by user.")
            break

# -----------------------------
# 13) Release
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved output video to: {output_path}")
print(f"Total frames read: {frame_idx}")
print(f"Frames actually processed by model: {processed_count}")