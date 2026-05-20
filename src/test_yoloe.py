import cv2
from ultralytics import YOLOE, YOLO

# -----------------------------
# 1) Load models
# -----------------------------
# YOLOE = main model
model_yoloe = YOLOE("yoloe-26s-seg.pt")
model_yoloe.set_classes(["person", "helmet"])

# best.pt = helper only (not shown in final image)
model_best = YOLO("best.pt")

# -----------------------------
# 2) Image path
# -----------------------------
image_path = r"D:\yoloedetection\test.jpg"
output_path = r"D:\yoloedetection\result_worker_helmet_status.jpg"

# -----------------------------
# 3) Run prediction
# -----------------------------
# Main YOLOE prediction
results_yoloe = model_yoloe.predict(
    image_path,
    conf=0.35,
    agnostic_nms=False,
    verbose=False
)

# Helper best.pt prediction
results_best = model_best.predict(
    image_path,
    conf=0.25,
    agnostic_nms=False,
    verbose=False
)

# -----------------------------
# 4) Read original image
# -----------------------------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# -----------------------------
# 5) Collect YOLOE detections
# -----------------------------
boxes_yoloe = results_yoloe[0].boxes
names_yoloe = results_yoloe[0].names

persons = []
yoloe_helmets = []

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
    elif label == "helmet":
        yoloe_helmets.append(det)

# -----------------------------
# 6) Collect helper helmet detections from best.pt
# -----------------------------
helper_helmets = []
boxes_best = results_best[0].boxes
names_best = results_best[0].names

for box in boxes_best:
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())
    label = names_best[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    if label == "helmet":
        helper_helmets.append({
            "label": label,
            "conf": conf,
            "box": (x1, y1, x2, y2)
        })

# -----------------------------
# 7) Helper function:
#    Check if helmet belongs to person
# -----------------------------
def helmet_matches_person(person_box, helmet_box):
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box

    # helmet center
    hc_x = (hx1 + hx2) / 2
    hc_y = (hy1 + hy2) / 2

    # person's top region (upper 35%)
    person_h = py2 - py1
    top_region_y2 = py1 + 0.35 * person_h

    inside_x = px1 <= hc_x <= px2
    inside_head_region = py1 <= hc_y <= top_region_y2

    return inside_x and inside_head_region

# -----------------------------
# 8) First try YOLOE helmets
#    If not found, use best.pt helper helmets
# -----------------------------
results_person_status = []

used_yoloe_helmets = set()
used_helper_helmets = set()

for person in persons:
    pbox = person["box"]
    matched_helmet = None
    best_conf = -1

    # 8A) Try YOLOE helmets first
    best_idx_yoloe = -1
    for i, helmet in enumerate(yoloe_helmets):
        if i in used_yoloe_helmets:
            continue

        hbox = helmet["box"]
        if helmet_matches_person(pbox, hbox):
            if helmet["conf"] > best_conf:
                best_conf = helmet["conf"]
                matched_helmet = helmet
                best_idx_yoloe = i

    if matched_helmet is not None:
        used_yoloe_helmets.add(best_idx_yoloe)
        status = "worker with helmet"
        color = (0, 255, 0)
    else:
        # 8B) If YOLOE missed, use helper helmet detections
        best_idx_helper = -1
        best_conf_helper = -1
        matched_helper = None

        for j, helmet in enumerate(helper_helmets):
            if j in used_helper_helmets:
                continue

            hbox = helmet["box"]
            if helmet_matches_person(pbox, hbox):
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
# 9) Draw final result
#    Show only final worker status
# -----------------------------
for item in results_person_status:
    x1, y1, x2, y2 = item["person_box"]
    status = item["status"]
    color = item["color"]
    conf = item["person_conf"]

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = f"{status} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    text_y1 = max(y1 - th - 10, 0)
    text_y2 = y1

    cv2.rectangle(img, (x1, text_y1), (x1 + tw, text_y2), color, -1)
    cv2.putText(
        img,
        text,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

# -----------------------------
# 10) Save and show
# -----------------------------
cv2.imwrite(output_path, img)

cv2.imshow("Worker Helmet Status", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# 11) Print summary
# -----------------------------
print(f"Saved result to: {output_path}")
print(f"YOLOE persons: {len(persons)}")
print(f"YOLOE helmets: {len(yoloe_helmets)}")
print(f"Helper helmets (best.pt): {len(helper_helmets)}")