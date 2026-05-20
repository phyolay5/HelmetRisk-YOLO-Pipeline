import matplotlib.pyplot as plt
import numpy as np

# =========================
# INPUT DATA
# =========================

# YOLOE
yoloe_avg_person = 0.7584278732538223
yoloe_total_persons = 18
yoloe_with_helmet = 16
yoloe_no_helmet = 2
yoloe_helmet_conf_list = [
    0.889430642, 0.767738879, 0.82238698, 0.852970362, 0.80960077,
    0.847872853, 0.872055709, 0.902839899, 0.656840146, 0.73871249,
    0.800954282, 0.845366418, 0.771242738, 0.579711318, 0.832187653,
    0.749139905
]

# YOLOv8
yolov8_avg_person = 0.728831
yolov8_total_persons = 11
yolov8_with_helmet = 11
yolov8_no_helmet = 0
yolov8_helmet_conf_list = [
    0.875797, 0.830507, 0.851698, 0.874704, 0.818696,
    0.818317, 0.85681, 0.812522, 0.77855, 0.730622, 0.770277
]

# =========================
# CALCULATE
# =========================
yoloe_avg_helmet = np.mean(yoloe_helmet_conf_list)
yolov8_avg_helmet = np.mean(yolov8_helmet_conf_list)

# =========================
# FIGURE
# =========================
fig = plt.figure(figsize=(16, 10), facecolor="#eaeaea")
gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.3], hspace=0.35, wspace=0.08)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

fig.suptitle("Comparison of YOLOE and YOLOv8 for Helmet Detection",
             fontsize=24, fontweight="bold", y=0.98)

# =========================
# LEFT BAR CHART
# =========================
models = ["YOLOE", "YOLOv8"]
person_vals = [yoloe_avg_person, yolov8_avg_person]

bars1 = ax1.bar(models, person_vals,
                color=["#1f77b4", "#ff7f0e"])

ax1.set_title("Average Person Confidence", fontsize=20, fontweight="bold")
ax1.set_ylim(0, 1.05)

for bar, val in zip(bars1, person_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", fontsize=14)

# ↑ Higher arrow (person)
if yoloe_avg_person > yolov8_avg_person:
    ax1.text(0, yoloe_avg_person + 0.08, "↑ Higher",
             ha="center", fontsize=14, color="green")
else:
    ax1.text(1, yolov8_avg_person + 0.08, "↑ Higher",
             ha="center", fontsize=14, color="green")

# =========================
# RIGHT BAR CHART
# =========================
helmet_vals = [yoloe_avg_helmet, yolov8_avg_helmet]

bars2 = ax2.bar(models, helmet_vals,
                color=["#1f77b4", "#ff7f0e"])

ax2.set_title("Average Helmet Confidence", fontsize=20, fontweight="bold")
ax2.set_ylim(0, 1.05)

for bar, val in zip(bars2, helmet_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", fontsize=14)

# ↑ Higher arrow (helmet)
if yoloe_avg_helmet > yolov8_avg_helmet:
    ax2.text(0, yoloe_avg_helmet + 0.08, "↑ Higher",
             ha="center", fontsize=14, color="green")
else:
    ax2.text(1, yolov8_avg_helmet + 0.08, "↑ Higher",
             ha="center", fontsize=14, color="green")

# =========================
# TABLE
# =========================
ax_table.axis("off")

table_data = [
    ["Total Persons", yoloe_total_persons, yolov8_total_persons],
    ["With Helmet", yoloe_with_helmet, yolov8_with_helmet],
    ["No Helmet", yoloe_no_helmet, yolov8_no_helmet],
    ["Avg Person Conf", f"{yoloe_avg_person:.4f}", f"{yolov8_avg_person:.4f}"],
    ["Avg Helmet Conf", f"{yoloe_avg_helmet:.4f}", f"{yolov8_avg_helmet:.4f}"],
]

col_labels = ["Metric", "YOLOE", "YOLOv8"]

table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2)

# =========================
# SAVE + SHOW
# =========================
plt.savefig("comparison.png", dpi=300, bbox_inches="tight")
plt.show()