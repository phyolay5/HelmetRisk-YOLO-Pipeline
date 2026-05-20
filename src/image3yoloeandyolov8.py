import matplotlib.pyplot as plt
import numpy as np

# =========================
# INPUT DATA
# =========================

# YOLOE
yoloe_avg_person = 0.8781742930412293
yoloe_total_persons = 10
yoloe_with_helmet = 10
yoloe_no_helmet = 0
yoloe_helmet_conf_list = [
    0.91906482,
    0.805062294,
    0.845969081,
    0.857241869,
    0.844514728,
    0.846584201,
    0.838953614,
    0.65871948,
    0.822437167,
    0.814173758
]

# YOLOv8
yolov8_avg_person = 0.653734
yolov8_total_persons = 9
yolov8_with_helmet = 9
yolov8_no_helmet = 0
yolov8_helmet_conf_list = [
    0.859283,
    0.851324,
    0.88362,
    0.759542,
    0.813685,
    0.680567,
    0.262521,
    0.781037,
    0.811669
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

bars1 = ax1.bar(models, person_vals, color=["#1f77b4", "#ff7f0e"])

ax1.set_title("Average Person Confidence", fontsize=20, fontweight="bold")
ax1.set_ylim(0, 1.05)
ax1.tick_params(axis='both', labelsize=14)

for bar, val in zip(bars1, person_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=17, fontweight="bold")

if yoloe_avg_person > yolov8_avg_person:
    ax1.text(0, yoloe_avg_person + 0.08, "↑ Higher",
             ha="center", fontsize=16, color="green")
else:
    ax1.text(1, yolov8_avg_person + 0.08, "↑ Higher",
             ha="center", fontsize=16, color="green")

# =========================
# RIGHT BAR CHART
# =========================
helmet_vals = [yoloe_avg_helmet, yolov8_avg_helmet]

bars2 = ax2.bar(models, helmet_vals, color=["#1f77b4", "#ff7f0e"])

ax2.set_title("Average Helmet Confidence", fontsize=20, fontweight="bold")
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis='both', labelsize=14)

for bar, val in zip(bars2, helmet_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=17, fontweight="bold")

if yoloe_avg_helmet > yolov8_avg_helmet:
    ax2.text(0, yoloe_avg_helmet + 0.08, "↑ Higher",
             ha="center", fontsize=16, color="green")
else:
    ax2.text(1, yolov8_avg_helmet + 0.08, "↑ Higher",
             ha="center", fontsize=16, color="green")

# =========================
# TABLE
# =========================
ax_table.axis("off")

table_data = [
    ["Total Persons", f"{yoloe_total_persons}", f"{yolov8_total_persons}"],
    ["With Helmet", f"{yoloe_with_helmet}", f"{yolov8_with_helmet}"],
    ["No Helmet", f"{yoloe_no_helmet}", f"{yolov8_no_helmet}"],
    ["Avg Person Conf", f"{yoloe_avg_person:.4f}", f"{yolov8_avg_person:.4f}"],
    ["Avg Helmet Conf", f"{yoloe_avg_helmet:.4f}", f"{yolov8_avg_helmet:.4f}"],
]

col_labels = ["Metric", "YOLOE", "YOLOv8"]

table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colLoc="center",
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(15)
table.scale(1, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(1.2)
    if row == 0:
        cell.set_fontsize(16)

# =========================
# SAVE + SHOW
# =========================
output_file = "comparison_yoloe_vs_yolov8.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

print(f"Saved: {output_file}")