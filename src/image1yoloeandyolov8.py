import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ["YOLOE", "YOLOv8"]
person_conf = [0.9215, 0.8329]
helmet_conf = [0.906, 0.898]

# Table
table_df = pd.DataFrame({
    "Metric": ["Total Persons", "With Helmet", "No Helmet", "Avg Person Conf"],
    "YOLOE": [3, 2, 1, 0.9215],
    "YOLOv8": [3, 2, 1, 0.8329]
})

# Colors
colors = ["#1f77b4", "#ff7f0e"]

# Figure
fig = plt.figure(figsize=(12, 8))

# ===== Person Confidence =====
ax1 = plt.subplot(2, 2, 1)
bars = ax1.bar(models, person_conf, color=colors)
ax1.set_ylim(0, 1.05)
ax1.set_title("Average Person Confidence", fontweight="bold")

for i, v in enumerate(person_conf):
    ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# highlight winner
ax1.text(0, person_conf[0] + 0.08, "↑ Higher", ha='center', color='green')

# ===== Helmet Confidence =====
ax2 = plt.subplot(2, 2, 2)
bars = ax2.bar(models, helmet_conf, color=colors)
ax2.set_ylim(0, 1.05)
ax2.set_title("Average Helmet Confidence", fontweight="bold")

for i, v in enumerate(helmet_conf):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# ===== Table =====
ax3 = plt.subplot(2, 1, 2)
ax3.axis('off')

tbl = ax3.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc='center',
    cellLoc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2)

# Title
fig.suptitle(
    "Comparison of YOLOE and YOLOv8 for Helmet Detection",
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("comparison_pro.png", dpi=300)
plt.show()