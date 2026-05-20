# HelmetRisk-YOLO-Pipeline

YOLO-based helmet detection, person-helmet matching, and scene-level safety risk scoring pipeline.

## Overview

This repository stores an experimental computer-vision pipeline for construction worker helmet safety analysis. The project compares YOLOv8 and YOLOE-based detection outputs, performs person-helmet matching, and estimates scene-level safety risk using violation probability, crowd density, and Noisy-OR aggregation.

The purpose of this repository is to preserve the full experimental workflow, including Python scripts, model weights, framework diagrams, sample annotation files, and result figures.

---

## Main Components

- YOLOv8-based helmet and person detection
- YOLOE-based visual comparison
- Person-helmet matching
- Per-person helmet violation probability estimation
- Crowd density scaling
- Noisy-OR scene-level risk aggregation
- Qualitative visual comparison figures
- Video-based safety analysis scripts

---

## Repository Structure

```text

HelmetRisk-YOLO-Pipeline/
├── data_samples/
│   └── yoloe.xml
│
├── docs/
│   └── README.md
│
├── figures/
│   ├── 1_comparison_pro.png
│   ├── 2_comparison.png
│   ├── 3_comparison_yoloe_vs_yolov8.png
│   ├── Image helmet.png
│   ├── Dataset Preparation to Risk-2026-02-13-021439.png
│   ├── viber_image_2026-03-20_16-44-57-512.jpg
│   └── yolo_3d_rotate.gif
│
├── models/
│   ├── best.pt
│   ├── yoloe-11s-seg.pt
│   ├── yoloe-26m-seg.pt
│   ├── yoloe-26s-seg.pt
│   └── yolov8n.pt
│
├── results/
│   └── README.md
│
├── src/
│   ├── image1yoloeandyolov8.py
│   ├── image2yoloeandyolov8.py
│   ├── image3yoloeandyolov8.py
│   ├── test_videoyoloe.py
│   ├── test_videoyolov8.py
│   ├── test_yoloe.py
│   ├── test_yolov8.py
│   ├── yolo_worker_safety_analysis.py
│   └── yoloe_video_safety_analysis.py
│
├── videos/
│   └── README.md
│
├── README.md
├── LICENSE
└── .gitignore
