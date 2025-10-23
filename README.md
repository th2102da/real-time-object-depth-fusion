# 🚀 Real-Time Object & Depth Fusion

> **Real-time object detection and distance estimation system using Faster R-CNN and Depth Anything Model v2.**  
> Faster R-CNN 모델을 활용한 객체 인식과 Depth Anything v2 기반 거리 추정을 융합한 실시간 비전 시스템입니다.

---

## 📖 Overview

This project integrates **Faster R-CNN** (for object detection) and **Depth Anything Model v2** (for depth estimation) to achieve **real-time spatial perception**.  
It detects objects in a scene, estimates their relative or metric distances, and fuses the results to provide a unified 3D-aware output.

이 프로젝트는 **Faster R-CNN**을 이용해 객체를 인식하고, **Depth Anything v2** 모델로 각 객체의 **거리(깊이)** 를 추정하여  
실시간으로 **객체별 거리 정보를 융합(Fusion)** 하는 시스템을 구현합니다.

---

## 🧠 Features

✅ **Object Detection** — Detects multiple objects using the Faster R-CNN model  
✅ **Depth Estimation** — Predicts distance maps using Depth Anything Model v2  
✅ **Model Fusion** — Combines detection results with per-object depth values  
✅ **Visualization** — Displays object bounding boxes with distance overlays  
✅ **Real-Time Processing** — Optimized for live camera or video input

---

## 🧩 Architecture
```
┌────────────────────────┐
│ Input (Camera/Video) │
└────────────┬───────────┘
│
▼
┌────────────────────────┐
│ Faster R-CNN │
│ → Object detection (class + bbox) │
└────────────────────────┘
│
▼
┌────────────────────────────┐
│ Depth Anything Model v2 │
│ → Depth map prediction │
└────────────────────────────┘
│
▼
┌────────────────────────────┐
│ Fusion Module │
│ → Combine detections + depth info │
└────────────────────────────┘
│
▼
┌────────────────────────────┐
│ Visualization Output │
│ → Bounding boxes + Distance labels │
└────────────────────────────┘

```

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/th2102da/real-time-object-depth-fusion.git
cd real-time-object-depth-fusion

# 2️⃣ Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
