# 🚦 TrafficIQ Chat — Vehicle Counting, Speed Estimation & Conversational Traffic Analytics

A real-time traffic analysis system built using **YOLOv8** and **DeepSORT**. The pipeline detects vehicles in video, tracks each one with a persistent ID, estimates its real-world speed, counts traffic by direction, and lets you ask plain-English (or Roman Urdu) questions about the results through an **LLM-powered chatbot**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-purple)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq-teal)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Jupyter%20%2F%20CLI%20%2F%20Web-lightgrey)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Technology Stack](#️-technology-stack)
- [Detection Classes](#-detection-classes)
- [Pipeline Architecture](#-pipeline-architecture)
- [Configuration](#️-configuration)
- [Testing & Validation](#-testing--validation)
- [System Architecture](#-system-architecture)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Calibration Tips](#-calibration-tips)
- [Known Limitations](#-known-limitations)
- [Troubleshooting](#-troubleshooting)

---

## 📌 Project Overview

This system uses **YOLOv8** for vehicle detection and **DeepSORT** for multi-object tracking to turn raw traffic video into structured, queryable data — vehicle counts, per-vehicle speed, and direction of travel — with an LLM chatbot on top to query it in natural language.

### 🎯 Objectives
- Detect and track every vehicle in a video with a stable ID across frames
- Estimate each vehicle's real-world speed (km/h) from a one-time pixel calibration
- Count vehicles crossing a line, split by direction (incoming/outgoing)
- Let a user ask questions about the results in plain English or Roman Urdu, answered from the real computed data — not guessed by the LLM

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Detection | Ultralytics YOLOv8 (nano/small) |
| Tracking | DeepSORT (motion + appearance re-ID) |
| Speed Estimation | Custom pixel-to-real-world calibration |
| Web Dashboard | Streamlit |
| Chatbot / LLM | Groq API (`openai/gpt-oss-20b`) |
| Data Handling | Pandas |
| Language | Python 3.11 |
| Notebook | Jupyter |

---

## 📂 Detection Classes

Uses YOLOv8's pretrained COCO weights, filtered to vehicle classes only — no custom training/dataset required:

| Class ID | Vehicle Type |
|---|---|
| 2 | Car |
| 3 | Motorcycle |
| 5 | Bus |
| 7 | Truck |

---

## 🤖 Pipeline Architecture

- **Detector:** YOLOv8n (`yolov8n.pt`) by default — swappable for `yolov8s.pt` / `yolov8m.pt` for higher accuracy at the cost of speed
- **Tracker:** DeepSORT, tuned specifically for fast-moving highway traffic (see [Configuration](#️-configuration) below)
- **Speed model:** frame-to-frame pixel displacement × calibrated meters-per-pixel × FPS, smoothed over a rolling window
- **Counter:** virtual line-crossing detector, counts each track ID once, split by direction of travel

---

## ⚙️ Configuration

### DeepSORT tracker (`modules/tracker.py`)
| Parameter | Value | Why |
|---|---|---|
| `max_age` | 60 | Raised from DeepSORT's default (30) — highway traffic gets briefly occluded/missed for several frames; too low re-assigns a new ID instead of recovering the old one |
| `n_init` | 2 | Lowered so fast-moving vehicles confirm a track before they've already crossed the counting line |
| `max_iou_distance` | 0.9 | Raised — vehicles filmed from a distance move many pixels between frames relative to box size, so a strict threshold treats them as new objects every frame |
| `max_cosine_distance` | 0.3 | Leniency for appearance-based re-identification when recovering a briefly lost track |

### Speed estimator (`modules/speed_estimator.py`)
| Parameter | Value | Why |
|---|---|---|
| `smoothing_window` | 5 samples | Rolling average to reduce jitter from detection/tracking noise |
| `max_plausible_kmh` | 180 km/h | Any single-frame reading above this is discarded as a tracking glitch, not a real speed |

### Detector (`modules/detector.py`)
| Parameter | Default | Notes |
|---|---|---|
| `conf_threshold` | 0.35 (0.5 recommended) | Minimum confidence to keep a detection |
| `roi_top_pct` | 0.0 | Optional — excludes the top N% of frame from detection, for footage with a distant/cluttered horizon |

---

## 📊 Testing & Validation

Unlike a custom-trained model with benchmark metrics, this project uses YOLOv8's pretrained weights — so validation here means **testing the full pipeline against real footage** and confirming the numbers are physically believable, not just theoretically working:

| Test | Method | Result |
|---|---|---|
| Speed calibration | Measured real-world reference (lane markings) against pixel distance via `calibration_helper.py`, cross-checked against realistic highway speeds | Corrected from an initial ~160 km/h overestimate down to a verified ~90–125 km/h range, matching real US/UK highway traffic |
| Overcounting from ID switching | Compared vehicle counts before/after excluding a distant horizon cluster (`roi_top_pct`) | Root-caused to DeepSORT re-issuing IDs for tiny/distant vehicles; confirmed the horizon-crop fix works better than confidence-threshold tuning alone |
| Chatbot accuracy | Cross-checked 10 real chatbot answers against the underlying CSV data by hand | Found and fixed two real bugs: an LLM-narration mismatch on multi-vehicle answers (fixed by formatting multi-row results deterministically in Python instead of trusting free-form LLM prose), and a vehicle-mislabeling bug in `build_vehicle_summary()` (was using each track's *first* detected frame's label instead of its most common one) |

---

## 🏗️ System Architecture

### Processing Pipeline:
```
Video Input → YOLOv8 Detection (per frame)
→ DeepSORT Tracking (persistent IDs)
→ Speed Estimation (pixel displacement → km/h)
→ Line-Crossing Counter (by direction)
→ Annotated Video + CSV Log + Vehicle Summary
```

### Chatbot Pipeline (text-to-pandas, not "LLM does math"):
```
User Question → LLM writes ONE pandas expression
→ AST safety check (no imports/file-access/dunders)
→ pandas executes it against the real vehicle_summary table
→ Scalar result: LLM phrases it in plain language
→ Multi-row result: formatted directly in Python (zero hallucination risk)
```

---

## 🚀 How to Run

### Step 1 — Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
First run auto-downloads YOLOv8 nano weights (`yolov8n.pt`, ~6MB).

### Step 2 — Calibrate speed (per video)
```bash
python calibration_helper.py --video sample_data/your_video.mp4 --output grid.png
```
Measure a known real-world distance against the pixel grid it generates.

### Step 3 — Run it (3 ways)
```bash
# Live web dashboard
streamlit run app.py

# CLI: video in, annotated video + CSV out
python process_video.py --input sample_data/traffic_sample.mp4 --output output_annotated.mp4 \
    --real_distance_m 6 --pixel_distance 46

# Or open vehicle_speed_estimation.ipynb for a step-by-step walkthrough
```

### Step 4 — Enable the chatbot (optional)
```bash
cp .env.example .env
```
Add your free Groq API key (no credit card needed) at [console.groq.com/keys](https://console.groq.com/keys), then ask questions like *"how many trucks crossed the line?"* or *"sabse tez gaadi ki speed kitni thi?"*

---

## 📁 Project Structure

```
trafficiq-chat/
├── app.py                          # Streamlit live dashboard
├── process_video.py                # CLI: video in -> annotated video + CSV out
├── calibration_helper.py           # Pixel-grid overlay for speed calibration
├── vehicle_speed_estimation.ipynb  # Notebook walkthrough of the full pipeline
├── requirements.txt
├── .env.example                    # Copy to .env, add your GROQ_API_KEY
├── sample_data/                    # Put a test video here
├── modules/
│   ├── detector.py          # YOLOv8 wrapper -> vehicle detections
│   ├── tracker.py           # DeepSORT wrapper -> persistent track IDs
│   ├── speed_estimator.py   # Pixel displacement -> real-world speed
│   ├── utils.py             # Line-crossing counter + drawing helpers
│   └── chatbot.py           # Text-to-pandas Q&A over the results (Groq LLM)
├── .gitignore
└── README.md
```

---

## 🎯 Calibration Tips
- Best accuracy comes from a top-down or fixed-angle camera (CCTV, overhead traffic cam)
- **Don't guess the pixel distance** — always run `calibration_helper.py` first
- Measure your real-world reference near the counting line specifically (perspective changes pixel-per-meter across the frame)
- Recalibrate for every new video/camera — a calibration from one video does not transfer to another

---

## ⚠️ Known Limitations

- **Accuracy is best for clear, near-camera vehicles.** Distant vehicles near the horizon are small and low-contrast, so detection is less consistent frame-to-frame.
- **Overcounting from ID switching** can still occur for distant/dense traffic — the `roi_top_pct` option (excluding the horizon from detection) is the confirmed, effective fix.
- **Speed accuracy depends on a single meters-per-pixel calibration**, which is only exactly correct at the distance in the frame where it was measured — an expected limitation of monocular (single fixed camera, no depth sensor) speed estimation, not a bug.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Speeds look way too low/high | Recalibrate with `calibration_helper.py`, measuring near the counting line |
| Vehicle count way too high / IDs in the hundreds | Set `roi_top_pct` to exclude a distant/cluttered horizon from detection |
| `pkg_resources` / setuptools error on install | `pip install "setuptools<81"` — a `deep-sort-realtime` dependency issue |
| `UnpicklingError` loading the YOLO model | Fixed automatically in `detector.py` — a PyTorch 2.6+/older-Ultralytics compatibility patch |
| Chatbot: "empty response" or "safety check failed" | Usually a transient model hiccup — the app auto-retries once; try asking again |
| Notebook: `ModuleNotFoundError` for installed packages | Wrong Jupyter kernel selected — pick the project's `venv`/`.venv` interpreter, not the global Python |

---

## 🌍 Real-World Applications

- **Traffic monitoring** — automated vehicle counting and speed enforcement
- **Smart city infrastructure** — real-time congestion and flow analytics
- **Urban planning** — data-driven insight into traffic patterns by direction and time
- **Law enforcement** — non-intrusive speed monitoring from existing camera feeds

---

## 👨‍💻 Author

**Hamza Asif**  
BS Artificial Intelligence — DUET, Karachi  
[![GitHub](https://img.shields.io/badge/GitHub-Hamza--Asif--ai-black?style=flat-square&logo=github)](https://github.com/Hamza-Asif-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hamza%20Asif-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/hamzaasif-ai)

---
