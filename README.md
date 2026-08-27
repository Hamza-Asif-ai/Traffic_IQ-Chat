<div align="center">

# 🚦 TrafficIQ Chat

**Vehicle Counting, Speed Estimation & Conversational Traffic Analytics**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com/)

A computer-vision pipeline that watches traffic video and turns it into
structured, queryable data — detection, tracking, speed, and directional
counts, plus a built-in chatbot to ask about the results in plain English
or Roman Urdu.

</div>

---

### ✨ Features
- 🚗 **Vehicle detection** — YOLOv8, filtered to car / motorcycle / bus / truck
- 🔗 **Persistent tracking** — DeepSORT keeps a stable ID per vehicle across frames
- 📏 **Speed estimation** — real-world km/h from a one-time pixel calibration
- ↔️ **Directional counting** — incoming vs. outgoing traffic, tracked separately
- 📊 **Live dashboard** — Streamlit app with charts, CSV export, and annotated video
- 💬 **LLM chatbot** — ask questions about the results in plain English *or* Roman Urdu, answered via a safe text-to-pandas pipeline (no hallucinated numbers)
- 📓 **Three ways to run it** — Streamlit app, CLI script, or a walkthrough notebook — all sharing one codebase

### 📚 Table of contents
- [Project structure](#project-structure)
- [Setup](#setup)
- [Three ways to run this](#three-ways-to-run-this)
- [How it works](#how-it-works)
- [Directional counting](#directional-counting-both-carriageways-separately)
- [Calibration tips](#calibration-tips)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting-unrealistic-results)
- [💬 Chatbot: ask your traffic data questions](#-bonus-feature-ask-your-traffic-data-questions-llm-chatbot)
- [Possible extensions](#possible-extensions-good-future-work-talking-points-in-interviews)
- [Resume bullet](#resume-bullet)

---

## Project structure
```
trafficiq-chat/
├── app.py                          # Streamlit live dashboard (webcam or upload, no file saved)
├── process_video.py                # CLI script: video IN -> fully annotated video OUT + CSV
├── calibration_helper.py           # generates a pixel-grid overlay to help you calibrate speed
├── vehicle_speed_estimation.ipynb  # Jupyter/Colab notebook walkthrough of the whole pipeline
├── requirements.txt
├── .env.example                    # copy to .env and add your GROQ_API_KEY
├── sample_data/                    # put a test video here
├── modules/
│   ├── detector.py          # YOLOv8 wrapper -> vehicle detections
│   ├── tracker.py           # DeepSORT wrapper -> persistent track IDs
│   ├── speed_estimator.py   # pixel displacement -> real-world speed
│   ├── utils.py             # line-crossing counter + drawing helpers
│   └── chatbot.py           # text-to-pandas Q&A over the results (Groq LLM)
├── .gitignore
└── README.md
```

All three entry points (`app.py`, `process_video.py`, the notebook) reuse the
exact same `modules/` code — one pipeline, three ways to run it.

---

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run auto-downloads YOLOv8 nano weights (`yolov8n.pt`, ~6MB).

## Three ways to run this

### 1. Live interactive dashboard (Streamlit)
```bash
streamlit run app.py
```
Opens in your browser. Upload a video or use your webcam. Shows the live
annotated feed, running count, FPS, and a speed chart — nothing is saved to
disk automatically here (this is the "watch it happen live" mode).

### 2. Video-in, video-out (command line) — this is what most people want
Give it a video file, it gives you back a new annotated video file plus a
CSV of every vehicle's speed:
```bash
python process_video.py \
    --input sample_data/traffic_sample.mp4 \
    --output output_annotated.mp4 \
    --real_distance_m 3.0 \
    --pixel_distance 100 \
    --line_pct 60
```
- `--real_distance_m` / `--pixel_distance`: your speed calibration (see below)
- `--line_pct`: where the counting line sits (60 = 60% down the frame)
- `--roi_top_pct`: optional (default 0) — excludes the top N% of the frame
  from detection. Use this if your footage has a distant, cluttered horizon
  causing overcounting (see Troubleshooting below).
- Output: `output_annotated.mp4` (every frame has boxes, IDs, speed labels,
  the counting line, and the running count burned in) + `output_annotated_speed_log.csv`

### 3. Notebook walkthrough (Jupyter / Google Colab)
Open `vehicle_speed_estimation.ipynb`. It runs the same pipeline step by
step with explanations, previews your first frame so you can pick a good
counting-line position, and produces the same annotated output video +
charts inline. Good for demoing the project or exploring/tweaking things
interactively.

---

## How it works
1. **Detection** — Each frame is passed to YOLOv8, filtered to vehicle classes
   (car, motorcycle, bus, truck).
2. **Tracking** — Detections go into DeepSORT, which uses motion + appearance
   embeddings to keep a stable ID on each vehicle across frames (so the same
   car isn't recounted).
3. **Counting** — A virtual horizontal line is drawn across the frame. When a
   tracked vehicle's centroid crosses it, the count increments once per ID.
4. **Speed estimation** — You calibrate meters-per-pixel once (measure a known
   real-world distance, e.g. lane width, and its pixel length in the frame).
   Each vehicle's frame-to-frame pixel displacement is converted to real-world
   distance and divided by elapsed time (using video FPS) to get speed, then
   smoothed over a short rolling window to reduce jitter.
5. **Dashboard** — Streamlit shows the annotated live feed, running count,
   processing FPS, an average-speed-by-vehicle-type chart, and a downloadable
   CSV log of every tracked vehicle's speed.

## Directional counting (both carriageways separately)
If your video shows traffic moving in two opposite directions (e.g. a highway
with a near and far carriageway), the counting line reports each direction
separately, not just one combined total:
- **`app.py`**: sidebar lets you label each direction (e.g. "Towards camera" /
  "Away from camera"), and the dashboard shows both counts side by side.
- **`process_video.py`**: use `--label_down` and `--label_up` to name the two
  directions; the printed summary and the burned-in video overlay show both.
- **Chatbot**: the per-vehicle summary table includes a `crossing_direction`
  column, so you can ask things like "how many vehicles went away from the
  camera?"

Direction is determined by which way a vehicle's box moved across the line
(top-to-bottom vs bottom-to-top on screen) — rename the default "down"/"up"
labels to whatever matches your camera angle (e.g. northbound/southbound).

## Calibration tips
- Best accuracy comes from a top-down or fixed-angle camera (e.g. CCTV,
  overhead traffic cam) where perspective distortion is minimal.
- **Don't guess the pixel distance.** Run `calibration_helper.py` first:
  ```bash
  python calibration_helper.py --video sample_data/your_video.mp4 --output grid.png
  ```
  This overlays a labeled pixel grid on your video's first frame so you can
  accurately read off distances instead of eyeballing them.
- Use a real reference near your counting line specifically (perspective
  makes pixel-per-meter different at the top vs. bottom of the frame):
  lane width (~3.5-3.7m on most highways/UK motorways), distance between
  road markings, or a fixed object of known size.
- For serious accuracy, apply a perspective transform (homography) instead of
  a single meters-per-pixel ratio — noted as a possible extension below.

## Known limitations
Tested against real motorway/highway CCTV footage (distant, elevated, oblique
camera angle) — honest results, not just theory:
- **Accuracy is best for clear, near-camera vehicles.** Distant vehicles near
  the horizon are small and low-contrast, so detection confidence is lower
  and less consistent frame-to-frame.
- **Overcounting from ID switching is a real issue** for distant/dense
  traffic: when the tracker briefly loses a small vehicle, it gets a new ID
  on re-detection, and the line counter — correctly, from its point of view —
  counts each ID once. Raising `conf_threshold` alone helps only a little.
  The effective fix (confirmed by testing): use the `roi_top_pct` option to
  exclude the distant horizon region from detection entirely — see
  Troubleshooting below.
- **Speed accuracy depends on a single meters-per-pixel calibration**, which
  is only exactly correct at the distance in the frame where you measured it
  — perspective means it's an approximation everywhere else in the frame. In
  practice this is enough to get vehicles into a believable speed range, not
  frame-perfect radar-gun accuracy.
- This is a known, expected limitation of **monocular (single fixed camera,
  no depth sensor) speed estimation** — not a bug. The homography-based
  perspective correction listed under "Possible extensions" below is the
  standard way production systems solve this properly.

---

## Troubleshooting: unrealistic results
- **Speeds look way too low/high:** your calibration is off. Re-check it with
  `calibration_helper.py` and measure near the counting line, not far from it.
- **Vehicle count seems way too high / IDs climbing into the hundreds for a
  short clip:** this is DeepSORT losing and re-creating tracks (ID switching)
  for small/distant vehicles near the horizon. **The fix that actually works**
  (confirmed by testing, works much better than threshold tuning alone): set
  `roi_top_pct` (Streamlit slider, or `--roi_top_pct` in `process_video.py`)
  to exclude that distant region from detection entirely — e.g. `40` excludes
  the top 40% of the frame. Leave it at `0` for footage without a distant
  horizon cluster. Raising `conf_threshold` and `tracker.py`'s `max_age`
  (already tuned to 60) can help marginally too, but don't rely on them alone.

---

## 💬 Bonus feature: Ask your traffic data questions (LLM chatbot)

This is the differentiator most similar LinkedIn projects don't have: after
processing a video, you can ask plain-English (or Roman Urdu) questions like:
- "How many trucks crossed the line?"
- "What was the fastest vehicle's speed?"
- "Average car speed after the first 20 seconds?"

**How it works (and why it's accurate, not guessed):** instead of dumping raw
numbers into an LLM and hoping it does math correctly, this uses a
**text-to-pandas** pattern:
1. The LLM reads only the table's *schema* (column names/types + a few sample
   rows) and writes one pandas expression that would answer your question.
2. That expression is checked against a strict allow-list (AST-based — no
   imports, no file access, no dunder attributes, and file I/O methods like
   `read_csv`/`to_pickle` are explicitly blocked even though they aren't
   dunders) before it's ever executed.
3. The real, computed pandas result is handed back to the LLM, which turns it
   into a short natural-language answer.

This means the actual counting/math is done by pandas (always correct), and
the LLM just handles the translation between English and code — a genuinely
useful "agentic" pattern, not just an LLM guessing at numbers.

Powered by **Groq** (free, no credit card required):
```bash
pip install groq python-dotenv   # already in requirements.txt
cp .env.example .env             # then edit .env with your real key
```
Get a free API key at https://console.groq.com/keys

**Use it:**
- In `app.py`: paste your API key in the sidebar, process a video, then use
  the "💬 Ask Your Traffic Data" box that appears at the bottom.
- From the command line:
  ```bash
  python process_video.py --input traffic.mp4 --output result.mp4 \
      --real_distance_m 3.0 --pixel_distance 100 \
      --ask "how many trucks crossed the line?"
  ```
- In the notebook: import `modules.chatbot` and call
  `chatbot.ask("your question", vehicle_summary_df)`.

---

## Possible extensions (good "future work" talking points in interviews)
- Homography-based perspective correction for angled camera views
- Per-lane speed limit violation alerts
- Export annotated video, not just the live view
- Swap DeepSORT for ByteTrack (lighter, no re-ID embedding needed)
- Deploy on edge devices via ONNX/TensorRT export for higher FPS

---

## Resume bullet
> Built a real-time vehicle counting and speed estimation system using YOLOv8
> and DeepSORT; deployed via Streamlit with a live analytics dashboard
> (per-vehicle speed tracking, CSV export, average speed by vehicle type) and
> a natural-language Q&A feature powered by an LLM (Groq), using a safe text-to-pandas
> pipeline for accurate, hallucination-free answers about the traffic data.

<div align="center">

---

Built by **Hamza Asif** · [GitHub](https://github.com/hamza93-ai)

</div>
