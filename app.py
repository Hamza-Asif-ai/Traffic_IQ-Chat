"""
app.py
Real-time Vehicle Counting & Speed Estimation Dashboard
YOLOv8 (detection) + DeepSORT (tracking) + Streamlit (UI/analytics)

Run with:
    streamlit run app.py
"""

import time
import tempfile

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.detector import VehicleDetector
from modules.tracker import VehicleTracker
from modules.speed_estimator import SpeedEstimator
from modules.utils import LineCounter, draw_annotations, build_vehicle_summary
from modules import chatbot

st.set_page_config(page_title="TrafficIQ Chat", layout="wide")

st.title("🚗 TrafficIQ Chat — Vehicle Counting, Speed Estimation & Conversational Traffic Analytics")
st.caption("YOLOv8 + DeepSORT + Groq — detects, tracks, counts, estimates speed, and answers questions about your traffic video.")

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    source_type = st.radio("Video source", ["Upload a video", "Webcam"])

    uploaded_file = None
    if source_type == "Upload a video":
        uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])

    model_size = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt (fastest)", "yolov8s.pt (balanced)", "yolov8m.pt (accurate, slower)"],
        index=0,
    )
    model_path = model_size.split(" ")[0]

    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.5, 0.05)
    roi_top_pct = st.slider(
        "Exclude distant horizon (% of frame height, from top)", 0, 60, 0, 5,
        help="For elevated/distant camera footage, cropping out the tiny, far-away "
             "traffic cluster near the horizon fixes overcounting and speed jitter "
             "far more effectively than raising the confidence threshold alone. "
             "Leave at 0 unless you see that specific problem."
    ) / 100.0

    st.subheader("Speed calibration")
    st.markdown(
        "Measure a known real-world distance (e.g. lane width, distance between "
        "two lane markings) and its pixel length in your video, then enter:"
    )
    real_distance_m = st.number_input("Known real-world distance (meters)", value=12.0, min_value=0.1)
    pixel_distance = st.number_input("Same distance in pixels (in the frame)", value=123.0, min_value=1.0)
    meters_per_pixel = real_distance_m / pixel_distance

    st.subheader("Counting line")
    line_y_pct = st.slider("Counting line position (% of frame height)", 10, 90, 60)
    st.caption("Vehicles moving top-to-bottom on screen vs bottom-to-top are counted separately (e.g. two opposite carriageways).")
    label_down = st.text_input("Label for downward-moving traffic (top → bottom)", value="Incoming")
    label_up = st.text_input("Label for upward-moving traffic (bottom → top)", value="Outgoing")

    st.subheader("Chatbot (optional)")
    groq_api_key = st.text_input(
        "Groq API key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        key="groq_api_key_input",
        help="Needed only if you want to ask questions about the results afterwards. "
             "Get a free key (no credit card needed) at https://console.groq.com/keys. "
             "Leave blank to skip. This stays filled in for the whole session, so you "
             "can ask as many questions as you like without re-entering it.",
    )
    if groq_api_key:
        st.session_state["groq_api_key"] = groq_api_key

    run_button = st.button("▶ Start Processing")

# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
frame_placeholder = st.empty()
col1, col2, col3, col4, col5 = st.columns(5)
count_metric = col1.empty()
dir1_metric = col2.empty()
dir2_metric = col3.empty()
fps_metric = col4.empty()
active_tracks_metric = col5.empty()

chart_placeholder = st.empty()

if run_button:
    if source_type == "Upload a video" and uploaded_file is None:
        st.warning("Please upload a video file first.")
        st.stop()

    # Resolve video source
    if source_type == "Webcam":
        video_source = 0
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_source = tfile.name

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detector = VehicleDetector(model_path=model_path, conf_threshold=conf_threshold, roi_top_pct=roi_top_pct)
    tracker = VehicleTracker()
    speed_estimator = SpeedEstimator(meters_per_pixel=meters_per_pixel, fps=fps)

    frame_idx = 0
    speed_log = []  # for the analytics dashboard: {track_id, frame, speed, label}
    counter = None

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if counter is None:
            h, w = frame.shape[:2]
            line_y = int(h * line_y_pct / 100)
            counter = LineCounter(line_y=line_y, direction_labels=(label_down, label_up))

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        speeds = {}
        active_ids = set()
        for track in tracks:
            tid = track["id"]
            active_ids.add(tid)
            speed = speed_estimator.update(tid, track["bbox"], frame_idx)
            speeds[tid] = speed
            counter.update(tid, track["bbox"], frame_idx)

            if speed is not None:
                speed_log.append({
                    "frame": frame_idx,
                    "track_id": tid,
                    "label": track["label"],
                    "speed_kmh": speed,
                })

        speed_estimator.cleanup(active_ids)

        annotated = draw_annotations(frame.copy(), tracks, speeds, counter.line_y, counter.count,
                                      counter.count_by_direction)

        # FPS calc (processing speed, not source fps)
        now = time.time()
        proc_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        count_metric.metric("Total Counted", counter.count)
        dir1_metric.metric(label_down, counter.count_by_direction[label_down])
        dir2_metric.metric(label_up, counter.count_by_direction[label_up])
        fps_metric.metric("Processing FPS", f"{proc_fps:.1f}")
        active_tracks_metric.metric("Active Tracks", len(tracks))

        # Live-updating analytics chart every 15 frames (avoid re-render overhead)
        if speed_log and frame_idx % 15 == 0:
            df = pd.DataFrame(speed_log)
            avg_by_label = df.groupby("label")["speed_kmh"].mean().reset_index()
            fig = px.bar(avg_by_label, x="label", y="speed_kmh",
                         title="Average Speed by Vehicle Type (km/h)",
                         labels={"speed_kmh": "Avg Speed (km/h)", "label": "Vehicle Type"})
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        frame_idx += 1

    cap.release()
    st.success(f"Done. Total vehicles counted: {counter.count if counter else 0}")

    if speed_log:
        df = pd.DataFrame(speed_log)
        vehicle_summary = build_vehicle_summary(speed_log, counter, fps)

        st.subheader("📊 Full Session Analytics")
        st.dataframe(vehicle_summary)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download raw speed log (CSV)", csv, "speed_log.csv", "text/csv")

        # Stash results so the chat section below (outside this if-block,
        # so it survives Streamlit reruns) can use them.
        st.session_state["vehicle_summary"] = vehicle_summary
        st.session_state["groq_api_key"] = groq_api_key
else:
    st.info("Configure settings in the sidebar, then click **Start Processing**.")

# ---------------------------------------------------------------------------
# Chat with your traffic data
# ---------------------------------------------------------------------------
if "vehicle_summary" in st.session_state and not st.session_state["vehicle_summary"].empty:
    st.divider()
    st.subheader("💬 Ask Your Traffic Data")
    st.caption(
        "Ask in plain English or Roman Urdu, e.g. \"how many trucks crossed the line?\", "
        "\"what was the fastest vehicle's speed?\", \"average car speed after 20 seconds\"."
    )

    question = st.text_input("Your question", key="chat_question")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        api_key = st.session_state.get("groq_api_key") or None
        try:
            with st.spinner("Thinking..."):
                result = chatbot.ask(question, st.session_state["vehicle_summary"], api_key=api_key)
            st.success(result["answer"])
            with st.expander("Show how this was calculated"):
                st.code(result["pandas_code"], language="python")
                st.write("Raw result:", result["raw_result"])
        except ValueError as e:
            st.error(str(e))