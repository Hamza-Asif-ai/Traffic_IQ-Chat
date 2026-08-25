"""
process_video.py
Batch/offline version of the pipeline: give it an input video, it writes out
a fully annotated OUTPUT VIDEO FILE (boxes, IDs, speed, count burned into
every frame) plus a CSV log of every vehicle's speed. No Streamlit needed —
useful for processing a video end-to-end and getting a shareable .mp4 back.

Usage:
    python process_video.py --input path/to/input_video.mp4 --output path/to/output_video.mp4 \
        --real_distance_m 3.0 --pixel_distance 100 --line_pct 60

Arguments:
    --input            path to source video
    --output           path to save the annotated output video (.mp4)
    --real_distance_m  known real-world distance used for calibration (meters)
    --pixel_distance   pixel length of that same distance in the frame
    --line_pct         counting line position, as % of frame height (default 60)
    --model            YOLOv8 weights to use (default yolov8n.pt)
    --conf             detection confidence threshold (default 0.35)
"""

import argparse
import cv2
import pandas as pd

from modules.detector import VehicleDetector
from modules.tracker import VehicleTracker
from modules.speed_estimator import SpeedEstimator
from modules.utils import LineCounter, draw_annotations, build_vehicle_summary


def process_video(input_path, output_path, real_distance_m, pixel_distance,
                   line_pct=60, model_path="yolov8n.pt", conf=0.35, csv_path=None,
                   label_down="down", label_up="up"):
    meters_per_pixel = real_distance_m / pixel_distance

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = VehicleDetector(model_path=model_path, conf_threshold=conf)
    tracker = VehicleTracker()
    speed_estimator = SpeedEstimator(meters_per_pixel=meters_per_pixel, fps=fps)
    counter = LineCounter(line_y=int(height * line_pct / 100), direction_labels=(label_down, label_up))

    speed_log = []
    frame_idx = 0

    print(f"Processing '{input_path}' -> '{output_path}' "
          f"({total_frames} frames @ {fps:.1f} fps)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                    "frame": frame_idx, "track_id": tid,
                    "label": track["label"], "speed_kmh": speed,
                })

        speed_estimator.cleanup(active_ids)

        annotated = draw_annotations(frame.copy(), tracks, speeds, counter.line_y, counter.count,
                                      counter.count_by_direction)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  ...{frame_idx}/{total_frames} frames done "
                  f"(count so far: {counter.count})")

    cap.release()
    writer.release()

    print(f"Done. Total vehicles counted: {counter.count}")
    print(f"  {label_down}: {counter.count_by_direction[label_down]}")
    print(f"  {label_up}: {counter.count_by_direction[label_up]}")
    print(f"Annotated video saved to: {output_path}")

    vehicle_summary = build_vehicle_summary(speed_log, counter, fps)

    if speed_log:
        df = pd.DataFrame(speed_log)
        if csv_path is None:
            csv_path = output_path.rsplit(".", 1)[0] + "_speed_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"Speed log saved to: {csv_path}")

    return counter.count, vehicle_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video: count vehicles & estimate speed.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save annotated output video (.mp4)")
    parser.add_argument("--real_distance_m", type=float, required=True,
                         help="Known real-world distance for calibration (meters)")
    parser.add_argument("--pixel_distance", type=float, required=True,
                         help="Pixel length of that same distance in the frame")
    parser.add_argument("--line_pct", type=float, default=60, help="Counting line position (%% of height)")
    parser.add_argument("--label_down", default="down", help="Label for top-to-bottom moving traffic")
    parser.add_argument("--label_up", default="up", help="Label for bottom-to-top moving traffic")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--csv", default=None, help="Optional custom path for the CSV speed log")
    parser.add_argument("--ask", default=None,
                         help="Optional: ask a natural-language question about the results "
                              "once processing finishes, e.g. --ask \"how many trucks crossed?\" "
                              "(requires GROQ_API_KEY environment variable, free at console.groq.com/keys)")
    args = parser.parse_args()

    _, vehicle_summary = process_video(
        input_path=args.input,
        output_path=args.output,
        real_distance_m=args.real_distance_m,
        pixel_distance=args.pixel_distance,
        line_pct=args.line_pct,
        model_path=args.model,
        conf=args.conf,
        csv_path=args.csv,
        label_down=args.label_down,
        label_up=args.label_up,
    )

    if args.ask:
        from modules import chatbot
        print(f"\nQ: {args.ask}")
        result = chatbot.ask(args.ask, vehicle_summary)
        print(f"A: {result['answer']}")
