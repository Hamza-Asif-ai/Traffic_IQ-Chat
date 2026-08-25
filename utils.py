"""
utils.py
Line-crossing vehicle counter + drawing helpers.
"""

import cv2


class LineCounter:
    """
    Counts a vehicle exactly once when its centroid crosses a horizontal
    counting line (defined by a single y-coordinate, spanning the frame width).
    Also tracks WHICH DIRECTION each vehicle was moving when it crossed
    (e.g. "towards the camera" vs "away from the camera"), since a single
    combined total doesn't distinguish opposing carriageways/lanes.
    """

    def __init__(self, line_y: int, direction_labels=("down", "up")):
        """
        direction_labels: (label_for_downward_crossing, label_for_upward_crossing)
            "downward" means the vehicle's y-coordinate increased (moved from
            above the line to below it, i.e. top-to-bottom on screen).
            Rename these to whatever makes sense for your camera angle, e.g.
            ("towards_camera", "away_from_camera") or ("northbound", "southbound").
        """
        self.line_y = line_y
        self.counted_ids = set()
        self.count = 0
        self.direction_labels = direction_labels
        self.count_by_direction = {direction_labels[0]: 0, direction_labels[1]: 0}
        self._last_side = {}       # track_id -> 'above'/'below'
        self.crossing_frame = {}   # track_id -> frame_idx at moment of crossing
        self.crossing_direction = {}  # track_id -> direction label at moment of crossing

    def update(self, track_id, bbox, frame_idx=None):
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2.0
        side = "above" if cy < self.line_y else "below"

        prev_side = self._last_side.get(track_id)
        if prev_side is not None and prev_side != side and track_id not in self.counted_ids:
            self.count += 1
            self.counted_ids.add(track_id)

            direction = self.direction_labels[0] if side == "below" else self.direction_labels[1]
            self.count_by_direction[direction] += 1
            self.crossing_direction[track_id] = direction

            if frame_idx is not None:
                self.crossing_frame[track_id] = frame_idx

        self._last_side[track_id] = side
        return self.count


def build_vehicle_summary(speed_log, counter: "LineCounter", fps: float):
    """
    Turns the raw per-frame speed_log (list of dicts) into one row per
    vehicle -- this compact table is what gets handed to the chatbot,
    since it's small enough to reason over reliably regardless of video
    length.

    Returns a pandas DataFrame with columns:
        track_id, label, avg_speed_kmh, max_speed_kmh,
        first_seen_sec, last_seen_sec, crossed_line, crossing_time_sec
    """
    import pandas as pd

    if not speed_log:
        return pd.DataFrame(columns=[
            "track_id", "label", "avg_speed_kmh", "max_speed_kmh",
            "first_seen_sec", "last_seen_sec", "crossed_line", "crossing_time_sec",
        ])

    df = pd.DataFrame(speed_log)
    summary = df.groupby("track_id").agg(
        # Use the most common label across all frames, not just the first one --
        # the first frame a vehicle is detected in is usually when it's smallest/
        # most distant/hardest to classify, so "first" is the LEAST reliable
        # single frame to trust for vehicles YOLO occasionally misclassifies
        # (e.g. large cars vs small trucks/vans flip-flopping frame to frame).
        label=("label", lambda s: s.value_counts().idxmax()),
        avg_speed_kmh=("speed_kmh", "mean"),
        max_speed_kmh=("speed_kmh", "max"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
    ).reset_index()

    summary["first_seen_sec"] = (summary["first_frame"] / fps).round(2)
    summary["last_seen_sec"] = (summary["last_frame"] / fps).round(2)
    summary["avg_speed_kmh"] = summary["avg_speed_kmh"].round(1)
    summary["max_speed_kmh"] = summary["max_speed_kmh"].round(1)

    summary["crossed_line"] = summary["track_id"].isin(counter.counted_ids)
    summary["crossing_time_sec"] = summary["track_id"].map(
        lambda tid: round(counter.crossing_frame[tid] / fps, 2) if tid in counter.crossing_frame else None
    )
    summary["crossing_direction"] = summary["track_id"].map(
        lambda tid: counter.crossing_direction.get(tid, None)
    )

    return summary.drop(columns=["first_frame", "last_frame"])


def draw_annotations(frame, tracks, speeds, line_y, count, count_by_direction=None):
    """Draws bounding boxes, IDs, speed labels, the counting line, and a
    single grouped stats box (total + directional counts) in the top-left
    corner, instead of scattered loose lines of text."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        tid = track["id"]
        label = track["label"]
        speed = speeds.get(tid)

        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"ID {tid} {label}"
        if speed is not None:
            text += f" {speed:.1f} km/h"

        cv2.putText(frame, text, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Grouped stats box, top-left corner ---
    lines = [f"Total: {count}"]
    if count_by_direction:
        for direction, dir_count in count_by_direction.items():
            lines.append(f"{direction}: {dir_count}")

    padding = 12
    line_height = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    text_widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    box_w = max(text_widths) + padding * 2
    box_h = line_height * len(lines) + padding * 2

    # Semi-transparent black background so text stays readable over any footage
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 255, 255), 1)

    for i, line in enumerate(lines):
        y = 10 + padding + line_height * i + int(line_height * 0.7)
        text_color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Total in cyan, rest in white
        cv2.putText(frame, line, (10 + padding, y), font, font_scale, text_color, thickness)

    return frame