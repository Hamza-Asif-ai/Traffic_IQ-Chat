"""
speed_estimator.py
Estimates each tracked vehicle's speed using the distance it travels
(in real-world meters) between frames, divided by elapsed time.

Calibration approach:
    You provide `meters_per_pixel`, i.e. how many real-world meters one
    pixel represents in the video. The simplest way to get this number:
    1. Pick two points in the frame whose real-world distance you know
       (e.g. lane markings ~3m apart, or a fixed object of known length).
    2. Measure the pixel distance between them in the image.
    3. meters_per_pixel = real_world_distance_m / pixel_distance

Outlier filtering:
    Occasionally the tracker briefly loses a vehicle and re-associates it
    a few pixels away from where a *different* nearby vehicle actually is
    (an ID/position glitch, not real motion). Naively computing speed from
    that single frame-to-frame jump can produce wildly unrealistic values
    (e.g. "500+ km/h"). Any single-frame speed above `max_plausible_kmh`
    is treated as tracking noise and discarded rather than being folded
    into the smoothed average.
"""

from collections import deque, defaultdict


class SpeedEstimator:
    def __init__(self, meters_per_pixel: float, fps: float, smoothing_window: int = 5,
                 max_plausible_kmh: float = 180.0):
        """
        meters_per_pixel: calibration factor (see module docstring)
        fps: video frame rate (frames per second)
        smoothing_window: number of recent speed samples to average, to
                           reduce jitter from detection/tracking noise
        max_plausible_kmh: any single-frame speed reading above this is
                           discarded as a tracking glitch rather than used.
                           Default 180 km/h comfortably covers real highway
                           traffic while rejecting impossible spikes.
        """
        self.meters_per_pixel = meters_per_pixel
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.max_plausible_kmh = max_plausible_kmh

        # per track_id: deque of recent centroid positions (x, y, frame_idx)
        self.history = defaultdict(lambda: deque(maxlen=smoothing_window + 1))
        # per track_id: deque of recent speed estimates (km/h)
        self.speed_history = defaultdict(lambda: deque(maxlen=smoothing_window))

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, track_id, bbox, frame_idx):
        """
        Call once per track, per frame. Returns the current smoothed
        speed estimate in km/h (float), or None if not enough data yet.
        """
        cx, cy = self._centroid(bbox)
        self.history[track_id].append((cx, cy, frame_idx))

        if len(self.history[track_id]) < 2:
            return None

        (x_prev, y_prev, f_prev), (x_curr, y_curr, f_curr) = (
            self.history[track_id][-2],
            self.history[track_id][-1],
        )

        frame_gap = f_curr - f_prev
        if frame_gap <= 0:
            return self.get_smoothed_speed(track_id)

        pixel_dist = ((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2) ** 0.5
        meters = pixel_dist * self.meters_per_pixel
        seconds = frame_gap / self.fps
        speed_mps = meters / seconds
        speed_kmh = speed_mps * 3.6

        if speed_kmh <= self.max_plausible_kmh:
            self.speed_history[track_id].append(speed_kmh)
        # else: likely a tracking glitch (sudden position jump), not real
        # motion -- skip it so it doesn't corrupt the smoothed average.

        return self.get_smoothed_speed(track_id)

    def get_smoothed_speed(self, track_id):
        history = self.speed_history[track_id]
        if not history:
            return None
        return sum(history) / len(history)

    def cleanup(self, active_track_ids):
        """Drop history for tracks that no longer exist, to avoid memory growth."""
        for tid in list(self.history.keys()):
            if tid not in active_track_ids:
                self.history.pop(tid, None)
                self.speed_history.pop(tid, None)
