"""
detector.py
Wraps a YOLOv8 model to detect vehicles in a single frame.
"""

import torch
from ultralytics import YOLO
import numpy as np

# --- PyTorch 2.6+ / ultralytics==8.2.0 compatibility fix ---------------------
# PyTorch 2.6 changed torch.load's default from weights_only=False to True.
# ultralytics==8.2.0 (May 2024) predates that change and doesn't pass
# weights_only=False itself, so loading yolov8n.pt/s/m raises
# `UnpicklingError: Weights only load failed` on any modern torch install.
# requirements.txt intentionally doesn't pin torch (heavy, platform-specific
# wheels), so we can't just pin our way out of this -- patch torch.load
# instead. Safe here: the .pt file is auto-downloaded straight from
# Ultralytics' official GitHub releases, not user-supplied.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
# -------------------------------------------------------------------------

# COCO class ids that count as "vehicles"
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35,
                 device: str = "cpu", roi_top_pct: float = 0.0):
        """
        model_path: path to a YOLOv8 weights file. 'yolov8n.pt' auto-downloads
                    the nano model on first run (fastest, good for real-time demos).
                    Swap for 'yolov8s.pt' / 'yolov8m.pt' for better accuracy.
        conf_threshold: minimum confidence to keep a detection.
        device: 'cpu', 'cuda', or 'mps'
        roi_top_pct: fraction (0.0-0.9) of the frame height to exclude from the
                     TOP of the frame before detection even runs. Use this for
                     footage with a distant horizon cluster of tiny/far vehicles
                     (e.g. an elevated highway camera) -- raising conf_threshold
                     alone often isn't enough to stop those from being detected
                     inconsistently frame-to-frame, which causes DeepSORT to
                     keep losing/re-creating IDs for them (ID churn), which in
                     turn causes the line counter to overcount the same vehicle
                     multiple times. Cropping the region out entirely removes
                     the problem instead of just filtering around its edges.
                     e.g. 0.35 excludes the top 35% of the frame.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.roi_top_pct = max(0.0, min(roi_top_pct, 0.9))

    def detect(self, frame: np.ndarray):
        """
        Runs YOLOv8 on a single BGR frame.

        Returns a list of detections in the format DeepSORT expects:
            [ ([x, y, w, h], confidence, class_name), ... ]
        where (x, y) is the top-left corner of the box, in ORIGINAL frame
        coordinates (already offset back if roi_top_pct was used).
        """
        h, _ = frame.shape[:2]
        y_offset = int(h * self.roi_top_pct)
        search_frame = frame[y_offset:, :] if y_offset > 0 else frame

        results = self.model.predict(
            search_frame,
            conf=self.conf_threshold,
            classes=list(VEHICLE_CLASS_IDS.keys()),
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            y1 += y_offset
            y2 += y_offset
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            w, ht = x2 - x1, y2 - y1
            label = VEHICLE_CLASS_IDS.get(cls_id, "vehicle")
            detections.append(([x1, y1, w, ht], conf, label))

        return detections