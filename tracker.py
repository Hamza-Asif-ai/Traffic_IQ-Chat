"""
tracker.py
Wraps deep-sort-realtime to assign persistent IDs to vehicles across frames.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(self, max_age: int = 60, n_init: int = 2,
                 max_iou_distance: float = 0.9, max_cosine_distance: float = 0.3):
        """
        max_age: how many frames a lost track is kept alive before deletion.
                 Raised from the DeepSORT default (30) because fast highway
                 traffic can be briefly occluded or missed by the detector
                 for several frames -- a low max_age causes the same vehicle
                 to be re-assigned a NEW id (inflating the count) instead of
                 recovering its old one.
        n_init: how many consecutive detections needed to confirm a new track.
                Lowered slightly so fast-moving vehicles get confirmed before
                they've already crossed the counting line.
        max_iou_distance: how lenient the motion-based (IOU) matching is.
                Raised because vehicles filmed from a distance/overhead move
                many pixels between frames relative to their box size, so a
                strict IOU threshold treats them as a new object every frame.
        max_cosine_distance: how lenient the appearance-based re-identification
                is when recovering a briefly lost track.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
        )

    def update(self, detections, frame):
        """
        detections: output of VehicleDetector.detect()
        frame: current BGR frame (used internally by DeepSORT for appearance embedding)

        Returns a list of confirmed tracks:
            [ {"id": track_id, "bbox": (x1, y1, x2, y2), "label": class_name}, ... ]
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            results.append({
                "id": track.track_id,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label": track.get_det_class() if track.get_det_class() else "vehicle",
            })
        return results
