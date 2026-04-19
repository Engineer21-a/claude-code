from __future__ import annotations

import numpy as np

from ..config import DetectorConfig
from ..models import BoundingBox, Detection, DetectionClass, FrameDetections
from .base import bgr_to_rgb


class PersonDetector:
    """Detects persons using YOLOv11 via the ultralytics library."""

    detection_class = DetectionClass.PERSON

    def __init__(self, cfg: DetectorConfig) -> None:
        from ultralytics import YOLO

        self._model = YOLO(cfg.person_model)
        self._model.to(cfg.device)
        self._conf = cfg.confidence_threshold

    def detect_batch(
        self,
        frames: list[np.ndarray],
        start_frame_index: int,
    ) -> list[FrameDetections]:
        from ultralytics import YOLO  # noqa: F401 — keep import local for testability

        results = self._model(
            [bgr_to_rgb(f) for f in frames],
            conf=self._conf,
            classes=[0],  # COCO class 0 == person
            verbose=False,
        )
        output: list[FrameDetections] = []
        for i, result in enumerate(results):
            fd = FrameDetections(frame_index=start_frame_index + i)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                fd.detections.append(
                    Detection(
                        bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                        detection_class=DetectionClass.PERSON,
                        confidence=float(box.conf[0]),
                        frame_index=start_frame_index + i,
                    )
                )
            output.append(fd)
        return output

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect_batch([dummy], start_frame_index=0)
