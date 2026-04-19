from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..models import BoundingBox, DetectionClass, FrameDetections


@runtime_checkable
class BaseDetector(Protocol):
    """Protocol all detectors must satisfy."""

    detection_class: DetectionClass

    def detect_batch(
        self,
        frames: list[np.ndarray],
        start_frame_index: int,
    ) -> list[FrameDetections]:
        """
        Run inference on a batch of BGR frames.
        Returns one FrameDetections per input frame, in the same order.
        """
        ...

    def warmup(self) -> None:
        """Run one dummy inference to initialise model caches."""
        ...


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Flip channel order from BGR (OpenCV) to RGB (most models)."""
    return frame[:, :, ::-1].copy()


def xyxy_to_bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Convert float XYXY coordinates to integer BoundingBox."""
    return BoundingBox(int(x1), int(y1), int(x2), int(y2))
