from __future__ import annotations

import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline

from ..config import DetectorConfig
from ..models import BoundingBox, Detection, DetectionClass, FrameDetections
from .base import bgr_to_rgb


class LogoDetector:
    """Detects brand logos and trademarks using a HuggingFace object-detection pipeline."""

    detection_class = DetectionClass.LOGO

    def __init__(self, cfg: DetectorConfig) -> None:
        self._pipe = hf_pipeline(
            "object-detection",
            model=cfg.logo_model,
            device=0 if cfg.device == "cuda" else -1,
            threshold=cfg.confidence_threshold,
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
        start_frame_index: int,
    ) -> list[FrameDetections]:
        output: list[FrameDetections] = []
        for i, frame in enumerate(frames):
            pil_img = Image.fromarray(bgr_to_rgb(frame))
            results = self._pipe(pil_img)
            fd = FrameDetections(frame_index=start_frame_index + i)
            for r in results:
                box = r["box"]
                fd.detections.append(
                    Detection(
                        bbox=BoundingBox(
                            int(box["xmin"]),
                            int(box["ymin"]),
                            int(box["xmax"]),
                            int(box["ymax"]),
                        ),
                        detection_class=DetectionClass.LOGO,
                        confidence=float(r["score"]),
                        frame_index=start_frame_index + i,
                    )
                )
            output.append(fd)
        return output

    def warmup(self) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detect_batch([dummy], start_frame_index=0)
