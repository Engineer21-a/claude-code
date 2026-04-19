from __future__ import annotations

import numpy as np

from ..config import DetectorConfig, TrackerConfig
from ..models import BoundingBox, Detection, DetectionClass, InteractiveSelection


class InteractiveTracker:
    """
    Accepts a user-defined bounding box on a seed frame, uses SAM2 to generate
    a precise segmentation mask, then returns a Detection to seed the tracker.

    SAM2 is loaded lazily to avoid the ~2 GB model download for runs that
    don't use interactive selection.
    """

    def __init__(self, det_cfg: DetectorConfig, tracker_cfg: TrackerConfig) -> None:
        self._det_cfg = det_cfg
        self._tracker_cfg = tracker_cfg
        self._predictor = None

    def _load_sam2(self) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-small",
            device=self._det_cfg.device,
        )

    def get_mask_for_selection(
        self,
        frame: np.ndarray,
        selection: InteractiveSelection,
    ) -> np.ndarray:
        """
        Run SAM2 with a bounding-box prompt on the seed frame (BGR).
        Returns a binary mask (H×W uint8, values 0 or 255).
        """
        if self._predictor is None:
            self._load_sam2()

        assert self._predictor is not None
        rgb = frame[:, :, ::-1].copy()
        self._predictor.set_image(rgb)
        box = np.array(
            [selection.bbox.x1, selection.bbox.y1, selection.bbox.x2, selection.bbox.y2],
            dtype=float,
        )
        masks, _scores, _ = self._predictor.predict(
            box=box,
            multimask_output=False,
        )
        return (masks[0] * 255).astype(np.uint8)

    def mask_to_bbox(self, mask: np.ndarray) -> BoundingBox:
        """Convert a binary mask to its tight bounding box."""
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not rows.any():
            return BoundingBox(0, 0, mask.shape[1], mask.shape[0])
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y1, y2 = int(y_indices[0]), int(y_indices[-1])
        x1, x2 = int(x_indices[0]), int(x_indices[-1])
        return BoundingBox(x1, y1, x2 + 1, y2 + 1)

    def create_seed_detection(
        self,
        frame: np.ndarray,
        selection: InteractiveSelection,
    ) -> Detection:
        """SAM2 → mask → tight bbox → Detection for tracker seeding."""
        mask = self.get_mask_for_selection(frame, selection)
        bbox = self.mask_to_bbox(mask)
        return Detection(
            bbox=bbox,
            detection_class=DetectionClass.INTERACTIVE,
            confidence=1.0,
            frame_index=selection.frame_index,
        )
