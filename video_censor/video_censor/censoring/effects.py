from __future__ import annotations

import cv2
import numpy as np

from ..config import CensorConfig
from ..models import BoundingBox, CensorMethod


def apply_censor(
    frame: np.ndarray,
    bbox: BoundingBox,
    cfg: CensorConfig,
) -> np.ndarray:
    """Apply censoring effect to one bounding box region. Returns a modified copy."""
    h, w = frame.shape[:2]
    padded = _pad_bbox(bbox, cfg.padding_px, w, h)
    if not padded.is_valid():
        return frame.copy()
    result = frame.copy()
    if cfg.method == CensorMethod.GAUSSIAN_BLUR:
        return _apply_gaussian_blur(result, padded, cfg.blur_kernel_size)
    if cfg.method == CensorMethod.PIXELATE:
        return _apply_pixelate(result, padded, cfg.pixelate_block_size)
    if cfg.method == CensorMethod.SOLID_BLACK:
        return _apply_solid_black(result, padded)
    raise ValueError(f"Unknown censor method: {cfg.method!r}")


def apply_censor_many(
    frame: np.ndarray,
    bboxes: list[BoundingBox],
    cfg: CensorConfig,
) -> np.ndarray:
    """Apply censoring to multiple bounding boxes sequentially."""
    result = frame.copy()
    for bbox in bboxes:
        result = apply_censor(result, bbox, cfg)
    return result


def _pad_bbox(bbox: BoundingBox, padding: int, frame_w: int, frame_h: int) -> BoundingBox:
    return BoundingBox(
        x1=max(0, bbox.x1 - padding),
        y1=max(0, bbox.y1 - padding),
        x2=min(frame_w, bbox.x2 + padding),
        y2=min(frame_h, bbox.y2 + padding),
    )


def _apply_gaussian_blur(
    frame: np.ndarray,
    bbox: BoundingBox,
    kernel_size: int,
) -> np.ndarray:
    # Ensure odd kernel size >= 3; bitwise OR with 1 makes any even number odd
    k = max(3, kernel_size | 1)
    roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    if roi.size == 0:
        return frame
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = blurred
    return frame


def _apply_pixelate(
    frame: np.ndarray,
    bbox: BoundingBox,
    block_size: int,
) -> np.ndarray:
    roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return frame
    bw = max(1, block_size)
    small_w = max(1, w // bw)
    small_h = max(1, h // bw)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = pixelated
    return frame


def _apply_solid_black(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = 0
    return frame
