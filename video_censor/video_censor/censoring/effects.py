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
    """Apply censoring effect to one bounding box. Returns a modified copy; never mutates input."""
    h, w = frame.shape[:2]
    padded = _pad_bbox(bbox, cfg.padding_px, w, h)
    if not padded.is_valid():
        return frame.copy()
    result = frame.copy()
    _apply_inplace(result, padded, cfg)
    return result


def apply_censor_many(
    frame: np.ndarray,
    bboxes: list[BoundingBox],
    cfg: CensorConfig,
) -> np.ndarray:
    """Apply censoring to multiple bounding boxes. Makes exactly one copy of the frame."""
    if not bboxes:
        return frame.copy()
    h, w = frame.shape[:2]
    result = frame.copy()
    for bbox in bboxes:
        padded = _pad_bbox(bbox, cfg.padding_px, w, h)
        if padded.is_valid():
            _apply_inplace(result, padded, cfg)
    return result


def _apply_inplace(frame: np.ndarray, bbox: BoundingBox, cfg: CensorConfig) -> None:
    """Mutate frame in-place within bbox. Caller is responsible for having made a copy."""
    if cfg.method == CensorMethod.GAUSSIAN_BLUR:
        _apply_gaussian_blur(frame, bbox, cfg.blur_kernel_size)
    elif cfg.method == CensorMethod.PIXELATE:
        _apply_pixelate(frame, bbox, cfg.pixelate_block_size)
    elif cfg.method == CensorMethod.SOLID_BLACK:
        _apply_solid_black(frame, bbox)
    else:
        raise ValueError(f"Unknown censor method: {cfg.method!r}")


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
) -> None:
    # Ensure odd kernel size >= 3; bitwise OR with 1 makes any even number odd
    k = max(3, kernel_size | 1)
    roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    if roi.size == 0:
        return
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = blurred


def _apply_pixelate(
    frame: np.ndarray,
    bbox: BoundingBox,
    block_size: int,
) -> None:
    roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return
    bw = max(1, block_size)
    small_w = max(1, w // bw)
    small_h = max(1, h // bw)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = pixelated


def _apply_solid_black(frame: np.ndarray, bbox: BoundingBox) -> None:
    frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = 0
