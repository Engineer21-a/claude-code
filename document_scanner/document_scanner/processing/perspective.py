from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def warp_perspective(
    image: np.ndarray,
    corners: List[Tuple[int, int]],
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Apply 4-corner perspective warp.

    corners: [TL, TR, BR, BL]
    output_size: (width, height); auto-computed from corner distances if None.
    """
    if output_size is None:
        output_size = _compute_output_size(corners)

    w, h = output_size
    src = np.array(corners, dtype=np.float32)
    dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)


def _compute_output_size(corners: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Estimate output (width, height) from corner distances."""
    tl, tr, br, bl = corners

    top_w = np.linalg.norm(np.array(tr) - np.array(tl))
    bot_w = np.linalg.norm(np.array(br) - np.array(bl))
    left_h = np.linalg.norm(np.array(bl) - np.array(tl))
    right_h = np.linalg.norm(np.array(br) - np.array(tr))

    w = max(int(top_w), int(bot_w))
    h = max(int(left_h), int(right_h))
    return max(w, 1), max(h, 1)
