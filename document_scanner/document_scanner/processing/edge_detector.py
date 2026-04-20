from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def auto_detect_corners(image: np.ndarray) -> List[Tuple[int, int]]:
    """Return 4 document corner points [TL, TR, BR, BL].

    Falls back to image boundary corners if no document contour is found.
    """
    h, w = image.shape[:2]
    fallback = [(0, 0), (w, 0), (w, h), (0, h)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny thresholds based on median pixel intensity
    median = float(np.median(blurred))
    lo = int(max(0, 0.67 * median))
    hi = int(min(255, 1.33 * median))
    # Ensure meaningful separation even for very dark/bright images
    if hi - lo < 20:
        lo, hi = 30, 150

    edges = cv2.Canny(blurred, lo, hi)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback

    # Sort by area descending; check top candidates for a 4-vertex polygon
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            return _order_corners(pts)

    return fallback


def _order_corners(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort 4 unordered points into [TL, TR, BR, BL] order."""
    pts_arr = np.array(pts, dtype=np.int32)
    s = pts_arr.sum(axis=1)       # x+y: TL=min, BR=max
    d = pts_arr[:, 0] - pts_arr[:, 1]  # x-y: TR=max, BL=min

    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return [tl, tr, br, bl]
