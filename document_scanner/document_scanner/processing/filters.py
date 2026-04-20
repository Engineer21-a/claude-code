from __future__ import annotations

from typing import Callable, Dict, Optional

import cv2
import numpy as np

FilterFn = Callable[[np.ndarray], np.ndarray]

# ── optional doxapy for fast Sauvola ────────────────────────────────────────
_DOXAPY_AVAILABLE = False
try:
    import doxapy as _doxapy
    _DOXAPY_AVAILABLE = True
except ImportError:
    pass


# ── individual filter functions ──────────────────────────────────────────────

def original(image: np.ndarray) -> np.ndarray:
    return image.copy()


def grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def bw_otsu(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def bw_sauvola(image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if _DOXAPY_AVAILABLE:
        binary = np.empty(gray.shape, gray.dtype)
        _doxapy.Binarize.updateToSauvola(gray, binary, {"window": window_size, "k": k})
    else:
        binary = _sauvola_numpy(gray, window_size=window_size, k=k)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _sauvola_numpy(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window_size, window_size))
    sq_mean = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(window_size, window_size))
    variance = np.maximum(sq_mean - mean ** 2, 0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    binary = np.where(gray_f >= threshold, 255, 0).astype(np.uint8)
    return binary


def enhanced_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def magic_color(image: np.ndarray, saturation_boost: float = 1.4) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return enhanced_clahe(result, clip_limit=1.5)


def blueprint(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blue = np.zeros((*gray.shape, 3), dtype=np.uint8)
    blue[:, :, 0] = 180  # B
    blue[:, :, 1] = 100  # G
    blue[:, :, 2] = 30   # R  → dark navy background
    # lines are white
    line_mask = thresh == 255
    blue[line_mask] = [255, 255, 255]
    return blue


def vintage(image: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32) / 255.0
    # Warm sepia matrix (BGR order)
    r = result[:, :, 2] * 0.393 + result[:, :, 1] * 0.769 + result[:, :, 0] * 0.189
    g = result[:, :, 2] * 0.349 + result[:, :, 1] * 0.686 + result[:, :, 0] * 0.168
    b = result[:, :, 2] * 0.272 + result[:, :, 1] * 0.534 + result[:, :, 0] * 0.131
    sepia = np.stack([b, g, r], axis=-1)
    sepia = np.clip(sepia, 0.0, 1.0)
    # Slight vignette
    h, w = sepia.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.5 * dist, 0.0, 1.0)
    sepia *= vignette[:, :, np.newaxis]
    return (sepia * 255).astype(np.uint8)


# ── custom adjustments ───────────────────────────────────────────────────────

def apply_custom(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    shadow: float = 0.0,
    highlight: float = 0.0,
    **_kwargs,
) -> np.ndarray:
    img = image.astype(np.float32) / 255.0

    # Brightness
    if brightness != 0.0:
        img = img + brightness

    # Contrast
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5

    img = np.clip(img, 0.0, 1.0)

    # Shadow / Highlight curve
    if shadow != 0.0 or highlight != 0.0:
        img = _apply_shadow_highlight(img, shadow, highlight)

    result = (img * 255).astype(np.uint8)

    # Saturation (in HSV)
    if saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def _apply_shadow_highlight(img: np.ndarray, shadow: float, highlight: float) -> np.ndarray:
    # shadow lifts dark tones; highlight compresses bright tones
    luma = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    shadow_weight = np.clip(1.0 - luma * 2.0, 0.0, 1.0)
    highlight_weight = np.clip((luma - 0.5) * 2.0, 0.0, 1.0)
    adjustment = shadow * shadow_weight[:, :, np.newaxis] + highlight * highlight_weight[:, :, np.newaxis]
    return np.clip(img + adjustment, 0.0, 1.0)


# ── dispatcher ───────────────────────────────────────────────────────────────

PRESETS: Dict[str, Optional[FilterFn]] = {
    "original": original,
    "grayscale": grayscale,
    "bw_otsu": bw_otsu,
    "bw_sauvola": bw_sauvola,
    "enhanced_clahe": enhanced_clahe,
    "magic_color": magic_color,
    "blueprint": blueprint,
    "vintage": vintage,
    "custom": None,  # sentinel — use apply_custom
}


def apply_filter(
    image: np.ndarray,
    preset: str = "original",
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    **kwargs,
) -> np.ndarray:
    fn = PRESETS.get(preset)
    if fn is None:
        return apply_custom(image, brightness=brightness, contrast=contrast,
                            saturation=saturation, **kwargs)
    return fn(image)
