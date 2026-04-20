from __future__ import annotations

import cv2
import numpy as np

# ── optional Real-ESRGAN ─────────────────────────────────────────────────────
_REALESRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer as _RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet as _RRDBNet
    _REALESRGAN_AVAILABLE = True
except ImportError:
    pass


def unsharp_mask(
    image: np.ndarray,
    radius: int = 2,
    amount: float = 1.5,
    threshold: int = 0,
) -> np.ndarray:
    """Unsharp masking: sharpened = original + amount*(original - blur)."""
    ksize = 2 * radius + 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        mask = np.abs(image.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharpened = np.where(mask, image, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _realesrgan_sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """2× upscale via Real-ESRGAN then downscale back → sharpening without ringing."""
    import torch

    model = _RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=2)
    upsampler = _RealESRGANer(
        scale=2,
        model_path=None,  # auto-downloads weights
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    upscaled, _ = upsampler.enhance(image, outscale=2)
    h, w = image.shape[:2]
    downscaled = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_AREA)

    # Blend with original based on strength
    if strength < 1.0:
        downscaled = cv2.addWeighted(image, 1.0 - strength, downscaled, strength, 0)
    return downscaled.astype(np.uint8)


class Enhancer:
    """Image sharpener with graceful fallback chain: Real-ESRGAN → unsharp mask."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            self._backend = "realesrgan" if _REALESRGAN_AVAILABLE else "unsharp"
        elif backend == "realesrgan" and not _REALESRGAN_AVAILABLE:
            raise RuntimeError("realesrgan is not installed")
        else:
            self._backend = backend

    @property
    def backend_name(self) -> str:
        return self._backend

    def sharpen(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Sharpen image. strength 0=none, 1=normal, 2=aggressive."""
        if strength < 0.01:
            return image.copy()
        if self._backend == "realesrgan":
            try:
                return _realesrgan_sharpen(image, strength)
            except Exception:
                pass  # fall through to unsharp mask
        # unsharp mask — scale amount by strength
        return unsharp_mask(image, radius=2, amount=1.5 * strength)
