from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── capability detection (never raises) ─────────────────────────────────────
_DOCUWARP_AVAILABLE = False
_PAGEDEWARP_AVAILABLE = False

try:
    from docuwarp.unwarp import Unwarp as _DocuwarpUnwarp
    _DOCUWARP_AVAILABLE = True
except ImportError:
    pass

try:
    _PAGEDEWARP_AVAILABLE = shutil.which("page-dewarp") is not None
except Exception:
    pass


def download_model_weights(cache_dir: Optional[Path] = None) -> None:
    """Pre-download docuwarp model weights (no-op if docuwarp not installed)."""
    if not _DOCUWARP_AVAILABLE:
        return
    try:
        _DocuwarpUnwarp()  # triggers weight download on first instantiation
    except Exception:
        pass


def _try_docuwarp(image: np.ndarray) -> np.ndarray:
    unwarp = _DocuwarpUnwarp()
    # docuwarp expects BGR ndarray; returns BGR ndarray
    result = unwarp.unwarp_image_array(image)
    if result is None:
        return image
    return result.astype(np.uint8)


def _try_pagedewarp(image: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.png"
        cv2.imwrite(str(in_path), image)
        try:
            subprocess.run(
                ["page-dewarp", str(in_path)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            # page-dewarp writes <stem>_thresh.png next to input
            out_path = in_path.parent / (in_path.stem + "_thresh.png")
            if out_path.exists():
                result = cv2.imread(str(out_path))
                if result is not None:
                    return result.astype(np.uint8)
        except Exception:
            pass
    return image


class Dewarper:
    """SOTA document dewarping with graceful fallback: docuwarp → page-dewarp → identity."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            if _DOCUWARP_AVAILABLE:
                self._backend = "docuwarp"
            elif _PAGEDEWARP_AVAILABLE:
                self._backend = "pagedewarp"
            else:
                self._backend = "none"
        else:
            self._backend = backend

    @property
    def backend_name(self) -> str:
        return self._backend

    def dewarp(self, image: np.ndarray) -> np.ndarray:
        """Dewarp image. Returns copy unchanged if no backend available."""
        if self._backend == "docuwarp" and _DOCUWARP_AVAILABLE:
            try:
                return _try_docuwarp(image)
            except Exception:
                pass
        if self._backend == "pagedewarp" and _PAGEDEWARP_AVAILABLE:
            try:
                return _try_pagedewarp(image)
            except Exception:
                pass
        return image.copy()
