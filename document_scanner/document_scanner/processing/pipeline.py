from __future__ import annotations

from dataclasses import asdict
from typing import Callable, List, Optional

import numpy as np

from document_scanner.models.document import DocumentImage
from document_scanner.processing.dewarper import Dewarper
from document_scanner.processing.enhancer import Enhancer
from document_scanner.processing.filters import apply_filter
from document_scanner.processing.perspective import warp_perspective


def _corners_are_identity(image: np.ndarray, corners: list) -> bool:
    """True when corners exactly match the image boundary."""
    h, w = image.shape[:2]
    identity = [(0, 0), (w, 0), (w, h), (0, h)]
    return list(corners) == identity


class ProcessingPipeline:
    """Orchestrates all processing steps for one DocumentImage."""

    def __init__(
        self,
        dewarper: Optional[Dewarper] = None,
        enhancer: Optional[Enhancer] = None,
    ) -> None:
        self._dewarper = dewarper or Dewarper()
        self._enhancer = enhancer or Enhancer()

    def process(self, doc: DocumentImage, preview: bool = False) -> np.ndarray:
        """Run the full pipeline. preview=True skips slow SOTA dewarping."""
        img = doc.original.copy()

        # Step 1: Perspective warp (only when corners are non-trivial)
        if doc.corners and not _corners_are_identity(img, doc.corners):
            img = warp_perspective(img, doc.corners)

        # Step 2: SOTA dewarping (skip in preview mode for speed)
        if doc.dewarp_enabled and not preview:
            img = self._dewarper.dewarp(img)

        # Step 3: Filter
        settings = asdict(doc.filter_settings)
        img = apply_filter(img, **settings)

        # Step 4: Sharpening
        if doc.sharpen_enabled and doc.filter_settings.sharpness > 0.01:
            img = self._enhancer.sharpen(img, strength=doc.filter_settings.sharpness)

        doc.processed = img
        doc.needs_reprocess = False
        return img

    def process_all(
        self,
        docs: List[DocumentImage],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        total = len(docs)
        for i, doc in enumerate(docs):
            self.process(doc)
            if progress_callback:
                progress_callback(i + 1, total)


# ── background worker (requires PySide6) ─────────────────────────────────────
try:
    from PySide6.QtCore import QThread, Signal

    class PipelineWorker(QThread):
        finished = Signal(object)   # np.ndarray
        error = Signal(str)
        progress = Signal(int, int)

        def __init__(
            self,
            pipeline: ProcessingPipeline,
            docs: List[DocumentImage],
            preview: bool = False,
        ) -> None:
            super().__init__()
            self._pipeline = pipeline
            self._docs = docs
            self._preview = preview
            self._stop_requested = False

        def run(self) -> None:
            try:
                total = len(self._docs)
                for i, doc in enumerate(self._docs):
                    if self._stop_requested:
                        break
                    result = self._pipeline.process(doc, preview=self._preview)
                    self.progress.emit(i + 1, total)
                    if len(self._docs) == 1:
                        self.finished.emit(result)
                if len(self._docs) > 1:
                    self.finished.emit(None)
            except Exception as exc:
                self.error.emit(str(exc))

        def stop(self) -> None:
            self._stop_requested = True

except ImportError:
    # PySide6 not available — PipelineWorker simply won't exist
    pass
