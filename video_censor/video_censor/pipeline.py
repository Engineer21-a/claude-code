from __future__ import annotations

import os
import tempfile
import time
from typing import Callable

import numpy as np

from .censoring.effects import apply_censor_many
from .config import AppConfig
from .detectors.base import BaseDetector
from .detectors.license_plate import LicensePlateDetector
from .detectors.logo import LogoDetector
from .detectors.person import PersonDetector
from .io.video_reader import VideoReader
from .io.video_writer import VideoWriter
from .models import FrameDetections, ProcessingStats
from .tracking.interactive import InteractiveTracker
from .tracking.tracker import MultiClassTracker

ProgressCallback = Callable[[int, int], None]


class CensorPipeline:
    """
    Top-level orchestrator.

    Build → Warm-up → Run:
      1. Read frames in batches.
      2. Run all active detectors on the batch.
      3. Merge per-detector results into one FrameDetections per frame.
      4. Inject interactive-selection seed detections at the right frame.
      5. Update the tracker frame-by-frame (ByteTrack is sequential).
      6. Clamp all bboxes to frame dimensions (guards against negative numpy indices).
      7. Apply censoring effects.
      8. Write to a temp file; rename atomically on success.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._detectors: list[BaseDetector] = []
        self._tracker = MultiClassTracker(cfg.tracker)
        self._interactive_tracker: InteractiveTracker | None = None
        self._interactive_seeds: dict[int, object] = {}  # frame_index → InteractiveSelection

    # ------------------------------------------------------------------
    # Setup

    def build(self) -> None:
        """Instantiate and warm-up all detectors. Call once before run()."""
        if self._cfg.censor_persons:
            self._detectors.append(PersonDetector(self._cfg.detector))
        if self._cfg.censor_license_plates:
            self._detectors.append(LicensePlateDetector(self._cfg.detector))
        if self._cfg.censor_logos:
            self._detectors.append(LogoDetector(self._cfg.detector))

        if self._cfg.interactive_selections:
            self._interactive_tracker = InteractiveTracker(
                self._cfg.detector, self._cfg.tracker
            )
            for sel in self._cfg.interactive_selections:
                self._interactive_seeds[sel.frame_index] = sel

        for det in self._detectors:
            det.warmup()

    # ------------------------------------------------------------------
    # Main loop

    def run(self, on_progress: ProgressCallback | None = None) -> ProcessingStats:
        """Process the input video and write the censored output. Returns stats."""
        stats = ProcessingStats()
        start_time = time.monotonic()

        output_path = self._cfg.io.output_path
        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."

        # Write to a temp file in the same directory so os.rename is atomic
        # (same filesystem). On success we rename; on failure the partial file is
        # discarded and the original output (if any) is not overwritten.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp.mp4")
        os.close(tmp_fd)

        try:
            self._process(tmp_path, stats, on_progress)
        except BaseException:
            # Remove the incomplete temp file on any error (including Ctrl-C)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        os.replace(tmp_path, output_path)
        return stats

    def _process(
        self,
        output_path: str,
        stats: ProcessingStats,
        on_progress: ProgressCallback | None,
    ) -> None:
        start_time = time.monotonic()

        with VideoReader(self._cfg.io.input_path) as reader:
            meta = reader.metadata

            if meta.width <= 0 or meta.height <= 0:
                raise ValueError(
                    f"Invalid video dimensions: {meta.width}×{meta.height}. "
                    "The file may be corrupt or unsupported."
                )

            stats.total_frames = meta.total_frames
            frame_w, frame_h = meta.width, meta.height

            with VideoWriter(output_path, meta) as writer:
                frame_index = 0
                batch_size = self._cfg.io.batch_size

                while True:
                    batch = reader.read_batch(batch_size)
                    if not batch:
                        break

                    merged = _merge_batch_detections(
                        [det.detect_batch(batch, frame_index) for det in self._detectors],
                        len(batch),
                        frame_index,
                    )

                    if self._interactive_tracker is not None:
                        for offset, frame in enumerate(batch):
                            fi = frame_index + offset
                            if fi in self._interactive_seeds:
                                seed_det = self._interactive_tracker.create_seed_detection(
                                    frame, self._interactive_seeds[fi]  # type: ignore[arg-type]
                                )
                                merged[offset].detections.append(seed_det)

                    tracked_batch: list[FrameDetections] = []
                    for fd in merged:
                        tracked_batch.append(self._tracker.update(fd))

                    for frame, tracked_fd in zip(batch, tracked_batch):
                        # Clamp all bboxes to frame dimensions before censoring.
                        # Models can produce out-of-bounds coords; negative numpy
                        # indices wrap around and would censor the wrong region,
                        # leaking private content.
                        bboxes = [
                            d.bbox.clamp(frame_w, frame_h)
                            for d in tracked_fd.detections
                            if d.bbox.clamp(frame_w, frame_h).is_valid()
                        ]
                        censored = apply_censor_many(frame, bboxes, self._cfg.censor)
                        writer.write_frame(censored)
                        stats.processed_frames += 1
                        stats.total_censored_regions += len(bboxes)
                        if on_progress is not None:
                            on_progress(stats.processed_frames, stats.total_frames)

                    frame_index += len(batch)

        elapsed = time.monotonic() - start_time
        stats.elapsed_seconds = elapsed
        stats.fps = stats.processed_frames / elapsed if elapsed > 0 else 0.0


def _merge_batch_detections(
    per_detector_results: list[list[FrameDetections]],
    batch_len: int,
    start_frame_index: int,
) -> list[FrameDetections]:
    """
    Merge per-detector FrameDetections lists into one FrameDetections per frame.

    per_detector_results: one list per detector, each containing one FrameDetections per frame.
    Returns: one FrameDetections per frame with all detectors' results concatenated.
    """
    merged = [
        FrameDetections(frame_index=start_frame_index + i) for i in range(batch_len)
    ]
    for detector_results in per_detector_results:
        for i, fd in enumerate(detector_results):
            if i < len(merged):
                merged[i].detections.extend(fd.detections)
    return merged
