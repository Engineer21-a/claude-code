from __future__ import annotations

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
      6. Apply censoring effects.
      7. Write the censored frame to the output video.
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

        with VideoReader(self._cfg.io.input_path) as reader:
            meta = reader.metadata
            stats.total_frames = meta.total_frames

            with VideoWriter(self._cfg.io.output_path, meta) as writer:
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
                        bboxes = [d.bbox for d in tracked_fd.detections]
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
        return stats


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
