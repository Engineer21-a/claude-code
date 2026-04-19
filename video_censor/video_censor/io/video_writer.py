from __future__ import annotations

import cv2
import numpy as np

from .video_reader import VideoMetadata


class VideoWriter:
    """Wraps cv2.VideoWriter. Writes MP4 with mp4v codec."""

    def __init__(self, path: str, metadata: VideoMetadata) -> None:
        self._path = path
        self._metadata = metadata
        self._writer: cv2.VideoWriter | None = None

    def open(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._path,
            fourcc,
            self._metadata.fps,
            (self._metadata.width, self._metadata.height),
        )
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Cannot open output video for writing: {self._path!r}")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter is not open. Use as context manager or call open().")
        self._writer.write(frame)
