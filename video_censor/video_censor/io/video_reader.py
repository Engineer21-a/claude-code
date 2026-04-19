from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    fourcc: int


class VideoReader:
    """Wraps cv2.VideoCapture for sequential or random-access frame reading."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            self._cap = None
            raise FileNotFoundError(f"Cannot open video: {self._path!r}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def metadata(self) -> VideoMetadata:
        self._assert_open()
        assert self._cap is not None
        return VideoMetadata(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fourcc=int(self._cap.get(cv2.CAP_PROP_FOURCC)),
        )

    def read_frame(self) -> np.ndarray | None:
        """Read the next frame. Returns None at EOF."""
        self._assert_open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        return frame if ok else None

    def read_batch(self, batch_size: int) -> list[np.ndarray]:
        """Read up to batch_size frames. Returns a shorter list (possibly empty) at EOF."""
        batch: list[np.ndarray] = []
        for _ in range(batch_size):
            frame = self.read_frame()
            if frame is None:
                break
            batch.append(frame)
        return batch

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame index (0-based)."""
        self._assert_open()
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def read_frame_at(self, frame_index: int) -> np.ndarray | None:
        """Read a specific frame by index. Restores the current position after reading."""
        self._assert_open()
        assert self._cap is not None
        saved = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.seek(frame_index)
        frame = self.read_frame()
        self.seek(saved)
        return frame

    def _assert_open(self) -> None:
        if self._cap is None:
            raise RuntimeError("VideoReader is not open. Use as context manager or call open().")
