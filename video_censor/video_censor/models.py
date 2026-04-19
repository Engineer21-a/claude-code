from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DetectionClass(str, Enum):
    PERSON = "person"
    LICENSE_PLATE = "license_plate"
    LOGO = "logo"
    INTERACTIVE = "interactive"


class CensorMethod(str, Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    PIXELATE = "pixelate"
    SOLID_BLACK = "solid_black"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coords: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: int
    y1: int
    x2: int
    y2: int

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def clamp(self, frame_w: int, frame_h: int) -> BoundingBox:
        return BoundingBox(
            x1=max(0, min(self.x1, frame_w)),
            y1=max(0, min(self.y1, frame_h)),
            x2=max(0, min(self.x2, frame_w)),
            y2=max(0, min(self.y2, frame_h)),
        )

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1


@dataclass
class Detection:
    bbox: BoundingBox
    detection_class: DetectionClass
    confidence: float
    track_id: Optional[int] = None
    frame_index: int = 0


@dataclass
class FrameDetections:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)


@dataclass
class InteractiveSelection:
    frame_index: int
    bbox: BoundingBox


@dataclass
class ProcessingStats:
    total_frames: int = 0
    processed_frames: int = 0
    total_censored_regions: int = 0
    elapsed_seconds: float = 0.0
    fps: float = 0.0
