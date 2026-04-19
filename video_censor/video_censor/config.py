from __future__ import annotations

from dataclasses import dataclass, field

from .models import CensorMethod, InteractiveSelection


@dataclass
class DetectorConfig:
    person_model: str = "yolo11n.pt"
    license_plate_model: str = "keremberke/yolov8-license-plate-detection"
    logo_model: str = "openfoodfacts/universal-logo-detector"
    confidence_threshold: float = 0.4
    device: str = "cpu"


@dataclass
class TrackerConfig:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30


@dataclass
class CensorConfig:
    method: CensorMethod = CensorMethod.GAUSSIAN_BLUR
    blur_kernel_size: int = 51
    pixelate_block_size: int = 15
    padding_px: int = 5


@dataclass
class IOConfig:
    input_path: str = ""
    output_path: str = ""
    batch_size: int = 8


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    censor: CensorConfig = field(default_factory=CensorConfig)
    io: IOConfig = field(default_factory=IOConfig)
    censor_persons: bool = True
    censor_license_plates: bool = True
    censor_logos: bool = True
    interactive_selections: list[InteractiveSelection] = field(default_factory=list)
    show_progress: bool = True
