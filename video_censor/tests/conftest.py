import numpy as np
import pytest

from video_censor.config import AppConfig, CensorConfig, DetectorConfig, IOConfig, TrackerConfig
from video_censor.models import (
    BoundingBox,
    CensorMethod,
    Detection,
    DetectionClass,
    FrameDetections,
    InteractiveSelection,
)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """480×640 BGR frame filled with mid-gray."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def black_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def sample_bbox() -> BoundingBox:
    return BoundingBox(x1=100, y1=50, x2=300, y2=200)


@pytest.fixture
def sample_detection(sample_bbox: BoundingBox) -> Detection:
    return Detection(
        bbox=sample_bbox,
        detection_class=DetectionClass.PERSON,
        confidence=0.9,
        track_id=1,
        frame_index=0,
    )


@pytest.fixture
def sample_frame_detections(sample_detection: Detection) -> FrameDetections:
    return FrameDetections(frame_index=0, detections=[sample_detection])


@pytest.fixture
def sample_interactive_selection() -> InteractiveSelection:
    return InteractiveSelection(
        frame_index=0,
        bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
    )


@pytest.fixture
def default_censor_config() -> CensorConfig:
    return CensorConfig()


@pytest.fixture
def solid_black_censor_config() -> CensorConfig:
    return CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)


@pytest.fixture
def default_detector_config() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def default_tracker_config() -> TrackerConfig:
    return TrackerConfig()


@pytest.fixture
def default_app_config(tmp_path) -> AppConfig:
    return AppConfig(
        io=IOConfig(
            input_path=str(tmp_path / "input.mp4"),
            output_path=str(tmp_path / "output.mp4"),
        )
    )
