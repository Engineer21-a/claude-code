import pytest

from video_censor.models import (
    BoundingBox,
    CensorMethod,
    Detection,
    DetectionClass,
    FrameDetections,
    InteractiveSelection,
    ProcessingStats,
)


class TestBoundingBox:
    def test_area_normal(self):
        bb = BoundingBox(0, 0, 100, 50)
        assert bb.area() == 5000

    def test_area_zero_width(self):
        bb = BoundingBox(10, 10, 10, 50)
        assert bb.area() == 0

    def test_area_zero_height(self):
        bb = BoundingBox(10, 10, 50, 10)
        assert bb.area() == 0

    def test_area_inverted_box_returns_zero(self):
        bb = BoundingBox(50, 50, 10, 10)
        assert bb.area() == 0

    def test_clamp_within_frame(self):
        bb = BoundingBox(10, 10, 100, 100)
        clamped = bb.clamp(640, 480)
        assert clamped == bb

    def test_clamp_left_top_edge(self):
        bb = BoundingBox(-10, -5, 100, 100)
        clamped = bb.clamp(640, 480)
        assert clamped.x1 == 0
        assert clamped.y1 == 0

    def test_clamp_right_bottom_edge(self):
        bb = BoundingBox(100, 100, 700, 500)
        clamped = bb.clamp(640, 480)
        assert clamped.x2 == 640
        assert clamped.y2 == 480

    def test_to_xyxy(self):
        bb = BoundingBox(1, 2, 3, 4)
        assert bb.to_xyxy() == (1, 2, 3, 4)

    def test_to_xywh(self):
        bb = BoundingBox(10, 20, 50, 80)
        x, y, w, h = bb.to_xywh()
        assert x == 10 and y == 20
        assert w == 40 and h == 60

    def test_is_valid_normal(self):
        assert BoundingBox(0, 0, 10, 10).is_valid()

    def test_is_valid_zero_width(self):
        assert not BoundingBox(5, 0, 5, 10).is_valid()

    def test_is_valid_zero_height(self):
        assert not BoundingBox(0, 5, 10, 5).is_valid()

    def test_is_frozen(self):
        bb = BoundingBox(0, 0, 10, 10)
        with pytest.raises((TypeError, AttributeError)):
            bb.x1 = 99  # type: ignore[misc]

    def test_equality(self):
        assert BoundingBox(1, 2, 3, 4) == BoundingBox(1, 2, 3, 4)
        assert BoundingBox(1, 2, 3, 4) != BoundingBox(1, 2, 3, 5)


class TestDetectionClass:
    def test_string_values(self):
        assert DetectionClass.PERSON.value == "person"
        assert DetectionClass.LICENSE_PLATE.value == "license_plate"
        assert DetectionClass.LOGO.value == "logo"
        assert DetectionClass.INTERACTIVE.value == "interactive"

    def test_from_value(self):
        assert DetectionClass("person") == DetectionClass.PERSON


class TestCensorMethod:
    def test_string_values(self):
        assert CensorMethod.GAUSSIAN_BLUR.value == "gaussian_blur"
        assert CensorMethod.PIXELATE.value == "pixelate"
        assert CensorMethod.SOLID_BLACK.value == "solid_black"


class TestDetection:
    def test_default_track_id_is_none(self):
        det = Detection(BoundingBox(0, 0, 10, 10), DetectionClass.PERSON, 0.9)
        assert det.track_id is None

    def test_default_frame_index_is_zero(self):
        det = Detection(BoundingBox(0, 0, 10, 10), DetectionClass.PERSON, 0.9)
        assert det.frame_index == 0

    def test_all_fields_set(self):
        bb = BoundingBox(1, 2, 3, 4)
        det = Detection(bb, DetectionClass.LOGO, 0.5, track_id=7, frame_index=42)
        assert det.bbox == bb
        assert det.detection_class == DetectionClass.LOGO
        assert det.confidence == 0.5
        assert det.track_id == 7
        assert det.frame_index == 42


class TestFrameDetections:
    def test_default_empty_detections(self):
        fd = FrameDetections(frame_index=5)
        assert fd.detections == []
        assert fd.frame_index == 5

    def test_detections_list_mutable(self):
        fd = FrameDetections(frame_index=0)
        fd.detections.append(
            Detection(BoundingBox(0, 0, 1, 1), DetectionClass.PERSON, 0.9)
        )
        assert len(fd.detections) == 1


class TestInteractiveSelection:
    def test_fields(self):
        bb = BoundingBox(10, 20, 30, 40)
        sel = InteractiveSelection(frame_index=5, bbox=bb)
        assert sel.frame_index == 5
        assert sel.bbox == bb


class TestProcessingStats:
    def test_defaults_are_zero(self):
        stats = ProcessingStats()
        assert stats.total_frames == 0
        assert stats.processed_frames == 0
        assert stats.total_censored_regions == 0
        assert stats.elapsed_seconds == 0.0
        assert stats.fps == 0.0

    def test_mutable(self):
        stats = ProcessingStats()
        stats.processed_frames = 100
        assert stats.processed_frames == 100
