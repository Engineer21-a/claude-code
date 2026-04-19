import numpy as np
import pytest

from video_censor.detectors.base import BaseDetector, bgr_to_rgb, xyxy_to_bbox
from video_censor.models import BoundingBox, DetectionClass, FrameDetections


class TestBgrToRgb:
    def test_channels_are_flipped(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 10  # B
        frame[:, :, 1] = 20  # G
        frame[:, :, 2] = 30  # R
        rgb = bgr_to_rgb(frame)
        assert rgb[0, 0, 0] == 30  # was R, now first channel
        assert rgb[0, 0, 2] == 10  # was B, now last channel

    def test_returns_copy(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb = bgr_to_rgb(frame)
        rgb[0, 0, 0] = 99
        assert frame[0, 0, 0] == 0

    def test_shape_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert bgr_to_rgb(frame).shape == (480, 640, 3)


class TestXyxyToBbox:
    def test_converts_floats_to_int_bbox(self):
        bb = xyxy_to_bbox(10.7, 20.3, 100.9, 200.1)
        assert bb == BoundingBox(10, 20, 100, 200)

    def test_zero_coords(self):
        bb = xyxy_to_bbox(0.0, 0.0, 0.0, 0.0)
        assert bb == BoundingBox(0, 0, 0, 0)

    def test_returns_bounding_box_type(self):
        bb = xyxy_to_bbox(1.0, 2.0, 3.0, 4.0)
        assert isinstance(bb, BoundingBox)


class TestBaseDetectorProtocol:
    def test_class_implementing_protocol_is_recognized(self):
        class MyDetector:
            detection_class = DetectionClass.PERSON

            def detect_batch(self, frames, start_frame_index):
                return []

            def warmup(self):
                pass

        assert isinstance(MyDetector(), BaseDetector)

    def test_class_missing_method_not_recognized(self):
        class BadDetector:
            detection_class = DetectionClass.PERSON
            # missing detect_batch and warmup

        assert not isinstance(BadDetector(), BaseDetector)
