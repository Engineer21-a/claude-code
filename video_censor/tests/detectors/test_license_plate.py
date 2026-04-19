import numpy as np
import pytest

from video_censor.config import DetectorConfig
from video_censor.models import DetectionClass


def _make_hf_result(xmin, ymin, xmax, ymax, score=0.9):
    return {"box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}, "score": score, "label": "license-plate"}


class TestLicensePlateDetector:
    @pytest.fixture
    def detector(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[_make_hf_result(50, 60, 200, 120)])
        mocker.patch("video_censor.detectors.license_plate.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.license_plate import LicensePlateDetector

        return LicensePlateDetector(DetectorConfig())

    @pytest.fixture
    def empty_detector(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[])
        mocker.patch("video_censor.detectors.license_plate.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.license_plate import LicensePlateDetector

        return LicensePlateDetector(DetectorConfig())

    def test_returns_one_fd_per_frame(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert len(results) == 3

    def test_frame_index_offset(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=7)
        assert results[0].frame_index == 7

    def test_detection_class_is_license_plate(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        for det in results[0].detections:
            assert det.detection_class == DetectionClass.LICENSE_PLATE

    def test_bbox_coords_parsed_correctly(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        bb = results[0].detections[0].bbox
        assert bb.x1 == 50 and bb.y1 == 60
        assert bb.x2 == 200 and bb.y2 == 120

    def test_confidence_score_parsed(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert abs(results[0].detections[0].confidence - 0.9) < 0.01

    def test_empty_detections(self, empty_detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = empty_detector.detect_batch(frames, start_frame_index=0)
        assert results[0].detections == []

    def test_warmup_runs_without_error(self, empty_detector):
        empty_detector.warmup()

    def test_uses_cuda_device_index(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[])
        patch = mocker.patch("video_censor.detectors.license_plate.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.license_plate import LicensePlateDetector

        LicensePlateDetector(DetectorConfig(device="cuda"))
        _, kwargs = patch.call_args
        assert kwargs.get("device") == 0

    def test_uses_cpu_device_minus_one(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[])
        patch = mocker.patch("video_censor.detectors.license_plate.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.license_plate import LicensePlateDetector

        LicensePlateDetector(DetectorConfig(device="cpu"))
        _, kwargs = patch.call_args
        assert kwargs.get("device") == -1
