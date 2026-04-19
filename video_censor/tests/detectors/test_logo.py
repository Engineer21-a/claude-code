import numpy as np
import pytest

from video_censor.config import DetectorConfig
from video_censor.models import DetectionClass


def _make_hf_result(xmin, ymin, xmax, ymax, score=0.75):
    return {"box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}, "score": score, "label": "logo"}


class TestLogoDetector:
    @pytest.fixture
    def detector(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[_make_hf_result(10, 20, 80, 90)])
        mocker.patch("video_censor.detectors.logo.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.logo import LogoDetector

        return LogoDetector(DetectorConfig())

    @pytest.fixture
    def empty_detector(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[])
        mocker.patch("video_censor.detectors.logo.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.logo import LogoDetector

        return LogoDetector(DetectorConfig())

    def test_returns_one_fd_per_frame(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert len(results) == 2

    def test_detection_class_is_logo(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        for det in results[0].detections:
            assert det.detection_class == DetectionClass.LOGO

    def test_bbox_coords_parsed(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        bb = results[0].detections[0].bbox
        assert bb.x1 == 10 and bb.y1 == 20
        assert bb.x2 == 80 and bb.y2 == 90

    def test_confidence_parsed(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert abs(results[0].detections[0].confidence - 0.75) < 0.01

    def test_frame_index_offset(self, detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=3)
        assert results[0].frame_index == 3

    def test_empty_detections(self, empty_detector):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = empty_detector.detect_batch(frames, start_frame_index=0)
        assert results[0].detections == []

    def test_warmup_runs_without_error(self, empty_detector):
        empty_detector.warmup()

    def test_cuda_device_mapping(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[])
        patch = mocker.patch("video_censor.detectors.logo.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.logo import LogoDetector

        LogoDetector(DetectorConfig(device="cuda"))
        _, kwargs = patch.call_args
        assert kwargs.get("device") == 0

    def test_cpu_device_mapping(self, mocker):
        mock_pipe = mocker.MagicMock(return_value=[])
        patch = mocker.patch("video_censor.detectors.logo.hf_pipeline", return_value=mock_pipe)
        from video_censor.detectors.logo import LogoDetector

        LogoDetector(DetectorConfig(device="mps"))
        _, kwargs = patch.call_args
        assert kwargs.get("device") == -1
