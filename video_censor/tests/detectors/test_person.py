import numpy as np
import pytest

from video_censor.config import DetectorConfig
from video_censor.models import DetectionClass


@pytest.fixture
def mock_yolo_result(mocker):
    """Simulate one person detection at [100, 50, 300, 200] with confidence 0.85."""
    import torch

    box = mocker.MagicMock()
    box.xyxy = [torch.tensor([100.0, 50.0, 300.0, 200.0])]
    box.conf = [torch.tensor(0.85)]

    result = mocker.MagicMock()
    result.boxes = [box]
    return result


@pytest.fixture
def empty_yolo_result(mocker):
    result = mocker.MagicMock()
    result.boxes = []
    return result


class TestPersonDetector:
    def _make_detector(self, mocker, model_return_value):
        mock_model = mocker.MagicMock()
        mock_model.return_value = model_return_value
        mocker.patch("video_censor.detectors.person.YOLO", return_value=mock_model)
        from video_censor.detectors.person import PersonDetector

        return PersonDetector(DetectorConfig())

    def test_detect_batch_returns_one_fd_per_frame(self, mocker, mock_yolo_result):
        detector = self._make_detector(mocker, [mock_yolo_result, mock_yolo_result])
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = detector.detect_batch(frames, start_frame_index=5)
        assert len(results) == 2

    def test_frame_indices_start_at_offset(self, mocker, mock_yolo_result):
        detector = self._make_detector(mocker, [mock_yolo_result])
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=10)
        assert results[0].frame_index == 10

    def test_detection_class_is_person(self, mocker, mock_yolo_result):
        detector = self._make_detector(mocker, [mock_yolo_result])
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        for det in results[0].detections:
            assert det.detection_class == DetectionClass.PERSON

    def test_detection_confidence_parsed(self, mocker, mock_yolo_result):
        detector = self._make_detector(mocker, [mock_yolo_result])
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert abs(results[0].detections[0].confidence - 0.85) < 0.01

    def test_detection_bbox_parsed(self, mocker, mock_yolo_result):
        detector = self._make_detector(mocker, [mock_yolo_result])
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        bb = results[0].detections[0].bbox
        assert bb.x1 == 100 and bb.y1 == 50
        assert bb.x2 == 300 and bb.y2 == 200

    def test_empty_detections_on_no_persons(self, mocker, empty_yolo_result):
        detector = self._make_detector(mocker, [empty_yolo_result])
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert results[0].detections == []

    def test_multiple_detections_per_frame(self, mocker):
        import torch

        def make_box(x1, y1, x2, y2, conf):
            b = mocker.MagicMock()
            b.xyxy = [torch.tensor([float(x1), float(y1), float(x2), float(y2)])]
            b.conf = [torch.tensor(conf)]
            return b

        result = mocker.MagicMock()
        result.boxes = [make_box(0, 0, 10, 10, 0.9), make_box(20, 20, 30, 30, 0.7)]
        mock_model = mocker.MagicMock()
        mock_model.return_value = [result]
        mocker.patch("video_censor.detectors.person.YOLO", return_value=mock_model)
        from video_censor.detectors.person import PersonDetector

        detector = PersonDetector(DetectorConfig())
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        results = detector.detect_batch(frames, start_frame_index=0)
        assert len(results[0].detections) == 2

    def test_warmup_calls_detect_batch(self, mocker, empty_yolo_result):
        detector = self._make_detector(mocker, [empty_yolo_result])
        detector.warmup()  # should not raise
