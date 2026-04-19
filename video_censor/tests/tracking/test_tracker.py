import numpy as np
import pytest

from video_censor.config import TrackerConfig
from video_censor.models import BoundingBox, Detection, DetectionClass, FrameDetections
from video_censor.tracking.tracker import (
    MultiClassTracker,
    _detections_to_sv,
    _sv_to_detections,
)


def _make_detection(x1=0, y1=0, x2=10, y2=10, cls=DetectionClass.PERSON, conf=0.9, frame_index=0):
    return Detection(BoundingBox(x1, y1, x2, y2), cls, conf, frame_index=frame_index)


class TestDetectionsToSv:
    def test_correct_xyxy_shape(self):
        dets = [_make_detection(0, 0, 10, 10), _make_detection(20, 20, 40, 40)]
        sv_dets = _detections_to_sv(dets)
        assert sv_dets.xyxy.shape == (2, 4)

    def test_correct_confidence_values(self):
        dets = [_make_detection(conf=0.9), _make_detection(conf=0.7)]
        sv_dets = _detections_to_sv(dets)
        assert abs(sv_dets.confidence[0] - 0.9) < 0.01
        assert abs(sv_dets.confidence[1] - 0.7) < 0.01

    def test_empty_list_returns_empty_detections(self):
        sv_dets = _detections_to_sv([])
        assert len(sv_dets) == 0

    def test_xyxy_coords_match_input(self):
        dets = [_make_detection(5, 10, 50, 100)]
        sv_dets = _detections_to_sv(dets)
        assert sv_dets.xyxy[0, 0] == 5
        assert sv_dets.xyxy[0, 1] == 10
        assert sv_dets.xyxy[0, 2] == 50
        assert sv_dets.xyxy[0, 3] == 100


class TestSvToDetections:
    def test_track_id_assigned(self, mocker):
        import supervision as sv

        sv_dets = mocker.MagicMock(spec=sv.Detections)
        sv_dets.__len__ = mocker.MagicMock(return_value=1)
        sv_dets.xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        sv_dets.tracker_id = np.array([42])
        sv_dets.confidence = np.array([0.9])

        originals = [_make_detection()]
        result = _sv_to_detections(sv_dets, originals, frame_index=5)
        assert result[0].track_id == 42

    def test_frame_index_set_correctly(self, mocker):
        import supervision as sv

        sv_dets = mocker.MagicMock(spec=sv.Detections)
        sv_dets.__len__ = mocker.MagicMock(return_value=1)
        sv_dets.xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        sv_dets.tracker_id = np.array([1])
        sv_dets.confidence = np.array([0.8])

        originals = [_make_detection()]
        result = _sv_to_detections(sv_dets, originals, frame_index=99)
        assert result[0].frame_index == 99

    def test_detection_class_preserved_from_original(self, mocker):
        import supervision as sv

        sv_dets = mocker.MagicMock(spec=sv.Detections)
        sv_dets.__len__ = mocker.MagicMock(return_value=1)
        sv_dets.xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        sv_dets.tracker_id = np.array([1])
        sv_dets.confidence = np.array([0.8])

        originals = [_make_detection(cls=DetectionClass.LICENSE_PLATE)]
        result = _sv_to_detections(sv_dets, originals, frame_index=0)
        assert result[0].detection_class == DetectionClass.LICENSE_PLATE

    def test_none_tracker_id_gives_none(self, mocker):
        import supervision as sv

        sv_dets = mocker.MagicMock(spec=sv.Detections)
        sv_dets.__len__ = mocker.MagicMock(return_value=1)
        sv_dets.xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        sv_dets.tracker_id = None
        sv_dets.confidence = np.array([0.8])

        originals = [_make_detection()]
        result = _sv_to_detections(sv_dets, originals, frame_index=0)
        assert result[0].track_id is None

    def test_extra_tracker_detections_trimmed_to_originals(self, mocker):
        """ByteTrack can return more detections than inputs (resurrected tracks).
        Extra entries must be dropped to avoid assigning the wrong detection class."""
        import supervision as sv

        sv_dets = mocker.MagicMock(spec=sv.Detections)
        sv_dets.__len__ = mocker.MagicMock(return_value=3)
        sv_dets.xyxy = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],  # extra resurrected track
        ])
        sv_dets.tracker_id = np.array([1, 2, 99])
        sv_dets.confidence = np.array([0.9, 0.8, 0.7])

        originals = [_make_detection(), _make_detection(x1=20, y1=20, x2=30, y2=30)]
        result = _sv_to_detections(sv_dets, originals, frame_index=0)
        assert len(result) == 2  # extra entry is dropped


class TestMultiClassTracker:
    @pytest.fixture
    def tracker(self):
        return MultiClassTracker(TrackerConfig())

    def test_update_empty_frame_returns_empty(self, tracker):
        fd = FrameDetections(frame_index=0, detections=[])
        result = tracker.update(fd)
        assert result.detections == []
        assert result.frame_index == 0

    def test_update_creates_separate_tracker_per_class(self, tracker, mocker):
        mocker.patch("supervision.ByteTrack")
        fd = FrameDetections(
            frame_index=0,
            detections=[
                _make_detection(cls=DetectionClass.PERSON),
                _make_detection(cls=DetectionClass.LICENSE_PLATE),
            ],
        )
        tracker.update(fd)
        assert "person" in tracker._trackers
        assert "license_plate" in tracker._trackers
        assert len(tracker._trackers) == 2

    def test_same_class_reuses_tracker(self, tracker, mocker):
        mocker.patch("supervision.ByteTrack")
        fd1 = FrameDetections(
            frame_index=0,
            detections=[_make_detection(cls=DetectionClass.PERSON)],
        )
        fd2 = FrameDetections(
            frame_index=1,
            detections=[_make_detection(cls=DetectionClass.PERSON)],
        )
        tracker.update(fd1)
        tracker.update(fd2)
        assert len(tracker._trackers) == 1

    def test_reset_clears_all_trackers(self, tracker, mocker):
        mocker.patch("supervision.ByteTrack")
        fd = FrameDetections(
            frame_index=0,
            detections=[_make_detection()],
        )
        tracker.update(fd)
        assert len(tracker._trackers) > 0
        tracker.reset()
        assert tracker._trackers == {}

    def test_update_returns_frame_index(self, tracker, mocker):
        mocker.patch("supervision.ByteTrack")
        fd = FrameDetections(frame_index=42, detections=[])
        result = tracker.update(fd)
        assert result.frame_index == 42

    def test_update_assigns_track_id(self, tracker, mocker):
        import supervision as sv

        tracked_sv = mocker.MagicMock(spec=sv.Detections)
        tracked_sv.__len__ = mocker.MagicMock(return_value=1)
        tracked_sv.xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        tracked_sv.tracker_id = np.array([7])
        tracked_sv.confidence = np.array([0.9])

        mock_bytetrack = mocker.MagicMock()
        mock_bytetrack.update_with_detections.return_value = tracked_sv
        mocker.patch("supervision.ByteTrack", return_value=mock_bytetrack)

        fd = FrameDetections(
            frame_index=0,
            detections=[_make_detection()],
        )
        result = tracker.update(fd)
        assert result.detections[0].track_id == 7
