import numpy as np
import pytest

from video_censor.config import DetectorConfig, TrackerConfig
from video_censor.models import BoundingBox, DetectionClass, InteractiveSelection
from video_censor.tracking.interactive import InteractiveTracker


@pytest.fixture
def tracker():
    return InteractiveTracker(DetectorConfig(), TrackerConfig())


class TestMaskToBbox:
    def test_full_mask_returns_full_bbox(self, tracker):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        bb = tracker.mask_to_bbox(mask)
        assert bb.x1 == 0 and bb.y1 == 0
        assert bb.x2 == 100 and bb.y2 == 100

    def test_partial_mask_tight_bbox(self, tracker):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:70] = 255
        bb = tracker.mask_to_bbox(mask)
        assert bb.x1 == 30
        assert bb.y1 == 20
        assert bb.x2 == 70  # x2 = last_col + 1
        assert bb.y2 == 50  # y2 = last_row + 1

    def test_empty_mask_returns_full_frame_bbox(self, tracker):
        mask = np.zeros((80, 120), dtype=np.uint8)
        bb = tracker.mask_to_bbox(mask)
        assert bb.x1 == 0 and bb.y1 == 0
        assert bb.x2 == 120 and bb.y2 == 80

    def test_single_pixel_mask(self, tracker):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 60] = 255
        bb = tracker.mask_to_bbox(mask)
        assert bb.x1 == 60 and bb.y1 == 50
        assert bb.x2 == 61 and bb.y2 == 51


class TestGetMaskForSelection:
    def test_calls_predictor_with_box_prompt(self, mocker, tracker):
        dummy_mask = np.ones((480, 640), dtype=bool)
        mock_predictor = mocker.MagicMock()
        mock_predictor.predict.return_value = (
            np.array([dummy_mask]),
            np.array([0.99]),
            None,
        )
        tracker._predictor = mock_predictor

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sel = InteractiveSelection(frame_index=0, bbox=BoundingBox(10, 20, 100, 200))
        mask = tracker.get_mask_for_selection(frame, sel)

        mock_predictor.set_image.assert_called_once()
        called_box = mock_predictor.predict.call_args[1]["box"]
        assert list(called_box) == [10, 20, 100, 200]
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)

    def test_mask_values_are_0_or_255(self, mocker, tracker):
        binary_mask = np.zeros((10, 10), dtype=bool)
        binary_mask[2:8, 2:8] = True
        mock_predictor = mocker.MagicMock()
        mock_predictor.predict.return_value = (np.array([binary_mask]), np.array([0.9]), None)
        tracker._predictor = mock_predictor

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        sel = InteractiveSelection(frame_index=0, bbox=BoundingBox(0, 0, 10, 10))
        mask = tracker.get_mask_for_selection(frame, sel)
        unique_values = set(mask.flatten().tolist())
        assert unique_values.issubset({0, 255})

    def test_loads_sam2_lazily(self, mocker, tracker):
        load_spy = mocker.patch.object(tracker, "_load_sam2")
        dummy_mask = np.ones((10, 10), dtype=bool)
        mock_predictor = mocker.MagicMock()
        mock_predictor.predict.return_value = (np.array([dummy_mask]), np.array([0.9]), None)

        # Simulate _load_sam2 setting the predictor
        def set_predictor():
            tracker._predictor = mock_predictor

        load_spy.side_effect = set_predictor

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        sel = InteractiveSelection(frame_index=0, bbox=BoundingBox(0, 0, 10, 10))
        tracker.get_mask_for_selection(frame, sel)
        load_spy.assert_called_once()


class TestCreateSeedDetection:
    def test_returns_interactive_class_detection(self, mocker, tracker):
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        dummy_mask[10:50, 20:80] = 255
        mocker.patch.object(tracker, "get_mask_for_selection", return_value=dummy_mask)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sel = InteractiveSelection(frame_index=3, bbox=BoundingBox(20, 10, 80, 50))
        det = tracker.create_seed_detection(frame, sel)

        assert det.detection_class == DetectionClass.INTERACTIVE
        assert det.confidence == 1.0
        assert det.frame_index == 3

    def test_bbox_matches_mask(self, mocker, tracker):
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        dummy_mask[10:50, 20:80] = 255
        mocker.patch.object(tracker, "get_mask_for_selection", return_value=dummy_mask)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sel = InteractiveSelection(frame_index=0, bbox=BoundingBox(20, 10, 80, 50))
        det = tracker.create_seed_detection(frame, sel)

        assert det.bbox.x1 == 20
        assert det.bbox.y1 == 10
