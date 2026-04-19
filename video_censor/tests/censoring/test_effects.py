import numpy as np
import pytest

from video_censor.censoring.effects import (
    _apply_gaussian_blur,
    _apply_inplace,
    _apply_pixelate,
    _apply_solid_black,
    _pad_bbox,
    apply_censor,
    apply_censor_many,
)
from video_censor.config import CensorConfig
from video_censor.models import BoundingBox, CensorMethod


@pytest.fixture
def gray_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:200, 100:300] = 200
    return frame


@pytest.fixture
def bbox() -> BoundingBox:
    return BoundingBox(100, 50, 300, 200)


# ---------------------------------------------------------------------------
# _pad_bbox
# ---------------------------------------------------------------------------

class TestPadBbox:
    def test_pad_within_frame(self):
        bb = BoundingBox(50, 50, 100, 100)
        padded = _pad_bbox(bb, padding=10, frame_w=640, frame_h=480)
        assert padded == BoundingBox(40, 40, 110, 110)

    def test_pad_clamps_to_left_top_edge(self):
        bb = BoundingBox(3, 3, 100, 100)
        padded = _pad_bbox(bb, padding=10, frame_w=640, frame_h=480)
        assert padded.x1 == 0
        assert padded.y1 == 0

    def test_pad_clamps_to_right_bottom_edge(self):
        bb = BoundingBox(100, 100, 637, 477)
        padded = _pad_bbox(bb, padding=10, frame_w=640, frame_h=480)
        assert padded.x2 == 640
        assert padded.y2 == 480

    def test_zero_padding_unchanged(self):
        bb = BoundingBox(10, 10, 20, 20)
        assert _pad_bbox(bb, 0, 640, 480) == bb

    def test_large_padding_does_not_exceed_frame(self):
        bb = BoundingBox(5, 5, 10, 10)
        padded = _pad_bbox(bb, 1000, 640, 480)
        assert padded.x1 == 0 and padded.y1 == 0
        assert padded.x2 == 640 and padded.y2 == 480


# ---------------------------------------------------------------------------
# _apply_gaussian_blur (in-place)
# ---------------------------------------------------------------------------

class TestApplyGaussianBlur:
    def test_roi_is_modified(self, gray_frame, bbox):
        original_roi = gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()
        frame = gray_frame.copy()
        _apply_gaussian_blur(frame, bbox, kernel_size=15)
        roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert not np.array_equal(roi, original_roi)

    def test_outside_roi_unchanged(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_gaussian_blur(frame, bbox, kernel_size=15)
        assert frame[0, 0, 0] == gray_frame[0, 0, 0]

    def test_even_kernel_adjusted_to_odd(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_gaussian_blur(frame, bbox, kernel_size=10)  # should not raise

    def test_kernel_smaller_than_3_clamped(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_gaussian_blur(frame, bbox, kernel_size=1)

    def test_empty_roi_is_noop(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        original = frame.copy()
        bb = BoundingBox(5, 5, 5, 5)  # zero-area
        _apply_gaussian_blur(frame, bb, kernel_size=15)
        assert np.array_equal(frame, original)

    def test_large_kernel_smaller_than_roi(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        bb = BoundingBox(0, 0, 100, 100)
        _apply_gaussian_blur(frame, bb, kernel_size=201)  # should not raise

    def test_returns_none(self, gray_frame, bbox):
        result = _apply_gaussian_blur(gray_frame, bbox, kernel_size=15)
        assert result is None


# ---------------------------------------------------------------------------
# _apply_pixelate (in-place)
# ---------------------------------------------------------------------------

class TestApplyPixelate:
    def test_roi_is_modified(self, gray_frame, bbox):
        original_roi = gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()
        frame = gray_frame.copy()
        _apply_pixelate(frame, bbox, block_size=10)
        roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert not np.array_equal(roi, original_roi)

    def test_outside_roi_unchanged(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_pixelate(frame, bbox, block_size=10)
        assert frame[0, 0, 0] == gray_frame[0, 0, 0]

    def test_pixel_blocks_are_uniform(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        bb = BoundingBox(0, 0, 100, 100)
        block = 10
        _apply_pixelate(frame, bb, block_size=block)
        for y in range(0, 100, block):
            for x in range(0, 100, block):
                tile = frame[y : y + block, x : x + block]
                assert np.all(tile == tile[0, 0]), f"Tile at ({x},{y}) not uniform"

    def test_block_size_one_is_identity(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_pixelate(frame, bbox, block_size=1)
        assert np.array_equal(frame, gray_frame)

    def test_empty_roi_is_noop(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        original = frame.copy()
        bb = BoundingBox(5, 5, 5, 10)  # zero-width
        _apply_pixelate(frame, bb, block_size=5)
        assert np.array_equal(frame, original)

    def test_returns_none(self, gray_frame, bbox):
        result = _apply_pixelate(gray_frame, bbox, block_size=10)
        assert result is None


# ---------------------------------------------------------------------------
# _apply_solid_black (in-place)
# ---------------------------------------------------------------------------

class TestApplySolidBlack:
    def test_roi_is_black(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_solid_black(frame, bbox)
        assert np.all(frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] == 0)

    def test_outside_roi_unchanged(self, gray_frame, bbox):
        frame = gray_frame.copy()
        _apply_solid_black(frame, bbox)
        assert frame[0, 0, 0] == gray_frame[0, 0, 0]

    def test_full_frame_black(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bb = BoundingBox(0, 0, 100, 100)
        _apply_solid_black(frame, bb)
        assert np.all(frame == 0)

    def test_returns_none(self, gray_frame, bbox):
        result = _apply_solid_black(gray_frame, bbox)
        assert result is None


# ---------------------------------------------------------------------------
# _apply_inplace dispatch
# ---------------------------------------------------------------------------

class TestApplyInplace:
    def test_dispatches_gaussian_blur(self, gray_frame, bbox):
        frame = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.GAUSSIAN_BLUR, blur_kernel_size=15)
        _apply_inplace(frame, bbox, cfg)
        assert not np.array_equal(frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2],
                                   gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2])

    def test_dispatches_pixelate(self, gray_frame, bbox):
        frame = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.PIXELATE, pixelate_block_size=10)
        _apply_inplace(frame, bbox, cfg)
        assert not np.array_equal(frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2],
                                   gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2])

    def test_dispatches_solid_black(self, gray_frame, bbox):
        frame = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK)
        _apply_inplace(frame, bbox, cfg)
        assert np.all(frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2] == 0)

    def test_unknown_method_raises_value_error(self, gray_frame, bbox):
        cfg = CensorConfig()
        cfg.method = "bad_method"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown censor method"):
            _apply_inplace(gray_frame, bbox, cfg)


# ---------------------------------------------------------------------------
# apply_censor (public, returns copy)
# ---------------------------------------------------------------------------

class TestApplyCensor:
    def test_dispatches_gaussian_blur(self, gray_frame, bbox):
        cfg = CensorConfig(method=CensorMethod.GAUSSIAN_BLUR, blur_kernel_size=15, padding_px=0)
        result = apply_censor(gray_frame, bbox, cfg)
        assert not np.array_equal(result, gray_frame)

    def test_dispatches_pixelate(self, gray_frame, bbox):
        cfg = CensorConfig(method=CensorMethod.PIXELATE, pixelate_block_size=10, padding_px=0)
        result = apply_censor(gray_frame, bbox, cfg)
        assert not np.array_equal(result, gray_frame)

    def test_dispatches_solid_black(self, gray_frame, bbox):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        result = apply_censor(gray_frame, bbox, cfg)
        roi = result[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert np.all(roi == 0)

    def test_does_not_mutate_input(self, gray_frame, bbox):
        original = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        apply_censor(gray_frame, bbox, cfg)
        assert np.array_equal(gray_frame, original)

    def test_invalid_bbox_after_padding_returns_copy(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        bb = BoundingBox(5, 5, 5, 5)  # zero-area
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        result = apply_censor(frame, bb, cfg)
        assert np.array_equal(result, frame)

    def test_padding_expands_censored_region(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        bb = BoundingBox(40, 40, 60, 60)
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=10)
        result = apply_censor(frame, bb, cfg)
        assert result[30, 30, 0] == 0


# ---------------------------------------------------------------------------
# apply_censor_many — one copy total
# ---------------------------------------------------------------------------

class TestApplyCensorMany:
    def test_two_regions_censored(self, gray_frame):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        bboxes = [BoundingBox(0, 0, 50, 50), BoundingBox(100, 100, 200, 200)]
        result = apply_censor_many(gray_frame, bboxes, cfg)
        assert np.all(result[0:50, 0:50] == 0)
        assert np.all(result[100:200, 100:200] == 0)

    def test_empty_bboxes_returns_copy(self, gray_frame):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        result = apply_censor_many(gray_frame, [], cfg)
        assert np.array_equal(result, gray_frame)

    def test_does_not_mutate_input(self, gray_frame):
        original = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        apply_censor_many(gray_frame, [BoundingBox(0, 0, 100, 100)], cfg)
        assert np.array_equal(gray_frame, original)

    def test_makes_exactly_one_copy(self, gray_frame):
        """Verify apply_censor_many does not make a copy per bbox."""
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        bboxes = [BoundingBox(i * 10, 0, i * 10 + 5, 5) for i in range(10)]
        # Should not raise or error even with many boxes
        result = apply_censor_many(gray_frame, bboxes, cfg)
        assert result is not gray_frame  # is a copy

    def test_overlapping_boxes(self, gray_frame):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        bboxes = [BoundingBox(0, 0, 100, 100), BoundingBox(50, 50, 150, 150)]
        result = apply_censor_many(gray_frame, bboxes, cfg)
        assert np.all(result[50:100, 50:100] == 0)
