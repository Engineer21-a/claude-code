import numpy as np
import pytest

from video_censor.censoring.effects import (
    _apply_gaussian_blur,
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
# _apply_gaussian_blur
# ---------------------------------------------------------------------------

class TestApplyGaussianBlur:
    def test_roi_is_modified(self, gray_frame, bbox):
        original_roi = gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()
        result = _apply_gaussian_blur(gray_frame.copy(), bbox, kernel_size=15)
        roi = result[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert not np.array_equal(roi, original_roi)

    def test_outside_roi_unchanged(self, gray_frame, bbox):
        result = _apply_gaussian_blur(gray_frame.copy(), bbox, kernel_size=15)
        assert result[0, 0, 0] == gray_frame[0, 0, 0]

    def test_even_kernel_adjusted_to_odd(self, gray_frame, bbox):
        # Should not raise even when an even kernel is supplied
        _apply_gaussian_blur(gray_frame.copy(), bbox, kernel_size=10)

    def test_kernel_smaller_than_3_clamped(self, gray_frame, bbox):
        _apply_gaussian_blur(gray_frame.copy(), bbox, kernel_size=1)

    def test_empty_roi_returns_frame_unchanged(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        bb = BoundingBox(5, 5, 5, 5)  # zero-area
        result = _apply_gaussian_blur(frame.copy(), bb, kernel_size=15)
        assert np.array_equal(result, frame)

    def test_large_kernel_smaller_than_roi(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        bb = BoundingBox(0, 0, 100, 100)
        _apply_gaussian_blur(frame.copy(), bb, kernel_size=201)


# ---------------------------------------------------------------------------
# _apply_pixelate
# ---------------------------------------------------------------------------

class TestApplyPixelate:
    def test_roi_is_modified(self, gray_frame, bbox):
        original_roi = gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()
        result = _apply_pixelate(gray_frame.copy(), bbox, block_size=10)
        roi = result[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert not np.array_equal(roi, original_roi)

    def test_outside_roi_unchanged(self, gray_frame, bbox):
        result = _apply_pixelate(gray_frame.copy(), bbox, block_size=10)
        assert result[0, 0, 0] == gray_frame[0, 0, 0]

    def test_pixel_blocks_are_uniform(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        bb = BoundingBox(0, 0, 100, 100)
        block = 10
        result = _apply_pixelate(frame.copy(), bb, block_size=block)
        for y in range(0, 100, block):
            for x in range(0, 100, block):
                tile = result[y : y + block, x : x + block]
                assert np.all(tile == tile[0, 0]), f"Tile at ({x},{y}) not uniform"

    def test_block_size_one_is_identity(self, gray_frame, bbox):
        result = _apply_pixelate(gray_frame.copy(), bbox, block_size=1)
        assert np.array_equal(result, gray_frame)

    def test_empty_roi_returns_frame_unchanged(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        bb = BoundingBox(5, 5, 5, 10)  # zero-width
        result = _apply_pixelate(frame.copy(), bb, block_size=5)
        assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# _apply_solid_black
# ---------------------------------------------------------------------------

class TestApplySolidBlack:
    def test_roi_is_black(self, gray_frame, bbox):
        result = _apply_solid_black(gray_frame.copy(), bbox)
        roi = result[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert np.all(roi == 0)

    def test_outside_roi_unchanged(self, gray_frame, bbox):
        result = _apply_solid_black(gray_frame.copy(), bbox)
        assert result[0, 0, 0] == gray_frame[0, 0, 0]

    def test_full_frame_black(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bb = BoundingBox(0, 0, 100, 100)
        result = _apply_solid_black(frame, bb)
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# apply_censor (dispatch)
# ---------------------------------------------------------------------------

class TestApplyCensor:
    def test_dispatches_gaussian_blur(self, gray_frame, bbox):
        cfg = CensorConfig(method=CensorMethod.GAUSSIAN_BLUR, blur_kernel_size=15, padding_px=0)
        result = apply_censor(gray_frame.copy(), bbox, cfg)
        assert not np.array_equal(result, gray_frame)

    def test_dispatches_pixelate(self, gray_frame, bbox):
        cfg = CensorConfig(method=CensorMethod.PIXELATE, pixelate_block_size=10, padding_px=0)
        result = apply_censor(gray_frame.copy(), bbox, cfg)
        assert not np.array_equal(result, gray_frame)

    def test_dispatches_solid_black(self, gray_frame, bbox):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        result = apply_censor(gray_frame.copy(), bbox, cfg)
        roi = result[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        assert np.all(roi == 0)

    def test_does_not_mutate_input(self, gray_frame, bbox):
        original = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        apply_censor(gray_frame, bbox, cfg)
        assert np.array_equal(gray_frame, original)

    def test_unknown_method_raises_value_error(self, gray_frame, bbox):
        cfg = CensorConfig()
        cfg.method = "not_a_method"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown censor method"):
            apply_censor(gray_frame, bbox, cfg)

    def test_padding_expands_censored_region(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        bb = BoundingBox(40, 40, 60, 60)
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=10)
        result = apply_censor(frame.copy(), bb, cfg)
        # The corner at (30,30) should be black because padding expands by 10
        assert result[30, 30, 0] == 0

    def test_invalid_bbox_after_padding_returns_copy(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
        # A box that collapses to zero-area after being clamped
        bb = BoundingBox(5, 5, 5, 5)
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        result = apply_censor(frame, bb, cfg)
        assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# apply_censor_many
# ---------------------------------------------------------------------------

class TestApplyCensorMany:
    def test_two_regions_censored(self, gray_frame):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        bboxes = [BoundingBox(0, 0, 50, 50), BoundingBox(100, 100, 200, 200)]
        result = apply_censor_many(gray_frame.copy(), bboxes, cfg)
        assert np.all(result[0:50, 0:50] == 0)
        assert np.all(result[100:200, 100:200] == 0)

    def test_empty_bboxes_returns_copy_of_frame(self, gray_frame):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        result = apply_censor_many(gray_frame.copy(), [], cfg)
        assert np.array_equal(result, gray_frame)

    def test_does_not_mutate_input(self, gray_frame):
        original = gray_frame.copy()
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        apply_censor_many(gray_frame, [BoundingBox(0, 0, 100, 100)], cfg)
        assert np.array_equal(gray_frame, original)

    def test_overlapping_boxes(self, gray_frame):
        cfg = CensorConfig(method=CensorMethod.SOLID_BLACK, padding_px=0)
        bboxes = [BoundingBox(0, 0, 100, 100), BoundingBox(50, 50, 150, 150)]
        result = apply_censor_many(gray_frame.copy(), bboxes, cfg)
        # Entire overlap region should be black
        assert np.all(result[50:100, 50:100] == 0)
