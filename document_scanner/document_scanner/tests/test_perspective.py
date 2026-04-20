import numpy as np
import pytest

from document_scanner.processing.perspective import _compute_output_size, warp_perspective


class TestComputeOutputSize:
    def test_square_document(self):
        corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
        w, h = _compute_output_size(corners)
        assert 90 <= w <= 110
        assert 90 <= h <= 110

    def test_portrait_document(self):
        corners = [(0, 0), (100, 0), (100, 200), (0, 200)]
        w, h = _compute_output_size(corners)
        assert h > w

    def test_returns_positive_size(self):
        corners = [(5, 5), (6, 5), (6, 6), (5, 6)]
        w, h = _compute_output_size(corners)
        assert w >= 1
        assert h >= 1


class TestWarpPerspective:
    def test_output_shape_has_three_channels(self, color_bgr):
        h, w = color_bgr.shape[:2]
        corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        result = warp_perspective(color_bgr, corners)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_explicit_output_size(self, color_bgr):
        h, w = color_bgr.shape[:2]
        corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        result = warp_perspective(color_bgr, corners, output_size=(200, 300))
        assert result.shape[:2] == (300, 200)  # (height, width)

    def test_identity_transform_preserves_content(self, color_bgr):
        h, w = color_bgr.shape[:2]
        # Use exact image-boundary corners so M ≈ identity
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        result = warp_perspective(color_bgr, corners, output_size=(w, h))
        diff = np.abs(result.astype(float) - color_bgr.astype(float))
        assert diff.mean() < 5.0

    def test_output_dtype_uint8(self, color_bgr):
        h, w = color_bgr.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        result = warp_perspective(color_bgr, corners)
        assert result.dtype == np.uint8

    def test_skewed_input_produces_rectangle(self, color_bgr):
        h, w = color_bgr.shape[:2]
        # Skewed: TL and BL shifted inward
        corners = [(20, 30), (w - 10, 10), (w - 5, h - 20), (10, h - 10)]
        result = warp_perspective(color_bgr, corners, output_size=(200, 250))
        assert result.shape == (250, 200, 3)
