import numpy as np
import pytest

from document_scanner.processing.edge_detector import _order_corners, auto_detect_corners


class TestOrderCorners:
    def test_already_ordered(self):
        pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        result = _order_corners(pts)
        assert result[0] == (0, 0)
        assert result[1] == (100, 0)
        assert result[2] == (100, 100)
        assert result[3] == (0, 100)

    def test_shuffled_input(self):
        pts = [(100, 100), (0, 0), (100, 0), (0, 100)]
        result = _order_corners(pts)
        assert result[0] == (0, 0)
        assert result[1] == (100, 0)
        assert result[2] == (100, 100)
        assert result[3] == (0, 100)

    def test_returns_four_points(self):
        pts = [(10, 20), (80, 5), (90, 95), (5, 85)]
        result = _order_corners(pts)
        assert len(result) == 4


class TestAutoDetectCorners:
    def test_returns_four_points(self, document_on_dark):
        corners = auto_detect_corners(document_on_dark)
        assert len(corners) == 4

    def test_each_point_is_tuple_of_two_ints(self, document_on_dark):
        corners = auto_detect_corners(document_on_dark)
        for pt in corners:
            assert len(pt) == 2
            assert isinstance(pt[0], int)
            assert isinstance(pt[1], int)

    def test_detects_document_region(self, document_on_dark):
        corners = auto_detect_corners(document_on_dark)
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        assert min(xs) < 100
        assert max(xs) > 150
        assert min(ys) < 100
        assert max(ys) > 250

    def test_fallback_on_blank_image(self, blank_bgr):
        h, w = blank_bgr.shape[:2]
        corners = auto_detect_corners(blank_bgr)
        assert len(corners) == 4
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        assert min(xs) <= 0
        assert max(xs) >= w - 1
        assert min(ys) <= 0
        assert max(ys) >= h - 1

    def test_points_within_image_bounds(self, color_bgr):
        h, w = color_bgr.shape[:2]
        corners = auto_detect_corners(color_bgr)
        for x, y in corners:
            assert 0 <= x <= w
            assert 0 <= y <= h

    def test_output_dtype_int(self, document_on_dark):
        corners = auto_detect_corners(document_on_dark)
        for x, y in corners:
            assert isinstance(x, (int, np.integer))
            assert isinstance(y, (int, np.integer))
