import numpy as np
import pytest

from document_scanner.processing.enhancer import (
    Enhancer,
    _REALESRGAN_AVAILABLE,
    unsharp_mask,
)


class TestUnsharpMask:
    def test_output_shape(self, color_bgr):
        result = unsharp_mask(color_bgr)
        assert result.shape == color_bgr.shape

    def test_output_dtype(self, color_bgr):
        result = unsharp_mask(color_bgr)
        assert result.dtype == np.uint8

    def test_sharpens_image(self, color_bgr):
        result = unsharp_mask(color_bgr, amount=2.0)
        assert not np.array_equal(result, color_bgr)

    def test_zero_amount_near_identity(self, color_bgr):
        result = unsharp_mask(color_bgr, amount=0.0)
        diff = np.abs(result.astype(float) - color_bgr.astype(float))
        assert diff.mean() < 2.0

    def test_output_values_clipped(self, color_bgr):
        result = unsharp_mask(color_bgr, amount=5.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_threshold_parameter(self, color_bgr):
        result = unsharp_mask(color_bgr, threshold=10)
        assert result.shape == color_bgr.shape
        assert result.dtype == np.uint8


class TestEnhancerUnsharpBackend:
    def test_sharpen_output_shape(self, color_bgr):
        e = Enhancer(backend="unsharp")
        result = e.sharpen(color_bgr, strength=1.0)
        assert result.shape == color_bgr.shape

    def test_sharpen_output_dtype(self, color_bgr):
        e = Enhancer(backend="unsharp")
        result = e.sharpen(color_bgr)
        assert result.dtype == np.uint8

    def test_sharpen_zero_strength_near_identity(self, color_bgr):
        e = Enhancer(backend="unsharp")
        result = e.sharpen(color_bgr, strength=0.0)
        np.testing.assert_array_equal(result, color_bgr)

    def test_backend_name(self):
        e = Enhancer(backend="unsharp")
        assert e.backend_name == "unsharp"

    def test_auto_backend_resolves(self):
        e = Enhancer(backend="auto")
        assert e.backend_name in {"realesrgan", "unsharp"}


@pytest.mark.skipif(not _REALESRGAN_AVAILABLE, reason="realesrgan not installed")
class TestRealESRGANBackend:
    def test_sharpen_output_shape(self, color_bgr):
        e = Enhancer(backend="realesrgan")
        result = e.sharpen(color_bgr, strength=1.0)
        assert result.shape == color_bgr.shape

    def test_sharpen_output_dtype(self, color_bgr):
        e = Enhancer(backend="realesrgan")
        result = e.sharpen(color_bgr)
        assert result.dtype == np.uint8
