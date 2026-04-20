import numpy as np
import pytest

from document_scanner.processing.filters import (
    PRESETS,
    apply_custom,
    apply_filter,
    blueprint,
    bw_otsu,
    bw_sauvola,
    enhanced_clahe,
    grayscale,
    magic_color,
    original,
    vintage,
)


class TestPresetRegistry:
    def test_all_presets_callable_or_none(self):
        for name, fn in PRESETS.items():
            if fn is not None:
                assert callable(fn), f"{name} should be callable"

    def test_required_presets_present(self):
        required = {
            "original", "grayscale", "bw_otsu", "bw_sauvola",
            "enhanced_clahe", "magic_color", "blueprint", "vintage", "custom",
        }
        assert required.issubset(set(PRESETS.keys()))

    def test_custom_sentinel_is_none(self):
        assert PRESETS["custom"] is None


@pytest.mark.parametrize(
    "preset_name",
    ["original", "grayscale", "bw_otsu", "bw_sauvola",
     "enhanced_clahe", "magic_color", "blueprint", "vintage"],
)
class TestPresetFunctions:
    def test_output_shape_preserved(self, preset_name, color_bgr):
        result = apply_filter(color_bgr, preset_name)
        assert result.shape == color_bgr.shape

    def test_output_dtype_uint8(self, preset_name, color_bgr):
        result = apply_filter(color_bgr, preset_name)
        assert result.dtype == np.uint8

    def test_output_values_valid(self, preset_name, color_bgr):
        result = apply_filter(color_bgr, preset_name)
        assert result.min() >= 0
        assert result.max() <= 255


class TestGrayscaleFilter:
    def test_all_channels_equal(self, color_bgr):
        result = grayscale(color_bgr)
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])


class TestBWOtsu:
    def test_output_is_binary(self, color_bgr):
        result = bw_otsu(color_bgr)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_output_shape(self, color_bgr):
        result = bw_otsu(color_bgr)
        assert result.shape == color_bgr.shape


class TestBWSauvola:
    def test_output_is_binary(self, color_bgr):
        result = bw_sauvola(color_bgr)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_output_shape(self, color_bgr):
        result = bw_sauvola(color_bgr)
        assert result.shape == color_bgr.shape


class TestCustomFilter:
    def test_identity_params_near_original(self, color_bgr):
        result = apply_custom(color_bgr, brightness=0.0, contrast=1.0, saturation=1.0)
        diff = np.abs(result.astype(float) - color_bgr.astype(float))
        assert diff.mean() < 5.0

    def test_brightness_increase_raises_values(self, blank_bgr):
        dark = np.full_like(blank_bgr, 50)
        result = apply_custom(dark, brightness=0.3)
        assert result.mean() > dark.mean()

    def test_brightness_decrease_lowers_values(self, blank_bgr):
        bright = np.full_like(blank_bgr, 200)
        result = apply_custom(bright, brightness=-0.3)
        assert result.mean() < bright.mean()

    def test_output_clipped_no_overflow(self, color_bgr):
        result = apply_custom(color_bgr, brightness=1.0, contrast=3.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_output_dtype_uint8(self, color_bgr):
        result = apply_custom(color_bgr)
        assert result.dtype == np.uint8

    def test_shadow_raises_dark_tones(self):
        dark_img = np.full((100, 100, 3), 30, dtype=np.uint8)
        result = apply_custom(dark_img, shadow=0.3)
        assert result.mean() > dark_img.mean()

    def test_saturation_zero_produces_gray(self, color_bgr):
        result = apply_custom(color_bgr, saturation=0.0)
        # All channels should be near-equal (grayscale)
        ch_std = result[:, :, 0].astype(float) - result[:, :, 1].astype(float)
        assert np.abs(ch_std).mean() < 10.0


class TestApplyFilterDispatch:
    def test_unknown_preset_falls_to_custom(self, color_bgr):
        result = apply_filter(color_bgr, preset="nonexistent_preset")
        assert result.shape == color_bgr.shape
        assert result.dtype == np.uint8

    def test_custom_preset_uses_apply_custom(self, color_bgr):
        result = apply_filter(color_bgr, preset="custom", brightness=0.1)
        assert result.shape == color_bgr.shape
