import numpy as np
import pytest

from document_scanner.processing.dewarper import (
    Dewarper,
    _DOCUWARP_AVAILABLE,
    _PAGEDEWARP_AVAILABLE,
)


class TestDewarperNoneBackend:
    def test_identity_returns_same_shape(self, color_bgr):
        d = Dewarper(backend="none")
        result = d.dewarp(color_bgr)
        assert result.shape == color_bgr.shape

    def test_identity_returns_uint8(self, color_bgr):
        d = Dewarper(backend="none")
        result = d.dewarp(color_bgr)
        assert result.dtype == np.uint8

    def test_backend_name(self):
        d = Dewarper(backend="none")
        assert d.backend_name == "none"

    def test_identity_content_unchanged(self, color_bgr):
        d = Dewarper(backend="none")
        result = d.dewarp(color_bgr)
        np.testing.assert_array_equal(result, color_bgr)


class TestDewarperAutoBackend:
    def test_auto_resolves_to_known_backend(self):
        d = Dewarper(backend="auto")
        assert d.backend_name in {"docuwarp", "pagedewarp", "none"}

    def test_auto_dewarp_returns_valid_image(self, color_bgr):
        d = Dewarper(backend="auto")
        result = d.dewarp(color_bgr)
        assert result.ndim == 3
        assert result.dtype == np.uint8


@pytest.mark.skipif(not _DOCUWARP_AVAILABLE, reason="docuwarp not installed")
class TestDocuwarpBackend:
    def test_dewarp_output_shape(self, document_on_dark):
        d = Dewarper(backend="docuwarp")
        result = d.dewarp(document_on_dark)
        assert result.ndim == 3
        assert result.dtype == np.uint8


@pytest.mark.skipif(not _PAGEDEWARP_AVAILABLE, reason="page-dewarp not installed")
class TestPagedewarpBackend:
    def test_dewarp_output_shape(self, document_on_dark):
        d = Dewarper(backend="pagedewarp")
        result = d.dewarp(document_on_dark)
        assert result.ndim == 3
        assert result.dtype == np.uint8
