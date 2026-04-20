from unittest.mock import MagicMock

import numpy as np
import pytest

from document_scanner.models.document import DocumentImage, FilterSettings
from document_scanner.processing.dewarper import Dewarper
from document_scanner.processing.enhancer import Enhancer
from document_scanner.processing.pipeline import ProcessingPipeline


@pytest.fixture
def pl():
    return ProcessingPipeline(
        dewarper=Dewarper(backend="none"),
        enhancer=Enhancer(backend="unsharp"),
    )


@pytest.fixture
def simple_doc(color_bgr, tmp_path):
    p = tmp_path / "img.jpg"
    p.touch()
    return DocumentImage(path=p, original=color_bgr)


class TestProcessingPipeline:
    def test_process_returns_ndarray(self, pl, simple_doc):
        result = pl.process(simple_doc)
        assert isinstance(result, np.ndarray)

    def test_process_returns_uint8(self, pl, simple_doc):
        result = pl.process(simple_doc)
        assert result.dtype == np.uint8

    def test_process_sets_processed_field(self, pl, simple_doc):
        pl.process(simple_doc)
        assert simple_doc.processed is not None

    def test_process_clears_needs_reprocess(self, pl, simple_doc):
        simple_doc.needs_reprocess = True
        pl.process(simple_doc)
        assert simple_doc.needs_reprocess is False

    def test_preview_mode_skips_dewarping(self, color_bgr, tmp_path):
        p = tmp_path / "img.jpg"
        p.touch()
        doc = DocumentImage(path=p, original=color_bgr, dewarp_enabled=True)

        mock_dewarper = MagicMock()
        mock_dewarper.dewarp.return_value = color_bgr.copy()

        pl2 = ProcessingPipeline(
            dewarper=mock_dewarper,
            enhancer=Enhancer(backend="unsharp"),
        )
        pl2.process(doc, preview=True)
        mock_dewarper.dewarp.assert_not_called()

    def test_full_mode_calls_dewarping(self, color_bgr, tmp_path):
        p = tmp_path / "img.jpg"
        p.touch()
        doc = DocumentImage(path=p, original=color_bgr, dewarp_enabled=True)

        mock_dewarper = MagicMock()
        mock_dewarper.dewarp.return_value = color_bgr.copy()

        pl2 = ProcessingPipeline(
            dewarper=mock_dewarper,
            enhancer=Enhancer(backend="unsharp"),
        )
        pl2.process(doc, preview=False)
        mock_dewarper.dewarp.assert_called_once()

    def test_dewarp_disabled_skips_dewarping(self, color_bgr, tmp_path):
        p = tmp_path / "img.jpg"
        p.touch()
        doc = DocumentImage(path=p, original=color_bgr, dewarp_enabled=False)

        mock_dewarper = MagicMock()
        mock_dewarper.dewarp.return_value = color_bgr.copy()

        pl2 = ProcessingPipeline(
            dewarper=mock_dewarper,
            enhancer=Enhancer(backend="unsharp"),
        )
        pl2.process(doc, preview=False)
        mock_dewarper.dewarp.assert_not_called()

    def test_perspective_applied_for_nontrivial_corners(self, color_bgr, tmp_path):
        p = tmp_path / "img.jpg"
        p.touch()
        h, w = color_bgr.shape[:2]
        corners = [(20, 20), (w - 20, 10), (w - 10, h - 20), (10, h - 10)]
        doc = DocumentImage(path=p, original=color_bgr, corners=corners)
        pl2 = ProcessingPipeline(
            dewarper=Dewarper(backend="none"),
            enhancer=Enhancer(backend="unsharp"),
        )
        result = pl2.process(doc)
        assert result is not None
        assert result.dtype == np.uint8

    def test_process_all_processes_each_doc(self, color_bgr, tmp_path):
        docs = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            p.touch()
            docs.append(DocumentImage(path=p, original=color_bgr.copy()))

        pl2 = ProcessingPipeline(
            dewarper=Dewarper(backend="none"),
            enhancer=Enhancer(backend="unsharp"),
        )
        pl2.process_all(docs)
        for doc in docs:
            assert doc.processed is not None

    def test_process_all_calls_progress_callback(self, color_bgr, tmp_path):
        calls = []
        docs = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            p.touch()
            docs.append(DocumentImage(path=p, original=color_bgr.copy()))

        pl2 = ProcessingPipeline(
            dewarper=Dewarper(backend="none"),
            enhancer=Enhancer(backend="unsharp"),
        )
        pl2.process_all(docs, progress_callback=lambda cur, tot: calls.append((cur, tot)))
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_sharpening_applied_when_strength_nonzero(self, color_bgr, tmp_path):
        p = tmp_path / "img.jpg"
        p.touch()
        doc = DocumentImage(
            path=p,
            original=color_bgr,
            sharpen_enabled=True,
            filter_settings=FilterSettings(sharpness=1.0),
        )
        mock_enhancer = MagicMock()
        mock_enhancer.sharpen.return_value = color_bgr.copy()

        pl2 = ProcessingPipeline(
            dewarper=Dewarper(backend="none"),
            enhancer=mock_enhancer,
        )
        pl2.process(doc)
        mock_enhancer.sharpen.assert_called_once()

    def test_sharpening_skipped_when_disabled(self, color_bgr, tmp_path):
        p = tmp_path / "img.jpg"
        p.touch()
        doc = DocumentImage(
            path=p,
            original=color_bgr,
            sharpen_enabled=False,
            filter_settings=FilterSettings(sharpness=1.0),
        )
        mock_enhancer = MagicMock()
        mock_enhancer.sharpen.return_value = color_bgr.copy()

        pl2 = ProcessingPipeline(
            dewarper=Dewarper(backend="none"),
            enhancer=mock_enhancer,
        )
        pl2.process(doc)
        mock_enhancer.sharpen.assert_not_called()
