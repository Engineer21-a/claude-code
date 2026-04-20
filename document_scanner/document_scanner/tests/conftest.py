"""Shared pytest fixtures for all test modules."""
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def blank_bgr() -> np.ndarray:
    return np.zeros((400, 300, 3), dtype=np.uint8)


@pytest.fixture
def color_bgr() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)


@pytest.fixture
def document_on_dark() -> np.ndarray:
    """White rectangle on dark background (corners ~(50,50)–(250,350))."""
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 350), (255, 255, 255), -1)
    return img


@pytest.fixture
def gray_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (400, 300), dtype=np.uint8)


@pytest.fixture
def pipeline():
    from document_scanner.processing.dewarper import Dewarper
    from document_scanner.processing.enhancer import Enhancer
    from document_scanner.processing.pipeline import ProcessingPipeline

    return ProcessingPipeline(
        dewarper=Dewarper(backend="none"),
        enhancer=Enhancer(backend="unsharp"),
    )


@pytest.fixture
def doc(color_bgr, tmp_path):
    from document_scanner.models.document import DocumentImage

    p = tmp_path / "test.jpg"
    p.touch()
    return DocumentImage(path=p, original=color_bgr)
