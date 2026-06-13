"""Settings dialog — edit and persist core preferences."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from app.config.settings import OcrEngine, OutputMode, Settings


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        form = QFormLayout(self)

        self.mode = QComboBox()
        self.mode.addItem("Maximum safe (flatten to image)", OutputMode.MAXIMUM_SAFE.value)
        self.mode.addItem("Structure-preserving (born-digital only)", OutputMode.STRUCTURE_PRESERVING.value)
        self.mode.setCurrentIndex(0 if settings.default_output_mode == OutputMode.MAXIMUM_SAFE else 1)
        form.addRow("Output mode", self.mode)

        self.ocr = QComboBox()
        self.ocr.addItem("RapidOCR (quality)", OcrEngine.RAPIDOCR.value)
        self.ocr.addItem("Tesseract (speed)", OcrEngine.TESSERACT.value)
        self.ocr.setCurrentIndex(0 if settings.ocr_engine == OcrEngine.RAPIDOCR else 1)
        form.addRow("OCR engine", self.ocr)

        self.dpi = QSpinBox()
        self.dpi.setRange(150, 600)
        self.dpi.setValue(settings.dpi)
        form.addRow("Render DPI", self.dpi)

        self.margin = QSpinBox()
        self.margin.setRange(0, 40)
        self.margin.setValue(settings.redaction_margin_px)
        form.addRow("Redaction margin (px)", self.margin)

        self.enable_regex = QCheckBox()
        self.enable_regex.setChecked(settings.enable_regex_de)
        form.addRow("German regex", self.enable_regex)

        self.enable_gliner = QCheckBox()
        self.enable_gliner.setChecked(settings.enable_gliner)
        form.addRow("GLiNER2 (AI)", self.enable_gliner)

        self.gliner_threshold = QDoubleSpinBox()
        self.gliner_threshold.setRange(0.05, 0.95)
        self.gliner_threshold.setSingleStep(0.05)
        self.gliner_threshold.setValue(settings.gliner_threshold)
        form.addRow("GLiNER threshold", self.gliner_threshold)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _apply(self) -> None:
        self.settings.default_output_mode = OutputMode(self.mode.currentData())
        self.settings.ocr_engine = OcrEngine(self.ocr.currentData())
        self.settings.dpi = self.dpi.value()
        self.settings.redaction_margin_px = self.margin.value()
        self.settings.enable_regex_de = self.enable_regex.isChecked()
        self.settings.enable_gliner = self.enable_gliner.isChecked()
        self.settings.gliner_threshold = self.gliner_threshold.value()
        self.settings.save()
        self.accept()
