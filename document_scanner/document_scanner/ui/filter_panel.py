from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from document_scanner.models.document import DocumentImage, FilterSettings
from document_scanner.processing.filters import PRESETS

_PRESET_LABELS = {
    "original": "Original",
    "grayscale": "Grayscale",
    "bw_otsu": "Black & White",
    "bw_sauvola": "B&W (Sauvola)",
    "enhanced_clahe": "Enhanced",
    "magic_color": "Magic Color",
    "blueprint": "Blueprint",
    "vintage": "Vintage",
    "custom": "Custom",
}


class FilterPanel(QWidget):
    filter_changed = Signal(object)    # FilterSettings
    dewarp_toggled = Signal(bool)
    sharpen_toggled = Signal(bool)
    apply_to_all_requested = Signal()

    # Slider (int) → float mapping
    _BRIGHTNESS_RANGE = (-100, 100)   # /100
    _CONTRAST_RANGE = (10, 300)       # /100
    _SATURATION_RANGE = (0, 300)      # /100
    _SHARPNESS_RANGE = (0, 200)       # /100
    _SHADOW_RANGE = (-100, 100)       # /100
    _HIGHLIGHT_RANGE = (-100, 100)    # /100

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(270)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        # ── Preset selector ──────────────────────────────────────────────────
        layout.addWidget(QLabel("<b>Filter Preset</b>"))
        self._preset_combo = QComboBox()
        for key, label in _PRESET_LABELS.items():
            self._preset_combo.addItem(label, key)
        layout.addWidget(self._preset_combo)

        # ── Custom adjustments ───────────────────────────────────────────────
        self._custom_group = QGroupBox("Custom Adjustments")
        cg_layout = QVBoxLayout(self._custom_group)

        self._brightness_slider, self._brightness_lbl = self._build_slider_row(
            cg_layout, "Brightness", *self._BRIGHTNESS_RANGE, 0)
        self._contrast_slider, self._contrast_lbl = self._build_slider_row(
            cg_layout, "Contrast", *self._CONTRAST_RANGE, 100)
        self._saturation_slider, self._saturation_lbl = self._build_slider_row(
            cg_layout, "Saturation", *self._SATURATION_RANGE, 100)
        self._sharpness_slider, self._sharpness_lbl = self._build_slider_row(
            cg_layout, "Sharpness", *self._SHARPNESS_RANGE, 0)
        self._shadow_slider, self._shadow_lbl = self._build_slider_row(
            cg_layout, "Shadow", *self._SHADOW_RANGE, 0)
        self._highlight_slider, self._highlight_lbl = self._build_slider_row(
            cg_layout, "Highlight", *self._HIGHLIGHT_RANGE, 0)

        layout.addWidget(self._custom_group)

        # ── Processing options ───────────────────────────────────────────────
        proc_group = QGroupBox("Processing")
        pg_layout = QVBoxLayout(proc_group)

        self._dewarp_check = QCheckBox("Dewarp (SOTA model)")
        self._dewarp_check.setChecked(True)
        pg_layout.addWidget(self._dewarp_check)

        self._sharpen_check = QCheckBox("Auto-sharpen")
        self._sharpen_check.setChecked(True)
        pg_layout.addWidget(self._sharpen_check)

        layout.addWidget(proc_group)

        # ── Apply to all ─────────────────────────────────────────────────────
        self._apply_all_btn = QPushButton("Apply Settings to All")
        layout.addWidget(self._apply_all_btn)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Wire signals
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        for sl in (self._brightness_slider, self._contrast_slider,
                   self._saturation_slider, self._sharpness_slider,
                   self._shadow_slider, self._highlight_slider):
            sl.valueChanged.connect(self._on_slider_changed)
        self._dewarp_check.toggled.connect(self.dewarp_toggled)
        self._sharpen_check.toggled.connect(self.sharpen_toggled)
        self._apply_all_btn.clicked.connect(self.apply_to_all_requested)

        self._on_preset_changed(0)

    # ── public API ────────────────────────────────────────────────────────────

    def set_document(self, doc: DocumentImage) -> None:
        fs = doc.filter_settings
        # Block all signals to prevent cascade emissions
        self.blockSignals(True)
        preset_keys = list(_PRESET_LABELS.keys())
        idx = preset_keys.index(fs.preset) if fs.preset in preset_keys else 0
        self._preset_combo.setCurrentIndex(idx)
        self._brightness_slider.setValue(int(fs.brightness * 100))
        self._contrast_slider.setValue(int(fs.contrast * 100))
        self._saturation_slider.setValue(int(fs.saturation * 100))
        self._sharpness_slider.setValue(int(fs.sharpness * 100))
        self._shadow_slider.setValue(int(fs.shadow * 100))
        self._highlight_slider.setValue(int(fs.highlight * 100))
        self._dewarp_check.setChecked(doc.dewarp_enabled)
        self._sharpen_check.setChecked(doc.sharpen_enabled)
        self.blockSignals(False)
        self._update_custom_group_visibility()

    def get_settings(self) -> FilterSettings:
        preset = self._preset_combo.currentData()
        return FilterSettings(
            preset=preset,
            brightness=self._brightness_slider.value() / 100.0,
            contrast=self._contrast_slider.value() / 100.0,
            saturation=self._saturation_slider.value() / 100.0,
            sharpness=self._sharpness_slider.value() / 100.0,
            shadow=self._shadow_slider.value() / 100.0,
            highlight=self._highlight_slider.value() / 100.0,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _on_preset_changed(self, _index: int) -> None:
        self._update_custom_group_visibility()
        if not self.signalsBlocked():
            self.filter_changed.emit(self.get_settings())

    def _on_slider_changed(self) -> None:
        if not self.signalsBlocked():
            lbl_map = {
                self._brightness_slider: self._brightness_lbl,
                self._contrast_slider: self._contrast_lbl,
                self._saturation_slider: self._saturation_lbl,
                self._sharpness_slider: self._sharpness_lbl,
                self._shadow_slider: self._shadow_lbl,
                self._highlight_slider: self._highlight_lbl,
            }
            sender = self.sender()
            if sender in lbl_map:
                lbl_map[sender].setText(f"{sender.value() / 100.0:.2f}")
            self.filter_changed.emit(self.get_settings())

    def _update_custom_group_visibility(self) -> None:
        preset = self._preset_combo.currentData()
        self._custom_group.setVisible(preset == "custom")

    def _build_slider_row(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        lo: int,
        hi: int,
        default: int,
    ) -> tuple[QSlider, QLabel]:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)

        name_lbl = QLabel(label)
        name_lbl.setFixedWidth(80)
        row.addWidget(name_lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        row.addWidget(slider)

        val_lbl = QLabel(f"{default / 100.0:.2f}")
        val_lbl.setFixedWidth(38)
        row.addWidget(val_lbl)

        parent_layout.addWidget(row_widget)
        return slider, val_lbl
