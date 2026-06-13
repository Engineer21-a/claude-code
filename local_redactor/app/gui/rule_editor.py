"""Per-term rule editor — edit a single UserWord's matching options."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QWidget,
)

from app.config.settings import UserWord


class RuleEditor(QDialog):
    def __init__(self, word: Optional[UserWord] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit redaction term")
        self.word = word or UserWord(text="")
        form = QFormLayout(self)

        self.text = QLineEdit(self.word.text)
        form.addRow("Term", self.text)

        self.case = QCheckBox()
        self.case.setChecked(self.word.case_insensitive)
        form.addRow("Case-insensitive", self.case)

        self.whole = QCheckBox()
        self.whole.setChecked(self.word.whole_word_only)
        form.addRow("Whole word only", self.whole)

        self.umlaut = QCheckBox()
        self.umlaut.setChecked(self.word.normalize_umlauts)
        form.addRow("Normalise umlauts (ä↔ae)", self.umlaut)

        self.full_line = QCheckBox()
        self.full_line.setChecked(self.word.redact_full_line)
        form.addRow("Redact full line", self.full_line)

        self.everywhere = QCheckBox()
        self.everywhere.setChecked(self.word.redact_everywhere)
        form.addRow("Redact everywhere once confirmed", self.everywhere)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _apply(self) -> None:
        self.word = UserWord(
            text=self.text.text().strip(),
            case_insensitive=self.case.isChecked(),
            whole_word_only=self.whole.isChecked(),
            normalize_umlauts=self.umlaut.isChecked(),
            redact_full_line=self.full_line.isChecked(),
            redact_everywhere=self.everywhere.isChecked(),
        )
        self.accept()
