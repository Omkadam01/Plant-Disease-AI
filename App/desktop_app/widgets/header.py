"""Application header widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from desktop_app.i18n import Translator


class HeaderWidget(QWidget):
    """Top section with bold application name only."""

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.title_label = QLabel()
        self.title_label.setObjectName("appTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.title_label.setText(self.translator.t("header.title"))
