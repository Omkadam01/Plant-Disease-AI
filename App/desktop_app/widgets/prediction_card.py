"""Primary prediction — compact single line with dropdown for alternates."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QToolButton

from desktop_app.i18n import Translator
from desktop_app.styles.theme import CARD_STYLE, COLORS
from inference.predictor import TopPrediction


class PredictionCard(QFrame):
    """Disease Name : <class>  <confidence>  (dropdown)."""

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self._predictions: list[TopPrediction] = []

        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ {CARD_STYLE} }}")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 10, 0)
        layout.setSpacing(8)

        self.label_prefix = QLabel()
        self.label_prefix.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {COLORS['text_secondary']};")

        self.primary_name = QLabel()
        self.primary_name.setStyleSheet("font-size: 14px; font-weight: 700;")

        self.primary_conf = QLabel()
        self.primary_conf.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {COLORS['accent']};"
        )

        self.dropdown_btn = QToolButton()
        self.dropdown_btn.setObjectName("predictionDropdown")
        self.dropdown_btn.setText("▾")
        self.dropdown_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.dropdown_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.dropdown_btn.setVisible(False)
        self.dropdown_btn.setFixedSize(28, 28)

        layout.addWidget(self.label_prefix)
        layout.addWidget(self.primary_name)
        layout.addStretch()
        layout.addWidget(self.primary_conf)
        layout.addWidget(self.dropdown_btn)

        self.retranslate_ui()
        self.clear()

    def _build_menu(self, extras: list[TopPrediction]) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
            f"padding: 4px; }} QMenu::item {{ padding: 6px 16px; }}"
        )
        for index, pred in enumerate(extras, start=2):
            label = self.translator.translate_class(pred.class_name)
            pct = self.translator.t("prediction.confidence", pct=pred.confidence * 100)
            action = QAction(f"{index}. {label}  —  {pct}", self)
            action.setEnabled(False)
            menu.addAction(action)
        self.dropdown_btn.setMenu(menu)

    def retranslate_ui(self) -> None:
        self.label_prefix.setText(self.translator.t("prediction.disease_label"))
        if self._predictions:
            self.set_predictions(self._predictions)
        else:
            self.clear()

    def set_predictions(self, predictions: list[TopPrediction]) -> None:
        self._predictions = predictions
        if not predictions:
            self.clear()
            return

        top = predictions[0]
        self.primary_name.setText(self.translator.translate_class(top.class_name))
        self.primary_conf.setText(self.translator.t("prediction.confidence", pct=top.confidence * 100))

        extras = predictions[1:3]
        has_extras = len(extras) > 0
        self.dropdown_btn.setVisible(has_extras)
        if has_extras:
            self._build_menu(extras)

    def clear(self) -> None:
        self._predictions = []
        self.primary_name.setText(self.translator.t("prediction.empty_name"))
        self.primary_conf.setText(self.translator.t("prediction.empty_percent"))
        self.dropdown_btn.setMenu(None)
        self.dropdown_btn.setVisible(False)
