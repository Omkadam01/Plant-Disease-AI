"""Left panel — crop and model selection."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFrame, QLabel, QVBoxLayout, QWidget

from desktop_app.i18n import Translator
from desktop_app.styles.theme import CARD_STYLE


class LeftPanel(QFrame):
    """Crop and model dropdown selectors."""

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ {CARD_STYLE} }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.crop_label = QLabel()
        self.crop_label.setObjectName("sectionLabel")
        self.crop_combo = QComboBox()
        self.crop_combo.addItem("", "grape")
        self.crop_combo.addItem("", "tomato")

        self.model_label = QLabel()
        self.model_label.setObjectName("sectionLabel")
        self.model_combo = QComboBox()
        self.model_combo.addItem("", "efficientnet_b0")

        layout.addWidget(self.crop_label)
        layout.addWidget(self.crop_combo)
        layout.addSpacing(8)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addStretch()

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.crop_label.setText(self.translator.t("left_panel.crop"))
        self.model_label.setText(self.translator.t("left_panel.model"))
        self.crop_combo.setItemText(0, self.translator.t("left_panel.crop_grape"))
        self.crop_combo.setItemText(1, self.translator.t("left_panel.crop_tomato"))
        self.model_combo.setItemText(0, self.translator.t("left_panel.model_efficientnet"))

    def selected_crop(self) -> str:
        return str(self.crop_combo.currentData())

    def selected_model(self) -> str:
        return str(self.model_combo.currentData())

    def selected_model_display(self) -> str:
        return self.model_combo.currentText()

    def selected_crop_display(self) -> str:
        return self.crop_combo.currentText()
