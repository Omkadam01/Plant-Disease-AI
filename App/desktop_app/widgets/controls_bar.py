"""Bottom toolbar — crop, model, language, upload, and capture."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from desktop_app.i18n import SUPPORTED_LANGUAGES, Translator
from desktop_app.styles.theme import CARD_STYLE


class ControlsBar(QFrame):
    """Horizontal control strip at the bottom of the main window."""

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ {CARD_STYLE} }}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        self.crop_label = QLabel()
        self.crop_label.setObjectName("sectionLabel")
        self.crop_combo = QComboBox()
        self.crop_combo.setMinimumWidth(120)
        self.crop_combo.addItem("", "grape")
        self.crop_combo.addItem("", "tomato")

        self.model_label = QLabel()
        self.model_label.setObjectName("sectionLabel")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.addItem("", "efficientnet_b0")

        self.language_label = QLabel()
        self.language_label.setObjectName("sectionLabel")
        self.language_combo = QComboBox()
        self.language_combo.setMinimumWidth(140)
        for code in SUPPORTED_LANGUAGES:
            self.language_combo.addItem("", code)

        self.upload_btn = QPushButton()
        self.capture_btn = QPushButton()
        self.capture_btn.setObjectName("secondaryBtn")

        layout.addWidget(self.crop_label)
        layout.addWidget(self.crop_combo)
        layout.addSpacing(8)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addSpacing(8)
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)
        layout.addStretch()
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.capture_btn)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.crop_label.setText(self.translator.t("left_panel.crop"))
        self.model_label.setText(self.translator.t("left_panel.model"))
        self.language_label.setText(self.translator.t("language.label"))
        self.crop_combo.setItemText(0, self.translator.t("left_panel.crop_grape"))
        self.crop_combo.setItemText(1, self.translator.t("left_panel.crop_tomato"))
        self.model_combo.setItemText(0, self.translator.t("left_panel.model_efficientnet"))
        for index, code in enumerate(SUPPORTED_LANGUAGES):
            self.language_combo.setItemText(index, self.translator.language_option(code))
        self.upload_btn.setText(self.translator.t("buttons.upload_image"))
        # Default label; controller may override while preview is active.
        if not hasattr(self, "_camera_preview_active") or not self._camera_preview_active:
            self.capture_btn.setText(self.translator.t("buttons.start_camera"))

    def set_camera_preview_active(self, active: bool) -> None:
        self._camera_preview_active = active
        if active:
            self.capture_btn.setText(self.translator.t("buttons.capture"))
            self.capture_btn.setObjectName("")
            self.capture_btn.style().unpolish(self.capture_btn)
            self.capture_btn.style().polish(self.capture_btn)
        else:
            self.capture_btn.setText(self.translator.t("buttons.start_camera"))
            self.capture_btn.setObjectName("secondaryBtn")
            self.capture_btn.style().unpolish(self.capture_btn)
            self.capture_btn.style().polish(self.capture_btn)

    def selected_crop(self) -> str:
        return str(self.crop_combo.currentData())

    def selected_model(self) -> str:
        return str(self.model_combo.currentData())

    def selected_model_display(self) -> str:
        return self.model_combo.currentText()

    def selected_crop_display(self) -> str:
        return self.crop_combo.currentText()
