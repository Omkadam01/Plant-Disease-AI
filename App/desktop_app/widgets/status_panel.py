"""Center-panel status display."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from desktop_app.i18n import Translator
from desktop_app.styles.theme import CARD_STYLE, COLORS


class StatusPanel(QFrame):
    """Shows prediction status, inference time, and selected model."""

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self._state = "ready"
        self._inference_ms: float | None = None
        self._model_name: str | None = None
        self._error_message: str | None = None

        self.setObjectName("card")
        self.setStyleSheet(
            f"QFrame#card {{ {CARD_STYLE} background-color: {COLORS['success_bg']}; "
            f"border-color: #A7F3D0; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)

        self.status_label = QLabel()
        self.time_label = QLabel()
        self.model_label = QLabel()

        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.time_label)
        layout.addSpacing(24)
        layout.addWidget(self.model_label)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        if self._state == "running":
            self.set_running()
        elif self._state == "complete" and self._inference_ms is not None and self._model_name:
            self.set_complete(self._inference_ms, self._model_name)
        elif self._state == "error" and self._error_message:
            self.set_error(self._error_message)
        else:
            self.set_ready()

    def set_ready(self) -> None:
        self._state = "ready"
        self.status_label.setText(self.translator.t("status_panel.ready"))
        self.status_label.setStyleSheet(f"font-weight: 700; color: {COLORS['text_secondary']};")
        self.time_label.setText(self.translator.t("status_panel.inference_time_placeholder"))
        self.model_label.setText(self.translator.t("status_panel.model_placeholder"))

    def set_running(self) -> None:
        self._state = "running"
        self.status_label.setText(self.translator.t("status_panel.running"))
        self.status_label.setStyleSheet(f"font-weight: 700; color: {COLORS['accent']};")

    def set_complete(
        self,
        inference_ms: float,
        model_name: str,
        *,
        crop_name: str = "",
        low_confidence: bool = False,
    ) -> None:
        self._state = "complete"
        self._inference_ms = inference_ms
        self._model_name = model_name
        self._crop_name = crop_name
        self._low_confidence = low_confidence
        self.status_label.setText(self.translator.t("status_panel.complete"))
        self.status_label.setStyleSheet(f"font-weight: 700; color: {COLORS['success']};")
        self.time_label.setText(self.translator.t("status_panel.inference_time", ms=inference_ms))
        if crop_name:
            self.model_label.setText(
                self.translator.t("status_panel.crop_model", crop=crop_name, name=model_name)
            )
        else:
            self.model_label.setText(self.translator.t("status_panel.model", name=model_name))
        if low_confidence:
            self.status_label.setText(self.translator.t("status_panel.low_confidence"))
            self.status_label.setStyleSheet("font-weight: 700; color: #D97706;")

    def set_model_info(self, crop_name: str, model_name: str) -> None:
        self._crop_name = crop_name
        self._model_name = model_name
        if self._state == "complete" and self._inference_ms is not None:
            self.set_complete(
                self._inference_ms,
                model_name,
                crop_name=crop_name,
                low_confidence=getattr(self, "_low_confidence", False),
            )
        elif crop_name:
            self.model_label.setText(
                self.translator.t("status_panel.crop_model", crop=crop_name, name=model_name)
            )
        else:
            self.model_label.setText(self.translator.t("status_panel.model", name=model_name))

    def set_error(self, message: str) -> None:
        self._state = "error"
        self._error_message = message
        self.status_label.setText(self.translator.t("status_panel.error", message=message))
        self.status_label.setStyleSheet("font-weight: 700; color: #DC2626;")

    def clear(self) -> None:
        self._error_message = None
        self._inference_ms = None
        self._model_name = None
        self.set_ready()
