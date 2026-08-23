"""Horizontal prevention / remedies / tips in one compact row."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from desktop_app.i18n import Translator
from desktop_app.services.groq_service import GroqExplanation
from desktop_app.styles.theme import CARD_STYLE, COLORS


class _ExplanationColumn(QWidget):
    def __init__(self, title_key: str, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self.title_key = title_key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: 700; font-size: 12px;")

        self.body_label = QLabel()
        self.body_label.setWordWrap(True)
        self.body_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._apply_body_style(muted=True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.body_label, 1)

        self.retranslate_ui()

    def _apply_body_style(self, muted: bool = False) -> None:
        if muted:
            self.body_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-style: italic; font-size: 11px;"
            )
        else:
            self.body_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 11px; line-height: 1.35;"
            )

    def set_text(self, text: str, *, muted: bool = False) -> None:
        self.body_label.setText(text)
        self._apply_body_style(muted=muted)

    def retranslate_ui(self) -> None:
        self.title_label.setText(self.translator.t(self.title_key))


class ExplanationPanel(QFrame):
    """Prevention, remedies, and tips in one compact horizontal card."""

    PANEL_HEIGHT = 176

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self._mode = "waiting"
        self._explanation: GroqExplanation | None = None

        self.setObjectName("card")
        self.setStyleSheet(f"QFrame#card {{ {CARD_STYLE} }}")
        from utils.platform import is_raspberry_pi

        height = 120 if is_raspberry_pi() else self.PANEL_HEIGHT
        self.setFixedHeight(height)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.prevention_section = _ExplanationColumn("explanation.prevention", translator)
        self.remedies_section = _ExplanationColumn("explanation.remedies", translator)
        self.tips_section = _ExplanationColumn("explanation.tips", translator)

        for index, column in enumerate(
            (self.prevention_section, self.remedies_section, self.tips_section)
        ):
            if index > 0:
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.VLine)
                divider.setStyleSheet(f"color: {COLORS['border']};")
                outer.addWidget(divider)
            outer.addWidget(column, 1)

        self.reset()

    def _bullets(self, items: list[str]) -> str:
        if not items:
            return ""
        return "\n".join(f"• {item}" for item in items)

    def _format_prevention(self, explanation: GroqExplanation) -> str:
        return self._bullets(explanation.prevention) or self.translator.t("explanation.unavailable")

    def _format_remedies(self, explanation: GroqExplanation) -> str:
        parts: list[str] = []
        organic = self._bullets(explanation.remedies_organic)
        chemical = self._bullets(explanation.remedies_chemical)
        if organic:
            parts.append(f"{self.translator.t('explanation.organic_remedies')}\n{organic}")
        if chemical:
            parts.append(f"{self.translator.t('explanation.chemical_remedies')}\n{chemical}")
        return "\n\n".join(parts) or self.translator.t("explanation.unavailable")

    def _format_tips(self, explanation: GroqExplanation) -> str:
        parts: list[str] = []
        if explanation.overview:
            parts.append(explanation.overview)
        tips = self._bullets(explanation.tips)
        if tips:
            parts.append(tips)
        return "\n\n".join(parts) or self.translator.t("explanation.unavailable")

    def retranslate_ui(self) -> None:
        self.prevention_section.retranslate_ui()
        self.remedies_section.retranslate_ui()
        self.tips_section.retranslate_ui()
        if self._mode == "loading":
            self.set_loading()
        elif self._mode == "unavailable":
            self.set_unavailable()
        elif self._mode == "content" and self._explanation is not None:
            self.set_explanation(self._explanation)
        else:
            self.reset()

    def set_loading(self) -> None:
        self._mode = "loading"
        loading = self.translator.t("explanation.loading")
        self.prevention_section.set_text(loading, muted=True)
        self.remedies_section.set_text(loading, muted=True)
        self.tips_section.set_text(loading, muted=True)

    def set_explanation(self, explanation: GroqExplanation) -> None:
        self._mode = "content"
        self._explanation = explanation
        self.prevention_section.set_text(self._format_prevention(explanation), muted=False)
        self.remedies_section.set_text(self._format_remedies(explanation), muted=False)
        self.tips_section.set_text(self._format_tips(explanation), muted=False)

    def set_unavailable(self) -> None:
        self._mode = "unavailable"
        unavailable = self.translator.t("explanation.unavailable")
        self.prevention_section.set_text(unavailable, muted=True)
        self.remedies_section.set_text(unavailable, muted=True)
        self.tips_section.set_text(unavailable, muted=True)

    def reset(self) -> None:
        self._mode = "waiting"
        self._explanation = None
        waiting = self.translator.t("explanation.remedies_waiting")
        self.prevention_section.set_text(waiting, muted=True)
        self.remedies_section.set_text(waiting, muted=True)
        self.tips_section.set_text(waiting, muted=True)
