"""Side-by-side image display widgets."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from desktop_app.i18n import Translator
from desktop_app.styles.theme import IMAGE_PLACEHOLDER_STYLE


class ImageDisplay(QFrame):
    """Single image panel with label."""

    IMAGE_HEIGHT = 300
    IMAGE_HEIGHT_SMALL = 160

    def __init__(
        self,
        title_key: str,
        translator: Translator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.translator = translator
        self.title_key = title_key
        self._current_path: Path | str | None = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: 600; font-size: 12px; color: #5F6B7A;")

        from utils.platform import is_raspberry_pi

        min_h = self.IMAGE_HEIGHT_SMALL if is_raspberry_pi() else self.IMAGE_HEIGHT

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(min_h)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet(IMAGE_PLACEHOLDER_STYLE)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.title_label.setText(self.translator.t(self.title_key))
        pix = self.image_label.pixmap()
        if self._current_path is None and (pix is None or pix.isNull()) and not self.image_label.text():
            self.image_label.setText(self.translator.t("image_panel.no_image"))
        elif self._current_path is None and (pix is None or pix.isNull()):
            # Keep live-preview hint if showing text only
            if self.image_label.text() in {
                "",
                self.translator.t("image_panel.no_image"),
            }:
                pass

    def set_live_frame(self, pixmap: QPixmap) -> None:
        """Show a live camera frame (no file path)."""
        self._current_path = None
        if pixmap.isNull():
            self.image_label.setText(self.translator.t("image_panel.live_preview"))
            return
        self._apply_scaled_pixmap(pixmap)

    def set_image(self, path: Path | str | None) -> None:
        self._current_path = path
        # Drop previous pixmap before loading a new one (helps repeated captures on Pi).
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())

        if path is None or not Path(path).exists():
            self.image_label.setText(self.translator.t("image_panel.no_image"))
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image_label.setText(self.translator.t("image_panel.failed_to_load"))
            return

        self._apply_scaled_pixmap(pixmap)

    def _apply_scaled_pixmap(self, pixmap: QPixmap) -> None:
        from utils.platform import is_raspberry_pi

        mode = (
            Qt.TransformationMode.FastTransformation
            if is_raspberry_pi()
            else Qt.TransformationMode.SmoothTransformation
        )
        scaled = pixmap.scaled(
            max(self.image_label.width() - 8, 1),
            max(self.image_label.height() - 8, 1),
            Qt.AspectRatioMode.KeepAspectRatio,
            mode,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_path and Path(self._current_path).exists():
            pixmap = QPixmap(str(self._current_path))
            if not pixmap.isNull():
                self._apply_scaled_pixmap(pixmap)


class ImagePairPanel(QWidget):
    """Original and Grad-CAM overlay displayed side by side on one row."""

    def __init__(self, translator: Translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.translator = translator
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.original_display = ImageDisplay("image_panel.original", translator)
        self.overlay_display = ImageDisplay("image_panel.gradcam", translator)

        layout.addWidget(self.original_display, 1)
        layout.addWidget(self.overlay_display, 1)

    def retranslate_ui(self) -> None:
        self.original_display.retranslate_ui()
        self.overlay_display.retranslate_ui()

    def set_images(self, original_path: Path | str | None, overlay_path: Path | str | None) -> None:
        self.original_display.set_image(original_path)
        self.overlay_display.set_image(overlay_path)

    def clear(self) -> None:
        self.set_images(None, None)
