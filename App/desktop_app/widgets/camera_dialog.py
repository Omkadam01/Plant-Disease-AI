"""Webcam / Pi Camera capture dialog."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from desktop_app.i18n import Translator
from desktop_app.services.camera_service import CameraBackend, create_camera_backend
from utils.config import AppConfig
from utils.platform import is_raspberry_pi


class CameraCaptureDialog(QDialog):
    """Capture a single frame from USB webcam or Raspberry Pi Camera Module."""

    def __init__(
        self,
        config: AppConfig,
        translator: Translator,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.translator = translator
        self.captured_path: Path | None = None
        self._camera: CameraBackend | None = None
        self._timer = QTimer(self)
        self._on_pi = is_raspberry_pi()

        # Smaller dialog / slower preview on Pi to avoid OOM crashes.
        if self._on_pi:
            self.setFixedSize(520, 440)
            preview_min = (480, 320)
            self._preview_interval_ms = 150
            self._scale_mode = Qt.TransformationMode.FastTransformation
        else:
            self.setFixedSize(640, 520)
            preview_min = (600, 400)
            self._preview_interval_ms = 30
            self._scale_mode = Qt.TransformationMode.SmoothTransformation

        layout = QVBoxLayout(self)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(*preview_min)
        self.preview.setStyleSheet("background: #F0F2F5; border-radius: 8px;")

        btn_row = QHBoxLayout()
        self.capture_btn = QPushButton()
        self.cancel_btn = QPushButton()
        self.cancel_btn.setObjectName("secondaryBtn")
        btn_row.addStretch()
        btn_row.addWidget(self.capture_btn)
        btn_row.addWidget(self.cancel_btn)

        layout.addWidget(self.preview)
        layout.addLayout(btn_row)

        self.capture_btn.clicked.connect(self._capture)
        self.cancel_btn.clicked.connect(self.reject)
        self._timer.timeout.connect(self._update_frame)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.translator.t("camera.title"))
        self.capture_btn.setText(self.translator.t("buttons.capture"))
        self.cancel_btn.setText(self.translator.t("buttons.cancel"))
        if self._camera is None or not self._camera.is_open:
            if not self.preview.pixmap() or self.preview.pixmap().isNull():
                self.preview.setText(self.translator.t("camera.initializing"))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.preview.setText(self.translator.t("camera.initializing"))
        # Longer defer on Pi so previous camera session can fully release.
        delay_ms = 400 if self._on_pi else 50
        QTimer.singleShot(delay_ms, self._open_camera)

    def _open_camera(self) -> None:
        if self._camera is not None:
            return
        try:
            self._camera = create_camera_backend(self.config)
        except Exception:
            self._camera = None
        if self._camera is None or not self._camera.is_open:
            self.preview.setText(self.translator.t("camera.not_available"))
            self.capture_btn.setEnabled(False)
            return
        self._timer.start(self._preview_interval_ms)

    def _release_camera(self) -> None:
        self._timer.stop()
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        self.preview.clear()
        if self._on_pi:
            import gc

            gc.collect()

    def closeEvent(self, event) -> None:
        self._release_camera()
        super().closeEvent(event)

    def reject(self) -> None:
        self._release_camera()
        super().reject()

    def _frame_to_pixmap(self, rgb_frame) -> QPixmap:
        h, w, ch = rgb_frame.shape
        if ch != 3:
            return QPixmap()
        image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image).scaled(
            self.preview.width(),
            self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            self._scale_mode,
        )

    def _update_frame(self) -> None:
        if self._camera is None:
            return
        try:
            frame = self._camera.read_rgb()
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                return
            self.preview.setPixmap(self._frame_to_pixmap(frame))
        except Exception:
            # Never let a bad frame kill the whole Pi desktop session.
            return

    def _capture_dir(self) -> Path:
        """Always save under the project folder (pendrive when app runs from USB)."""
        capture_dir = self.config.project_root / "logs" / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        return capture_dir

    def _capture(self) -> None:
        if self._camera is None:
            return
        try:
            frame = self._camera.read_rgb()
        except Exception:
            return
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return

        capture_dir = self._capture_dir()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = capture_dir / f"capture_{timestamp}.jpg"

        # Release camera before disk write + inference to free RAM on Pi.
        self._release_camera()

        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            self.preview.setText(self.translator.t("camera.not_available"))
            return

        self.captured_path = path
        self.accept()
