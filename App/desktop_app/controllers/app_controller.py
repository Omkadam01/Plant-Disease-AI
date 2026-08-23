"""Application controller — connects UI events to services."""

from __future__ import annotations

import gc
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from desktop_app.services.camera_service import CameraBackend, create_camera_backend
from desktop_app.services.groq_service import GroqExplanation, GroqExplanationService
from desktop_app.services.inference_service import InferenceService
from desktop_app.services.model_manager import ModelManager
from desktop_app.services.workers import GroqWorker, InferenceWorker, ModelLoadWorker
from inference.explainable_predictor import ExplainablePredictionResult
from utils.config import AppConfig
from utils.platform import is_raspberry_pi, prepare_image_for_pi_inference

logger = logging.getLogger(__name__)


class AppController:
    """
    Mediates between the main window UI and the inference service.

    Keeps inference logic out of widget code.
    """

    def __init__(self, config: AppConfig, window: QWidget) -> None:
        self.config = config
        self.window = window
        self.model_manager = ModelManager(config.project_root)
        self.groq_service = GroqExplanationService(config.project_root)
        self.service: InferenceService | None = None
        self._current_crop = config.crop_name
        self._inference_worker: InferenceWorker | None = None
        self._groq_worker: GroqWorker | None = None
        self._load_worker: ModelLoadWorker | None = None
        self._current_image_path: Path | None = None
        self._last_result: ExplainablePredictionResult | None = None
        self._inference_generation = 0
        self._busy = False

        self._camera: CameraBackend | None = None
        self._camera_preview_active = False
        self._last_preview_frame: np.ndarray | None = None
        self._preview_timer = QTimer(window)
        self._preview_timer.timeout.connect(self._update_live_preview)
        self._preview_interval_ms = 200 if is_raspberry_pi() else 40

        self._bind_window()

    def _bind_window(self) -> None:
        w = self.window
        w.upload_btn.clicked.connect(self.on_upload)
        w.capture_btn.clicked.connect(self.on_capture)
        w.left_panel.crop_combo.currentIndexChanged.connect(self.on_crop_changed)

    def _t(self, key: str, **kwargs) -> str:
        return self.window.translator.t(key, **kwargs)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if self._camera_preview_active:
            self.window.upload_btn.setEnabled(False)
            self.window.capture_btn.setEnabled(True)
        else:
            self.window.upload_btn.setEnabled(not busy)
            self.window.capture_btn.setEnabled(not busy)

    def _cleanup_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def start_model_loading(self) -> None:
        self._load_crop_model(self._current_crop)

    def on_crop_changed(self, _index: int = 0) -> None:
        self._stop_camera_preview()
        crop = self.window.left_panel.selected_crop()
        if crop == self._current_crop and self.service is not None:
            return
        self._inference_generation += 1
        self._current_crop = crop
        self._last_result = None
        self._current_image_path = None
        self.window.clear_inference_state()
        self._load_crop_model(crop)

    def _load_crop_model(self, crop: str) -> None:
        if self.model_manager.is_cached(crop):
            self.config = self.model_manager.load_config(crop)
            self.service = self.model_manager.get_cached(crop)
            self.window.set_model_ready(self.service)
            logger.info("Model loaded from cache | crop=%s", crop)
            return

        self.window.set_model_loading()
        self._load_worker = ModelLoadWorker(crop, self.model_manager, self.window)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_load_error)
        self._load_worker.start()

    def _on_model_loaded(self, crop: str, service: InferenceService) -> None:
        self._current_crop = crop
        self.config = self.model_manager.load_config(crop)
        self.service = service
        self.window.set_model_ready(service)
        logger.info("Model loaded for desktop app | crop=%s", crop)

    def _on_model_load_error(self, error: str) -> None:
        self.window.set_model_error(error)
        logger.error("Model load failed: %s", error)
        QMessageBox.critical(
            self.window,
            self._t("dialog.model_load_error_title"),
            self._t("dialog.model_load_error_body", error=error),
        )

    def on_upload(self) -> None:
        if self._busy:
            return
        self._stop_camera_preview()
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            self._t("file_dialog.select_leaf"),
            "",
            self._t("file_dialog.image_filter"),
        )
        if path:
            self.run_inference(Path(path))

    def on_capture(self) -> None:
        """Toggle: Start Camera → live preview; Capture → save frame + inference."""
        if self._busy and not self._camera_preview_active:
            return

        if not self._camera_preview_active:
            self._start_camera_preview()
        else:
            self._capture_from_preview()

    def _start_camera_preview(self) -> None:
        self._cleanup_memory()
        self.window.image_panel.overlay_display.set_image(None)
        self.window.image_panel.original_display.set_image(None)
        self.window.image_panel.original_display.image_label.setText(
            self._t("camera.initializing")
        )

        try:
            camera = create_camera_backend(self.config)
        except Exception as exc:
            QMessageBox.critical(
                self.window,
                self._t("camera.title"),
                f"Failed to open camera: {exc}",
            )
            return

        if camera is None or not camera.is_open:
            QMessageBox.warning(
                self.window,
                self._t("camera.title"),
                self._t("camera.not_available"),
            )
            return

        self._camera = camera
        self._camera_preview_active = True
        self.window.left_panel.set_camera_preview_active(True)
        self.window.upload_btn.setEnabled(False)
        self._preview_timer.start(self._preview_interval_ms)
        logger.info("Live camera preview started | backend=%s", camera.name)

    def _update_live_preview(self) -> None:
        if self._camera is None or not self._camera_preview_active:
            return
        try:
            frame = self._camera.read_rgb()
        except Exception:
            return
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return

        self._last_preview_frame = frame
        h, w, ch = frame.shape
        image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.window.image_panel.original_display.set_live_frame(pixmap)

    def _stop_camera_preview(self) -> None:
        self._preview_timer.stop()
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        self._camera_preview_active = False
        self._last_preview_frame = None
        self.window.left_panel.set_camera_preview_active(False)
        if not self._busy:
            self.window.upload_btn.setEnabled(True)
            self.window.capture_btn.setEnabled(True)
        self._cleanup_memory()

    def _capture_from_preview(self) -> None:
        frame = self._last_preview_frame
        if frame is None and self._camera is not None:
            try:
                frame = self._camera.read_rgb()
            except Exception:
                frame = None

        self._stop_camera_preview()

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            QMessageBox.warning(
                self.window,
                self._t("camera.title"),
                self._t("camera.not_available"),
            )
            return

        capture_dir = self.config.project_root / "logs" / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = capture_dir / f"capture_{timestamp}.jpg"
        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            QMessageBox.critical(
                self.window,
                self._t("camera.title"),
                "Failed to save captured image.",
            )
            return

        # Show captured still in Original Image panel, then run AI.
        self.window.image_panel.original_display.set_image(path)
        self.run_inference(path)

    def run_inference(self, image_path: Path) -> None:
        selected_crop = self.window.left_panel.selected_crop()

        if self.service is None or not self.service.is_loaded:
            QMessageBox.warning(
                self.window,
                self._t("dialog.model_not_ready_title"),
                self._t("dialog.model_not_ready_body"),
            )
            return

        if self.service.crop_name != selected_crop:
            QMessageBox.warning(
                self.window,
                self._t("dialog.crop_mismatch_title"),
                self._t(
                    "dialog.crop_mismatch_body",
                    selected=self.window.left_panel.selected_crop_display(),
                    loaded=self.service.crop_name.capitalize(),
                ),
            )
            return

        if self._busy:
            return

        self._inference_generation += 1
        inference_generation = self._inference_generation
        prepared = prepare_image_for_pi_inference(
            image_path,
            output_dir=self.config.project_root / "logs" / "captures",
            max_side=512,
        )
        self._current_image_path = prepared
        if os.getenv("PLANT_DISEASE_INFERENCE_DEBUG", "").lower() in ("1", "true", "yes"):
            logger.info(
                "[INFERENCE_DEBUG] selected_crop=%s display=%s service_crop=%s weights=%s",
                selected_crop,
                self.window.left_panel.selected_crop_display(),
                self.service.crop_name,
                self.service.weights_path,
            )
        self._set_busy(True)
        self.window.set_inference_running(prepared)
        self._inference_worker = InferenceWorker(
            self.service,
            prepared,
            inference_generation,
            self.window,
        )
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.start()

    def _on_inference_finished(self, generation: int, result: ExplainablePredictionResult) -> None:
        self._set_busy(False)
        if generation != self._inference_generation:
            logger.info("Discarding stale inference result (crop changed during prediction)")
            return
        if self.service and self.service.crop_name != self.window.left_panel.selected_crop():
            logger.warning("Discarding inference result due to crop mismatch")
            return

        self._last_result = result
        low_confidence = result.confidence < 0.75
        if len(result.top_predictions) >= 2:
            margin = result.top_predictions[0].confidence - result.top_predictions[1].confidence
            low_confidence = low_confidence or margin < 0.15

        self.window.display_result(result, low_confidence=low_confidence)
        logger.info(
            "Inference complete | crop=%s prediction=%s confidence=%.2f",
            self._current_crop,
            result.predicted_class,
            result.confidence,
        )
        self._cleanup_memory()
        self._request_groq_explanation(result)

    def _request_groq_explanation(self, result: ExplainablePredictionResult) -> None:
        if self._current_image_path is None:
            return

        if self._groq_worker is not None and self._groq_worker.isRunning():
            self._groq_worker.wait(1000)

        self.window.set_groq_loading()
        crop_display = self.window.left_panel.selected_crop_display()
        language = self.window.translator.locale

        self._groq_worker = GroqWorker(
            self.groq_service,
            self._current_image_path,
            crop_display,
            self._current_crop,
            result,
            language,
            self.window,
        )
        self._groq_worker.finished.connect(self._on_groq_finished)
        self._groq_worker.error.connect(self._on_groq_error)
        self._groq_worker.start()

    def on_language_changed(self) -> None:
        if self._last_result is None or self._current_image_path is None:
            return
        self._request_groq_explanation(self._last_result)

    def _on_groq_finished(self, explanation: GroqExplanation) -> None:
        self.window.display_groq_explanation(explanation)
        if not explanation.from_cache:
            logger.info(
                "Groq explanation displayed | crop=%s language=%s latency_ms=%.0f",
                self._current_crop,
                self.window.translator.locale,
                explanation.latency_ms,
            )

    def _on_groq_error(self, error: str) -> None:
        logger.error("Groq explanation failed: %s", error)
        self.window.display_groq_unavailable()

    def _on_inference_error(self, error: str) -> None:
        self._set_busy(False)
        self._cleanup_memory()
        self.window.set_inference_error(error)
        logger.error("Inference error: %s", error)
        QMessageBox.critical(
            self.window,
            self._t("dialog.inference_error_title"),
            self._t("dialog.inference_error_body", error=error),
        )
