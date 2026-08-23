"""Main application window — vertical layout with bottom controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from desktop_app.controllers.app_controller import AppController
from desktop_app.i18n import SUPPORTED_LANGUAGES, Translator
from desktop_app.services.inference_service import InferenceService
from desktop_app.styles.theme import STYLESHEET
from desktop_app.widgets.controls_bar import ControlsBar
from desktop_app.widgets.explanation_panel import ExplanationPanel
from desktop_app.widgets.header import HeaderWidget
from desktop_app.widgets.image_panel import ImagePairPanel
from desktop_app.widgets.prediction_card import PredictionCard
from inference.explainable_predictor import ExplainablePredictionResult
from utils.config import AppConfig
from utils.platform import is_raspberry_pi


class MainWindow(QMainWindow):
    """PlantDiseaseAI desktop application (auto-sizes for small Pi displays)."""

    WIDTH = 1400
    HEIGHT = 780

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.translator = Translator()

        self._status_model_name = self.translator.t("left_panel.model_efficientnet")
        self._status_crop_name = ""
        self._inference_service: InferenceService | None = None
        self._last_result: ExplainablePredictionResult | None = None

        self._apply_window_size()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        # Tighter margins on small screens.
        margin = 8 if is_raspberry_pi() else 20
        root.setContentsMargins(margin, 8, margin, 8)
        root.setSpacing(6 if is_raspberry_pi() else 8)

        self.header = HeaderWidget(self.translator)
        root.addWidget(self.header)

        self.image_panel = ImagePairPanel(self.translator)
        root.addWidget(self.image_panel, 1)

        self.prediction_card = PredictionCard(self.translator)
        root.addWidget(self.prediction_card)

        self.explanation_panel = ExplanationPanel(self.translator)
        root.addWidget(self.explanation_panel)

        self.controls_bar = ControlsBar(self.translator)
        self.left_panel = self.controls_bar
        self.upload_btn = self.controls_bar.upload_btn
        self.capture_btn = self.controls_bar.capture_btn
        root.addWidget(self.controls_bar)

        self._set_initial_crop(config.crop_name)
        self.setStyleSheet(STYLESHEET)

        self.translator.on_language_changed(self._on_language_changed)
        self.controls_bar.language_combo.currentIndexChanged.connect(self._on_language_combo_changed)
        self._sync_language_combo()

        self.controller = AppController(config, self)
        self.retranslate_ui()
        self.controller.start_model_loading()

    def _apply_window_size(self) -> None:
        """Fit window to the active screen — critical for 7\" HDMI/DSI panels."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(self.WIDTH, self.HEIGHT)
            return

        available = screen.availableGeometry()
        # Leave a small margin so window managers / panels don't clip us.
        max_w = max(640, available.width() - 16)
        max_h = max(480, available.height() - 16)

        if is_raspberry_pi() or max_w < self.WIDTH or max_h < self.HEIGHT:
            width = min(self.WIDTH, max_w)
            height = min(self.HEIGHT, max_h)
            self.setMinimumSize(min(640, width), min(480, height))
            self.resize(width, height)
            # Prefer maximized on very small panels so UI stays usable.
            if max_w <= 1024 or max_h <= 600:
                self.showMaximized()
        else:
            self.setFixedSize(self.WIDTH, self.HEIGHT)
    def _set_initial_crop(self, crop_name: str) -> None:
        combo = self.left_panel.crop_combo
        combo.blockSignals(True)
        for index in range(combo.count()):
            if combo.itemData(index) == crop_name:
                combo.setCurrentIndex(index)
                break
        combo.blockSignals(False)

    def _sync_language_combo(self) -> None:
        codes = list(SUPPORTED_LANGUAGES.keys())
        if self.translator.locale in codes:
            self.controls_bar.language_combo.blockSignals(True)
            self.controls_bar.language_combo.setCurrentIndex(codes.index(self.translator.locale))
            self.controls_bar.language_combo.blockSignals(False)

    def _on_language_combo_changed(self, index: int) -> None:
        codes = list(SUPPORTED_LANGUAGES.keys())
        if 0 <= index < len(codes):
            self.translator.set_language(codes[index])

    def _on_language_changed(self) -> None:
        self.retranslate_ui()
        self.controller.on_language_changed()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.translator.t("app.title"))
        self.header.retranslate_ui()
        self.left_panel.retranslate_ui()
        self.image_panel.retranslate_ui()
        self.prediction_card.retranslate_ui()
        self.explanation_panel.retranslate_ui()

    def set_model_loading(self) -> None:
        self.upload_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)

    def set_model_ready(self, service: InferenceService) -> None:
        self._inference_service = service
        self._status_model_name = self.translator.t("left_panel.model_efficientnet")
        self._status_crop_name = self.left_panel.selected_crop_display()
        self.upload_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

    def set_model_error(self, _error: str) -> None:
        pass

    def set_inference_running(self, image_path: Path) -> None:
        self.upload_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.image_panel.original_display.set_image(image_path)
        self.image_panel.overlay_display.set_image(None)
        self.prediction_card.clear()
        self.explanation_panel.reset()

    def display_result(self, result: ExplainablePredictionResult, *, low_confidence: bool = False) -> None:
        self._last_result = result
        self.upload_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

        original = result.original_output_path or result.image_path
        self.image_panel.set_images(original, result.overlay_path)
        self.prediction_card.set_predictions(result.top_predictions)

    def set_groq_loading(self) -> None:
        self.explanation_panel.set_loading()

    def display_groq_explanation(self, explanation) -> None:
        from desktop_app.services.groq_service import GroqExplanation

        assert isinstance(explanation, GroqExplanation)
        self.explanation_panel.set_explanation(explanation)

    def display_groq_unavailable(self) -> None:
        self.explanation_panel.set_unavailable()

    def set_inference_error(self, message: str) -> None:
        self.upload_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

    def clear_inference_state(self) -> None:
        self._last_result = None
        self.image_panel.clear()
        self.prediction_card.clear()
        self.explanation_panel.reset()
