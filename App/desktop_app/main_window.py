"""Main application window."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from desktop_app.styles import STYLESHEET
from desktop_app.widgets import DiseaseInfoPanel, DropZone, PredictionCard
from desktop_app.workers.inference_worker import InferenceWorker, PredictorLoader
from inference.predictor import Predictor, PredictionResult
from utils.config import AppConfig

logger = logging.getLogger(__name__)

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    """PlantDiseaseAI main application window."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.predictor: Predictor | None = None
        self.current_image_path: Path | None = None
        self.current_result: PredictionResult | None = None
        self._inference_worker: InferenceWorker | None = None

        title = str(config.get("desktop_app.title", "PlantDiseaseAI"))
        self.setWindowTitle(title)
        self.setMinimumSize(
            int(config.get("desktop_app.window.min_width", 900)),
            int(config.get("desktop_app.window.min_height", 600)),
        )
        self.resize(
            int(config.get("desktop_app.window.width", 1200)),
            int(config.get("desktop_app.window.height", 800)),
        )
        self.setStyleSheet(STYLESHEET)

        self._setup_ui()
        self._setup_statusbar()
        self._load_predictor()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Header
        header = QVBoxLayout()
        title = QLabel("PlantDiseaseAI")
        title.setObjectName("appTitle")
        subtitle = QLabel("Offline Grape Leaf Disease Detection")
        subtitle.setObjectName("appSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        main_layout.addLayout(header)

        # Content area
        content = QHBoxLayout()
        content.setSpacing(20)

        # Left panel - upload
        left_panel = QVBoxLayout()
        self.drop_zone = DropZone()
        self.drop_zone.imageDropped.connect(self._on_image_selected)
        self.drop_zone.clicked.connect(self._browse_image)

        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self._browse_image)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._run_inference)
        self.save_btn = QPushButton("Save Prediction")
        self.save_btn.setObjectName("secondaryButton")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_prediction)
        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.analyze_btn)
        btn_row.addWidget(self.save_btn)

        left_panel.addWidget(self.drop_zone)
        left_panel.addLayout(btn_row)
        content.addLayout(left_panel, stretch=2)

        # Right panel - results
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.prediction_card = PredictionCard()
        self.disease_panel = DiseaseInfoPanel()
        self.disease_panel.clear()

        right_layout.addWidget(self.prediction_card)
        right_layout.addWidget(self.disease_panel)
        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        content.addWidget(right_scroll, stretch=3)

        main_layout.addLayout(content)

    def _setup_statusbar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Loading model...")

    def _load_predictor(self) -> None:
        self._loader = PredictorLoader(self.config, self)
        self._loader.loaded.connect(self._on_predictor_loaded)
        self._loader.error.connect(self._on_predictor_error)
        self._loader.start()

    def _on_predictor_loaded(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.statusbar.showMessage("Ready — upload a grape leaf image to begin")
        logger.info("Predictor loaded successfully")

    def _on_predictor_error(self, error_msg: str) -> None:
        self.statusbar.showMessage("Model not loaded — train and export a model first")
        logger.error("Failed to load predictor: %s", error_msg)
        QMessageBox.warning(
            self,
            "Model Not Available",
            "Could not load the disease detection model.\n\n"
            "Please ensure you have:\n"
            "1. Preprocessed the dataset\n"
            "2. Trained the model\n"
                "3. Saved weights to Models/weights/grape/best_model.pth\n\n"
            f"Error: {error_msg.split(chr(10))[0]}",
        )

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Grape Leaf Image", "", IMAGE_FILTER)
        if path:
            self._on_image_selected(path)

    def _on_image_selected(self, path: str) -> None:
        self.current_image_path = Path(path)
        self.drop_zone.set_preview(path)
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.prediction_card.clear()
        self.disease_panel.clear()
        self.statusbar.showMessage(f"Selected: {Path(path).name}")

    def _run_inference(self) -> None:
        if not self.current_image_path or not self.predictor:
            return

        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.statusbar.showMessage("Analyzing image...")

        self._inference_worker = InferenceWorker(
            self.predictor, self.current_image_path, self
        )
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.start()

    def _on_inference_finished(self, result: PredictionResult) -> None:
        self.current_result = result
        self.prediction_card.set_result(result)
        self.disease_panel.set_disease(result.predicted_class)
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.statusbar.showMessage(
            f"Prediction: {result.predicted_class} ({result.confidence:.1%}) "
            f"— {result.inference_time_ms:.0f} ms"
        )

    def _on_inference_error(self, error_msg: str) -> None:
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.statusbar.showMessage("Inference failed")
        logger.error("Inference error: %s", error_msg)
        QMessageBox.critical(self, "Inference Error", f"Failed to analyze image:\n{error_msg.split(chr(10))[0]}")

    def _save_prediction(self) -> None:
        if not self.current_result:
            return

        save_dir = self.config.project_root / self.config.get(
            "desktop_app.save_predictions_dir", "logs/predictions"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{timestamp}.json"
        output_path = save_dir / filename

        data = self.current_result.to_dict()
        data["saved_at"] = datetime.now(timezone.utc).isoformat()
        data["disease_info"] = {
            "description": self.disease_panel.description.text(),
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.statusbar.showMessage(f"Prediction saved: {output_path.name}")
        QMessageBox.information(self, "Saved", f"Prediction saved to:\n{output_path}")
