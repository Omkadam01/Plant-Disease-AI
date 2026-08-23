"""Custom UI widgets for PlantDiseaseAI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from desktop_app.disease_info import DiseaseInfo, get_disease_info
from desktop_app.styles import COLORS
from inference.predictor import PredictionResult


class DropZone(QFrame):
    """Drag-and-drop image upload zone."""

    imageDropped = Signal(str)
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setProperty("dragActive", False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_label = QLabel("🍃")
        self.icon_label.setStyleSheet("font-size: 48px;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel("Drag & drop a grape leaf image here")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 15px;")

        self.hint_label = QLabel("or click to browse files")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(280, 200)
        self.preview_label.setScaledContents(True)
        self.preview_label.hide()

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.preview_label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and self._is_image(url.toLocalFile()):
                event.acceptProposedAction()
                self.setProperty("dragActive", True)
                self.style().unpolish(self)
                self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if self._is_image(path):
                self.set_preview(path)
                self.imageDropped.emit(path)
                event.acceptProposedAction()

    @staticmethod
    def _is_image(path: str) -> bool:
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def set_preview(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.preview_label.show()
            self.icon_label.hide()
            self.text_label.setText(Path(image_path).name)


class ConfidenceBar(QWidget):
    """Visual confidence indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        self.label = QLabel("Confidence")
        self.label.setStyleSheet("font-weight: 600;")
        self.value_label = QLabel("0%")
        self.value_label.setStyleSheet(f"color: {COLORS['primary']}; font-weight: 700;")
        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.value_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFormat("%p%")

        layout.addLayout(header)
        layout.addWidget(self.bar)

    def set_confidence(self, confidence: float) -> None:
        pct = int(confidence * 100)
        self.bar.setValue(pct)
        self.value_label.setText(f"{pct}%")

        if confidence >= 0.8:
            color = COLORS["confidence_high"]
        elif confidence >= 0.5:
            color = COLORS["confidence_medium"]
        else:
            color = COLORS["confidence_low"]

        self.bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}"
        )


class PredictionCard(QFrame):
    """Displays prediction results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("predictionCard")

        layout = QVBoxLayout(self)

        self.title = QLabel("Prediction")
        self.title.setObjectName("sectionTitle")

        self.class_label = QLabel("—")
        self.class_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {COLORS['primary_dark']};"
        )

        self.confidence_bar = ConfidenceBar()

        self.top_k_label = QLabel("")
        self.top_k_label.setWordWrap(True)
        self.top_k_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        layout.addWidget(self.title)
        layout.addWidget(self.class_label)
        layout.addWidget(self.confidence_bar)
        layout.addWidget(self.top_k_label)
        layout.addWidget(self.time_label)

    def set_result(self, result: PredictionResult) -> None:
        self.class_label.setText(result.predicted_class)
        self.confidence_bar.set_confidence(result.confidence)

        top_k_lines = []
        for i, pred in enumerate(result.top_predictions, 1):
            top_k_lines.append(f"{i}. {pred.class_name} — {pred.confidence:.1%}")
        self.top_k_label.setText("\n".join(top_k_lines))
        self.time_label.setText(f"Inference time: {result.inference_time_ms:.1f} ms")

    def clear(self) -> None:
        self.class_label.setText("—")
        self.confidence_bar.set_confidence(0.0)
        self.top_k_label.setText("")
        self.time_label.setText("")


class DiseaseInfoPanel(QFrame):
    """Displays disease description, symptoms, treatment, and prevention."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("infoCard")

        layout = QVBoxLayout(self)

        self.disease_title = QLabel("Disease Information")
        self.disease_title.setObjectName("sectionTitle")

        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.symptoms_title = QLabel("Symptoms")
        self.symptoms_title.setStyleSheet("font-weight: 600; margin-top: 8px;")
        self.symptoms = QLabel("")
        self.symptoms.setWordWrap(True)

        self.treatment_title = QLabel("Treatment")
        self.treatment_title.setStyleSheet("font-weight: 600; margin-top: 8px;")
        self.treatment = QLabel("")
        self.treatment.setWordWrap(True)

        self.prevention_title = QLabel("Prevention")
        self.prevention_title.setStyleSheet("font-weight: 600; margin-top: 8px;")
        self.prevention = QLabel("")
        self.prevention.setWordWrap(True)

        layout.addWidget(self.disease_title)
        layout.addWidget(self.description)
        layout.addWidget(self.symptoms_title)
        layout.addWidget(self.symptoms)
        layout.addWidget(self.treatment_title)
        layout.addWidget(self.treatment)
        layout.addWidget(self.prevention_title)
        layout.addWidget(self.prevention)

    @staticmethod
    def _format_list(items: list[str]) -> str:
        return "\n".join(f"• {item}" for item in items)

    def set_disease(self, class_name: str) -> None:
        info: DiseaseInfo = get_disease_info(class_name)
        self.disease_title.setText(info.name)
        self.description.setText(info.description)
        self.symptoms.setText(self._format_list(info.symptoms))
        self.treatment.setText(self._format_list(info.treatment))
        self.prevention.setText(self._format_list(info.prevention))

    def clear(self) -> None:
        self.disease_title.setText("Disease Information")
        self.description.setText("Upload an image to see disease information.")
        self.symptoms.setText("")
        self.treatment.setText("")
        self.prevention.setText("")
