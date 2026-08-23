"""Reusable inference predictor supporting multiple backends."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.factory import ModelFactory
from training.transforms import IMAGENET_MEAN, IMAGENET_STD, get_inference_transforms
from utils.config import AppConfig
from utils.device import get_device
from utils.image_utils import read_image_rgb

logger = logging.getLogger(__name__)

INFERENCE_DEBUG = os.getenv("PLANT_DISEASE_INFERENCE_DEBUG", "").lower() in ("1", "true", "yes")


class InferenceBackend(str, Enum):
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"


@dataclass(frozen=True)
class TopPrediction:
    """Single top-k prediction entry."""

    class_id: int
    class_name: str
    confidence: float


@dataclass
class PredictionResult:
    """Complete prediction result."""

    predicted_class: str
    predicted_class_id: int
    confidence: float
    top_predictions: list[TopPrediction] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "predicted_class_id": self.predicted_class_id,
            "confidence": self.confidence,
            "top_predictions": [
                {
                    "class_id": p.class_id,
                    "class_name": p.class_name,
                    "confidence": p.confidence,
                }
                for p in self.top_predictions
            ],
            "inference_time_ms": self.inference_time_ms,
            "image_path": self.image_path,
        }


class Predictor:
    """
    Production inference predictor.

    Supports PyTorch, TorchScript, and ONNX backends.
    Returns predicted class, confidence, top-3 predictions, and inference time.
    """

    def __init__(
        self,
        config: AppConfig,
        weights_path: Path | None = None,
        backend: str | None = None,
        class_names_path: Path | None = None,
    ) -> None:
        self.config = config
        self.device = get_device(config)
        self.backend = InferenceBackend(
            (backend or config.get("inference.backend", "pytorch")).lower()
        )
        self.top_k = int(config.get("inference.top_k", 3))
        self.confidence_threshold = float(config.get("inference.confidence_threshold", 0.0))

        if weights_path:
            self.weights_path = weights_path
        else:
            default = config.get(
                "inference.weights_path",
                f"Models/weights/{config.crop_name}/best_model.pth",
            )
            self.weights_path = config.project_root / default

        self.class_names = self._load_class_names(class_names_path)
        self.id_to_name = {i: name for i, name in enumerate(self.class_names)}
        self.image_size = int(config.get("inference.image_size", config.get("training.image_size", 224)))
        self.transform = get_inference_transforms(self.image_size)

        self._model: nn.Module | torch.jit.ScriptModule | None = None
        self._onnx_session = None
        self._load_backend()

        if INFERENCE_DEBUG:
            self._log_load_summary()

    def _log_load_summary(self) -> None:
        logger.info(
            "[INFERENCE_DEBUG] crop=%s weights=%s class_mapping=%s image_size=%s "
            "num_classes=%s mean=%s std=%s backend=%s device=%s",
            self.config.crop_name,
            self.weights_path,
            self.class_names_path,
            self.image_size,
            len(self.class_names),
            IMAGENET_MEAN,
            IMAGENET_STD,
            self.backend.value,
            self.device,
        )
        logger.info("[INFERENCE_DEBUG] class_names=%s", self.class_names)

    def _log_predict_debug(
        self,
        *,
        logits: np.ndarray,
        probs: np.ndarray,
        pred_idx: int,
        top_indices: np.ndarray,
    ) -> None:
        if not INFERENCE_DEBUG:
            return
        logger.info("[INFERENCE_DEBUG] raw_logits=%s", np.array2string(logits, precision=6, separator=", "))
        logger.info("[INFERENCE_DEBUG] softmax_probs=%s", np.array2string(probs, precision=6, separator=", "))
        logger.info(
            "[INFERENCE_DEBUG] predicted_class_index=%d mapped_class_name=%s confidence=%.6f",
            pred_idx,
            self.id_to_name.get(pred_idx, f"Class_{pred_idx}"),
            float(probs[pred_idx]),
        )
        top3 = [
            {
                "index": int(idx),
                "name": self.id_to_name.get(int(idx), f"Class_{idx}"),
                "confidence": float(probs[idx]),
            }
            for idx in top_indices
        ]
        logger.info("[INFERENCE_DEBUG] top3_predictions=%s", top3)

    def _load_class_names(self, class_names_path: Path | None) -> list[str]:
        if class_names_path is None:
            path_str = self.config.get(
                "inference.class_names_path",
                f"Dataset/datasets/{self.config.crop_name}/reports/class_mapping.json",
            )
            class_names_path = self.config.project_root / path_str

        self.class_names_path = class_names_path.resolve()

        if class_names_path.exists():
            with class_names_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            id_to_display = data.get("id_to_display", {})
            if id_to_display:
                sorted_items = sorted(
                    id_to_display.items(),
                    key=lambda item: int(item[0]),
                )
                return [item[1] for item in sorted_items]
            return [c["display_name"] for c in data.get("classes", [])]

        logger.warning("Class mapping not found at %s — falling back to config", class_names_path)
        return self.config.class_names

    def _load_backend(self) -> None:
        if self.backend == InferenceBackend.PYTORCH:
            self._load_pytorch()
        elif self.backend == InferenceBackend.TORCHSCRIPT:
            self._load_torchscript()
        elif self.backend == InferenceBackend.ONNX:
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_pytorch(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        class_mapping = checkpoint.get("class_mapping", {})
        model_name = checkpoint.get("model_name") or class_mapping.get(
            "model_name", self.config.get("training.model_name", "efficientnet_b0")
        )
        self.image_size = class_mapping.get("image_size", self.image_size)
        self.transform = get_inference_transforms(self.image_size)

        model = ModelFactory.create(
            model_name=str(model_name),
            num_classes=len(self.class_names),
            pretrained=False,
            device=self.device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model = model

        checkpoint_classes = class_mapping.get("id_to_display") or class_mapping.get("folder_to_id")
        if checkpoint_classes and len(self.class_names) != len(checkpoint_classes):
            logger.warning(
                "Class count mismatch: mapping file has %d classes, checkpoint metadata implies %d",
                len(self.class_names),
                len(checkpoint_classes),
            )

    def _load_torchscript(self) -> None:
        export_dir = self.config.project_root / self.config.get(
            "export.output_dir", f"exports/{self.config.crop_name}"
        )
        prefix = self.config.get("export.model_name_prefix", f"{self.config.crop_name}_disease")
        ts_path = export_dir / f"{prefix}.torchscript.pt"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"TorchScript model not found: {ts_path}. Run export first."
            )

        self._model = torch.jit.load(str(ts_path), map_location=self.device)
        self._model.eval()

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        export_dir = self.config.project_root / self.config.get(
            "export.output_dir", f"exports/{self.config.crop_name}"
        )
        prefix = self.config.get("export.model_name_prefix", f"{self.config.crop_name}_disease")
        onnx_path = export_dir / f"{prefix}.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}. Run export first."
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)

    def _preprocess(self, image_path: Path) -> torch.Tensor:
        image = read_image_rgb(image_path)
        transformed = self.transform(image=image)
        tensor = transformed["image"].unsqueeze(0)
        return tensor

    @torch.no_grad()
    def _infer_pytorch(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None
        tensor = tensor.to(self.device)
        outputs = self._model(tensor)
        probs = torch.softmax(outputs, dim=1)
        return outputs, probs

    @torch.no_grad()
    def _infer_torchscript(self, tensor: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        tensor = tensor.to(self.device)
        outputs = self._model(tensor)
        if isinstance(outputs, torch.Tensor):
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
        probs = torch.softmax(outputs[0], dim=1)
        return probs.cpu().numpy()

    def _infer_onnx(self, tensor: torch.Tensor) -> np.ndarray:
        assert self._onnx_session is not None
        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: tensor.numpy()})
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    def predict(self, image_path: Path | str) -> PredictionResult:
        """
        Run inference on a single image.

        Returns predicted class, confidence, top-k predictions, and timing.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        tensor = self._preprocess(path)

        if INFERENCE_DEBUG:
            logger.info(
                "[INFERENCE_DEBUG] preprocessing crop=%s image=%s resize=%dx%d rgb=True "
                "normalize_mean=%s normalize_std=%s tensor_shape=%s",
                self.config.crop_name,
                path,
                self.image_size,
                self.image_size,
                IMAGENET_MEAN,
                IMAGENET_STD,
                tuple(tensor.shape),
            )

        start = time.perf_counter()
        if self.backend == InferenceBackend.PYTORCH:
            logits_t, probs_t = self._infer_pytorch(tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000

            pred_idx = int(torch.argmax(probs_t, dim=1).item())
            topk_k = min(self.top_k, probs_t.shape[1])
            topk = torch.topk(probs_t, k=topk_k, dim=1)
            top_indices = topk.indices[0].cpu().numpy()
            prob_array = probs_t[0].cpu().numpy()
            logits = logits_t[0].cpu().numpy()

            self._log_predict_debug(
                logits=logits,
                probs=prob_array,
                pred_idx=pred_idx,
                top_indices=top_indices,
            )

            top_predictions = [
                TopPrediction(
                    class_id=int(idx),
                    class_name=self.id_to_name.get(int(idx), f"Class_{idx}"),
                    confidence=float(prob_array[idx]),
                )
                for idx in top_indices
            ]
            best = top_predictions[0]
            return PredictionResult(
                predicted_class=best.class_name,
                predicted_class_id=best.class_id,
                confidence=best.confidence,
                top_predictions=top_predictions,
                inference_time_ms=elapsed_ms,
                image_path=str(path),
            )

        if self.backend == InferenceBackend.TORCHSCRIPT:
            probs = self._infer_torchscript(tensor)
        else:
            probs = self._infer_onnx(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000

        prob_array = probs[0]
        top_indices = np.argsort(prob_array)[::-1][: self.top_k]

        top_predictions = [
            TopPrediction(
                class_id=int(idx),
                class_name=self.id_to_name.get(int(idx), f"Class_{idx}"),
                confidence=float(prob_array[idx]),
            )
            for idx in top_indices
        ]

        best = top_predictions[0]
        return PredictionResult(
            predicted_class=best.class_name,
            predicted_class_id=best.class_id,
            confidence=best.confidence,
            top_predictions=top_predictions,
            inference_time_ms=elapsed_ms,
            image_path=str(path),
        )

    def predict_batch(self, image_paths: list[Path | str]) -> list[PredictionResult]:
        """Run inference on multiple images sequentially."""
        return [self.predict(p) for p in image_paths]
