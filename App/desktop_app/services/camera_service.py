"""Cross-platform camera backends for desktop capture (OpenCV + Pi Camera)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from utils.config import AppConfig
from utils.platform import is_linux, is_raspberry_pi

logger = logging.getLogger(__name__)


class CameraBackend(ABC):
    """Minimal camera interface used by the capture dialog."""

    name: str = "base"

    @abstractmethod
    def open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_rgb(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError


class OpenCVCameraBackend(CameraBackend):
    name = "opencv"

    def __init__(
        self,
        *,
        index: int = 0,
        device_path: str | None = None,
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 5,
    ) -> None:
        self.index = index
        self.device_path = device_path
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _open_capture(self, source: int | str) -> cv2.VideoCapture | None:
        if is_linux():
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def open(self) -> bool:
        self.close()
        candidates: list[int | str] = []

        if self.device_path:
            candidates.append(self.device_path)
        candidates.append(self.index)
        if is_linux():
            candidates.extend([f"/dev/video{self.index}", "/dev/video0", "/dev/video1"])

        seen: set[str | int] = set()
        for source in candidates:
            if source in seen:
                continue
            seen.add(source)

            cap = self._open_capture(source)
            if cap is None:
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            for _ in range(max(self.warmup_frames, 1)):
                cap.read()

            if cap.isOpened():
                self._cap = cap
                logger.info("OpenCV camera opened | source=%s", source)
                return True
            cap.release()

        return False

    def read_rgb(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class Picamera2Backend(CameraBackend):
    name = "picamera2"

    def __init__(self, *, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._camera = None
        self._stream_name = "main"

    @property
    def is_open(self) -> bool:
        return self._camera is not None

    @staticmethod
    def _to_rgb_u8(frame: np.ndarray) -> np.ndarray | None:
        """Normalize Picamera2 buffers to contiguous HxWx3 RGB uint8."""
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim != 3:
            return None
        if arr.shape[2] == 4:
            # Common on Pi: XRGB/BGRX — drop alpha and swap if needed later.
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def open(self) -> bool:
        self.close()
        try:
            from picamera2 import Picamera2
        except ImportError:
            logger.info("picamera2 not installed; skipping Pi Camera backend")
            return False

        try:
            camera = Picamera2()
            # Keep buffer count low to reduce RAM pressure on Pi 4/5.
            config = camera.create_preview_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                buffer_count=2,
            )
            camera.configure(config)
            camera.start()
            self._camera = camera
            self._stream_name = "main"
            logger.info(
                "Pi Camera opened via picamera2 | size=%sx%s",
                self.width,
                self.height,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to open Pi Camera via picamera2: %s", exc)
            self.close()
            return False

    def read_rgb(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        try:
            frame = self._camera.capture_array(self._stream_name)
            return self._to_rgb_u8(frame)
        except Exception as exc:
            logger.warning("Pi Camera read failed: %s", exc)
            return None

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
            # Give libcamera time to release the sensor before the next open.
            if is_raspberry_pi():
                import time

                time.sleep(0.4)
                import gc

                gc.collect()


def _camera_settings(config: AppConfig) -> dict:
    return {
        "backend": str(config.get("desktop_app.camera.backend", "auto")).lower(),
        "index": int(config.get("desktop_app.camera.index", 0)),
        "device_path": config.get("desktop_app.camera.device_path"),
        "width": int(config.get("desktop_app.camera.width", 640)),
        "height": int(config.get("desktop_app.camera.height", 480)),
        "warmup_frames": int(config.get("desktop_app.camera.warmup_frames", 5)),
    }


def create_camera_backend(config: AppConfig) -> CameraBackend | None:
    """Try configured camera backends in a Pi-friendly order."""
    settings = _camera_settings(config)
    backend_pref = settings["backend"]

    backend_order: list[str] = []
    if backend_pref == "auto":
        if is_raspberry_pi():
            backend_order.extend(["picamera2", "opencv"])
        else:
            backend_order.append("opencv")
    elif backend_pref in {"picamera2", "opencv"}:
        backend_order.append(backend_pref)
    else:
        backend_order.append("opencv")

    for backend_name in backend_order:
        backend: CameraBackend
        if backend_name == "picamera2":
            backend = Picamera2Backend(
                width=settings["width"],
                height=settings["height"],
            )
        else:
            backend = OpenCVCameraBackend(
                index=settings["index"],
                device_path=settings["device_path"],
                width=settings["width"],
                height=settings["height"],
                warmup_frames=settings["warmup_frames"],
            )

        if backend.open():
            return backend
        backend.close()

    logger.error("No camera backend could be opened | tried=%s", backend_order)
    return None
