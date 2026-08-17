"""
TigerDetector — Pench Eye

Interface:
    detect(image: np.ndarray) -> List[TigerDetection]

Demo mode: deterministic detections based on image hash (tests / explicit demo only).
Production mode: YOLO inference with fail-closed behaviour when weights are missing.

Production MUST NEVER invent a tiger detection. Missing weights or inference
errors return an empty list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np

# Only these class labels count as tiger. Never map person/animal/unknown → tiger.
TIGER_CLASS_NAMES = frozenset({"tiger", "panthera_tigris", "panthera tigris"})


def _project_weights_dir() -> Path:
    # ml/detection/tiger_detector.py → ml/weights
    return Path(__file__).resolve().parents[1] / "weights"


def default_tiger_yolo_path() -> Path:
    return _project_weights_dir() / "tiger_yolo.pt"


@dataclass
class TigerDetection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    confidence: float  # 0-1
    class_name: str
    class_id: int
    cropped_image: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def bbox_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@dataclass
class ModelDetection:
    """Unfiltered YOLO output used only for diagnostics and tiger filtering."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int


def is_tiger_class(name: Optional[str]) -> bool:
    if not name:
        return False
    return name.strip().lower().replace("-", "_") in TIGER_CLASS_NAMES or name.strip().lower() == "tiger"


class TigerDetector:
    def __init__(
        self,
        ml_mode: str = "demo",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ):
        self.ml_mode = ml_mode
        self.model_path = Path(model_path) if model_path else default_tiger_yolo_path()
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._load_error: Optional[str] = None
        self._class_names: Dict[int, str] = {}

    def detect(self, image: np.ndarray) -> List[TigerDetection]:
        """Detect tigers in image. Returns only tiger-class detections."""
        if image is None or getattr(image, "size", 0) == 0:
            return []

        if self.ml_mode == "demo":
            return self._detect_demo(image)
        return [
            TigerDetection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_name=detection.class_name,
                class_id=detection.class_id,
                cropped_image=image[
                    detection.bbox[1] : detection.bbox[3],
                    detection.bbox[0] : detection.bbox[2],
                ].copy(),
            )
            for detection in self.detect_all(image)
            if is_tiger_class(detection.class_name)
        ]

    def detect_all(self, image: np.ndarray) -> List[ModelDetection]:
        """Run production YOLO and return every labelled class for diagnostics."""
        if self.ml_mode != "production" or image is None or getattr(image, "size", 0) == 0:
            return []
        return self._detect_production(image)

    def status(self) -> Dict[str, Any]:
        """Diagnostic snapshot of the detector configuration."""
        weights_exist = self.model_path.exists()
        if self.ml_mode == "production" and self._model is None and weights_exist:
            self._load_model()
        return {
            "ml_mode": self.ml_mode,
            "model_path": str(self.model_path),
            "model_type": "Ultralytics YOLO",
            "weights_present": weights_exist,
            "model_loaded": self._model is not None,
            "load_error": self._load_error,
            "class_names": dict(self._class_names),
            "available_classes": dict(self._class_names),
            "tiger_class_ids": [
                cid for cid, name in self._class_names.items() if is_tiger_class(name)
            ],
            "tiger_class_names": sorted(
                name for name in self._class_names.values() if is_tiger_class(name)
            ),
            "tiger_class_id": next(
                (cid for cid, name in self._class_names.items() if is_tiger_class(name)), None
            ),
            "tiger_class_name": next(
                (name for name in self._class_names.values() if is_tiger_class(name)), None
            ),
            "confidence_threshold": self.confidence_threshold,
            "fail_closed": self.ml_mode == "production",
        }

    def _detect_demo(self, image: np.ndarray) -> List[TigerDetection]:
        """
        Demo-only path. Must not be used for production uploads.
        Deterministic hash scenarios — tests / /demo/simulate only.
        """
        import hashlib

        h, w = image.shape[:2]
        img_hash = int(hashlib.md5(image[:100, :100].tobytes()).hexdigest(), 16)
        random.seed(img_hash)

        scenario = img_hash % 10
        if scenario < 3:
            return []

        detections: List[TigerDetection] = []
        n_tigers = 2 if scenario >= 9 else 1

        for i in range(n_tigers):
            size_factor = 0.2 + (img_hash >> (i * 4) & 0xFF) / 255.0 * 0.4
            cx = int(w * (0.3 + (img_hash >> (i * 8 + 8) & 0xFF) / 255.0 * 0.4))
            cy = int(h * (0.3 + (img_hash >> (i * 8 + 16) & 0xFF) / 255.0 * 0.4))
            bw = int(w * size_factor)
            bh = int(h * size_factor * 0.6)
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)
            confidence = 0.75 + (img_hash >> 24 & 0xFF) / 255.0 * 0.24

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2].copy()
            detections.append(
                TigerDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=round(confidence, 3),
                    class_name="tiger",
                    class_id=0,
                    cropped_image=crop,
                )
            )
        return detections

    def _detect_production(self, image: np.ndarray) -> List[ModelDetection]:
        """Production YOLO inference. Empty list on missing weights or errors."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            # Fail closed — never invent a tiger.
            return []

        try:
            results = self._model(image, verbose=False, conf=self.confidence_threshold)
            detections: List[ModelDetection] = []
            for r in results:
                names = getattr(r, "names", None) or self._class_names
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < self.confidence_threshold:
                        continue
                    cls_id = int(box.cls[0])
                    if isinstance(names, dict):
                        cls_name = names.get(cls_id)
                    else:
                        cls_name = names[cls_id] if names and cls_id < len(names) else None

                    if cls_name is None:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append(
                        ModelDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_name=str(cls_name),
                            class_id=cls_id,
                        )
                    )
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections
        except Exception as exc:
            self._load_error = f"Inference error: {exc}"
            return []

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            if not self.model_path.exists():
                self._load_error = (
                    f"Tiger YOLO weights not found at {self.model_path}. "
                    "Production detection is fail-closed until tiger_yolo.pt is installed."
                )
                self._model = None
                return

            self._model = YOLO(str(self.model_path))
            names = getattr(self._model, "names", None) or {}
            if isinstance(names, dict):
                self._class_names = {int(k): str(v) for k, v in names.items()}
            else:
                self._class_names = {i: str(n) for i, n in enumerate(names)}

            tiger_ids = [cid for cid, name in self._class_names.items() if is_tiger_class(name)]
            if not tiger_ids:
                self._load_error = (
                    f"Loaded YOLO weights at {self.model_path} but no tiger class was found "
                    f"in names={self._class_names}. Refusing to treat other classes as tiger."
                )
                self._model = None
                return

            self._load_error = None
        except Exception as exc:
            self._load_error = f"Could not load tiger YOLO: {exc}"
            self._model = None
