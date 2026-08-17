"""
Canonical Re-ID preprocessing — the single source of truth for turning a flank
crop into a model input tensor.

Training and inference **must** call the same function, otherwise embeddings
computed at inference time are not comparable with those the model was trained
on. `ml/reid/stripe_processor.py` delegates here so the classical-CV path and
the neural path cannot drift apart.

`PREPROCESSING_VERSION` is written into every checkpoint and every extracted
embedding. Bump it whenever the transform changes so stale embeddings can be
detected instead of silently mixed with new ones.

OpenCV is optional: CLAHE needs it, everything else falls back to PIL/NumPy.
The realised setting is recorded in the config so a checkpoint trained with
CLAHE is never served without it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

PREPROCESSING_VERSION = "reid-preproc-v1"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:  # pragma: no cover - environment dependent
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False


@dataclass
class PreprocessConfig:
    """Deterministic transform applied identically at train and inference time."""

    image_size: Tuple[int, int] = (224, 224)  # (height, width)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD
    version: str = PREPROCESSING_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        data["clahe_grid_size"] = list(self.clahe_grid_size)
        data["mean"] = list(self.mean)
        data["std"] = list(self.std)
        data["clahe_effective"] = self.use_clahe and CV2_AVAILABLE
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PreprocessConfig":
        if not data:
            return cls()
        return cls(
            image_size=tuple(data.get("image_size", (224, 224))),  # type: ignore[arg-type]
            use_clahe=bool(data.get("use_clahe", True)),
            clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
            clahe_grid_size=tuple(data.get("clahe_grid_size", (8, 8))),  # type: ignore[arg-type]
            mean=tuple(data.get("mean", IMAGENET_MEAN)),  # type: ignore[arg-type]
            std=tuple(data.get("std", IMAGENET_STD)),  # type: ignore[arg-type]
            version=str(data.get("version", PREPROCESSING_VERSION)),
        )


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as HWC uint8 RGB."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def apply_clahe_rgb(image_rgb: np.ndarray, clip_limit: float, grid: Sequence[int]) -> np.ndarray:
    """CLAHE on the L channel in LAB space. No-op when OpenCV is unavailable."""
    if not CV2_AVAILABLE or image_rgb.ndim != 3:
        return image_rgb
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(grid[0]), int(grid[1])))
    merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def resize_rgb(image_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize to (height, width)."""
    height, width = int(size[0]), int(size[1])
    if image_rgb.shape[0] == height and image_rgb.shape[1] == width:
        return image_rgb
    if CV2_AVAILABLE:
        return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    from PIL import Image

    return np.array(Image.fromarray(image_rgb).resize((width, height), Image.BILINEAR))


def preprocess_rgb(image_rgb: np.ndarray, config: Optional[PreprocessConfig] = None) -> np.ndarray:
    """
    RGB uint8 HWC → normalised CHW float32 model input.

    Order: resize → optional CLAHE → scale to [0,1] → ImageNet normalise → CHW.
    """
    config = config or PreprocessConfig()
    resized = resize_rgb(np.ascontiguousarray(image_rgb), config.image_size)
    if config.use_clahe:
        resized = apply_clahe_rgb(resized, config.clahe_clip_limit, config.clahe_grid_size)

    scaled = resized.astype(np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    normalized = (scaled - mean) / std
    return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))


def preprocess_bgr(image_bgr: np.ndarray, config: Optional[PreprocessConfig] = None) -> np.ndarray:
    """OpenCV-ordered (BGR) entry point, used by `StripeProcessor`."""
    if image_bgr.ndim == 2:
        rgb = np.stack([image_bgr] * 3, axis=-1)
    else:
        rgb = image_bgr[:, :, ::-1]
    return preprocess_rgb(rgb, config)


def preprocess_path(path: str, config: Optional[PreprocessConfig] = None) -> np.ndarray:
    return preprocess_rgb(load_image_rgb(path), config)


def sharpness_score(image_rgb: np.ndarray) -> float:
    """
    Laplacian-variance sharpness in [0, 1]; a blur proxy for quality gating.

    Mirrors the scale used by `StripeProcessor._quality_score` so quality values
    remain comparable across the classical and neural paths.
    """
    if image_rgb.ndim == 3:
        gray = (
            0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
        ).astype(np.float32)
    else:
        gray = image_rgb.astype(np.float32)

    if CV2_AVAILABLE:
        variance = float(cv2.Laplacian(gray.astype(np.float64, copy=False), cv2.CV_64F).var())
    else:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        padded = np.pad(gray, 1, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        variance = float((windows * kernel).sum(axis=(-1, -2)).var())
    return float(np.clip(variance / 500.0, 0.0, 1.0))
