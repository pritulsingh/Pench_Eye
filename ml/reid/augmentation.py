"""
Training-time augmentation for tiger flank crops.

Stripe geometry *is* the identity signal, so this module is deliberately
conservative. Photometric and sensor-level noise is safe; anything that
rearranges stripe layout is not.

Deliberately excluded:

* **Vertical flip** — never occurs in camera-trap imagery.
* **Large rotation / shear / perspective / elastic warp** — distorts the stripe
  spacing the model must key on.
* **Grayscale, channel shuffle, heavy colour jitter** — a tiger's coat colour is
  a useful prior; destroying it costs accuracy for no robustness gain.

`horizontal_flip` defaults to **off**. A tiger's left and right flanks carry
different stripe patterns, so mirroring a left flank fabricates a right flank
that individual does not have. Enable it only if you train flank-agnostic and
accept the label noise; if your data has reliable flank labels, prefer training
per-flank galleries instead.

Implemented with NumPy + PIL so no extra dependency (albumentations etc.) is
introduced.
"""
from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    enabled: bool = True
    # Mirroring fabricates the opposite flank — see module docstring.
    horizontal_flip: bool = False
    horizontal_flip_prob: float = 0.5
    random_resized_crop: bool = True
    crop_scale: Tuple[float, float] = (0.80, 1.0)
    crop_prob: float = 0.5
    brightness: float = 0.20
    contrast: float = 0.20
    photometric_prob: float = 0.6
    blur_prob: float = 0.15
    blur_max_radius: float = 1.2
    jpeg_prob: float = 0.20
    jpeg_quality: Tuple[int, int] = (45, 90)
    rotation_prob: float = 0.25
    rotation_degrees: float = 7.0
    cutout_prob: float = 0.25
    cutout_scale: Tuple[float, float] = (0.02, 0.12)
    cutout_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["crop_scale"] = list(self.crop_scale)
        data["jpeg_quality"] = list(self.jpeg_quality)
        data["cutout_scale"] = list(self.cutout_scale)
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AugmentationConfig":
        if not data:
            return cls()
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        for key in ("crop_scale", "jpeg_quality", "cutout_scale"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)  # type: ignore[arg-type]

    @classmethod
    def disabled(cls) -> "AugmentationConfig":
        return cls(enabled=False)


class Augmenter:
    """Applies `AugmentationConfig` to an RGB uint8 HWC array."""

    def __init__(self, config: Optional[AugmentationConfig] = None, seed: Optional[int] = None):
        self.config = config or AugmentationConfig()
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        cfg = self.config
        if not cfg.enabled or image_rgb.size == 0:
            return image_rgb

        out = image_rgb
        if cfg.random_resized_crop and self._chance(cfg.crop_prob):
            out = self._random_resized_crop(out, cfg.crop_scale)
        if cfg.rotation_degrees > 0 and self._chance(cfg.rotation_prob):
            out = self._rotate(out, cfg.rotation_degrees)
        if cfg.horizontal_flip and self._chance(cfg.horizontal_flip_prob):
            out = out[:, ::-1]
        if self._chance(cfg.photometric_prob):
            out = self._photometric(out, cfg.brightness, cfg.contrast)
        if cfg.blur_prob > 0 and self._chance(cfg.blur_prob):
            out = self._blur(out, cfg.blur_max_radius)
        if cfg.jpeg_prob > 0 and self._chance(cfg.jpeg_prob):
            out = self._jpeg(out, cfg.jpeg_quality)
        if cfg.cutout_prob > 0 and self._chance(cfg.cutout_prob):
            out = self._cutout(out, cfg.cutout_scale, cfg.cutout_count)
        return np.ascontiguousarray(out)

    def _chance(self, probability: float) -> bool:
        return probability > 0 and self._rng.random() < probability

    def _random_resized_crop(self, image: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
        height, width = image.shape[:2]
        area_scale = self._rng.uniform(*scale)
        # Keep aspect ratio close to the original so stripe spacing is preserved.
        new_h = max(8, int(round(height * (area_scale ** 0.5))))
        new_w = max(8, int(round(width * (area_scale ** 0.5))))
        top = self._rng.randint(0, max(0, height - new_h))
        left = self._rng.randint(0, max(0, width - new_w))
        return image[top : top + new_h, left : left + new_w]

    def _rotate(self, image: np.ndarray, max_degrees: float) -> np.ndarray:
        from PIL import Image

        angle = self._rng.uniform(-max_degrees, max_degrees)
        rotated = Image.fromarray(image).rotate(angle, resample=Image.BILINEAR, expand=False)
        return np.array(rotated)

    def _photometric(self, image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        out = image.astype(np.float32)
        if brightness > 0:
            out *= 1.0 + self._rng.uniform(-brightness, brightness)
        if contrast > 0:
            factor = 1.0 + self._rng.uniform(-contrast, contrast)
            mean = out.mean()
            out = (out - mean) * factor + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    def _blur(self, image: np.ndarray, max_radius: float) -> np.ndarray:
        from PIL import Image, ImageFilter

        radius = self._rng.uniform(0.3, max(0.3, max_radius))
        blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred)

    def _jpeg(self, image: np.ndarray, quality_range: Tuple[int, int]) -> np.ndarray:
        from PIL import Image

        quality = self._rng.randint(int(quality_range[0]), int(quality_range[1]))
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            return np.array(reopened.convert("RGB"))

    def _cutout(self, image: np.ndarray, scale: Tuple[float, float], count: int) -> np.ndarray:
        out = image.copy()
        height, width = out.shape[:2]
        for _ in range(max(1, count)):
            area = self._rng.uniform(*scale) * height * width
            box_h = max(1, min(height - 1, int(round(area ** 0.5))))
            box_w = max(1, min(width - 1, int(round(area ** 0.5))))
            top = self._rng.randint(0, height - box_h)
            left = self._rng.randint(0, width - box_w)
            # Fill with per-image mean rather than black: closer to occlusion by
            # vegetation than to a dead sensor region.
            out[top : top + box_h, left : left + box_w] = out.mean(axis=(0, 1)).astype(np.uint8)
        return out
