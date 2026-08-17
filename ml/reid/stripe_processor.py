"""
StripeProcessor — normalises a flank crop into a Re-ID model input.

The actual transform lives in `ml/reid/preprocessing.py` so that training and
inference cannot drift apart; this class is the OpenCV-oriented wrapper that the
rest of the ML package already uses. `preprocessing_version` is carried on the
result so downstream code can tell whether a cached tensor is still valid.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ml.reid.preprocessing import (
    PREPROCESSING_VERSION,
    PreprocessConfig,
    apply_clahe_rgb,
    preprocess_rgb,
    resize_rgb,
    sharpness_score,
)


@dataclass
class ProcessedStripe:
    tensor: np.ndarray            # CHW float32, normalized
    original_resized: np.ndarray  # HWC uint8 (BGR) for visualization
    quality_score: float
    contrast_enhanced: bool
    target_size: Tuple[int, int]
    preprocessing_version: str = PREPROCESSING_VERSION


class StripeProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        use_clahe: bool = True,
        use_background_suppression: bool = False,
        config: Optional[PreprocessConfig] = None,
    ):
        self.config = config or PreprocessConfig(
            image_size=(int(target_size[0]), int(target_size[1])), use_clahe=use_clahe
        )
        self.target_size = self.config.image_size
        self.use_clahe = self.config.use_clahe
        self.use_background_suppression = use_background_suppression

    def process(self, flank_image: np.ndarray) -> ProcessedStripe:
        """BGR flank crop → ProcessedStripe with a model-ready tensor."""
        if flank_image is None or getattr(flank_image, "size", 0) == 0:
            height, width = self.target_size
            return ProcessedStripe(
                tensor=np.zeros((3, height, width), dtype=np.float32),
                original_resized=np.zeros((height, width, 3), dtype=np.uint8),
                quality_score=0.0,
                contrast_enhanced=False,
                target_size=self.target_size,
                preprocessing_version=self.config.version,
            )

        rgb = flank_image[:, :, ::-1] if flank_image.ndim == 3 else np.stack([flank_image] * 3, axis=-1)
        resized = resize_rgb(np.ascontiguousarray(rgb), self.config.image_size)
        enhanced = (
            apply_clahe_rgb(resized, self.config.clahe_clip_limit, self.config.clahe_grid_size)
            if self.config.use_clahe
            else resized
        )

        return ProcessedStripe(
            tensor=preprocess_rgb(rgb, self.config),
            original_resized=np.ascontiguousarray(enhanced[:, :, ::-1]),
            quality_score=sharpness_score(enhanced),
            contrast_enhanced=self.config.use_clahe,
            target_size=self.target_size,
            preprocessing_version=self.config.version,
        )
