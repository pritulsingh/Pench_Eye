"""
FlankExtractor — Tiger Intelligence System

Extracts left or right flank from a tiger detection crop.

Current strategy: configurable crop (aspect-ratio based heuristic).
Future: Replace with segmentation model.

Interface is stable. Implementation is replaceable.

OpenCV is optional; without it, edge density falls back to a NumPy gradient.
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass

try:  # pragma: no cover - environment dependent
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

class FlankSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"

@dataclass
class FlankResult:
    flank_image: np.ndarray
    flank_side: FlankSide
    confidence: float
    method: str   # "heuristic" or "segmentation"

class FlankExtractor:
    """
    Extracts normalized flank region from tiger crop.
    
    Heuristic strategy:
    1. Estimate orientation from bbox aspect ratio
    2. If width > height (landscape, tiger walking sideways): take center 70% width
    3. Classify left/right by analyzing which side has more edge detail
    4. Crop and return the dominant flank side
    """
    
    def extract(self, tiger_crop: np.ndarray) -> FlankResult:
        """Extract flank from tiger crop."""
        if tiger_crop is None or tiger_crop.size == 0:
            return FlankResult(
                flank_image=np.zeros((10, 10, 3), dtype=np.uint8),
                flank_side=FlankSide.UNKNOWN,
                confidence=0.0,
                method="heuristic"
            )
            
        flank_side = self._estimate_flank_side(tiger_crop)
        normalized_crop = self._normalize_crop(tiger_crop)
        
        return FlankResult(
            flank_image=normalized_crop,
            flank_side=flank_side,
            confidence=0.75, # Heuristic confidence
            method="heuristic"
        )
    
    def _estimate_flank_side(self, image: np.ndarray) -> FlankSide:
        """Determine which flank is visible using edge density analysis."""
        h, w = image.shape[:2]
        if w < 10 or h < 10:
            return FlankSide.UNKNOWN

        edges = self._edge_map(image)
        mid = w // 2
        left_edges = float(np.count_nonzero(edges[:, :mid]))
        right_edges = float(np.count_nonzero(edges[:, mid:]))

        if left_edges > right_edges * 1.1:
            return FlankSide.LEFT
        elif right_edges > left_edges * 1.1:
            return FlankSide.RIGHT

        return FlankSide.UNKNOWN

    @staticmethod
    def _edge_map(image: np.ndarray) -> np.ndarray:
        """Binary edge map; Canny when OpenCV is present, gradient magnitude otherwise."""
        if image.ndim == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = (
                    0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]
                ).astype(np.uint8)
        else:
            gray = image

        if CV2_AVAILABLE:
            return cv2.Canny(gray, 100, 200)

        gray_f = gray.astype(np.float32)
        dy, dx = np.gradient(gray_f)
        magnitude = np.hypot(dx, dy)
        return (magnitude > max(20.0, float(magnitude.mean()) * 2.0)).astype(np.uint8)
    
    def _normalize_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop to central flank region, removing head/tail."""
        h, w = image.shape[:2]
        if w < 10 or h < 10:
            return image
            
        # Center 80% horizontally, 70-90% vertically (center 80%)
        x1 = int(w * 0.1)
        x2 = int(w * 0.9)
        y1 = int(h * 0.1)
        y2 = int(h * 0.9)
        
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2].copy()
        return image
