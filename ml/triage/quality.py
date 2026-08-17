from dataclasses import dataclass
from typing import List
import numpy as np

from .features import ImageFeatures, ImageFeatureExtractor

@dataclass
class BlankCheckResult:
    is_definitely_blank: bool      # True only if very confident
    blank_probability: float       # 0-1
    features: ImageFeatures
    reasons: List[str]             # human-readable explanation
    stage: str = "cv"              # "cv" or "model"

class BlankImageQualityChecker:
    """Stage A - fast CV quality and blank image checker."""
    
    def __init__(self):
        self.extractor = ImageFeatureExtractor()
        
    def check(self, image: np.ndarray, threshold: float = 0.95) -> BlankCheckResult:
        """Check image quality and determine if it's blank."""
        if image is None or image.size == 0:
            return BlankCheckResult(
                is_definitely_blank=True,
                blank_probability=1.0,
                features=self.extractor.extract(np.zeros((1, 1), dtype=np.uint8)),
                reasons=["Empty or corrupted image array."],
                stage="cv"
            )

        features = self.extractor.extract(image)
        reasons = []
        is_definitely_blank = False
        blank_prob = 0.0
        
        # Hard rules
        if features.dark_pixel_ratio > 0.95:
            is_definitely_blank = True
            blank_prob = 0.99
            reasons.append("Image is almost completely black.")
        elif features.bright_pixel_ratio > 0.95:
            is_definitely_blank = True
            blank_prob = 0.98
            reasons.append("Image is overexposed (almost completely white).")
        elif features.blur_score < 0.02:
            is_definitely_blank = (threshold <= 0.92)
            blank_prob = 0.92
            reasons.append("Image is extremely blurry or solid color.")
            
        # If hard rules didn't trigger a definitive blank, compute probability
        if not is_definitely_blank:
            blank_prob = (
                0.3 * (1.0 - features.entropy) +
                0.3 * (1.0 - features.edge_density) +
                0.2 * (features.dark_pixel_ratio + features.bright_pixel_ratio) / 2.0 +
                0.2 * (1.0 - features.contrast)
            )
            blank_prob = float(np.clip(blank_prob, 0.0, 1.0))
            is_definitely_blank = blank_prob >= threshold
            
            if is_definitely_blank:
                reasons.append(f"Image lacks detail or contrast (score: {blank_prob:.2f}).")
            else:
                reasons.append(f"Image appears to have content (blank score: {blank_prob:.2f}).")
                
        return BlankCheckResult(
            is_definitely_blank=is_definitely_blank,
            blank_probability=blank_prob,
            features=features,
            reasons=reasons,
            stage="cv"
        )
