from dataclasses import dataclass
from typing import List
import numpy as np
from pathlib import Path

from .quality import BlankImageQualityChecker, BlankCheckResult
from .features import ImageFeatures

@dataclass
class BlankClassificationResult:
    is_blank: bool
    blank_probability: float
    subject_probability: float
    stage_used: str
    features: ImageFeatures
    reasons: List[str]
    quality_score: float

class BlankImageClassifier:
    """
    Two-stage blank image classifier.
    
    Stage A: Fast CV metrics (always runs)
    Stage B: Learned classifier (optional, used for ambiguous cases)
    
    The model is replaceable. The interface is stable.
    """
    
    def __init__(self, ml_mode: str = "demo", blank_threshold: float = 0.95):
        self.ml_mode = ml_mode
        self.blank_threshold = blank_threshold
        self.quality_checker = BlankImageQualityChecker()
        self._model = None  # loaded lazily in production mode
        
    def classify(self, image: np.ndarray) -> BlankClassificationResult:
        """
        Classify an image as blank or subject.
        Returns BlankClassificationResult.
        """
        if self.ml_mode == "demo":
            return self._classify_demo(image)
            
        stage_a_res = self.quality_checker.check(image, self.blank_threshold)
        if stage_a_res.is_definitely_blank:
            return self._classify_stage_a(image, stage_a_res)
            
        return self._classify_stage_b(image, stage_a_res)

    def _classify_demo(self, image: np.ndarray) -> BlankClassificationResult:
        """Deterministic demo mode: uses CV features only."""
        stage_a_res = self.quality_checker.check(image, self.blank_threshold)
        return self._classify_stage_a(image, stage_a_res)

    def _classify_stage_a(self, image: np.ndarray, stage_a_res: BlankCheckResult = None) -> BlankClassificationResult:
        """Fast CV Stage A."""
        if stage_a_res is None:
            stage_a_res = self.quality_checker.check(image, self.blank_threshold)
            
        return BlankClassificationResult(
            is_blank=stage_a_res.is_definitely_blank,
            blank_probability=stage_a_res.blank_probability,
            subject_probability=1.0 - stage_a_res.blank_probability,
            stage_used=stage_a_res.stage,
            features=stage_a_res.features,
            reasons=stage_a_res.reasons,
            quality_score=stage_a_res.features.quality_score
        )

    def _classify_stage_b(self, image: np.ndarray, stage_a: BlankCheckResult) -> BlankClassificationResult:
        """
        Stage B: lightweight CNN classifier.
        In demo/production-without-model: falls back to Stage A result.
        In production with model: runs MobileNetV2-based classifier.
        """
        if self._model is None:
            try:
                import torch
                model_path = Path("ml/weights/blank_classifier.pt")
                if model_path.exists():
                    self._model = torch.jit.load(str(model_path))
                    self._model.eval()
            except Exception as e:
                print(f"[BlankImageClassifier] Could not load model: {e}. Falling back to Stage A.")
                self._model = "failed"
                
        if self._model is None or self._model == "failed":
            return self._classify_stage_a(image, stage_a)
            
        try:
            import torch
            import cv2
            # Simple preprocess for mobilenet
            resized = cv2.resize(image, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)
            
            with torch.no_grad():
                out = self._model(tensor)
                prob = torch.sigmoid(out).item()
                
            is_blank = prob >= self.blank_threshold
            reasons = stage_a.reasons + [f"CNN blank probability: {prob:.2f}"]
            
            return BlankClassificationResult(
                is_blank=is_blank,
                blank_probability=prob,
                subject_probability=1.0 - prob,
                stage_used="model",
                features=stage_a.features,
                reasons=reasons,
                quality_score=stage_a.features.quality_score
            )
        except Exception as e:
            print(f"[BlankImageClassifier] Model inference failed: {e}. Falling back to Stage A.")
            return self._classify_stage_a(image, stage_a)
