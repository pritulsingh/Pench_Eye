"""
Quality gating for Re-ID embeddings.

A neural similarity score is not a biological identification. Two crops can be
highly similar because they show the same tiger, or because both are dark,
blurred and mostly foliage. This module produces the evidence needed to hold
back a confident-looking match that rests on weak input.

Checks applied:

* **Sharpness** — Laplacian variance; motion blur destroys stripe detail.
* **Resolution** — an upscaled 40 px crop cannot carry stripe geometry.
* **Contrast / exposure** — flat or blown-out crops give unstable embeddings.
* **Flank mismatch** — left and right flanks are *different* patterns. Comparing
  across flanks is not evidence of the same animal.
* **Gallery sufficiency** — one enrolled image per tiger makes any match fragile.
* **Domain shift** — embeddings far from the training distribution (proxied by
  low top-1 similarity across the whole gallery) mean the model is extrapolating.

Nothing here overrides the decision engine; it annotates. Callers decide whether
to downgrade auto-match to human review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Minimum flank crop edge in pixels below which stripe detail is unreliable.
MIN_CROP_EDGE_PX = 96
# Laplacian-variance sharpness (0-1 scale from ml.reid.preprocessing).
MIN_SHARPNESS = 0.15
# Standard deviation of grayscale intensity; below this the crop is near-flat.
MIN_CONTRAST_STD = 12.0
# Gallery images per identity below which a match is considered weakly supported.
MIN_GALLERY_IMAGES = 3


@dataclass
class QualityAssessment:
    """Per-image quality verdict attached to an embedding."""

    usable: bool = True
    quality_score: float = 0.0
    sharpness: float = 0.0
    contrast_std: float = 0.0
    min_edge_px: int = 0
    flank: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_crop(
    image_rgb: np.ndarray,
    *,
    flank: str = "unknown",
    min_edge_px: int = MIN_CROP_EDGE_PX,
    min_sharpness: float = MIN_SHARPNESS,
    min_contrast_std: float = MIN_CONTRAST_STD,
) -> QualityAssessment:
    """Assess a flank crop *before* embedding it."""
    from ml.reid.preprocessing import sharpness_score

    assessment = QualityAssessment(flank=flank or "unknown")
    if image_rgb is None or image_rgb.size == 0:
        assessment.usable = False
        assessment.blocking_reasons.append("empty_crop")
        return assessment

    height, width = image_rgb.shape[:2]
    assessment.min_edge_px = int(min(height, width))
    assessment.sharpness = sharpness_score(image_rgb)

    gray = (
        0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
        if image_rgb.ndim == 3
        else image_rgb.astype(np.float32)
    )
    assessment.contrast_std = float(np.std(gray))
    mean_intensity = float(np.mean(gray))

    if assessment.min_edge_px < min_edge_px:
        assessment.blocking_reasons.append(
            f"crop_too_small({assessment.min_edge_px}px<{min_edge_px}px)"
        )
    if assessment.sharpness < min_sharpness:
        assessment.warnings.append(f"low_sharpness({assessment.sharpness:.3f})")
    if assessment.contrast_std < min_contrast_std:
        assessment.warnings.append(f"low_contrast(std={assessment.contrast_std:.1f})")
    if mean_intensity < 25:
        assessment.warnings.append("underexposed")
    elif mean_intensity > 230:
        assessment.warnings.append("overexposed")
    if (flank or "unknown") == "unknown":
        assessment.warnings.append("flank_side_unknown")

    assessment.usable = not assessment.blocking_reasons
    # Composite score: sharpness and contrast, penalised by each warning.
    contrast_component = float(np.clip(assessment.contrast_std / 60.0, 0.0, 1.0))
    base = 0.6 * assessment.sharpness + 0.4 * contrast_component
    assessment.quality_score = float(np.clip(base * (0.9 ** len(assessment.warnings)), 0.0, 1.0))
    return assessment


@dataclass
class MatchReliability:
    """Caveats on a similarity result, independent of the score itself."""

    reliable: bool = True
    warnings: List[str] = field(default_factory=list)
    recommend_human_review: bool = False
    gallery_size: int = 0
    top_similarity: float = 0.0
    score_gap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_match(
    *,
    top_similarity: float,
    runner_up_similarity: Optional[float] = None,
    gallery_size: int = 0,
    query_flank: str = "unknown",
    match_flank: str = "unknown",
    query_quality: Optional[QualityAssessment] = None,
    min_gallery_images: int = MIN_GALLERY_IMAGES,
    domain_shift_threshold: float = 0.25,
) -> MatchReliability:
    """
    Flag structural reasons a match may be untrustworthy even when the cosine
    score looks high.
    """
    result = MatchReliability(
        gallery_size=gallery_size,
        top_similarity=float(top_similarity),
        score_gap=float(top_similarity - runner_up_similarity) if runner_up_similarity is not None else 0.0,
    )

    if gallery_size == 0:
        result.warnings.append("empty_gallery_unknown_individual")
        result.reliable = False
        result.recommend_human_review = True
        return result

    if gallery_size < min_gallery_images:
        result.warnings.append(f"sparse_gallery({gallery_size}<{min_gallery_images})")
        result.recommend_human_review = True

    known = {"left", "right"}
    if query_flank in known and match_flank in known and query_flank != match_flank:
        # Opposite flanks carry different stripe patterns entirely.
        result.warnings.append(f"flank_mismatch({query_flank}_vs_{match_flank})")
        result.reliable = False
        result.recommend_human_review = True

    if runner_up_similarity is not None and result.score_gap < 0.05:
        result.warnings.append(f"ambiguous_top2(gap={result.score_gap:.3f})")
        result.recommend_human_review = True

    if top_similarity < domain_shift_threshold:
        # Nothing in the gallery resembles this crop: unknown tiger, or the model
        # is being applied outside the imagery it was trained on.
        result.warnings.append(f"possible_domain_shift_or_unknown_individual({top_similarity:.3f})")
        result.recommend_human_review = True

    if query_quality is not None:
        if not query_quality.usable:
            result.warnings.extend(query_quality.blocking_reasons)
            result.reliable = False
            result.recommend_human_review = True
        elif query_quality.quality_score < 0.25:
            result.warnings.append(f"low_query_quality({query_quality.quality_score:.3f})")
            result.recommend_human_review = True

    return result
