"""
Inference abstraction boundary.

Everything the application knows about "AI" goes through `InferencePipeline`.
Two implementations share one interface:

  DemoInference        — deterministic, hash-seeded, no model weights, no GPU.
                         Clearly marked `is_demo=True`. ONLY for explicit demo
                         endpoints / unit tests. NEVER used for real uploads.
  ProductionInference  — real CV triage + YOLO tiger detection + MegaDescriptor.
                         Fail-closed: missing weights / errors do NOT invent
                         tigers, embeddings, or identities.

Swapping in a real model means implementing this interface; no API, database
or UI change is required.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import MLMode, settings


@dataclass
class DetectionOutput:
    present: bool
    species: str = "unknown"
    detection_type: str = "unknown"
    confidence: float = 0.0
    bbox: Optional[List[int]] = None
    reason: Optional[str] = None
    raw_detections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TriageOutput:
    is_blank: bool
    blank_probability: Optional[float]
    quality_score: Optional[float]
    reason: str
    stage: Optional[str] = None


@dataclass
class IdentityOutput:
    embedding: List[float] = field(default_factory=list)
    flank_side: str = "unknown"
    model_version: str = "demo-v0"
    similarity: float = 0.0
    suggested_tiger_code: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    is_demo: bool = True
    preprocessing_version: Optional[str] = None
    quality_score: Optional[float] = None
    quality_warnings: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    triage: TriageOutput
    detection: DetectionOutput
    identity: Optional[IdentityOutput]
    is_demo: bool
    model_version: str


# Non-tiger species used by the demo classifier — Pench genuinely hosts these,
# but any assignment produced here is simulated, not observed. Tiger is weighted
# because the demo narrative centres on individual tiger identification.
DEMO_SPECIES = [
    "tiger",
    "tiger",
    "tiger",
    "tiger",
    "leopard",
    "sambar",
    "chital",
    "wild_dog",
    "gaur",
    "sloth_bear",
]


def _seed_from(payload: bytes | str | None) -> int:
    if payload is None:
        return 0
    data = payload.encode() if isinstance(payload, str) else payload
    return int(hashlib.sha256(data).hexdigest()[:16], 16)


class DemoInference:
    """Deterministic simulated inference. Same input always yields same output."""

    model_version = "demo-inference-v1"
    is_demo = True

    def triage(self, *, image_hash: Optional[str], width: int = 0, height: int = 0) -> TriageOutput:
        seed = _seed_from(image_hash)
        blank_prob = round(((seed >> 8) % 1000) / 1000, 3)
        is_blank = blank_prob >= settings.BLANK_THRESHOLD
        return TriageOutput(
            is_blank=is_blank,
            blank_probability=blank_prob,
            quality_score=round(0.45 + ((seed >> 20) % 500) / 1000, 3),
            reason="blank_frame_simulated" if is_blank else "subject_detected_simulated",
            stage="demo",
        )

    def detect(self, *, image_hash: Optional[str]) -> DetectionOutput:
        seed = _seed_from(image_hash)
        species = DEMO_SPECIES[(seed >> 4) % len(DEMO_SPECIES)]
        confidence = round(0.72 + ((seed >> 12) % 270) / 1000, 3)
        detection_type = "tiger" if species == "tiger" else "other_wildlife"
        return DetectionOutput(
            present=True,
            species=species,
            detection_type=detection_type,
            confidence=confidence,
            bbox=[
                60 + (seed % 40),
                50 + ((seed >> 3) % 40),
                420 + ((seed >> 6) % 60),
                380 + ((seed >> 9) % 60),
            ],
            reason="demo_hash_detection",
        )

    def identify(
        self,
        *,
        image_hash: Optional[str],
        known_tiger_codes: List[str],
    ) -> IdentityOutput:
        seed = _seed_from(image_hash)
        rng_vec = [
            ((seed >> (i % 32)) % 1000) / 1000 - 0.5 for i in range(settings.EMBEDDING_DIM)
        ]
        norm = sum(v * v for v in rng_vec) ** 0.5 or 1.0
        embedding = [v / norm for v in rng_vec]

        similarity = round(0.58 + ((seed >> 16) % 410) / 1000, 3)
        suggested = known_tiger_codes[(seed >> 5) % len(known_tiger_codes)] if known_tiger_codes else None
        candidates = []
        if known_tiger_codes:
            ordered = known_tiger_codes[(seed >> 5) % len(known_tiger_codes) :] + known_tiger_codes[
                : (seed >> 5) % len(known_tiger_codes)
            ]
            for rank, code in enumerate(ordered[:3], start=1):
                candidates.append(
                    {
                        "tiger_code": code,
                        "score": round(max(0.0, similarity - 0.06 * (rank - 1)), 3),
                        "rank": rank,
                    }
                )

        return IdentityOutput(
            embedding=embedding,
            flank_side=["left", "right", "unknown"][(seed >> 7) % 3],
            model_version=self.model_version,
            similarity=similarity,
            suggested_tiger_code=suggested,
            candidates=candidates,
            is_demo=True,
        )

    def reid_status(self) -> Dict[str, Any]:
        return {
            "ml_mode": "demo",
            "model_version": self.model_version,
            "available": True,
            "is_demo": True,
            "embedding_dim": settings.EMBEDDING_DIM,
            "disclaimer": "Demo embeddings are deterministic placeholders, not tiger identification.",
        }

    def detector_status(self) -> Dict[str, Any]:
        return {
            "ml_mode": "demo",
            "available": True,
            "is_demo": True,
            "disclaimer": "Demo detector invents hash-seeded tiger boxes. Not for real uploads.",
        }


class ReIDUnavailable(RuntimeError):
    """ML_MODE=production but no trained Re-ID checkpoint is installed."""


class MegaDescriptorUnavailable(RuntimeError):
    """MegaDescriptor model failed to load or generate embedding."""


class DetectorUnavailable(RuntimeError):
    """Tiger YOLO weights missing or invalid — production detection cannot invent results."""


class ProductionInference:
    """
    Real-model path for MVP using MegaDescriptor.

    Fail-closed rules:
    - Triage uses real CV heuristics (Stage A). No hash simulation.
    - Detection uses YOLO with tiger class filtering only. Missing weights → no tiger.
    - Identification uses MegaDescriptor only after a real tiger detection.
    - Never falls back to DemoInference for any stage.
    """

    model_version = "production-megadescriptor-v1"
    is_demo = False

    def __init__(self) -> None:
        self._classifier = None
        self._detector = None
        self._megadescriptor = None
        self._megadescriptor_error: Optional[str] = None
        self._classifier_error: Optional[str] = None
        self._detector_error: Optional[str] = None
        try:
            from ml.triage.blank_classifier import BlankImageClassifier

            self._classifier = BlankImageClassifier(
                ml_mode="production", blank_threshold=settings.BLANK_THRESHOLD
            )
        except Exception as exc:
            self._classifier = None
            self._classifier_error = str(exc)
        try:
            from ml.detection.tiger_detector import TigerDetector
            # Only instantiate the production detector when an explicit
            # weights override is provided. This prevents silently loading a
            # repository-default weights file during unit tests or CI where
            # the operator did not opt-in to real detection.
            if settings.TIGER_YOLO_WEIGHTS:
                model_path = settings.TIGER_YOLO_WEIGHTS
                self._detector = TigerDetector(
                    ml_mode="production",
                    model_path=model_path,
                )
                status = self._detector.status()
                if not status.get("model_loaded"):
                    self._detector_error = status.get("load_error") or "Tiger YOLO not loaded."
            else:
                self._detector = None
                # Leave _detector_error unset so detect_frame falls back to the
                # canonical 'tiger_detector_unavailable' reason expected by
                # tests and diagnostic consumers.
                self._detector_error = None
        except Exception as exc:
            self._detector = None
            self._detector_error = str(exc)

    def _get_megadescriptor(self):
        """Lazily load the MegaDescriptor model."""
        if self._megadescriptor is not None:
            return self._megadescriptor
        try:
            from app.services.megadescriptor_service import MegaDescriptorEmbeddingService

            self._megadescriptor = MegaDescriptorEmbeddingService(
                model_name=settings.MEGADESCRIPTOR_MODEL_NAME,
                cache_dir=settings.MEGADESCRIPTOR_CACHE_DIR,
            )
        except Exception as exc:
            self._megadescriptor_error = f"Could not load MegaDescriptor: {exc}"
            return None
        return self._megadescriptor

    def reid_status(self) -> Dict[str, Any]:
        """Status of the embedding model (MegaDescriptor for MVP)."""
        md = self._get_megadescriptor()
        if md is None:
            return {
                "ml_mode": "production",
                "model_version": "megadescriptor-unavailable",
                "available": False,
                "is_demo": False,
                "embedding_dim": settings.EMBEDDING_DIM,
                "error": self._megadescriptor_error or "MegaDescriptor unavailable.",
            }
        return {
            "ml_mode": "production",
            "model_version": self.model_version,
            "available": True,
            "is_demo": False,
            "embedding_dim": settings.EMBEDDING_DIM,
            "model_name": settings.MEGADESCRIPTOR_MODEL_NAME,
            "disclaimer": (
                "MegaDescriptor is a pretrained vision model. Similarity scores are "
                "cosine similarities, not calibrated identity probabilities."
            ),
        }

    def detector_status(self) -> Dict[str, Any]:
        if self._detector is None:
            return {
                "ml_mode": "production",
                "available": False,
                "is_demo": False,
                "error": self._detector_error or "TigerDetector failed to construct.",
            }
        status = self._detector.status()
        status["available"] = bool(status.get("model_loaded"))
        status["is_demo"] = False
        if self._detector_error:
            status["error"] = self._detector_error
        return status

    def _tta_views(self, pixels: Any) -> List[Any]:
        """
        Build identity-preserving views of an RGB crop for test-time augmentation.

        Views are always returned in this fixed order so TTA_WEIGHTS can align:
        original, horizontal-flip, vertical-flip, each rotation angle, each crop.
        Views disabled by config are skipped (and their weight slot skipped too).
        Rotations/crops that would degenerate a tiny crop are silently dropped.
        """
        import numpy as np

        arr = np.asarray(pixels)
        views: List[Any] = [arr]

        if settings.ENABLE_HORIZONTAL_FLIP_TTA:
            views.append(arr[:, ::-1].copy())
        # Vertical flip is a control only; a tiger is rarely upside-down and it
        # destroys stripe identity. Off by default (ENABLE_VERTICAL_FLIP_TTA).
        if settings.ENABLE_VERTICAL_FLIP_TTA:
            views.append(arr[::-1, :].copy())

        angles = settings.rotation_angles_list
        if angles and arr.ndim == 3:
            try:
                from PIL import Image as _PILImage

                base = _PILImage.fromarray(arr.astype("uint8"), "RGB")
                for deg in angles:
                    rotated = base.rotate(deg, resample=_PILImage.BILINEAR, expand=False)
                    views.append(np.asarray(rotated))
            except Exception:
                # Rotation is best-effort; fall back to the geometric views we have.
                pass

        if settings.ENABLE_CROP_TTA and arr.ndim == 3:
            h, w = arr.shape[:2]
            for frac in settings.tta_crop_fractions_list:
                cw, ch = int(w * frac), int(h * frac)
                if cw < 8 or ch < 8:
                    continue
                left, top = (w - cw) // 2, (h - ch) // 2
                views.append(arr[top:top + ch, left:left + cw].copy())

        return views

    def _aggregate_embeddings(self, embeddings: List[Any]) -> Any:
        """
        L2-normalize each embedding, average (optionally weighted), then
        L2-normalize the result. Never averages unnormalized vectors.
        """
        import numpy as np

        mats = []
        for emb in embeddings:
            v = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n == 0:
                continue
            mats.append(v / n)
        if not mats:
            raise MegaDescriptorUnavailable("No valid embeddings to aggregate.")

        stacked = np.vstack(mats)
        if settings.TTA_AGGREGATION_METHOD == "weighted" and settings.tta_weights_list:
            weights = np.asarray(settings.tta_weights_list[: stacked.shape[0]], dtype=np.float32)
            if weights.shape[0] < stacked.shape[0]:
                weights = np.concatenate(
                    [weights, np.ones(stacked.shape[0] - weights.shape[0], dtype=np.float32)]
                )
            wsum = float(weights.sum())
            mean = (stacked * weights[:, None]).sum(axis=0) / (wsum if wsum else 1.0)
        else:
            mean = stacked.mean(axis=0)

        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm) or norm == 0:
            raise MegaDescriptorUnavailable("Aggregated embedding is degenerate.")
        return mean / norm

    def identify_frame(
        self,
        pixels: Any,
        image_hash: Optional[str],
        known_tiger_codes: List[str],
        flank_side: str = "unknown",
    ) -> IdentityOutput:
        """
        Generate a MegaDescriptor embedding from a tiger crop.
        Must only be called after a real tiger detection.

        Applies test-time augmentation (TTA): the embedding is aggregated over
        identity-preserving views (horizontal flip, small rotations, centre
        crops) so a flipped/rotated/cropped capture of the same tiger stays
        close to its enrolled embedding instead of being scored as a new
        individual. Each view is L2-normalized before averaging and the mean is
        renormalized. TTA is disabled when all view flags are off.
        """
        md = self._get_megadescriptor()
        if md is None:
            raise MegaDescriptorUnavailable(
                self._megadescriptor_error
                or "MegaDescriptor unavailable. Check that transformers/torch libraries are installed."
            )

        # Quality gate on the raw crop before embedding (reuse ml.reid.quality).
        quality_score: Optional[float] = None
        quality_warnings: List[str] = []
        try:
            import numpy as np

            from ml.reid.quality import assess_crop

            crop_rgb = np.asarray(pixels)
            if crop_rgb.ndim == 3:
                assessment = assess_crop(crop_rgb, flank=flank_side or "unknown")
                quality_score = float(assessment.quality_score)
                quality_warnings = list(assessment.warnings) + list(assessment.blocking_reasons)
        except Exception:
            # Quality gating is advisory; never block embedding on its failure.
            quality_score = None

        views = self._tta_views(pixels)
        tta_used = len(views) > 1
        try:
            view_embeddings = [md.get_embedding(v) for v in views]
            aggregated = self._aggregate_embeddings(view_embeddings)
            embedding_list = aggregated.tolist()
        except MegaDescriptorUnavailable:
            raise
        except Exception as exc:
            raise MegaDescriptorUnavailable(f"Embedding generation failed: {exc}") from exc

        preprocessing_version = (
            f"megadescriptor-tta-v1(n={len(views)})" if tta_used else "megadescriptor-pil"
        )

        return IdentityOutput(
            embedding=embedding_list,
            flank_side="unknown",
            model_version=self.model_version,
            similarity=0.0,
            suggested_tiger_code=None,
            candidates=[],
            is_demo=False,
            preprocessing_version=preprocessing_version,
            quality_score=quality_score,
            quality_warnings=quality_warnings,
        )

    def triage_frame(self, pixels: Any, image_hash: Optional[str] = None) -> TriageOutput:
        """Real CV triage. Fail closed to blank/quarantine if classifier unavailable."""
        if self._classifier is None or pixels is None:
            return TriageOutput(
                is_blank=True,
                blank_probability=None,
                quality_score=None,
                reason=(
                    "triage_unavailable: "
                    + (self._classifier_error or "blank classifier not loaded; refusing to invent a subject")
                ),
                stage="unavailable",
            )
        try:
            res = self._classifier.classify(pixels)
            stage = getattr(res, "stage_used", "cv")
            return TriageOutput(
                is_blank=bool(res.is_blank),
                # CV Stage A emits a heuristic score, not a calibrated probability.
                blank_probability=(float(res.blank_probability) if stage == "model" else None),
                quality_score=float(getattr(res, "quality_score", 0.5)),
                reason=", ".join(getattr(res, "reasons", []) or ["classified"]),
                stage=stage,
            )
        except Exception as exc:
            return TriageOutput(
                is_blank=True,
                blank_probability=None,
                quality_score=None,
                reason=f"triage_failed: {exc}",
                stage="error",
            )

    def detect_frame(self, pixels: Any, image_hash: Optional[str] = None) -> DetectionOutput:
        """
        Real YOLO tiger detection. Fail closed — never invent a tiger.
        Non-tiger classes (e.g. person) are ignored for the tiger pipeline.
        """
        if self._detector is None or pixels is None:
            return DetectionOutput(
                present=False,
                species="unknown",
                detection_type="unknown",
                confidence=0.0,
                reason=self._detector_error or "tiger_detector_unavailable",
            )
        try:
            status = self._detector.status()
            if not status.get("model_loaded"):
                return DetectionOutput(
                    present=False,
                    species="unknown",
                    detection_type="unknown",
                    confidence=0.0,
                    reason="tiger_detector_unavailable",
                )

            all_detections = self._detector.detect_all(pixels)
            raw = []
            for det in all_detections:
                raw.append(
                    {
                        "class_name": det.class_name,
                        "class_id": det.class_id,
                        "confidence": float(det.confidence),
                        "bbox": list(det.bbox),
                    }
                )
            from ml.detection.tiger_detector import is_tiger_class

            detections = [det for det in all_detections if is_tiger_class(det.class_name)]

            if not detections:
                return DetectionOutput(
                    present=False,
                    species="unknown",
                    detection_type="unknown",
                    confidence=0.0,
                    reason="no_tiger_detected",
                    raw_detections=raw,
                )

            best = max(detections, key=lambda detection: detection.confidence)
            return DetectionOutput(
                present=True,
                species="tiger",
                detection_type="tiger",
                confidence=float(best.confidence),
                bbox=list(best.bbox),
                reason="tiger_detected",
                raw_detections=raw,
            )
        except Exception as exc:
            return DetectionOutput(
                present=False,
                species="unknown",
                detection_type="unknown",
                confidence=0.0,
                reason=f"detection_failed: {exc}",
            )


def build_pipeline():
    if settings.ML_MODE == MLMode.PRODUCTION:
        return ProductionInference()
    return DemoInference()


inference_pipeline = build_pipeline()


def pipeline_info() -> Dict[str, Any]:
    reid = (
        inference_pipeline.reid_status()
        if hasattr(inference_pipeline, "reid_status")
        else {"available": True, "is_demo": True}
    )
    detector = (
        inference_pipeline.detector_status()
        if hasattr(inference_pipeline, "detector_status")
        else {"available": False, "is_demo": True}
    )
    return {
        "ml_mode": settings.ML_MODE.value,
        "model_version": inference_pipeline.model_version,
        "is_demo": inference_pipeline.is_demo,
        "reid": reid,
        "reid_available": bool(reid.get("available")),
        "detector": detector,
        "detector_available": bool(detector.get("available")),
        "disclaimer": (
            "Demo inference produces deterministic simulated results. It is not a "
            "scientifically validated tiger identification model. Real image uploads "
            "are rejected while ML_MODE=demo."
            if inference_pipeline.is_demo
            else (
                f"Production pipeline active. Detector: "
                f"{'loaded' if detector.get('available') else 'UNAVAILABLE (fail-closed)'}. "
                f"MegaDescriptor: {'available' if reid.get('available') else 'unavailable'}."
            )
        ),
    }
