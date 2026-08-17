"""
TigerReIDEncoder — produces the 512-d L2-normalised embedding the application
stores and searches.

Two modes, and the difference is never hidden:

* `ml_mode="demo"` → deterministic pseudo-embedding derived from image content.
  Stable and useful for demonstrations. It is **not** tiger identification:
  similarity between two different tigers is essentially arbitrary.
  Results are marked `is_demo=True` and versioned `demo-*`.

* `ml_mode="production"` → a checkpoint trained by `ml/reid/train.py`, loaded via
  `ml/reid/checkpoint.py`. If no checkpoint is present, `encode()` raises
  `ReIDModelUnavailable` rather than silently returning a demo vector. Callers
  must surface that as an unavailable stage.

Preprocessing comes from `ml.reid.preprocessing`, the same module the trainer
uses, so a checkpoint sees identical inputs at train and inference time.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml.reid.preprocessing import PREPROCESSING_VERSION, PreprocessConfig, preprocess_bgr

logger = logging.getLogger(__name__)

DEMO_MODEL_VERSION = "demo-deterministic-v1.1"


class ReIDModelUnavailable(RuntimeError):
    """Production mode was requested but no usable trained checkpoint exists."""


@dataclass
class EmbeddingResult:
    embedding: List[float]          # 512-d L2-normalised
    model_version: str
    inference_time_ms: float
    is_demo: bool
    preprocessing_version: str = PREPROCESSING_VERSION
    embedding_dim: int = 512
    metadata: Dict[str, Any] = field(default_factory=dict)


class TigerReIDEncoder:
    EMBEDDING_DIM = 512

    def __init__(
        self,
        ml_mode: str = "demo",
        model_path: Optional[str] = None,
        embedding_dim: int = EMBEDDING_DIM,
        *,
        strict: bool = True,
    ):
        """
        `strict=True` (default) makes production mode fail loudly when the
        checkpoint is missing. Set it to False only where a caller has its own
        explicit fallback and labels the result as simulated.
        """
        self.ml_mode = (ml_mode or "demo").lower()
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.strict = strict

        self._model = None
        self._device = "cpu"
        self._checkpoint_payload = None
        self._preprocess: PreprocessConfig = PreprocessConfig()
        self._load_error: Optional[str] = None
        self._load_attempted = False

    # ── Introspection ─────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ml_mode == "production"

    @property
    def model_version(self) -> str:
        if not self.is_production:
            return DEMO_MODEL_VERSION
        if self._checkpoint_payload is not None:
            return self._checkpoint_payload.model_version
        return "tiger-reid-unavailable"

    @property
    def preprocess_config(self) -> PreprocessConfig:
        return self._preprocess

    def is_available(self) -> bool:
        """True when this encoder can produce embeddings of its declared kind."""
        if not self.is_production:
            return True
        self._ensure_model()
        return self._model is not None

    def status(self) -> Dict[str, Any]:
        """Machine-readable state for /health and the demo-status endpoint."""
        available = self.is_available()
        info: Dict[str, Any] = {
            "ml_mode": self.ml_mode,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "preprocessing_version": self._preprocess.version,
            "is_demo": not self.is_production,
            "available": available,
            "checkpoint_path": str(self._resolved_path) if self._resolved_path else None,
        }
        if self.is_production:
            payload = self._checkpoint_payload
            info["trained_epochs"] = payload.epoch if payload else None
            info["training_metrics"] = payload.metrics if payload else None
            info["known_identities"] = len(payload.identity_to_index) if payload else 0
            info["validated"] = bool(payload and payload.metrics.get("val_rank1") is not None)
            if not available:
                info["error"] = self._load_error or "No trained Re-ID checkpoint found."
        else:
            info["disclaimer"] = (
                "Demo embeddings are deterministic placeholders, not tiger identification."
            )
        return info

    # ── Encoding ──────────────────────────────────────────────────────────
    def encode(self, processed_stripe: Any) -> EmbeddingResult:
        if self.is_production:
            return self._encode_production(processed_stripe)
        return self._encode_demo(processed_stripe)

    def encode_batch(self, stripes: List[Any]) -> List[EmbeddingResult]:
        return [self.encode(s) for s in stripes]

    def _encode_demo(self, stripe: Any) -> EmbeddingResult:
        """
        Deterministic content-seeded vector: same image always yields the same
        embedding. Provides stable behaviour for demos, not identification.
        """
        start = time.time()
        image = self._stripe_to_bgr(stripe)

        gray = image.mean(axis=2) if image.ndim == 3 else image.astype(np.float32)
        small = self._resize_gray(gray, 32)
        digest = hashlib.sha256(small.astype(np.uint8).tobytes()).hexdigest()
        rng = np.random.default_rng(seed=int(digest, 16) % (2**32))
        vector = rng.normal(0.0, 1.0, self.embedding_dim).astype(np.float32)

        # Blend in coarse intensity structure so visually similar crops land
        # nearer each other than pure noise would.
        block = small.reshape(4, 8, 4, 8).mean(axis=(1, 3)).flatten()
        span = float(np.abs(block).max())
        if span > 0:
            vector[: block.size] += (block / span).astype(np.float32)

        norm = float(np.linalg.norm(vector))
        embedding = (vector / norm) if norm > 0 else vector

        return EmbeddingResult(
            embedding=embedding.tolist(),
            model_version=DEMO_MODEL_VERSION,
            inference_time_ms=(time.time() - start) * 1000.0,
            is_demo=True,
            preprocessing_version=self._preprocess.version,
            embedding_dim=self.embedding_dim,
            metadata={
                "simulated": True,
                "disclaimer": "Deterministic demo embedding — not tiger identification.",
            },
        )

    def _encode_production(self, stripe: Any) -> EmbeddingResult:
        self._ensure_model()
        if self._model is None:
            raise ReIDModelUnavailable(
                self._load_error
                or "ML_MODE=production but no trained Re-ID checkpoint was found. "
                "Train one with `python -m ml.reid.train` (see docs/reid_training.md), "
                "or set ML_MODE=demo."
            )

        import torch

        start = time.time()
        tensor = self._to_model_tensor(stripe)
        with torch.no_grad():
            embedding = self._model(tensor.to(self._device))
        vector = embedding.cpu().numpy().flatten().astype(np.float32)

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        payload = self._checkpoint_payload
        return EmbeddingResult(
            embedding=vector.tolist(),
            model_version=self.model_version,
            inference_time_ms=(time.time() - start) * 1000.0,
            is_demo=False,
            preprocessing_version=self._preprocess.version,
            embedding_dim=int(vector.size),
            metadata={
                "simulated": False,
                "backbone": payload.model_config.backbone if payload else None,
                "trained_epochs": payload.epoch if payload else None,
                "validated": bool(payload and payload.metrics.get("val_rank1") is not None),
            },
        )

    # ── Model loading ─────────────────────────────────────────────────────
    @property
    def _resolved_path(self) -> Optional[Path]:
        from ml.reid.checkpoint import resolve_checkpoint_path

        return resolve_checkpoint_path(self.model_path)

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_attempted:
            return
        self._load_attempted = True
        self._load_model()

    def _load_model(self) -> None:
        """Load a trained checkpoint. Records the failure reason on error."""
        from ml.reid.checkpoint import load_model_for_inference, resolve_checkpoint_path

        path = resolve_checkpoint_path(self.model_path)
        if path is None:
            self._load_error = (
                "No Re-ID checkpoint found. Looked for ml/weights/tiger_reid/best.pt, "
                "ml/weights/tiger_reid/latest.pt and ml/weights/tiger_reid.pt."
            )
            logger.error("[TigerReIDEncoder] %s", self._load_error)
            return

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, payload = load_model_for_inference(path, device=device)
        except Exception as exc:
            self._load_error = f"Failed to load checkpoint {path}: {exc}"
            logger.error("[TigerReIDEncoder] %s", self._load_error)
            return

        if payload.model_config.embedding_dim != self.embedding_dim:
            self._load_error = (
                f"Checkpoint embedding_dim={payload.model_config.embedding_dim} does not match the "
                f"application contract ({self.embedding_dim}). Retrain with "
                f"--embedding-dim {self.embedding_dim}."
            )
            logger.error("[TigerReIDEncoder] %s", self._load_error)
            return

        self._model = model
        self._device = device
        self._checkpoint_payload = payload
        self._preprocess = payload.preprocess_config
        self._load_error = None
        logger.info(
            "[TigerReIDEncoder] Loaded %s from %s (epoch %d, device %s).",
            payload.model_version,
            path,
            payload.epoch,
            device,
        )
        if payload.metrics.get("val_rank1") is None:
            logger.warning(
                "[TigerReIDEncoder] Checkpoint %s has no recorded validation Rank-1; "
                "it is UNVALIDATED. Run `python -m ml.reid.evaluate` before relying on it.",
                path,
            )

    # ── Helpers ───────────────────────────────────────────────────────────
    def _stripe_to_bgr(self, stripe: Any) -> np.ndarray:
        """Accept a ProcessedStripe, a raw array, or None."""
        if stripe is None:
            height, width = self._preprocess.image_size
            return np.zeros((height, width, 3), dtype=np.uint8)
        image = getattr(stripe, "original_resized", stripe)
        if image is None:
            height, width = self._preprocess.image_size
            return np.zeros((height, width, 3), dtype=np.uint8)
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return array

    def _to_model_tensor(self, stripe: Any):
        """
        Build the model input, reusing a precomputed tensor only when it came
        from the current preprocessing version.
        """
        import torch

        tensor = getattr(stripe, "tensor", None)
        version_ok = getattr(stripe, "preprocessing_version", None) == self._preprocess.version
        if tensor is not None and version_ok:
            array = np.asarray(tensor, dtype=np.float32)
        else:
            array = preprocess_bgr(self._stripe_to_bgr(stripe), self._preprocess)

        out = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        return out.unsqueeze(0) if out.dim() == 3 else out

    @staticmethod
    def _resize_gray(gray: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        clipped = np.clip(gray, 0, 255).astype(np.uint8)
        return np.array(Image.fromarray(clipped).resize((size, size), Image.BILINEAR)).astype(np.float32)
