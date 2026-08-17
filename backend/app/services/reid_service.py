"""
ReID Service — Tiger Intelligence System
Wraps the full re-identification pipeline:
  flank extraction → stripe processing → embedding → similarity search → identity decision
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.embedding import Embedding
from typing import List, Dict, Any, Optional
import uuid

# ── ML imports with graceful fallback ──────────────────────────────────────
try:
    from ml.reid.flank_extractor import FlankExtractor
    from ml.reid.stripe_processor import StripeProcessor
    from ml.reid.tiger_reid_encoder import TigerReIDEncoder
    from ml.reid.identity_decision_engine import (
        IdentityDecisionEngine,
        CandidateMatch,
        IdentityDecision,
    )
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    FlankExtractor = None  # type: ignore
    StripeProcessor = None  # type: ignore

    class TigerReIDEncoder:  # type: ignore
        """Fallback encoder — deterministic hash-based 512-dim vectors."""
        def __init__(self, ml_mode: str = "demo", model_path: Optional[str] = None):
            self.ml_mode = ml_mode

        def encode(self, processed_stripe: Any) -> Any:
            import numpy as np
            import hashlib

            if processed_stripe is None:
                vec = np.random.rand(512).astype(np.float32)
            else:
                img = (
                    processed_stripe.original_resized
                    if hasattr(processed_stripe, "original_resized")
                    else processed_stripe
                )
                raw = img.tobytes()[:1024] if hasattr(img, "tobytes") else b"default"
                h = int(hashlib.sha256(raw).hexdigest(), 16)
                rng = np.random.default_rng(seed=h % (2 ** 32))
                vec = rng.normal(0, 1, 512).astype(np.float32)

            norm = np.linalg.norm(vec)
            embedding = (vec / norm).tolist() if norm > 0 else vec.tolist()

            class _Result:
                pass

            r = _Result()
            r.embedding = embedding  # type: ignore
            r.model_version = "demo-fallback"  # type: ignore
            r.inference_time_ms = 0.0  # type: ignore
            r.is_demo = True  # type: ignore
            return r

    class CandidateMatch:  # type: ignore
        def __init__(self, tiger_id: str, similarity_score: float, observation_count: int, rank: int):
            self.tiger_id = tiger_id
            self.similarity_score = similarity_score
            self.observation_count = observation_count
            self.rank = rank

    class IdentityDecision:  # type: ignore
        AUTO_MATCH = "auto_match"
        HUMAN_REVIEW = "human_review"
        NEW_INDIVIDUAL = "new_individual"

    class IdentityDecisionEngine:  # type: ignore
        """Fallback threshold-based decision engine."""
        def __init__(
            self,
            auto_match_threshold: float = 0.90,
            review_threshold: float = 0.75,
            new_individual_threshold: float = 0.60,
        ):
            self.auto_match_threshold = auto_match_threshold
            self.review_threshold = review_threshold
            self.new_individual_threshold = new_individual_threshold

        def decide(self, candidates: list) -> Any:
            class _Result:
                pass

            r = _Result()
            if not candidates:
                r.decision = IdentityDecision.NEW_INDIVIDUAL  # type: ignore
                r.matched_tiger_id = None  # type: ignore
                r.candidates = []  # type: ignore
                r.top_score = 0.0  # type: ignore
                r.confidence = 1.0  # type: ignore
                r.reason = "No candidates found."  # type: ignore
                return r

            sorted_c = sorted(candidates, key=lambda c: c.similarity_score, reverse=True)
            top = sorted_c[0]
            top_score = top.similarity_score

            if top_score >= self.auto_match_threshold:
                gap = (sorted_c[0].similarity_score - sorted_c[1].similarity_score) if len(sorted_c) > 1 else 1.0
                if gap >= 0.05:
                    r.decision = IdentityDecision.AUTO_MATCH  # type: ignore
                    r.matched_tiger_id = top.tiger_id  # type: ignore
                    r.reason = f"Auto matched: {top.tiger_id} (score {top_score:.3f})"  # type: ignore
                else:
                    r.decision = IdentityDecision.HUMAN_REVIEW  # type: ignore
                    r.matched_tiger_id = None  # type: ignore
                    r.reason = f"High score but gap too small — needs review."  # type: ignore
            elif top_score >= self.review_threshold:
                r.decision = IdentityDecision.HUMAN_REVIEW  # type: ignore
                r.matched_tiger_id = None  # type: ignore
                r.reason = f"Score {top_score:.3f} in review zone."  # type: ignore
            else:
                r.decision = IdentityDecision.NEW_INDIVIDUAL  # type: ignore
                r.matched_tiger_id = None  # type: ignore
                r.reason = f"Score {top_score:.3f} below all thresholds — new individual."  # type: ignore

            r.candidates = sorted_c  # type: ignore
            r.top_score = top_score  # type: ignore
            r.confidence = float(min(1.0, top_score))  # type: ignore
            return r


# ── ReID Service ───────────────────────────────────────────────────────────
class ReIDService:
    """Orchestrates the full Re-ID pipeline end-to-end."""

    def __init__(self) -> None:
        from app.core.config import settings
        ml_mode = settings.ML_MODE.value if hasattr(settings.ML_MODE, "value") else str(settings.ML_MODE)
        self.encoder = TigerReIDEncoder(ml_mode=ml_mode)
        self.decision_engine = IdentityDecisionEngine(
            auto_match_threshold=settings.AUTO_MATCH_THRESHOLD,
            review_threshold=settings.REVIEW_THRESHOLD,
            new_individual_threshold=settings.NEW_INDIVIDUAL_THRESHOLD,
        )
        if _ML_AVAILABLE:
            self.flank_extractor: Any = FlankExtractor()
            self.stripe_processor: Any = StripeProcessor()
        else:
            self.flank_extractor = None
            self.stripe_processor = None

    async def extract_embedding(
        self,
        image_np: Any,
        detection_bbox: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Extract a 512-dim embedding from a tiger crop (or full image).
        Returns dict with embedding, flank_side, quality_score, model_version, is_demo.
        """
        import numpy as np

        # Crop to detection region if provided
        if detection_bbox is not None and image_np is not None:
            x1, y1, x2, y2 = detection_bbox
            crop = image_np[y1:y2, x1:x2]
        else:
            crop = image_np if image_np is not None else np.zeros((224, 224, 3), dtype=np.uint8)

        # Stage 1: Flank extraction
        flank_side = "unknown"
        if self.flank_extractor is not None:
            try:
                flank_result = await asyncio.to_thread(self.flank_extractor.extract, crop)
                crop = flank_result.flank_image
                flank_side = (
                    flank_result.flank_side.value
                    if hasattr(flank_result.flank_side, "value")
                    else str(flank_result.flank_side)
                )
            except Exception:
                pass  # Fall through with original crop

        # Stage 2: Stripe processing
        quality_score = 0.5
        if self.stripe_processor is not None:
            try:
                processed = await asyncio.to_thread(self.stripe_processor.process, crop)
                quality_score = getattr(processed, "quality_score", 0.5)
            except Exception:
                processed = _make_dummy_processed(crop)
        else:
            processed = _make_dummy_processed(crop)

        # Stage 3: Embedding
        emb_result = await asyncio.to_thread(self.encoder.encode, processed)

        return {
            "embedding": emb_result.embedding,
            "flank_side": flank_side,
            "quality_score": quality_score,
            "model_version": emb_result.model_version,
            "is_demo": emb_result.is_demo,
        }

    async def save_embedding(
        self,
        db: AsyncSession,
        observation_id: str,
        tiger_id: str,
        emb_data: Dict[str, Any],
    ) -> Embedding:
        """Persist embedding vector to the database."""
        embedding = Embedding(
            embedding_id=f"EMB-{uuid.uuid4()}",
            observation_id=observation_id,
            tiger_id=tiger_id,
            embedding=emb_data["embedding"],
            flank_side=emb_data.get("flank_side", "unknown"),
            model_version=emb_data.get("model_version", "unknown"),
        )
        db.add(embedding)
        await db.commit()
        await db.refresh(embedding)
        return embedding

    async def search_similar(
        self,
        db: AsyncSession,
        embedding_vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        pgvector cosine similarity search.
        Returns list of {tiger_id, observation_id, score, rank}.
        """
        vector_str = "[" + ",".join(map(str, embedding_vector)) + "]"
        query = text(
            """
            SELECT
                tiger_id,
                observation_id,
                1 - (embedding <=> :query_vec::vector) AS similarity
            FROM embeddings
            WHERE tiger_id IS NOT NULL
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :k
            """
        )
        try:
            result = await db.execute(query, {"query_vec": vector_str, "k": top_k})
            rows = result.all()
            return [
                {
                    "tiger_id": str(row.tiger_id),
                    "observation_id": str(row.observation_id),
                    "score": float(row.similarity),
                    "rank": i + 1,
                }
                for i, row in enumerate(rows)
            ]
        except Exception:
            return []

    async def decide_identity(
        self, similarity_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run the IdentityDecisionEngine on similarity search results.
        Aggregates by tiger_id (best score per tiger) before deciding.
        """
        tiger_scores: Dict[str, float] = {}
        for r in similarity_results:
            tid = r["tiger_id"]
            if tid not in tiger_scores or r["score"] > tiger_scores[tid]:
                tiger_scores[tid] = r["score"]

        candidates = [
            CandidateMatch(tiger_id=tid, similarity_score=score, observation_count=1, rank=0)
            for tid, score in tiger_scores.items()
        ]

        result = self.decision_engine.decide(candidates)
        decision_val = (
            result.decision.value
            if hasattr(result.decision, "value")
            else str(result.decision)
        )

        return {
            "decision": decision_val,
            "matched_tiger_id": getattr(result, "matched_tiger_id", None),
            "candidates": [
                {"tiger_id": c.tiger_id, "score": c.similarity_score, "rank": i + 1}
                for i, c in enumerate(getattr(result, "candidates", candidates))
            ],
            "top_score": getattr(result, "top_score", 0.0),
            "confidence": getattr(result, "confidence", 0.0),
            "reason": getattr(result, "reason", ""),
        }


def _make_dummy_processed(crop: Any) -> Any:
    """Creates a minimal ProcessedStripe-compatible object for fallback use."""
    class _DummyProcessed:
        original_resized = crop
        tensor = None
        quality_score = 0.5
        contrast_enhanced = False
        target_size = (224, 224)
    return _DummyProcessed()


# Singleton
reid_service = ReIDService()