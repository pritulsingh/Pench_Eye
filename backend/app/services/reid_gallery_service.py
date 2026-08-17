"""
Re-ID gallery service: enrol embeddings and retrieve nearest candidates.

Storage is unchanged — `Embedding.embedding` remains 512-d (`vector(512)` under
pgvector, JSON on SQLite). Similarity search uses pgvector's cosine operator when
available and falls back to an in-Python cosine scan on SQLite.

`model_version` filtering matters: embeddings from different checkpoints or
different preprocessing occupy different spaces, so comparing across them
produces meaningless scores. Callers should pass the active model version.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import IS_SQLITE
from app.models.embedding import Embedding
from app.models.observation import Observation
from app.models.tiger import Tiger

logger = logging.getLogger(__name__)


@dataclass
class GalleryCandidate:
    tiger_id: str            # database UUID as string
    tiger_code: Optional[str]
    similarity: float
    rank: int
    embedding_id: Optional[str] = None
    observation_id: Optional[str] = None
    gallery_size: int = 0
    flank_side: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tiger_id": self.tiger_id,
            "tiger_code": self.tiger_code,
            "score": round(self.similarity, 4),
            "rank": self.rank,
            "gallery_size": self.gallery_size,
            "flank_side": self.flank_side,
        }


@dataclass
class SearchResult:
    candidates: List[GalleryCandidate] = field(default_factory=list)
    gallery_total: int = 0
    model_version: Optional[str] = None
    reliability: Optional[Dict[str, Any]] = None

    @property
    def top(self) -> Optional[GalleryCandidate]:
        return self.candidates[0] if self.candidates else None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class ReIDGalleryService:
    @staticmethod
    async def store_embedding(
        db: AsyncSession,
        *,
        observation_id,
        tiger_id=None,
        embedding: Sequence[float],
        model_version: str,
        flank_side: str = "unknown",
    ) -> Embedding:
        """Persist one embedding, validating the dimension contract first."""
        vector = [float(v) for v in embedding]
        if len(vector) != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding has {len(vector)} dimensions but the schema expects "
                f"{settings.EMBEDDING_DIM}."
            )

        row = Embedding(
            embedding_id=f"EMB-{uuid.uuid4().hex[:12].upper()}",
            observation_id=observation_id,
            tiger_id=tiger_id,
            embedding=vector,
            model_version=model_version,
            flank_side=flank_side,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row

    @staticmethod
    async def assign_to_tiger(db: AsyncSession, embedding_id: str, tiger_id) -> Optional[Embedding]:
        """Attach an embedding to an individual (e.g. after human review)."""
        row = (
            await db.execute(select(Embedding).where(Embedding.embedding_id == embedding_id))
        ).scalar_one_or_none()
        if row is None:
            return None
        row.tiger_id = tiger_id
        await db.commit()
        await db.refresh(row)
        return row

    @staticmethod
    async def gallery_size(db: AsyncSession, model_version: Optional[str] = None) -> int:
        query = select(func.count(Embedding.id)).where(Embedding.tiger_id.isnot(None))
        if model_version:
            query = query.where(Embedding.model_version == model_version)
        return int((await db.execute(query)).scalar_one_or_none() or 0)

    @staticmethod
    async def search(
        db: AsyncSession,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        model_version: Optional[str] = None,
        exclude_observation_id=None,
        query_flank: str = "unknown",
        query_quality: Optional[Any] = None,
    ) -> SearchResult:
        """
        Nearest enrolled identities by cosine similarity, best score per tiger.

        The result carries a reliability assessment so callers can downgrade a
        high-scoring but weakly-supported match to human review.
        """
        vector = [float(v) for v in embedding]
        gallery_total = await ReIDGalleryService.gallery_size(db, model_version)
        if gallery_total == 0 or not vector:
            return SearchResult(candidates=[], gallery_total=gallery_total, model_version=model_version)

        rows = (
            await ReIDGalleryService._search_sqlite(db, vector, model_version, exclude_observation_id)
            if IS_SQLITE
            else await ReIDGalleryService._search_pgvector(
                db, vector, model_version, exclude_observation_id, top_k
            )
        )

        # Best score per tiger, plus that tiger's gallery depth.
        best: Dict[str, Dict[str, Any]] = {}
        per_tiger_counts: Dict[str, int] = {}
        for item in rows:
            key = item["tiger_id"]
            per_tiger_counts[key] = per_tiger_counts.get(key, 0) + 1
            if key not in best or item["similarity"] > best[key]["similarity"]:
                best[key] = item

        ordered = sorted(best.values(), key=lambda r: r["similarity"], reverse=True)[:top_k]

        codes = await ReIDGalleryService._tiger_codes(db, [r["tiger_id"] for r in ordered])
        candidates = [
            GalleryCandidate(
                tiger_id=item["tiger_id"],
                tiger_code=codes.get(item["tiger_id"]),
                similarity=float(item["similarity"]),
                rank=index + 1,
                embedding_id=item.get("embedding_id"),
                observation_id=item.get("observation_id"),
                gallery_size=per_tiger_counts.get(item["tiger_id"], 0),
                flank_side=item.get("flank_side"),
            )
            for index, item in enumerate(ordered)
        ]

        reliability = None
        if candidates:
            try:
                from ml.reid.quality import assess_match

                reliability = assess_match(
                    top_similarity=candidates[0].similarity,
                    runner_up_similarity=candidates[1].similarity if len(candidates) > 1 else None,
                    gallery_size=candidates[0].gallery_size,
                    query_flank=query_flank,
                    match_flank=candidates[0].flank_side or "unknown",
                    query_quality=query_quality,
                ).to_dict()
            except Exception as exc:  # quality gating must never break search
                logger.debug("Match reliability assessment unavailable: %s", exc)

        return SearchResult(
            candidates=candidates,
            gallery_total=gallery_total,
            model_version=model_version,
            reliability=reliability,
        )

    @staticmethod
    async def _search_pgvector(
        db: AsyncSession,
        vector: List[float],
        model_version: Optional[str],
        exclude_observation_id,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        vector_literal = "[" + ",".join(map(str, vector)) + "]"
        conditions = ["tiger_id IS NOT NULL"]
        params: Dict[str, Any] = {"query_vec": vector_literal, "k": max(top_k * 5, 50)}
        if model_version:
            conditions.append("model_version = :model_version")
            params["model_version"] = model_version
        if exclude_observation_id is not None:
            conditions.append("observation_id <> :exclude_obs")
            params["exclude_obs"] = str(exclude_observation_id)

        query = text(
            f"""
            SELECT embedding_id, tiger_id, observation_id, flank_side,
                   1 - (embedding <=> :query_vec::vector) AS similarity
            FROM embeddings
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :k
            """
        )
        try:
            result = await db.execute(query, params)
            return [
                {
                    "embedding_id": row.embedding_id,
                    "tiger_id": str(row.tiger_id),
                    "observation_id": str(row.observation_id) if row.observation_id else None,
                    "flank_side": row.flank_side,
                    "similarity": float(row.similarity),
                }
                for row in result.all()
            ]
        except Exception as exc:
            logger.warning("pgvector search failed (%s); falling back to in-Python cosine.", exc)
            return await ReIDGalleryService._search_sqlite(
                db, vector, model_version, exclude_observation_id
            )

    @staticmethod
    async def _search_sqlite(
        db: AsyncSession,
        vector: List[float],
        model_version: Optional[str],
        exclude_observation_id,
    ) -> List[Dict[str, Any]]:
        """In-Python cosine scan. Fine for prototype gallery sizes."""
        query = select(Embedding).where(Embedding.tiger_id.isnot(None))
        if model_version:
            query = query.where(Embedding.model_version == model_version)
        rows = (await db.execute(query)).scalars().all()

        out: List[Dict[str, Any]] = []
        for row in rows:
            if exclude_observation_id is not None and str(row.observation_id) == str(
                exclude_observation_id
            ):
                continue
            stored = row.embedding
            if not stored or len(stored) != len(vector):
                continue
            out.append(
                {
                    "embedding_id": row.embedding_id,
                    "tiger_id": str(row.tiger_id),
                    "observation_id": str(row.observation_id) if row.observation_id else None,
                    "flank_side": row.flank_side,
                    "similarity": cosine_similarity(vector, stored),
                }
            )
        return out

    @staticmethod
    async def _tiger_codes(db: AsyncSession, tiger_ids: Sequence[str]) -> Dict[str, str]:
        if not tiger_ids:
            return {}
        rows = (await db.execute(select(Tiger.id, Tiger.tiger_id))).all()
        wanted = {str(t) for t in tiger_ids}
        return {str(row[0]): row[1] for row in rows if str(row[0]) in wanted}

    @staticmethod
    async def decide_identity(
        db: AsyncSession,
        embedding: Sequence[float],
        *,
        model_version: Optional[str] = None,
        query_flank: str = "unknown",
        query_quality: Optional[Any] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Search the gallery and apply the configured thresholds via
        `IdentityDecisionEngine`, then attach quality caveats.
        """
        search = await ReIDGalleryService.search(
            db,
            embedding,
            top_k=top_k,
            model_version=model_version,
            query_flank=query_flank,
            query_quality=query_quality,
        )

        from ml.reid.identity_decision_engine import CandidateMatch, IdentityDecisionEngine

        engine = IdentityDecisionEngine(
            auto_match_threshold=settings.AUTO_MATCH_THRESHOLD,
            review_threshold=settings.REVIEW_THRESHOLD,
            new_individual_threshold=settings.NEW_INDIVIDUAL_THRESHOLD,
        )
        decision = engine.decide(
            [
                CandidateMatch(
                    tiger_id=c.tiger_id,
                    similarity_score=c.similarity,
                    observation_count=c.gallery_size,
                    rank=c.rank,
                )
                for c in search.candidates
            ]
        )

        decision_value = (
            decision.decision.value if hasattr(decision.decision, "value") else str(decision.decision)
        )
        reliability = search.reliability or {}
        # Structural doubt overrides a confident-looking score.
        if decision_value == "auto_match" and reliability.get("recommend_human_review"):
            decision_value = "human_review"

        codes = {c.tiger_id: c.tiger_code for c in search.candidates}
        return {
            "decision": decision_value,
            "tiger_id": decision.matched_tiger_id if decision_value == "auto_match" else None,
            "tiger_code": codes.get(decision.matched_tiger_id) if decision_value == "auto_match" else None,
            "top_score": decision.top_score,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "candidates": [c.to_dict() for c in search.candidates],
            "gallery_total": search.gallery_total,
            "reliability": reliability,
            "thresholds": {
                "auto_match": settings.AUTO_MATCH_THRESHOLD,
                "review": settings.REVIEW_THRESHOLD,
                "new_individual": settings.NEW_INDIVIDUAL_THRESHOLD,
            },
        }


reid_gallery_service = ReIDGalleryService()
