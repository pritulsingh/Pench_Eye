from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from app.models.camera_station import CameraStation
from app.models.image import Image
from app.models.review_queue import ReviewQueue, QueueStatus
from app.models.observation import Observation, MatchType, ReviewStatus
from app.models.tiger import Tiger
from app.models.embedding import Embedding
from app.services.tiger_service import TigerService
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


def _serialize_review(
    review: ReviewQueue,
    obs: Optional[Observation],
    image: Optional[Image],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": review.id,
        "review_id": review.review_id,
        "observation_id": str(review.observation_id) if review.observation_id else None,
        "observation_code": obs.observation_id if obs else None,
        "camera_id": obs.camera_id if obs else None,
        "timestamp": obs.timestamp if obs else None,
        "image_id": image.image_id if image else None,
        "image_url": f"/api/v1/images/{image.image_id}/file" if image else None,
        "status": review.status.value if review.status else None,
        "candidates": candidates,
        "review_note": review.review_note,
        "reviewed_by": review.reviewed_by,
        "reviewed_at": review.reviewed_at,
        "created_at": review.created_at,
    }


class ReviewService:
    @staticmethod
    async def create_review_item(db: AsyncSession, observation_id, candidates: list) -> ReviewQueue:
        codes = [
            c.get("tiger_code") or c.get("tiger_id")
            for c in candidates
            if isinstance(c, dict)
        ]
        scores = {
            (c.get("tiger_code") or c.get("tiger_id")): c.get("score")
            for c in candidates
            if isinstance(c, dict)
        }
        review_item = ReviewQueue(
            review_id=f"REV-{uuid.uuid4().hex[:10].upper()}",
            observation_id=observation_id,
            candidate_tiger_ids=codes,
            candidate_scores=scores,
            alternative_candidates_json=candidates,
            status=QueueStatus.PENDING,
        )
        db.add(review_item)
        await db.commit()
        await db.refresh(review_item)
        return review_item

    @staticmethod
    async def list_reviews(
        db: AsyncSession,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[Dict[str, Any]], int]:
        query = (
            select(ReviewQueue, Observation, Image)
            .outerjoin(Observation, Observation.id == ReviewQueue.observation_id)
            .outerjoin(Image, Image.id == Observation.image_id)
        )
        count_query = select(func.count(ReviewQueue.id))

        if status:
            cond = ReviewQueue.status == QueueStatus(status)
            query = query.where(cond)
            count_query = count_query.where(cond)

        total = (await db.execute(count_query)).scalar_one_or_none() or 0
        rows = (
            await db.execute(
                query.order_by(ReviewQueue.created_at.desc()).offset(skip).limit(limit)
            )
        ).all()

        tiger_rows = (await db.execute(select(Tiger.tiger_id, Tiger.name))).all()
        tiger_names = {r[0]: r[1] for r in tiger_rows}

        items = []
        for review, obs, image in rows:
            raw = review.alternative_candidates_json or []
            candidates = []
            if isinstance(raw, list) and raw:
                for c in raw:
                    code = c.get("tiger_code") or c.get("tiger_id")
                    candidates.append(
                        {
                            "tiger_id": code,
                            "tiger_code": code,
                            "tiger_name": tiger_names.get(code),
                            "score": c.get("score"),
                        }
                    )
            else:
                scores = review.candidate_scores or {}
                for code in review.candidate_tiger_ids or []:
                    candidates.append(
                        {
                            "tiger_id": code,
                            "tiger_code": code,
                            "tiger_name": tiger_names.get(code),
                            "score": scores.get(code) if isinstance(scores, dict) else None,
                        }
                    )
            items.append(_serialize_review(review, obs, image, candidates))
        return items, int(total)

    @staticmethod
    async def get_review(db: AsyncSession, review_id: str) -> Optional[Dict[str, Any]]:
        items, _ = await ReviewService.list_reviews(db, limit=500)
        for item in items:
            if item["review_id"] == review_id:
                return item
        return None

    @staticmethod
    async def get_pending_reviews(db: AsyncSession, skip: int = 0, limit: int = 50) -> List[ReviewQueue]:
        result = await db.execute(
            select(ReviewQueue).where(ReviewQueue.status == QueueStatus.PENDING).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def approve_review(db: AsyncSession, review_id: str, tiger_id: str, reviewer: str, note: str = None):
        result = await db.execute(select(ReviewQueue).where(ReviewQueue.review_id == review_id))
        review = result.scalar_one_or_none()
        if not review:
            return None
            
        review.status = QueueStatus.APPROVED
        review.reviewed_by = reviewer
        review.reviewed_at = datetime.now(timezone.utc)
        review.review_note = note
        
        # Update Observation
        obs = await db.get(Observation, review.observation_id)
        if obs:
            tiger_result = await db.execute(select(Tiger).where(Tiger.tiger_id == tiger_id))
            tiger = tiger_result.scalar_one_or_none()
            if tiger:
                obs.tiger_id = tiger.id
                for embedding in list(
                    (await db.execute(select(Embedding).where(Embedding.observation_id == obs.id)))
                    .scalars()
                    .all()
                ):
                    embedding.tiger_id = tiger.id
                obs.match_type = MatchType.HUMAN_VERIFIED
                obs.review_status = ReviewStatus.APPROVED
                await db.commit()
                await TigerService.update_tiger_stats(db, tiger.id)
                
        await db.commit()
        return review

    @staticmethod
    async def reject_review(db: AsyncSession, review_id: str, reviewer: str, note: str = None):
        result = await db.execute(select(ReviewQueue).where(ReviewQueue.review_id == review_id))
        review = result.scalar_one_or_none()
        if review:
            review.status = QueueStatus.REJECTED
            review.reviewed_by = reviewer
            review.reviewed_at = datetime.now(timezone.utc)
            review.review_note = note
            
            obs = await db.get(Observation, review.observation_id)
            if obs:
                obs.review_status = ReviewStatus.REJECTED
                
            await db.commit()
        return review

    @staticmethod
    async def create_new_tiger_from_review(db: AsyncSession, review_id: str, reviewer: str, note: str = None):
        result = await db.execute(select(ReviewQueue).where(ReviewQueue.review_id == review_id))
        review = result.scalar_one_or_none()
        if not review:
            return None
            
        review.status = QueueStatus.NEW_TIGER
        review.reviewed_by = reviewer
        review.reviewed_at = datetime.now(timezone.utc)
        review.review_note = note
        
        # Create new tiger
        tiger = await TigerService.create_tiger(db, notes=note)
        
        # Update Observation
        obs = await db.get(Observation, review.observation_id)
        if obs:
            obs.tiger_id = tiger.id
            for embedding in list(
                (await db.execute(select(Embedding).where(Embedding.observation_id == obs.id)))
                .scalars()
                .all()
            ):
                embedding.tiger_id = tiger.id
            obs.match_type = MatchType.NEW_INDIVIDUAL
            obs.review_status = ReviewStatus.APPROVED
            await db.commit()
            await TigerService.update_tiger_stats(db, tiger.id)
            
        await db.commit()
        return review
