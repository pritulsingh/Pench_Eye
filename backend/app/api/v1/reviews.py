from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import PaginatedResponse
from app.schemas.review import (
    NewTigerRequest,
    ReviewApproveRequest,
    ReviewQueueItem,
    ReviewRejectRequest,
    ReviewResponse,
)
from app.services.review_service import ReviewService

router = APIRouter()


@router.get("", response_model=PaginatedResponse[ReviewQueueItem])
async def list_reviews(
    status: Optional[str] = Query("pending"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    try:
        items, total = await ReviewService.list_reviews(db, status=status, skip=skip, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)


@router.get("/{review_id}", response_model=ReviewQueueItem)
async def get_review(review_id: str, db: AsyncSession = Depends(get_db)):
    review = await ReviewService.get_review(db, review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


@router.post("/{review_id}/approve", response_model=ReviewResponse)
async def approve_review(review_id: str, data: ReviewApproveRequest, db: AsyncSession = Depends(get_db)):
    review = await ReviewService.approve_review(db, review_id, data.tiger_id, data.reviewer, data.note)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewResponse(success=True, status="approved", detail=f"Assigned to {data.tiger_id}")


@router.post("/{review_id}/reject", response_model=ReviewResponse)
async def reject_review(review_id: str, data: ReviewRejectRequest, db: AsyncSession = Depends(get_db)):
    review = await ReviewService.reject_review(db, review_id, data.reviewer, data.note)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewResponse(success=True, status="rejected")


@router.post("/{review_id}/new-tiger", response_model=ReviewResponse)
async def new_tiger_review(review_id: str, data: NewTigerRequest, db: AsyncSession = Depends(get_db)):
    review = await ReviewService.create_new_tiger_from_review(db, review_id, data.reviewer, data.note)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewResponse(success=True, status="new_tiger")
