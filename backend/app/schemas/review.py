from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID
from enum import Enum
from app.schemas.common import PaginatedResponse  # noqa: F401 — re-exported for routes


class ReviewStatusEnum(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEW_TIGER = "new_tiger"


class ReviewCandidate(BaseModel):
    tiger_id: str
    tiger_code: Optional[str] = None
    tiger_name: Optional[str] = None
    score: Optional[float] = None


class ReviewQueueItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    review_id: str
    observation_id: Optional[str] = None
    observation_code: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    image_id: Optional[str] = None
    image_url: Optional[str] = None
    status: Optional[str] = None
    candidates: List[ReviewCandidate] = []
    review_note: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class ReviewQueueResponse(ReviewQueueItem):
    pass


class ReviewApproveRequest(BaseModel):
    tiger_id: str
    reviewer: str = "demo-reviewer"
    note: Optional[str] = None


class ReviewRejectRequest(BaseModel):
    reviewer: str = "demo-reviewer"
    note: Optional[str] = None


class NewTigerRequest(BaseModel):
    reviewer: str = "demo-reviewer"
    note: Optional[str] = None
    name: Optional[str] = None
    sex: Optional[str] = None


class ReviewResponse(BaseModel):
    success: bool
    status: str
    detail: Optional[str] = None