from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from enum import Enum
from app.schemas.common import PaginatedResponse  # noqa: F401 — re-exported for routes


class ImageStatusEnum(str, Enum):
    PENDING = "pending"
    TRIAGED = "triaged"
    QUARANTINED = "quarantined"
    DELETED = "deleted"
    PROCESSED = "processed"


class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    image_id: str
    original_filename: Optional[str] = None
    camera_id: Optional[str] = None
    status: Optional[str] = None
    blank_probability: Optional[float] = None
    quality_score: Optional[float] = None
    triage_reason: Optional[str] = None
    file_size_bytes: Optional[int] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    sha256_hash: Optional[str] = None
    timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None
    is_demo: bool = False
    url: Optional[str] = None


class ImageSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    image_id: str
    original_filename: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    status: Optional[str] = None
    blank_probability: Optional[float] = None
    triage_reason: Optional[str] = None
    timestamp: Optional[datetime] = None
    url: Optional[str] = None
    species: Optional[str] = None
    tiger_code: Optional[str] = None
    tiger_name: Optional[str] = None
    identity_confidence: Optional[float] = None
    is_demo: bool = False


class ImageUploadResponse(BaseModel):
    image_id: str
    status: str
    blank_probability: Optional[float] = None
    is_blank: bool = False
    triage_reason: Optional[str] = None
    triage_stage: Optional[str] = None
    reason: Optional[str] = None
    observation_id: Optional[str] = None
    tiger_code: Optional[str] = None
    similarity: Optional[float] = None
    decision: Optional[str] = None
    candidate_tiger: Optional[str] = None
    species: Optional[str] = None
    detection_confidence: Optional[float] = None
    alerts_created: int = 0
    megadescriptor_ran: bool = False
    raw_detections: Optional[list] = None
    identity_error: Optional[str] = None
    message: str = "Image processed successfully"


class BatchUploadResponse(BaseModel):
    total: int
    processed: int
    blank_count: int
    subject_count: int
    failed_count: int
    images: List[ImageUploadResponse] = []