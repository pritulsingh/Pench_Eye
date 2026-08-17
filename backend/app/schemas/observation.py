from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID
from app.schemas.common import PaginatedResponse  # noqa: F401 — re-exported for routes

class ObservationResponse(BaseModel):
    id: UUID
    observation_id: str
    tiger_id: Optional[UUID] = None
    tiger_code: Optional[str] = None
    tiger_name: Optional[str] = None
    image_id: Optional[UUID] = None
    image_code: Optional[str] = None
    image_url: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zone: Optional[str] = None
    species: Optional[str] = None
    detection_type: Optional[str] = None
    detection_confidence: Optional[float] = None
    identity_confidence: Optional[float] = None
    match_type: Optional[str] = None
    review_status: Optional[str] = None
    flank_side: Optional[str] = None
    bounding_box_json: Optional[Any] = None
    model_version: Optional[str] = None
    is_demo: bool = False
    model_config = ConfigDict(from_attributes=True)

class ObservationCreate(BaseModel):
    tiger_id: UUID
    image_id: UUID
    camera_id: str

class ObservationSummary(BaseModel):
    observation_id: str
    tiger_id: Optional[UUID] = None

class CandidateMatch(BaseModel):
    tiger_id: str
    score: float
    rank: int
