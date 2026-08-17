from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from app.schemas.common import PaginatedResponse  # noqa: F401 — re-exported for routes

class TigerCreate(BaseModel):
    name: Optional[str] = None
    sex: str = "unknown"
    notes: Optional[str] = None

class TigerUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None

class TigerResponse(BaseModel):
    id: UUID
    tiger_id: str
    name: Optional[str] = None
    sex: Optional[str] = None
    status: Optional[str] = None
    estimated_age_years: Optional[float] = None
    total_observations: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    notes: Optional[str] = None
    is_demo: bool = False
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class TigerSummary(BaseModel):
    id: UUID
    tiger_id: str
    name: Optional[str] = None
    sex: Optional[str] = None
    status: Optional[str] = None
    total_observations: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    mean_confidence: Optional[float] = None
    camera_count: int = 0
    is_demo: bool = False
    model_config = ConfigDict(from_attributes=True)

class TigerCameraUsage(BaseModel):
    camera_id: str
    camera_name: Optional[str] = None
    detections: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class TigerProfile(TigerResponse):
    mean_confidence: Optional[float] = None
    camera_count: int = 0
    zone_distribution: List[dict] = []
    frequent_cameras: List[TigerCameraUsage] = []
    recent_observations: List[dict] = []
    detections_by_month: List[dict] = []
