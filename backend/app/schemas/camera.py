from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from app.schemas.common import PaginatedResponse  # noqa: F401 — re-exported for routes

class CameraStationBase(BaseModel):
    camera_id: str
    name: str
    zone: str
    zone_code: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    altitude_m: Optional[float] = None
    status: str = "active"
    description: Optional[str] = None

class CameraStationCreate(CameraStationBase):
    pass

class CameraStationUpdate(BaseModel):
    name: Optional[str] = None
    zone: Optional[str] = None
    zone_code: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    altitude_m: Optional[float] = None

class CameraStationResponse(CameraStationBase):
    id: UUID
    battery_percent: Optional[int] = None
    installed_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    last_detection_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_demo: bool = False
    model_config = ConfigDict(from_attributes=True)

class CameraStationSummary(BaseModel):
    id: UUID
    camera_id: str
    name: str
    zone: Optional[str] = None
    zone_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Optional[str] = None
    marker_state: str = "active"
    battery_percent: Optional[int] = None
    last_active_at: Optional[datetime] = None
    last_detection_at: Optional[datetime] = None
    observation_count: int = 0
    image_count: int = 0
    open_alert_count: int = 0
    is_demo: bool = False
    model_config = ConfigDict(from_attributes=True)

class CameraTimelinePoint(BaseModel):
    date: str
    detections: int

class CameraGalleryItem(BaseModel):
    image_id: str
    url: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: Optional[str] = None
    blank_probability: Optional[float] = None
    tiger_code: Optional[str] = None
    identity_confidence: Optional[float] = None

class CameraRecentDetection(BaseModel):
    observation_id: str
    timestamp: Optional[datetime] = None
    tiger_code: Optional[str] = None
    tiger_name: Optional[str] = None
    species: Optional[str] = None
    identity_confidence: Optional[float] = None
    detection_confidence: Optional[float] = None
    image_id: Optional[str] = None

class CameraDetail(CameraStationResponse):
    marker_state: str = "active"
    observation_count: int = 0
    image_count: int = 0
    unique_tigers: int = 0
    open_alert_count: int = 0
    recent_detections: List[CameraRecentDetection] = []
    recent_images: List[CameraGalleryItem] = []
    detection_timeline: List[CameraTimelinePoint] = []
