from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID


class ZoneResponse(BaseModel):
    zone_code: str
    name: str
    zone_type: str
    description: Optional[str] = None
    center_latitude: Optional[float] = None
    center_longitude: Optional[float] = None
    area_km2: Optional[float] = None
    style_color: Optional[str] = None
    geometry_json: Optional[Dict[str, Any]] = None
    camera_count: int = 0
    observation_count: int = 0
    is_demo: bool = False
    model_config = ConfigDict(from_attributes=True)


class GateResponse(BaseModel):
    code: str
    name: str
    latitude: float
    longitude: float
    gate_type: str


class MapCamera(BaseModel):
    camera_id: str
    name: str
    zone: Optional[str] = None
    zone_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Optional[str] = None
    marker_state: str
    battery_percent: Optional[int] = None
    last_active_at: Optional[datetime] = None
    last_detection_at: Optional[datetime] = None
    observation_count: int = 0
    open_alert_count: int = 0
    is_demo: bool = False


class MapSighting(BaseModel):
    observation_id: str
    tiger_code: Optional[str] = None
    tiger_name: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zone: Optional[str] = None
    species: Optional[str] = None
    detection_type: Optional[str] = None
    identity_confidence: Optional[float] = None
    detection_confidence: Optional[float] = None
    image_id: Optional[str] = None
    image_url: Optional[str] = None
    is_demo: bool = False


class MovementLeg(BaseModel):
    from_camera_id: Optional[str] = None
    from_camera_name: Optional[str] = None
    from_latitude: Optional[float] = None
    from_longitude: Optional[float] = None
    from_timestamp: Optional[datetime] = None
    to_camera_id: Optional[str] = None
    to_camera_name: Optional[str] = None
    to_latitude: Optional[float] = None
    to_longitude: Optional[float] = None
    to_timestamp: Optional[datetime] = None
    distance_km: float = 0.0
    hours_elapsed: float = 0.0


class MovementPoint(BaseModel):
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[datetime] = None


class MovementTrack(BaseModel):
    tiger_code: str
    tiger_name: Optional[str] = None
    observations: List[MovementPoint] = []
    legs: List[MovementLeg] = []
    total_distance_km: float = 0.0
    sighting_count: int = 0


class MapOverview(BaseModel):
    center: Optional[List[float]] = None
    bounds: List[List[float]]
    data_source: str
    disclaimer: str
    zones: List[ZoneResponse] = []
    gates: List[GateResponse] = []
    cameras: List[MapCamera] = []
    sightings: List[MapSighting] = []
    tracks: List[MovementTrack] = []
