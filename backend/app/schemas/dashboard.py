from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class RecentIdentification(BaseModel):
    observation_id: str
    tiger_id: Optional[str] = None
    tiger_code: Optional[str] = None
    tiger_name: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    timestamp: Optional[str] = None
    identity_confidence: Optional[float] = None
    match_type: Optional[str] = None
    image_id: Optional[str] = None
    image_url: Optional[str] = None


class LabeledCount(BaseModel):
    label: str
    count: int


class DetectionTrendPoint(BaseModel):
    date: str
    detections: int
    blanks: int = 0


class CameraHealth(BaseModel):
    active: int = 0
    offline: int = 0
    maintenance: int = 0
    warning: int = 0
    total: int = 0


class AlertPreview(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    status: str
    title: str
    message: str
    camera_id: Optional[str] = None
    zone_code: Optional[str] = None
    created_at: Optional[str] = None


class DashboardStats(BaseModel):
    total_images: int = 0
    blank_images: int = 0
    subject_images: int = 0
    quarantined_images: int = 0
    total_tigers: int = 0
    active_tigers: int = 0
    total_observations: int = 0
    detections_last_7_days: int = 0
    total_cameras: int = 0
    active_cameras: int = 0
    pending_reviews: int = 0
    open_alerts: int = 0
    storage_saved_bytes: int = 0
    total_storage_bytes: int = 0
    mean_identity_confidence: Optional[float] = None
    demo_mode: bool = True
    data_source: str = "demo"
    camera_health: CameraHealth = CameraHealth()
    recent_identifications: List[RecentIdentification] = []
    recent_alerts: List[AlertPreview] = []
    recent_images: List[Dict[str, Any]] = []
    images_by_camera: List[LabeledCount] = []
    detections_by_zone: List[LabeledCount] = []
    detection_trend: List[DetectionTrendPoint] = []
    most_active_tigers: List[LabeledCount] = []