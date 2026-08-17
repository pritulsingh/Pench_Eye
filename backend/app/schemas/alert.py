from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import UUID


class AlertResponse(BaseModel):
    id: UUID
    alert_id: str
    alert_type: str
    severity: str
    status: str
    title: str
    message: str
    camera_id: Optional[str] = None
    tiger_id: Optional[UUID] = None
    zone_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    detail_json: Optional[Dict[str, Any]] = None
    is_demo: bool = False
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


class AlertUpdate(BaseModel):
    status: str
    actor: Optional[str] = None


class AlertSummary(BaseModel):
    open: int = 0
    acknowledged: int = 0
    resolved: int = 0
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0


class AlertEvaluationResult(BaseModel):
    created: int
    summary: AlertSummary
