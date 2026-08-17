from sqlalchemy import Column, String, Float, DateTime, Enum as SQLEnum, Text, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID, JSONType


class AlertSeverity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, enum.Enum):
    HIGH_PRIORITY_DETECTION = "high_priority_detection"
    CAMERA_OFFLINE = "camera_offline"
    UNUSUAL_MOVEMENT = "unusual_movement"
    HIGH_ACTIVITY = "high_activity"
    LOW_CONFIDENCE = "low_confidence"


class AlertStatus(str, enum.Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    alert_type = Column(SQLEnum(AlertType), nullable=False, index=True)
    severity = Column(SQLEnum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM, index=True)
    status = Column(SQLEnum(AlertStatus), nullable=False, default=AlertStatus.OPEN, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    camera_id = Column(String(50), ForeignKey("camera_stations.camera_id"), nullable=True, index=True)
    tiger_id = Column(GUID(), ForeignKey("tigers.id"), nullable=True)
    observation_id = Column(GUID(), ForeignKey("observations.id"), nullable=True)
    zone_code = Column(String(50), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    detail_json = Column(JSONType, nullable=True)
    # Deduplication key so rule evaluation is idempotent across runs.
    dedupe_key = Column(String(200), unique=True, nullable=True, index=True)
    is_demo = Column(Boolean, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    camera = relationship("CameraStation", back_populates="alerts")
    tiger = relationship("Tiger")
    observation = relationship("Observation")


Index("ix_alerts_status_created", Alert.status, Alert.created_at)
