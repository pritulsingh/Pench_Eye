from sqlalchemy import Boolean, Column, String, Float, DateTime, Enum as SQLEnum, Text, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID

class CameraStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"

class CameraZone(str, enum.Enum):
    CORE = "core"
    BUFFER = "buffer"
    VILLAGE_ADJACENT = "village_adjacent"

class CameraStation(Base):
    __tablename__ = "camera_stations"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    camera_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    zone = Column(SQLEnum(CameraZone), nullable=False, default=CameraZone.CORE)
    zone_code = Column(String(50), nullable=True, index=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude_m = Column(Float, nullable=True)
    status = Column(SQLEnum(CameraStatus), nullable=False, default=CameraStatus.ACTIVE)
    description = Column(Text, nullable=True)
    battery_percent = Column(Integer, nullable=True)
    installed_at = Column(DateTime(timezone=True), nullable=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    last_detection_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    # Synthetic stations remain available to explicit demo workflows only.
    is_demo = Column(Boolean, nullable=False, default=False)
    
    images = relationship("Image", back_populates="camera", lazy="dynamic")
    observations = relationship("Observation", back_populates="camera", lazy="dynamic")
    alerts = relationship("Alert", back_populates="camera", lazy="dynamic")
