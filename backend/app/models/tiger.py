from sqlalchemy import Column, String, Float, DateTime, Enum as SQLEnum, Text, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID

class TigerSex(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class TigerStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DECEASED = "deceased"
    UNKNOWN = "unknown"

class Tiger(Base):
    __tablename__ = "tigers"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    tiger_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)
    sex = Column(SQLEnum(TigerSex), default=TigerSex.UNKNOWN)
    estimated_age_years = Column(Float, nullable=True)
    status = Column(SQLEnum(TigerStatus), default=TigerStatus.ACTIVE)
    total_observations = Column(Integer, default=0)
    first_seen = Column(DateTime(timezone=True), nullable=True)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    representative_image_id = Column(GUID(), ForeignKey("images.id"), nullable=True)
    notes = Column(Text, nullable=True)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    observations = relationship("Observation", back_populates="tiger")
    embeddings = relationship("Embedding", back_populates="tiger")
    registration_sessions = relationship("TigerRegistrationSession", back_populates="tiger")
    movements = relationship("TigerMovement", back_populates="tiger")
    observed_areas = relationship("TigerObservedArea", back_populates="tiger")
