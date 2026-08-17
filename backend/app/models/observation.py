from sqlalchemy import Column, String, Float, DateTime, Enum as SQLEnum, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID, JSONType
from .camera_station import CameraZone

class MatchType(str, enum.Enum):
    AUTO_MATCH = "auto_match"
    HUMAN_VERIFIED = "human_verified"
    NEW_INDIVIDUAL = "new_individual"
    DEMO = "demo"

class ReviewStatus(str, enum.Enum):
    APPROVED = "approved"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"

class FlankSide(str, enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"

class DetectionType(str, enum.Enum):
    TIGER = "tiger"
    OTHER_WILDLIFE = "other_wildlife"
    HUMAN = "human"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"

class Observation(Base):
    __tablename__ = "observations"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    observation_id = Column(String(100), unique=True, nullable=False, index=True)
    tiger_id = Column(GUID(), ForeignKey("tigers.id"), nullable=True, index=True)
    image_id = Column(GUID(), ForeignKey("images.id"), nullable=False)
    camera_id = Column(String(50), ForeignKey("camera_stations.camera_id"), nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=True, index=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    zone = Column(SQLEnum(CameraZone), nullable=True)
    species = Column(String(80), nullable=True, default="tiger", index=True)
    detection_type = Column(SQLEnum(DetectionType), nullable=True, default=DetectionType.TIGER)
    detection_confidence = Column(Float, nullable=True)
    identity_confidence = Column(Float, nullable=True)
    match_type = Column(SQLEnum(MatchType), nullable=True)
    review_status = Column(SQLEnum(ReviewStatus), nullable=True)
    flank_side = Column(SQLEnum(FlankSide), nullable=True)
    bounding_box_json = Column(JSONType, nullable=True)
    model_version = Column(String(100), nullable=True)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    tiger = relationship("Tiger", back_populates="observations")
    image = relationship("Image", back_populates="observations")
    camera = relationship("CameraStation", back_populates="observations")
    review_queue = relationship("ReviewQueue", back_populates="observation", uselist=False)
    embeddings = relationship("Embedding", back_populates="observation")

Index("ix_observations_tiger_timestamp", Observation.tiger_id, Observation.timestamp)
