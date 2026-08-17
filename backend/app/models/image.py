from sqlalchemy import Column, String, Float, DateTime, Enum as SQLEnum, Text, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID

class ImageStatus(str, enum.Enum):
    PENDING = "pending"
    TRIAGED = "triaged"
    QUARANTINED = "quarantined"
    DELETED = "deleted"
    PROCESSED = "processed"

class SourceType(str, enum.Enum):
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"
    STREAM_FRAME = "stream_frame"

class ProcessingStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Image(Base):
    __tablename__ = "images"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    image_id = Column(String(100), unique=True, nullable=False, index=True)
    storage_key = Column(String(500), nullable=True)
    quarantine_key = Column(String(500), nullable=True)
    original_filename = Column(String(500), nullable=True)
    camera_id = Column(String(50), ForeignKey("camera_stations.camera_id"), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    file_size_bytes = Column(Integer, default=0)
    width_px = Column(Integer, default=0)
    height_px = Column(Integer, default=0)
    sha256_hash = Column(String(64), nullable=True, index=True)
    perceptual_hash = Column(String(64), nullable=True)
    quality_score = Column(Float, nullable=True)
    blank_probability = Column(Float, nullable=True)
    blank_threshold_used = Column(Float, nullable=True)
    triage_reason = Column(Text, nullable=True)
    status = Column(SQLEnum(ImageStatus), default=ImageStatus.PENDING)
    source_type = Column(SQLEnum(SourceType), default=SourceType.IMAGE)
    source_filename = Column(String(500), nullable=True)
    frame_number = Column(Integer, nullable=True)
    video_id = Column(String(100), nullable=True)
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.QUEUED)
    error_message = Column(Text, nullable=True)
    triage_run_id = Column(GUID(), ForeignKey("triage_runs.id"), nullable=True)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    camera = relationship("CameraStation", back_populates="images")
    triage_run = relationship("TriageRun", back_populates="images")
    observations = relationship("Observation", back_populates="image", lazy="dynamic")
