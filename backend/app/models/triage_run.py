from sqlalchemy import Column, String, Integer, Float, DateTime, Enum as SQLEnum, Text, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID

class RunStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TriageRun(Base):
    __tablename__ = "triage_runs"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    run_number = Column(Integer, autoincrement=True, unique=True)
    status = Column(SQLEnum(RunStatus), default=RunStatus.RUNNING)
    total_images = Column(Integer, default=0)
    blank_count = Column(Integer, default=0)
    subject_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    quarantined_count = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    original_storage_bytes = Column(BigInteger, default=0)
    active_storage_bytes = Column(BigInteger, default=0)
    storage_saved_bytes = Column(BigInteger, default=0)
    triage_processing_seconds = Column(Float, default=0.0)
    blank_threshold_used = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    images = relationship("Image", back_populates="triage_run", lazy="dynamic")
