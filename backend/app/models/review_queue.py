from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.core.types import GUID, JSONType

class QueueStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEW_TIGER = "new_tiger"

class ReviewQueue(Base):
    __tablename__ = "review_queue"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    review_id = Column(String(100), unique=True, nullable=False, index=True)
    observation_id = Column(GUID(), ForeignKey("observations.id"), nullable=False)
    candidate_tiger_ids = Column(JSONType, nullable=True)
    candidate_scores = Column(JSONType, nullable=True)
    alternative_candidates_json = Column(JSONType, nullable=True)
    status = Column(SQLEnum(QueueStatus), default=QueueStatus.PENDING)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    observation = relationship("Observation", back_populates="review_queue")
