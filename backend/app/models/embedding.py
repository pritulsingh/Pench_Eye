from sqlalchemy import Boolean, Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.core.database import Base
from app.core.types import GUID, VectorType
from app.core.config import settings

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    embedding_id = Column(String(100), unique=True, nullable=False, index=True)
    observation_id = Column(GUID(), ForeignKey("observations.id"), nullable=False)
    tiger_id = Column(GUID(), ForeignKey("tigers.id"), nullable=True)
    embedding = Column(VectorType(settings.EMBEDDING_DIM), nullable=False)
    model_version = Column(String(100), nullable=True)
    flank_side = Column(String(20), nullable=True)
    is_demo = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    observation = relationship("Observation", back_populates="embeddings")
    tiger = relationship("Tiger", back_populates="embeddings")
