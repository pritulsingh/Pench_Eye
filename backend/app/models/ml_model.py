import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.core.types import GUID


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    model_version = Column(String(80), unique=True, nullable=False, index=True)
    model_type = Column(String(40), default="reid")
    backbone = Column(String(80), default="resnet50")
    embedding_dimension = Column(Integer, default=512)
    checkpoint_path = Column(String(500), nullable=False)
    dataset_id = Column(String(80), ForeignKey("ml_datasets.dataset_id"), nullable=True, index=True)
    training_run_id = Column(String(80), ForeignKey("training_runs.run_id"), nullable=True, index=True)
    rank1 = Column(Float, nullable=True)
    rank5 = Column(Float, nullable=True)
    rank10 = Column(Float, nullable=True)
    map = Column(Float, nullable=True)
    status = Column(String(40), default="trained")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    activated_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("MLDataset", back_populates="model_versions")
    training_run = relationship("TrainingRun", back_populates="models")
