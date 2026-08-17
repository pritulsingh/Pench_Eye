import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.core.types import GUID


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(80), unique=True, nullable=False, index=True)
    dataset_id = Column(String(80), ForeignKey("ml_datasets.dataset_id"), nullable=False, index=True)
    status = Column(String(40), default="queued")
    backbone = Column(String(80), default="resnet50")
    epochs = Column(Integer, default=1)
    batch_size = Column(Integer, default=32)
    learning_rate = Column(Float, default=0.0005)
    current_epoch = Column(Integer, default=0)
    train_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    rank1 = Column(Float, nullable=True)
    rank5 = Column(Float, nullable=True)
    map_value = Column(Float, nullable=True, name="map")
    checkpoint_path = Column(String(500), nullable=True)
    model_version = Column(String(80), nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    dataset = relationship("MLDataset", back_populates="training_runs")
    models = relationship("MLModel", back_populates="training_run")
