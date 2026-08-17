import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.core.types import GUID


class MLDataset(Base):
    __tablename__ = "ml_datasets"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(String(80), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    source_path = Column(String(500), nullable=True)
    extracted_path = Column(String(500), nullable=True)
    status = Column(String(40), default="uploaded")
    identity_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)
    sequence_count = Column(Integer, default=0)
    train_identities = Column(Integer, default=0)
    train_images = Column(Integer, default=0)
    val_identities = Column(Integer, default=0)
    val_images = Column(Integer, default=0)
    test_identities = Column(Integer, default=0)
    test_images = Column(Integer, default=0)
    min_images_per_identity = Column(Integer, default=0)
    median_images_per_identity = Column(Integer, default=0)
    mean_images_per_identity = Column(Float, default=0.0)
    max_images_per_identity = Column(Integer, default=0)
    corrupted_images = Column(Integer, default=0)
    duplicate_images = Column(Integer, default=0)
    suspected_sequence_leakage = Column(Integer, default=0)
    manifest_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    training_runs = relationship("TrainingRun", back_populates="dataset")
    model_versions = relationship("MLModel", back_populates="dataset")
