import uuid
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, JSON, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.core.types import GUID


class TigerRegistrationSession(Base):
    __tablename__ = "tiger_registration_sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    registration_id = Column(String(80), unique=True, nullable=False, index=True)
    tiger_id = Column(GUID(), ForeignKey("tigers.id"), nullable=True, index=True)
    tiger_code = Column(String(80), nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    capture_timestamp = Column(DateTime(timezone=True), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    zone = Column(String(40), nullable=True)
    status = Column(String(40), default="draft")
    quality = Column(Float, default=0.0)
    embedding = Column(JSON, nullable=True)
    image_payload = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    tiger = relationship("Tiger", back_populates="registration_sessions")


class TigerMovement(Base):
    __tablename__ = "tiger_movements"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    tiger_id = Column(GUID(), ForeignKey("tigers.id"), nullable=False, index=True)
    from_camera = Column(String(80), nullable=True)
    to_camera = Column(String(80), nullable=True)
    from_timestamp = Column(DateTime(timezone=True), nullable=True)
    to_timestamp = Column(DateTime(timezone=True), nullable=True)
    distance_km = Column(Float, default=0.0)
    elapsed_minutes = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    tiger = relationship("Tiger", back_populates="movements")


class TigerObservedArea(Base):
    __tablename__ = "tiger_observed_areas"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    tiger_id = Column(GUID(), ForeignKey("tigers.id"), nullable=False, index=True)
    geometry_json = Column(JSON, nullable=False)
    calculation_method = Column(String(80), default="convex_hull")
    camera_count = Column(Integer, default=0)
    observation_count = Column(Integer, default=0)
    first_observation = Column(DateTime(timezone=True), nullable=True)
    last_observation = Column(DateTime(timezone=True), nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(Text, nullable=True)
    tiger = relationship("Tiger", back_populates="observed_areas")
