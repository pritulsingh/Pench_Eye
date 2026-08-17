from sqlalchemy import Column, String, Float, DateTime, Boolean, Text
from sqlalchemy.sql import func
import uuid
from app.core.database import Base
from app.core.types import GUID, JSONType


class Zone(Base):
    """
    Management zone / area of the reserve (core, buffer, corridor, gate area).

    Geometry is stored as a GeoJSON geometry object so the same rows work on
    PostGIS and SQLite. `is_demo` marks simulated boundaries.
    """

    __tablename__ = "zones"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    zone_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    zone_type = Column(String(50), nullable=False, default="core")
    description = Column(Text, nullable=True)
    center_latitude = Column(Float, nullable=True)
    center_longitude = Column(Float, nullable=True)
    area_km2 = Column(Float, nullable=True)
    geometry_json = Column(JSONType, nullable=True)
    style_color = Column(String(20), nullable=True)
    is_demo = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
