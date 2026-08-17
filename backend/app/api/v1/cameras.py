from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.camera_station import CameraStation
from app.schemas.camera import (
    CameraDetail,
    CameraStationCreate,
    CameraStationResponse,
    CameraStationSummary,
    CameraStationUpdate,
)
from app.schemas.common import PaginatedResponse
from app.schemas.observation import ObservationResponse
from app.services.camera_service import CameraService
from app.services.observation_service import ObservationService

router = APIRouter()


@router.get("", response_model=PaginatedResponse[CameraStationSummary])
async def list_cameras(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    zone: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    items, total = await CameraService.list_cameras(
        db, skip=skip, limit=limit, zone=zone, status=status, search=search
    )
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)


@router.post("", response_model=CameraStationResponse, status_code=201)
async def create_camera(data: CameraStationCreate, db: AsyncSession = Depends(get_db)):
    existing = (
        await db.execute(select(CameraStation).where(CameraStation.camera_id == data.camera_id))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="camera_id already exists")
    payload = data.model_dump()
    # A freshly-registered camera is considered online until it misses check-ins,
    # so default last_active_at to now when the client didn't provide one.
    if payload.get("last_active_at") is None:
        payload["last_active_at"] = datetime.now(timezone.utc)
    camera = CameraStation(**payload)
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return camera


@router.get("/{camera_id}", response_model=CameraDetail)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    detail = await CameraService.get_camera_detail(db, camera_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Camera not found")
    return detail


@router.patch("/{camera_id}", response_model=CameraStationResponse)
async def update_camera(
    camera_id: str, data: CameraStationUpdate, db: AsyncSession = Depends(get_db)
):
    camera = (
        await db.execute(select(CameraStation).where(CameraStation.camera_id == camera_id))
    ).scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    # exclude_unset preserves omitted fields while still allowing an explicit
    # null to clear an incorrect coordinate or altitude.
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    await db.commit()
    await db.refresh(camera)
    return camera


@router.get("/{camera_id}/observations", response_model=PaginatedResponse[ObservationResponse])
async def get_camera_observations(
    camera_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    items, total = await ObservationService.query_observations(
        db, skip=skip, limit=limit, camera_id=camera_id
    )
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)
