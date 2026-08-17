from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import PaginatedResponse
from app.schemas.image import ImageSummary
from app.schemas.observation import ObservationResponse
from app.schemas.tiger import TigerCreate, TigerProfile, TigerResponse, TigerSummary
from app.services.observation_service import ObservationService
from app.services.tiger_service import TigerService

router = APIRouter()


@router.get("", response_model=PaginatedResponse[TigerSummary])
async def list_tigers(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    items, total = await TigerService.list_tigers(
        db, skip=skip, limit=limit, status=status, search=search
    )
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)


@router.post("", response_model=TigerResponse, status_code=201)
async def create_tiger(data: TigerCreate, db: AsyncSession = Depends(get_db)):
    return await TigerService.create_tiger(db, name=data.name, sex=data.sex, notes=data.notes)


@router.get("/{tiger_id}", response_model=TigerProfile)
async def get_tiger(tiger_id: str, db: AsyncSession = Depends(get_db)):
    profile = await TigerService.get_tiger_profile(db, tiger_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Tiger not found")
    return profile


@router.get("/{tiger_id}/observations", response_model=PaginatedResponse[ObservationResponse])
async def get_tiger_observations(
    tiger_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    tiger = await TigerService.get_tiger(db, tiger_id)
    if not tiger:
        raise HTTPException(status_code=404, detail="Tiger not found")
    items, total = await ObservationService.query_observations(
        db, skip=skip, limit=limit, tiger_code=tiger_id
    )
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)


@router.get("/{tiger_id}/gallery", response_model=List[ImageSummary])
async def get_tiger_gallery(
    tiger_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    tiger = await TigerService.get_tiger(db, tiger_id)
    if not tiger:
        raise HTTPException(status_code=404, detail="Tiger not found")

    from app.models.camera_station import CameraStation
    from app.models.image import Image
    from app.models.observation import Observation

    rows = (
        await db.execute(
            select(Image, Observation, CameraStation)
            .join(Observation, Observation.image_id == Image.id)
            .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
            .where(Observation.tiger_id == tiger.id)
            .order_by(Observation.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )
    ).all()

    return [
        {
            "id": img.id,
            "image_id": img.image_id,
            "original_filename": img.original_filename,
            "camera_id": img.camera_id,
            "camera_name": cam.name if cam else None,
            "status": img.status.value if img.status else None,
            "blank_probability": img.blank_probability,
            "triage_reason": img.triage_reason,
            "timestamp": img.timestamp or img.created_at,
            "url": f"/api/v1/images/{img.image_id}/file",
            "species": obs.species,
            "tiger_code": tiger.tiger_id,
            "tiger_name": tiger.name,
            "identity_confidence": obs.identity_confidence,
            "is_demo": bool(img.is_demo),
        }
        for img, obs, cam in rows
    ]
