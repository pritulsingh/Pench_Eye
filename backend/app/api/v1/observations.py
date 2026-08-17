from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import PaginatedResponse
from app.schemas.observation import ObservationResponse
from app.services.observation_service import ObservationService

router = APIRouter()


@router.get("", response_model=PaginatedResponse[ObservationResponse])
async def list_observations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    tiger_code: Optional[str] = None,
    camera_id: Optional[str] = None,
    zone: Optional[str] = None,
    species: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    days: Optional[int] = Query(None, ge=1, le=3650),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    items, total = await ObservationService.query_observations(
        db,
        skip=skip,
        limit=limit,
        tiger_code=tiger_code,
        camera_id=camera_id,
        zone=zone,
        species=species,
        min_confidence=min_confidence,
        days=days,
        date_from=date_from,
        date_to=date_to,
    )
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)


@router.get("/{observation_id}", response_model=ObservationResponse)
async def get_observation(observation_id: str, db: AsyncSession = Depends(get_db)):
    obs = await ObservationService.get_observation_detail(db, observation_id)
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found")
    return obs
