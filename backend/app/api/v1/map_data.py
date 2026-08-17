from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.map_data import (
    GateResponse,
    MapCamera,
    MapOverview,
    MapSighting,
    MovementTrack,
    ZoneResponse,
)
from app.services.map_service import MapService

router = APIRouter()


@router.get("/overview", response_model=MapOverview)
async def map_overview(
    sighting_limit: int = Query(300, ge=1, le=2000),
    tiger_code: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=3650),
    db: AsyncSession = Depends(get_db),
):
    """Everything the map needs in one round trip."""
    return await MapService.get_overview(
        db, sighting_limit=sighting_limit, tiger_code=tiger_code, days=days
    )


@router.get("/zones", response_model=List[ZoneResponse])
async def list_zones(db: AsyncSession = Depends(get_db)):
    return await MapService.get_zones(db)


@router.get("/zones/geojson")
async def zones_as_geojson(db: AsyncSession = Depends(get_db)):
    """GeoJSON generated only from non-demo zone records in the database."""
    zones = await MapService.get_zones(db)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": zone["geometry_json"],
                "properties": {
                    "zone_code": zone["zone_code"],
                    "name": zone["name"],
                    "zone_type": zone["zone_type"],
                },
            }
            for zone in zones
            if zone.get("geometry_json")
        ],
    }


@router.get("/gates", response_model=List[GateResponse])
async def list_gates():
    # There is no gate table in the current schema. Never substitute static coordinates.
    return []


@router.get("/cameras", response_model=List[MapCamera])
async def map_cameras(db: AsyncSession = Depends(get_db)):
    return await MapService.get_map_cameras(db)


@router.get("/sightings", response_model=List[MapSighting])
async def map_sightings(
    limit: int = Query(400, ge=1, le=2000),
    tiger_code: Optional[str] = None,
    species: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=3650),
    db: AsyncSession = Depends(get_db),
):
    return await MapService.get_map_sightings(
        db, limit=limit, tiger_code=tiger_code, days=days, species=species
    )


@router.get("/movement", response_model=List[MovementTrack])
async def movement_tracks(
    tiger_code: Optional[str] = None,
    max_tigers: int = Query(6, ge=1, le=25),
    db: AsyncSession = Depends(get_db),
):
    return await MapService.get_movement_tracks(db, tiger_code=tiger_code, max_tigers=max_tigers)
