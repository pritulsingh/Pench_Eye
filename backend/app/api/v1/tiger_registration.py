from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.camera_station import CameraStation
from app.models.tiger import Tiger, TigerStatus
from app.models.tiger_registration import TigerRegistrationSession
from app.services.tiger_service import TigerService

router = APIRouter()


async def _get_camera(db: AsyncSession, camera_id: str) -> CameraStation:
    camera = (
        await db.execute(select(CameraStation).where(CameraStation.camera_id == camera_id))
    ).scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.post("/sessions", status_code=201)
async def create_registration_session(payload: Dict[str, Any], db: AsyncSession = Depends(get_db)):
    camera_id = payload.get("camera_id") or "CAM-001"
    camera = await _get_camera(db, camera_id)
    tiger_code = payload.get("tiger_code") or f"TGR-{len(payload.get('images', [])) or 1:03d}"
    existing = (await db.execute(select(Tiger).where(Tiger.tiger_id == tiger_code))).scalar_one_or_none()
    tiger = existing or await TigerService.create_tiger(db, name=tiger_code, sex="unknown", tiger_id=tiger_code)

    registration_id = f"REG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{len(payload.get('images', [])) or 1:03d}"
    session = TigerRegistrationSession(
        registration_id=registration_id,
        tiger_code=tiger_code,
        tiger_id=tiger.id,
        camera_id=camera.camera_id,
        capture_timestamp=payload.get("capture_timestamp"),
        latitude=float(payload.get("latitude") or camera.latitude or 0.0),
        longitude=float(payload.get("longitude") or camera.longitude or 0.0),
        zone=str(payload.get("zone") or camera.zone.value if camera.zone else "core"),
        quality=float(payload.get("quality") or 0.85),
        embedding=payload.get("embedding"),
        image_payload=payload.get("images") or [],
        status="draft",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return {
        "registration_id": session.registration_id,
        "tiger_code": session.tiger_code,
        "tiger_id": tiger.tiger_id,
        "camera_id": session.camera_id,
        "latitude": session.latitude,
        "longitude": session.longitude,
        "zone": session.zone,
        "quality": session.quality,
        "status": session.status,
        "images": session.image_payload or [],
    }


@router.get("/sessions/{registration_id}")
async def get_registration_session(registration_id: str, db: AsyncSession = Depends(get_db)):
    session = (
        await db.execute(select(TigerRegistrationSession).where(TigerRegistrationSession.registration_id == registration_id))
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Registration session not found")
    return {
        "registration_id": session.registration_id,
        "tiger_code": session.tiger_code,
        "camera_id": session.camera_id,
        "latitude": session.latitude,
        "longitude": session.longitude,
        "zone": session.zone,
        "status": session.status,
        "quality": session.quality,
        "images": session.image_payload or [],
    }


@router.post("/sessions/{registration_id}/images")
async def add_registration_images(registration_id: str, payload: Dict[str, Any], db: AsyncSession = Depends(get_db)):
    session = (
        await db.execute(select(TigerRegistrationSession).where(TigerRegistrationSession.registration_id == registration_id))
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Registration session not found")
    current = list(session.image_payload or [])
    current.extend(payload.get("images") or [])
    session.image_payload = current
    session.status = "ready"
    await db.commit()
    return {"registration_id": registration_id, "status": session.status, "image_count": len(current)}


@router.post("/sessions/{registration_id}/finalize")
async def finalize_registration_session(registration_id: str, db: AsyncSession = Depends(get_db)):
    session = (
        await db.execute(select(TigerRegistrationSession).where(TigerRegistrationSession.registration_id == registration_id))
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Registration session not found")
    session.status = "finalized"
    await db.commit()
    return {
        "registration_id": registration_id,
        "tiger_code": session.tiger_code,
        "camera_id": session.camera_id,
        "status": "finalized",
        "latitude": session.latitude,
        "longitude": session.longitude,
    }
