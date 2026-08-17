from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.camera_station import CameraStation
from app.models.image import Image, ImageStatus
from app.models.observation import Observation
from app.models.tiger import Tiger
from app.schemas.common import PaginatedResponse
from app.schemas.image import (
    BatchUploadResponse,
    ImageResponse,
    ImageSummary,
    ImageUploadResponse,
)
from app.services.pipeline_service import ImageValidationError, PipelineService
from app.services.storage_service import storage_service

router = APIRouter()


@router.post("/upload", response_model=ImageUploadResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    captured_at: Optional[datetime] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    try:
        result = await PipelineService.process_image(
            db, content=content, filename=file.filename, camera_id=camera_id, captured_at=captured_at
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@router.post("/batch", response_model=BatchUploadResponse, status_code=201)
async def upload_batch(
    files: List[UploadFile] = File(...),
    camera_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    results: List[dict] = []
    failed = 0
    blank = 0
    for file in files:
        content = await file.read()
        try:
            result = await PipelineService.process_image(
                db, content=content, filename=file.filename, camera_id=camera_id
            )
        except ImageValidationError as exc:
            failed += 1
            results.append(
                {
                    "image_id": file.filename or "unknown",
                    "status": "rejected",
                    "is_blank": False,
                    "triage_reason": str(exc),
                    "message": str(exc),
                }
            )
            continue
        if result.get("is_blank"):
            blank += 1
        results.append(result)

    processed = len(results) - failed
    return BatchUploadResponse(
        total=len(files),
        processed=processed,
        blank_count=blank,
        subject_count=max(0, processed - blank),
        failed_count=failed,
        images=results,
    )


@router.get("", response_model=PaginatedResponse[ImageSummary])
async def list_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    species: Optional[str] = None,
    tiger_code: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    base = (
        select(Image, Observation, Tiger, CameraStation)
        .outerjoin(Observation, Observation.image_id == Image.id)
        .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
        .outerjoin(CameraStation, CameraStation.camera_id == Image.camera_id)
        .where(Image.is_demo.is_(False))
    )
    count_query = (
        select(func.count(func.distinct(Image.id)))
        .outerjoin(Observation, Observation.image_id == Image.id)
        .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
        .where(Image.is_demo.is_(False))
    )

    conditions = []
    if camera_id:
        conditions.append(Image.camera_id == camera_id)
    if status:
        try:
            conditions.append(Image.status == ImageStatus(status))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid status '{status}'") from exc
    if species:
        conditions.append(Observation.species == species)
    if tiger_code:
        conditions.append(Tiger.tiger_id == tiger_code)
    if date_from:
        conditions.append(Image.timestamp >= date_from)
    if date_to:
        conditions.append(Image.timestamp <= date_to)

    for cond in conditions:
        base = base.where(cond)
        count_query = count_query.where(cond)

    total = (await db.execute(count_query)).scalar_one_or_none() or 0
    rows = (
        await db.execute(
            base.order_by(Image.created_at.desc()).offset(skip).limit(limit)
        )
    ).all()

    items = [
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
            "species": obs.species if obs else None,
            "tiger_code": tiger.tiger_id if tiger else None,
            "tiger_name": tiger.name if tiger else None,
            "identity_confidence": obs.identity_confidence if obs else None,
            "is_demo": bool(img.is_demo),
        }
        for img, obs, tiger, cam in rows
    ]
    return PaginatedResponse(items=items, total=int(total), page=skip // limit + 1, size=limit)


async def _get_image_or_404(db: AsyncSession, image_id: str) -> Image:
    image = (
        await db.execute(select(Image).where(Image.image_id == image_id))
    ).scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(image_id: str, db: AsyncSession = Depends(get_db)):
    image = await _get_image_or_404(db, image_id)
    payload = ImageResponse.model_validate(image)
    payload.url = f"/api/v1/images/{image.image_id}/file"
    return payload


@router.get("/{image_id}/file")
async def get_image_file(image_id: str, db: AsyncSession = Depends(get_db)):
    image = await _get_image_or_404(db, image_id)
    key = image.storage_key or image.quarantine_key
    if not key:
        raise HTTPException(status_code=404, detail="No stored file for this image")
    try:
        content = await storage_service.download_image(key)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Stored file unavailable") from exc
    return Response(content=content, media_type="image/jpeg")


@router.post("/{image_id}/restore", response_model=ImageResponse)
async def restore_image(image_id: str, db: AsyncSession = Depends(get_db)):
    image = await _get_image_or_404(db, image_id)
    image.status = ImageStatus.TRIAGED
    await db.commit()
    await db.refresh(image)
    payload = ImageResponse.model_validate(image)
    payload.url = f"/api/v1/images/{image.image_id}/file"
    return payload


@router.post("/{image_id}/delete")
async def delete_image(image_id: str, db: AsyncSession = Depends(get_db)):
    image = await _get_image_or_404(db, image_id)
    image.status = ImageStatus.DELETED
    await db.commit()
    return {"status": "deleted", "image_id": image_id}
