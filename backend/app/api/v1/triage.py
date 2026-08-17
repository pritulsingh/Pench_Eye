from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.camera_station import CameraStation
from app.models.image import Image, ImageStatus
from app.models.triage_run import RunStatus, TriageRun
from app.schemas.image import ImageSummary
from app.schemas.triage import TriageReport, TriageRunCreate, TriageRunResponse

router = APIRouter()


@router.post("/run", response_model=TriageRunResponse, status_code=201)
async def run_triage(data: TriageRunCreate, db: AsyncSession = Depends(get_db)):
    """
    Record a triage run over images already ingested (triage itself happens at
    ingestion time inside the pipeline). Counts come from the database.
    """
    run = TriageRun(
        run_id=f"RUN-{int(datetime.now(timezone.utc).timestamp())}",
        status=RunStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        blank_threshold_used=settings.BLANK_THRESHOLD,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    row = (
        await db.execute(
            select(
                func.count(Image.id),
                func.sum(
                    case((Image.blank_probability >= settings.BLANK_THRESHOLD, 1), else_=0)
                ),
                func.sum(func.coalesce(Image.file_size_bytes, 0)),
            )
        )
    ).one()
    total, blanks, total_bytes = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)

    quarantined = (
        await db.execute(
            select(func.count(Image.id)).where(Image.status == ImageStatus.QUARANTINED)
        )
    ).scalar_one_or_none() or 0
    quarantined_bytes = (
        await db.execute(
            select(func.sum(func.coalesce(Image.file_size_bytes, 0))).where(
                Image.status == ImageStatus.QUARANTINED
            )
        )
    ).scalar_one_or_none() or 0

    run.status = RunStatus.COMPLETED
    run.completed_at = datetime.now(timezone.utc)
    run.total_images = total
    run.processed_count = total
    run.blank_count = blanks
    run.subject_count = max(0, total - blanks)
    run.quarantined_count = int(quarantined)
    run.original_storage_bytes = total_bytes
    run.storage_saved_bytes = int(quarantined_bytes)
    run.active_storage_bytes = max(0, total_bytes - int(quarantined_bytes))
    run.triage_processing_seconds = max(
        0.0, (run.completed_at - run.started_at).total_seconds()
    )
    await db.commit()
    await db.refresh(run)
    return run


@router.get("/runs", response_model=List[TriageRunResponse])
async def list_triage_runs(
    skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(TriageRun).order_by(TriageRun.created_at.desc()).offset(skip).limit(limit)
    )
    return list(result.scalars().all())


@router.get("/runs/{run_id}", response_model=TriageRunResponse)
async def get_triage_run(run_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TriageRun).where(TriageRun.run_id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/report", response_model=TriageReport)
async def latest_triage_report(db: AsyncSession = Depends(get_db)):
    """Current triage state across all ingested images."""
    total = (await db.execute(select(func.count(Image.id)))).scalar_one_or_none() or 0
    blanks = (
        await db.execute(
            select(func.count(Image.id)).where(
                Image.blank_probability >= settings.BLANK_THRESHOLD
            )
        )
    ).scalar_one_or_none() or 0
    quarantined = (
        await db.execute(
            select(func.count(Image.id)).where(Image.status == ImageStatus.QUARANTINED)
        )
    ).scalar_one_or_none() or 0
    saved = (
        await db.execute(
            select(func.sum(func.coalesce(Image.file_size_bytes, 0))).where(
                Image.status == ImageStatus.QUARANTINED
            )
        )
    ).scalar_one_or_none() or 0

    blanks_by_camera = (
        await db.execute(
            select(Image.camera_id, func.count(Image.id))
            .where(
                Image.camera_id.isnot(None),
                Image.blank_probability >= settings.BLANK_THRESHOLD,
            )
            .group_by(Image.camera_id)
            .order_by(func.count(Image.id).desc())
            .limit(12)
        )
    ).all()

    quality_rows = (
        await db.execute(select(Image.quality_score).where(Image.quality_score.isnot(None)))
    ).all()
    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for (score,) in quality_rows:
        idx = min(4, int(float(score) * 5))
        buckets[list(buckets.keys())[idx]] += 1

    return TriageReport(
        run_id="current",
        total_images=int(total),
        blank_count=int(blanks),
        subject_count=max(0, int(total) - int(blanks)),
        quarantined_count=int(quarantined),
        storage_saved_bytes=int(saved),
        blanks_by_camera=[{"label": r[0], "count": int(r[1])} for r in blanks_by_camera],
        quality_distribution=[{"range": k, "count": v} for k, v in buckets.items()],
    )


@router.get("/runs/{run_id}/report", response_model=TriageReport)
async def get_triage_run_report(run_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TriageRun).where(TriageRun.run_id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    report = await latest_triage_report(db)
    report.run_id = run.run_id
    return report


@router.get("/quarantine", response_model=List[ImageSummary])
async def list_quarantine(
    skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), db: AsyncSession = Depends(get_db)
):
    rows = (
        await db.execute(
            select(Image, CameraStation)
            .outerjoin(CameraStation, CameraStation.camera_id == Image.camera_id)
            .where(Image.status == ImageStatus.QUARANTINED)
            .order_by(Image.created_at.desc())
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
            "is_demo": bool(img.is_demo),
        }
        for img, cam in rows
    ]
