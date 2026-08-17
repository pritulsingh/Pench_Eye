from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.alert import AlertEvaluationResult, AlertResponse, AlertSummary, AlertUpdate
from app.schemas.common import PaginatedResponse
from app.services.alert_service import AlertService

router = APIRouter()

ALLOWED_STATUSES = {"open", "acknowledged", "resolved"}


@router.get("", response_model=PaginatedResponse[AlertResponse])
async def list_alerts(
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    try:
        items, total = await AlertService.list_alerts(
            db,
            status=status,
            severity=severity,
            alert_type=alert_type,
            camera_id=camera_id,
            skip=skip,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PaginatedResponse(items=items, total=total, page=skip // limit + 1, size=limit)


@router.get("/summary", response_model=AlertSummary)
async def alert_summary(db: AsyncSession = Depends(get_db)):
    return AlertSummary(**await AlertService.summary(db))


@router.post("/evaluate", response_model=AlertEvaluationResult)
async def evaluate_rules(db: AsyncSession = Depends(get_db)):
    """Run camera-offline / high-activity rules. Idempotent."""
    created = await AlertService.evaluate_system_rules(db)
    return AlertEvaluationResult(created=created, summary=AlertSummary(**await AlertService.summary(db)))


@router.patch("/{alert_id}", response_model=AlertResponse)
async def update_alert(alert_id: str, data: AlertUpdate, db: AsyncSession = Depends(get_db)):
    if data.status not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=422, detail=f"status must be one of {sorted(ALLOWED_STATUSES)}"
        )
    alert = await AlertService.update_status(db, alert_id, data.status, data.actor)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
