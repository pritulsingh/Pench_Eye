from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.analytics import AnalyticsOverview
from app.services.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/overview", response_model=AnalyticsOverview)
async def analytics_overview(
    days: int = Query(90, ge=1, le=1825),
    db: AsyncSession = Depends(get_db),
):
    return await AnalyticsService.overview(db, days=days)
