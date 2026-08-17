from fastapi import APIRouter

from app.api.v1 import (
    alerts,
    analytics,
    cameras,
    dashboard,
    images,
    map_data,
    ml,
    observations,
    reviews,
    search,
    tiger_registration,
    tigers,
    triage,
)

api_router = APIRouter()

api_router.include_router(images.router, prefix="/api/v1/images", tags=["images"])
api_router.include_router(triage.router, prefix="/api/v1/triage", tags=["triage"])
api_router.include_router(tigers.router, prefix="/api/v1/tigers", tags=["tigers"])
api_router.include_router(observations.router, prefix="/api/v1/observations", tags=["observations"])
api_router.include_router(reviews.router, prefix="/api/v1/reviews", tags=["reviews"])
api_router.include_router(cameras.router, prefix="/api/v1/cameras", tags=["cameras"])
api_router.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
api_router.include_router(search.router, prefix="/api/v1/search", tags=["search"])
api_router.include_router(map_data.router, prefix="/api/v1/map", tags=["map"])
api_router.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
api_router.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
api_router.include_router(ml.router, prefix="/api/v1/ml", tags=["ml"])
api_router.include_router(tiger_registration.router, prefix="/api/v1/tiger-registration", tags=["tiger-registration"])
