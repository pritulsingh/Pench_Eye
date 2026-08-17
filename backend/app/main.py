import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.database import create_tables
from app.services.inference_service import pipeline_info

logger = logging.getLogger("pench_eye")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    info = pipeline_info()
    detector = info.get("detector", {})
    logger.info(
        "YOLO diagnostic | path=%s | type=%s | classes=%s | tiger_class_id=%s | "
        "tiger_class_name=%s",
        detector.get("model_path"),
        detector.get("model_type"),
        detector.get("available_classes"),
        detector.get("tiger_class_id"),
        detector.get("tiger_class_name"),
    )
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Pench Eye — wildlife intelligence API for camera-trap triage, individual "
        "tiger identification, geospatial monitoring, alerting and analytics."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health():
    info = pipeline_info()
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "ml_mode": info["ml_mode"],
        "model_version": info["model_version"],
        "demo_mode": settings.is_demo_mode,
        "geo_data_source": settings.GEO_DATA_SOURCE,
        "reid_available": info["reid_available"],
        "reid_model_version": info["reid"].get("model_version"),
        "reid_validated": info["reid"].get("validated"),
        "disclaimer": info["disclaimer"],
    }
