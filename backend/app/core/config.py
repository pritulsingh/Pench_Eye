from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from enum import Enum
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent

# The `ml` package lives beside `backend/`, so make it importable regardless of
# the working directory uvicorn was started from.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class MLMode(str, Enum):
    DEMO = "demo"
    PRODUCTION = "production"

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Pench Eye — Tiger Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "changeme"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    # Database — SQLite by default so the prototype runs with zero infrastructure.
    # docker-compose overrides these with PostgreSQL + pgvector.
    DATABASE_URL: str = f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'storage' / 'pench_eye.db').as_posix()}"
    DATABASE_URL_SYNC: str = f"sqlite:///{(PROJECT_ROOT / 'storage' / 'pench_eye.db').as_posix()}"
    
    # MinIO / S3
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "tiger-images"
    MINIO_SECURE: bool = False
    
    # Storage
    LOCAL_STORAGE_PATH: str = str(PROJECT_ROOT / "storage")
    QUARANTINE_RETENTION_DAYS: int = 30
    
    # ML — production by default so real uploads never use hash-seeded demo inference.
    # Set ML_MODE=demo only for the explicit /api/v1/demo simulation UI.
    ML_MODE: MLMode = MLMode.PRODUCTION
    # Trained Re-ID checkpoint (file or run directory). Empty → probe the
    # defaults in ml/reid/checkpoint.py.
    REID_CHECKPOINT_PATH: str = ""
    # Optional override for tiger YOLO weights. Empty → ml/weights/tiger_yolo.pt
    TIGER_YOLO_WEIGHTS: str = ""

    # MegaDescriptor configuration
    MEGADESCRIPTOR_MODEL_NAME: str = "hf-hub:BVRA/MegaDescriptor-T-224"
    MEGADESCRIPTOR_CACHE_DIR: str = str(PROJECT_ROOT / ".cache" / "megadescriptor")
    # Initial MVP operating points only. These are cosine similarities, not
    # probabilities, and must be calibrated with labelled Amur tiger data.
    HIGH_MATCH_THRESHOLD: float = 0.85

    # Triage thresholds
    BLANK_THRESHOLD: float = 0.95

    # Re-ID thresholds (used when REID_CHECKPOINT_PATH is set)
    AUTO_MATCH_THRESHOLD: float = 0.90
    REVIEW_THRESHOLD: float = 0.70
    NEW_INDIVIDUAL_THRESHOLD: float = 0.60

    # MegaDescriptor embedding dimension. Keep this aligned with the existing
    # production model and the embeddings stored by the MVP pipeline.
    EMBEDDING_DIM: int = 768

    # Uploads
    MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024
    ALLOWED_UPLOAD_EXTENSIONS: str = ".jpg,.jpeg,.png,.webp"

    @property
    def allowed_upload_extensions(self) -> List[str]:
        return [e.strip().lower() for e in self.ALLOWED_UPLOAD_EXTENSIONS.split(",") if e.strip()]

    # Alerting
    CAMERA_OFFLINE_HOURS: int = 48
    HIGH_ACTIVITY_DETECTIONS_PER_DAY: int = 5
    LOW_CONFIDENCE_THRESHOLD: float = 0.80

    # Geo defaults identify the real Pench landscape. Operational cameras,
    # observations and zones still come only from the database.
    RESERVE_CENTER_LAT: float = 21.7440
    RESERVE_CENTER_LON: float = 79.2940
    RESERVE_SOUTH_LAT: float = 21.5400
    RESERVE_WEST_LON: float = 79.1000
    RESERVE_NORTH_LAT: float = 21.9500
    RESERVE_EAST_LON: float = 79.5100
    GEO_DATA_SOURCE: str = "OpenStreetMap + database"

    @property
    def is_demo_mode(self) -> bool:
        return self.ML_MODE == MLMode.DEMO

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

settings = Settings()
