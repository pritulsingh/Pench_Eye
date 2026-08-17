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

    # --- Test-time augmentation (TTA) for identity-preserving robustness ---
    # A single MegaDescriptor embedding is not invariant to horizontal flips,
    # small rotations or crop-tightness differences. Averaging L2-normalized
    # embeddings over these views (then renormalizing) makes the query stable
    # without lowering thresholds or averaging raw unnormalized vectors.
    ENABLE_HORIZONTAL_FLIP_TTA: bool = True
    # Vertical flip is OFF by default: a tiger is rarely upside-down and it
    # collapses same-tiger similarity (measured ~0.56), destroying identity.
    ENABLE_VERTICAL_FLIP_TTA: bool = False
    ENABLE_CROP_TTA: bool = True
    # Small rotations only. Comma-separated degrees; empty disables rotation TTA.
    ROTATION_ANGLES: str = "-5,5"
    # Centre-crop fractions used when ENABLE_CROP_TTA is on.
    TTA_CROP_FRACTIONS: str = "0.9"
    # Aggregation across views: "mean" or "weighted" (uses TTA_WEIGHTS).
    TTA_AGGREGATION_METHOD: str = "mean"
    # Optional per-view weights for "weighted" aggregation, ordered as views are
    # generated (original, hflip, vflip, rotations…, crops…). Empty → uniform.
    TTA_WEIGHTS: str = ""

    @property
    def rotation_angles_list(self) -> List[float]:
        out: List[float] = []
        for part in self.ROTATION_ANGLES.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                continue
        return out

    @property
    def tta_crop_fractions_list(self) -> List[float]:
        out: List[float] = []
        for part in self.TTA_CROP_FRACTIONS.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                frac = float(part)
            except ValueError:
                continue
            if 0.0 < frac < 1.0:
                out.append(frac)
        return out

    @property
    def tta_weights_list(self) -> List[float]:
        out: List[float] = []
        for part in self.TTA_WEIGHTS.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                continue
        return out

    # --- Detection crop padding ---
    # Fraction of bbox width/height added on every side before embedding, so
    # crop-tightness differences between captures do not shift the embedding.
    # Clamped to image bounds. 0 → raw bbox (previous behaviour).
    BBOX_PADDING_PERCENT: float = 0.10

    # --- Multi-embedding gallery + MATCH/UNCERTAIN/NEW decision ---
    # How many enrolled embeddings to keep per identity. Similarity to an
    # identity is the MAX cosine over its stored views, so several views absorb
    # geometric variance without averaging away discriminative detail.
    MAX_EMBEDDINGS_PER_IDENTITY: int = 8
    # Number of top identities to surface as candidates for review.
    TOP_K: int = 5
    # Absolute cosine at/above which a match is accepted (subject to margin +
    # quality gates below). Separate from the legacy HIGH_MATCH_THRESHOLD so the
    # multi-embedding path can be calibrated independently.
    MATCH_THRESHOLD: float = 0.80
    # Minimum gap between the best and second-best identity for a confident
    # MATCH. Below this the result is UNCERTAIN (queued for human review).
    UNCERTAINTY_MARGIN: float = 0.05
    # Query crops below this quality score are forced to UNCERTAIN.
    QUALITY_THRESHOLD: float = 0.25

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
