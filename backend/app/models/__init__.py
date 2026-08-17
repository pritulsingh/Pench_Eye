# Models package — import all models so SQLAlchemy metadata is populated
from app.models.camera_station import CameraStation, CameraStatus, CameraZone
from app.models.image import Image, ImageStatus, SourceType, ProcessingStatus
from app.models.tiger import Tiger, TigerSex, TigerStatus
from app.models.observation import Observation, MatchType, ReviewStatus, FlankSide, DetectionType
from app.models.embedding import Embedding
from app.models.triage_run import TriageRun, RunStatus
from app.models.review_queue import ReviewQueue, QueueStatus
from app.models.zone import Zone
from app.models.alert import Alert, AlertSeverity, AlertStatus, AlertType
from app.models.ml_dataset import MLDataset
from app.models.training_run import TrainingRun
from app.models.ml_model import MLModel
from app.models.tiger_registration import TigerRegistrationSession, TigerMovement, TigerObservedArea

__all__ = [
    "CameraStation", "CameraStatus", "CameraZone",
    "Image", "ImageStatus", "SourceType", "ProcessingStatus",
    "Tiger", "TigerSex", "TigerStatus",
    "Observation", "MatchType", "ReviewStatus", "FlankSide", "DetectionType",
    "Embedding",
    "TriageRun", "RunStatus",
    "ReviewQueue", "QueueStatus",
    "Zone",
    "Alert", "AlertSeverity", "AlertStatus", "AlertType",
    "MLDataset", "TrainingRun", "MLModel",
    "TigerRegistrationSession", "TigerMovement", "TigerObservedArea",
]