from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID

class TriageRunCreate(BaseModel):
    path: Optional[str] = None
    batch_id: Optional[str] = None
    camera_id: Optional[str] = None

class TriageRunResponse(BaseModel):
    id: UUID
    run_id: str
    status: str
    total_images: int = 0
    blank_count: int = 0
    subject_count: int = 0
    quarantined_count: int = 0
    storage_saved_bytes: int = 0
    triage_processing_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class TriageRunSummary(BaseModel):
    run_id: str
    status: str

class LabeledCount(BaseModel):
    label: str
    count: int

class QualityBucket(BaseModel):
    range: str
    count: int

class TriageReport(BaseModel):
    run_id: str
    total_images: int = 0
    blank_count: int = 0
    subject_count: int = 0
    quarantined_count: int = 0
    storage_saved_bytes: int = 0
    blanks_by_camera: List[LabeledCount] = []
    quality_distribution: List[QualityBucket] = []

class ImageTriageResult(BaseModel):
    image_id: str
    status: str
