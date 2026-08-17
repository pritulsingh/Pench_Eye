from pydantic import BaseModel
from typing import List, Optional


class LabeledCount(BaseModel):
    label: str
    count: int


class TimeSeriesPoint(BaseModel):
    date: str
    detections: int
    tigers: int = 0
    blanks: int = 0


class ConfidenceBucket(BaseModel):
    range: str
    count: int


class MovementPair(BaseModel):
    from_camera: str
    to_camera: str
    transitions: int


class AnalyticsOverview(BaseModel):
    range_days: int
    detections_over_time: List[TimeSeriesPoint] = []
    detections_by_camera: List[LabeledCount] = []
    detections_by_zone: List[LabeledCount] = []
    detections_by_hour: List[LabeledCount] = []
    detections_by_weekday: List[LabeledCount] = []
    species_distribution: List[LabeledCount] = []
    top_tigers: List[LabeledCount] = []
    confidence_distribution: List[ConfidenceBucket] = []
    movement_frequency: List[MovementPair] = []
    camera_activity: List[LabeledCount] = []
    mean_identity_confidence: Optional[float] = None
    is_demo_data: bool = True
