from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from pathlib import Path
from datetime import datetime
import abc

class SourceType(str, Enum):
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"
    STREAM_FRAME = "stream_frame"

@dataclass
class ImageFrame:
    pixels: np.ndarray
    source_type: SourceType = SourceType.IMAGE
    source_filename: Optional[str] = None
    original_path: Optional[Path] = None
    timestamp: Optional[datetime] = None
    camera_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    frame_number: Optional[int] = None
    video_id: Optional[str] = None
    exif_data: dict = field(default_factory=dict)
    file_size_bytes: int = 0
    sha256_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        if self.pixels is not None:
            h, w = self.pixels.shape[:2]
            self.height = h
            self.width = w

class InputSource(abc.ABC):
    @abc.abstractmethod
    async def iter_frames(self):
        ...
    
    @property
    @abc.abstractmethod
    def source_type(self) -> SourceType:
        ...
