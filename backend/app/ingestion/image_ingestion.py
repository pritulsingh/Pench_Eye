import os
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash
from typing import Optional, AsyncGenerator
from .base import ImageFrame, SourceType

class ImageIngestionService:
    def __init__(self):
        self.allowed_extensions = {".jpg", ".jpeg", ".png"}

    def _compute_hashes(self, img: Image.Image, file_bytes: bytes):
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        p_hash = str(imagehash.phash(img))
        return sha256_hash, p_hash

    def _extract_exif(self, img: Image.Image) -> dict:
        exif = img.getexif()
        return dict(exif) if exif else {}

    def _infer_camera_id(self, path: Path) -> Optional[str]:
        # Simple heuristic: directory name might be CAM_001
        dirname = path.parent.name
        if "CAM" in dirname.upper():
            return dirname.upper().replace("_", "-")
        return None

    def process_upload_bytes(self, filename: str, file_bytes: bytes) -> ImageFrame:
        import io
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # Convert RGB to BGR for OpenCV
        pixels = np.array(img)[:, :, ::-1].copy()
        sha256_hash, p_hash = self._compute_hashes(img, file_bytes)
        exif = self._extract_exif(img)
        
        return ImageFrame(
            pixels=pixels,
            source_type=SourceType.IMAGE,
            source_filename=filename,
            file_size_bytes=len(file_bytes),
            sha256_hash=sha256_hash,
            perceptual_hash=p_hash,
            exif_data=exif,
            timestamp=datetime.utcnow(),
            width=img.width,
            height=img.height
        )

    async def process_directory(self, path: str) -> AsyncGenerator[ImageFrame, None]:
        p = Path(path)
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.allowed_extensions:
                with open(f, "rb") as file:
                    file_bytes = file.read()
                frame = self.process_upload_bytes(f.name, file_bytes)
                frame.original_path = f
                frame.camera_id = self._infer_camera_id(f)
                yield frame
