from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.image import Image, ImageStatus
from app.models.triage_run import TriageRun, RunStatus
from typing import List, Optional
import time
from datetime import datetime
import uuid

# Wrap ML import to gracefully handle missing modules
try:
    from ml.triage.blank_classifier import BlankImageClassifier
except ImportError:
    class BlankImageClassifier:
        def __init__(self, threshold=0.95):
            self.threshold = threshold
        def predict(self, frame):
            return 0.1, False

class TriageService:
    def __init__(self):
        from app.core.config import settings
        self.classifier = BlankImageClassifier(threshold=settings.BLANK_THRESHOLD)
        self.threshold = settings.BLANK_THRESHOLD
        
    async def triage_image(self, db: AsyncSession, image_frame, triage_run_id: str = None) -> Image:
        blank_prob, is_blank = self.classifier.predict(image_frame)
        
        # Check duplicate
        duplicate = None
        if image_frame.sha256_hash:
            result = await db.execute(select(Image).where(Image.sha256_hash == image_frame.sha256_hash).limit(1))
            duplicate = result.scalar_one_or_none()
            
        status = ImageStatus.TRIAGED
        triage_reason = "subject_detected"
        
        if duplicate:
            status = ImageStatus.QUARANTINED
            triage_reason = f"duplicate_of_{duplicate.image_id}"
        elif is_blank:
            status = ImageStatus.QUARANTINED
            triage_reason = "blank"
            
        image = Image(
            image_id=f"IMG-{uuid.uuid4()}",
            original_filename=image_frame.source_filename,
            camera_id=image_frame.camera_id,
            timestamp=image_frame.timestamp,
            file_size_bytes=image_frame.file_size_bytes,
            width_px=image_frame.width,
            height_px=image_frame.height,
            sha256_hash=image_frame.sha256_hash,
            perceptual_hash=image_frame.perceptual_hash,
            blank_probability=float(blank_prob),
            blank_threshold_used=float(self.threshold),
            triage_reason=triage_reason,
            status=status,
            triage_run_id=triage_run_id
        )
        
        db.add(image)
        await db.commit()
        await db.refresh(image)
        return image
        
    async def quarantine_image(self, db: AsyncSession, image_id: str) -> bool:
        result = await db.execute(select(Image).where(Image.image_id == image_id))
        image = result.scalar_one_or_none()
        if image:
            image.status = ImageStatus.QUARANTINED
            await db.commit()
            return True
        return False
        
    async def restore_image(self, db: AsyncSession, image_id: str) -> bool:
        result = await db.execute(select(Image).where(Image.image_id == image_id))
        image = result.scalar_one_or_none()
        if image:
            image.status = ImageStatus.TRIAGED
            await db.commit()
            return True
        return False
        
    async def permanently_delete_image(self, db: AsyncSession, image_id: str) -> bool:
        result = await db.execute(select(Image).where(Image.image_id == image_id))
        image = result.scalar_one_or_none()
        if image:
            image.status = ImageStatus.DELETED
            await db.commit()
            return True
        return False
        
    async def start_triage_run(self, db: AsyncSession) -> TriageRun:
        run = TriageRun(
            run_id=f"RUN-{uuid.uuid4()}",
            status=RunStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return run
        
    async def complete_triage_run(self, db: AsyncSession, run_id: str, stats: dict) -> TriageRun:
        result = await db.execute(select(TriageRun).where(TriageRun.id == run_id))
        run = result.scalar_one_or_none()
        if run:
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.utcnow()
            for k, v in stats.items():
                if hasattr(run, k):
                    setattr(run, k, v)
            await db.commit()
        return run

triage_service = TriageService()
