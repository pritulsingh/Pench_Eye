"""Detection service — thin wrapper around TigerDetector.

Production must fail closed. Never invent a tiger detection when the
detector cannot be imported or loaded.
"""
from typing import Any, List

try:
    from ml.detection.tiger_detector import TigerDetector
except ImportError:

    class TigerDetector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.ml_mode = "unavailable"

        def detect(self, frame) -> List[Any]:
            return []

        def status(self):
            return {"available": False, "error": "ml.detection.tiger_detector not importable"}


class DetectionService:
    def __init__(self):
        # Production fail-closed detector. Demo path is only for explicit demo tools.
        self.detector = TigerDetector(ml_mode="production")

    async def detect(self, image_frame) -> List[Any]:
        import asyncio

        return await asyncio.to_thread(self.detector.detect, image_frame)


detection_service = DetectionService()
