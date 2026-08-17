# Ingestion package
# ImageFrame is the core abstraction bridging raw inputs to the pipeline.
from app.ingestion.base import ImageFrame, InputSource, IngestionResult

__all__ = ["ImageFrame", "InputSource", "IngestionResult"]