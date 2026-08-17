"""
MegaDescriptor embedding service for the backend.

Wraps the MegaDescriptor model and provides caching + error handling.
Used by ProductionInference to generate embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global model instance (loaded once, reused)
_model_instance: Optional[object] = None


def get_megadescriptor_model(
    model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224",
    cache_dir: Optional[Union[str, Path]] = None,
) -> object:
    """
    Get the cached MegaDescriptor model instance.
    Loads on first call, reuses on subsequent calls.
    """
    global _model_instance
    
    if _model_instance is None:
        try:
            from ml.megadescriptor import MegaDescriptor
            
            logger.info("Initializing MegaDescriptor model...")
            _model_instance = MegaDescriptor(
                model_name=model_name,
                cache_dir=cache_dir,
                device=None,  # Auto-detect
            )
            logger.info("MegaDescriptor model ready")
        except Exception as e:
            logger.error(f"Failed to initialize MegaDescriptor: {e}")
            raise RuntimeError(f"MegaDescriptor initialization failed: {e}") from e
    
    return _model_instance


class MegaDescriptorEmbeddingService:
    """Service for generating embeddings using MegaDescriptor."""
    
    def __init__(
        self,
        model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the embedding service."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = get_megadescriptor_model(model_name, cache_dir)
    
    def get_embedding(
        self,
        image_data: Union[bytes, str, Path, Image.Image, np.ndarray],
    ) -> np.ndarray:
        """
        Generate embedding from image data.
        
        Args:
            image_data: Image as bytes, path, PIL Image, or numpy array
            
        Returns:
            L2-normalized embedding array of shape (768,)
        """
        try:
            if isinstance(image_data, bytes):
                import io
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return self.model.get_embedding(pil_image)
            elif isinstance(image_data, np.ndarray):
                # Assume it's RGB image array (H, W, 3)
                pil_image = Image.fromarray(image_data.astype("uint8"), "RGB")
                return self.model.get_embedding(pil_image)
            else:
                # Path or PIL Image
                return self.model.get_embedding(image_data)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: list[dict],
    ) -> Optional[dict]:
        """
        Find the most similar embedding in a list of references.
        
        Args:
            query_embedding: Query embedding array of shape (768,)
            reference_embeddings: List of dicts with 'embedding' and 'tiger_id' keys
            
        Returns:
            Dict with 'tiger_id', 'similarity' and 'index' keys, or None if empty
        """
        if not reference_embeddings:
            return None
        
        ref_arrays = np.array([r["embedding"] for r in reference_embeddings])
        similarities = self.model.cosine_similarities_batch(
            query_embedding,
            ref_arrays,
        )
        
        best_idx = int(np.argmax(similarities))
        
        return {
            "tiger_id": reference_embeddings[best_idx].get("tiger_id"),
            "similarity": float(similarities[best_idx]),
            "index": best_idx,
        }
