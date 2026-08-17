"""
MegaDescriptor model interface.

Downloads and caches the pretrained MegaDescriptor model from Meta/Hugging Face.
Generates L2-normalized 768-d embeddings for input images.

No fine-tuning. No training. Just inference.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# MegaDescriptor is published on the Hugging Face Hub by BVRA (WildlifeDatasets).
# It is a Swin Transformer trained for wildlife re-identification and is loaded
# via timm. The tiny/small 224 variants output a 768-d embedding.
DEFAULT_MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-T-224"
EMBEDDING_DIM = 768


class MegaDescriptor:
    """
    Pretrained MegaDescriptor model for generating image embeddings.
    
    Downloads and caches the model automatically. All embeddings are L2-normalized.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the MegaDescriptor model.
        
        Args:
            model_name: Hugging Face model ID or local path
            cache_dir: Directory to cache the model (default: ~/.cache/huggingface)
            device: torch device (e.g., 'cuda', 'cpu'). Auto-detect if None.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading MegaDescriptor from {model_name} on device={self.device}")

        try:
            # MegaDescriptor is a timm-compatible model published on the HF Hub.
            # num_classes=0 removes the classification head so the model returns
            # the pooled embedding vector directly.
            import timm

            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
            )
        except Exception as e:
            logger.warning(f"Failed to load via timm: {e}. Trying transformers...")
            try:
                # Fallback: Use transformers library
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    trust_remote_code=True,
                )
            except Exception as e2:
                logger.error(f"Failed to load MegaDescriptor: {e2}")
                raise RuntimeError(
                    f"Could not load MegaDescriptor. Make sure you have internet "
                    f"and the model is available. Error: {e2}"
                ) from e2
        
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"MegaDescriptor loaded successfully. Embedding dim: {EMBEDDING_DIM}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Load and preprocess an image for inference.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor of shape (1, 3, H, W)
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB") if isinstance(image, Image.Image) else image
        
        # Resize to standard dimensions (e.g., 224x224)
        # Most vision models use this standard size
        pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(pil_image)).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # ImageNet normalization
        img_tensor[0] = (img_tensor[0] / 255.0 - 0.485) / 0.229
        img_tensor[1] = (img_tensor[1] / 255.0 - 0.456) / 0.224
        img_tensor[2] = (img_tensor[2] / 255.0 - 0.406) / 0.225
        
        return img_tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def get_embedding(
        self,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Generate an L2-normalized embedding for an image.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            L2-normalized embedding as numpy array of shape (768,)
        """
        img_tensor = self.preprocess_image(image)
        
        # Forward pass through the model
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Extract embedding
        # MegaDescriptor returns either:
        # - (batch_size, embedding_dim) if no head
        # - Dict with 'embedding' or similar key
        if isinstance(output, dict):
            embedding = output.get("embedding", output.get("features", output))
        elif isinstance(output, (list, tuple)):
            embedding = output[0] if output else None
        else:
            embedding = output
        
        if embedding is None:
            raise RuntimeError("Model output could not be parsed as embedding")
        
        # Ensure it's a 2D tensor
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Return as numpy array, shape (768,)
        return embedding[0].cpu().numpy()
    
    @torch.no_grad()
    def get_embeddings_batch(
        self,
        images: list[Union[str, Path, Image.Image]],
    ) -> np.ndarray:
        """
        Generate L2-normalized embeddings for multiple images.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            Array of shape (n_images, 768) with L2-normalized embeddings
        """
        embeddings = []
        for img in images:
            emb = self.get_embedding(img)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    @staticmethod
    def cosine_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        
        Args:
            embedding1: Array of shape (768,)
            embedding2: Array of shape (768,)
            
        Returns:
            Cosine similarity score in range [0, 1]
        """
        # Since embeddings are L2-normalized, cosine similarity = dot product
        return float(np.dot(embedding1, embedding2))
    
    @staticmethod
    def cosine_similarities_batch(
        query_embedding: np.ndarray,
        reference_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarities between one query and many references.
        
        Args:
            query_embedding: Array of shape (768,)
            reference_embeddings: Array of shape (n, 768)
            
        Returns:
            Similarity scores of shape (n,)
        """
        # Dot product with L2-normalized vectors = cosine similarity
        similarities = np.dot(reference_embeddings, query_embedding)
        return np.clip(similarities, -1.0, 1.0)
