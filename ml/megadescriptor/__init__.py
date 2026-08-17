"""
MegaDescriptor: A simple pretrained embedding model for tiger re-identification.

This module provides a unified interface to load a pretrained MegaDescriptor model
and generate L2-normalized 768-d embeddings from camera-trap images.

The goal is MVP simplicity: no fine-tuning, no complex preprocessing, just:
  image → embedding → cosine similarity → match/new tiger
"""

from .model import MegaDescriptor

__all__ = ["MegaDescriptor"]
