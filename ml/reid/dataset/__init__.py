"""
Re-ID dataset package.

Loads identity-labelled tiger flank crops from either a directory layout
(`data/reid/{split}/{IDENTITY}/*.jpg`) or a flat CSV annotation file, and
produces splits that do not leak the same capture sequence across train and
evaluation.
"""
from ml.reid.dataset.discovery import (
    DatasetSummary,
    IdentityStats,
    ImageRecord,
    build_identity_mapping,
    discover_dataset,
    load_csv_annotations,
    summarize,
)
from ml.reid.dataset.splitting import split_records
from ml.reid.dataset.torch_dataset import ReIDDataset, build_dataloaders

__all__ = [
    "ImageRecord",
    "IdentityStats",
    "DatasetSummary",
    "discover_dataset",
    "load_csv_annotations",
    "build_identity_mapping",
    "summarize",
    "split_records",
    "ReIDDataset",
    "build_dataloaders",
]
