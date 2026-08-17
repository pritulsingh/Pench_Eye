"""
PyTorch `Dataset` / `DataLoader` wiring for tiger Re-ID.

Augmentation applies to train only; val/test go through the deterministic
preprocessing path so metrics are stable across runs.

`PKSampler` builds P-identities × K-instances batches, which triplet mining
needs — random batches often contain no positive pair at all, leaving the
triplet term contributing nothing.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ml.reid.augmentation import Augmenter, AugmentationConfig
from ml.reid.dataset.discovery import ImageRecord, build_identity_mapping
from ml.reid.preprocessing import PreprocessConfig, load_image_rgb, preprocess_rgb, sharpness_score

logger = logging.getLogger(__name__)


class ReIDDataset(Dataset):
    """Yields (image_tensor, label, index). Unreadable files become zero tensors."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        identity_to_index: Dict[str, int],
        preprocess: Optional[PreprocessConfig] = None,
        augmentation: Optional[AugmentationConfig] = None,
        seed: int = 42,
        return_quality: bool = False,
    ):
        self.records = list(records)
        self.identity_to_index = dict(identity_to_index)
        self.preprocess = preprocess or PreprocessConfig()
        self.augmentation = augmentation
        self.return_quality = return_quality
        self._augmenter = Augmenter(augmentation, seed=seed) if augmentation and augmentation.enabled else None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> List[int]:
        return [self.identity_to_index[r.identity] for r in self.records]

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            image = load_image_rgb(str(record.path))
        except Exception as exc:  # keep training alive on a bad file
            logger.warning("Unreadable image %s (%s); substituting a blank frame.", record.path, exc)
            height, width = self.preprocess.image_size
            image = np.zeros((height, width, 3), dtype=np.uint8)

        quality = sharpness_score(image) if self.return_quality else 0.0
        if self._augmenter is not None:
            image = self._augmenter(image)

        tensor = torch.from_numpy(preprocess_rgb(image, self.preprocess))
        label = self.identity_to_index[record.identity]
        if self.return_quality:
            return tensor, label, index, quality
        return tensor, label, index


class PKSampler(Sampler[List[int]]):
    """
    Batch sampler emitting P identities × K instances per batch.

    Guarantees positive pairs exist in-batch, which is what makes hard/semi-hard
    triplet mining meaningful.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_instances: int = 4,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if num_instances < 2:
            raise ValueError("num_instances must be >= 2 for triplet mining to work.")
        if batch_size < num_instances:
            raise ValueError("batch_size must be >= num_instances.")

        self.labels = list(labels)
        self.num_instances = num_instances
        self.num_identities = max(1, batch_size // num_instances)
        self.batch_size = self.num_identities * num_instances
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)

        self.index_by_label: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels):
            self.index_by_label.setdefault(label, []).append(idx)
        self._available_labels = [
            label for label, idxs in self.index_by_label.items() if len(idxs) >= 1
        ]
        self._num_batches = max(1, len(self.labels) // self.batch_size)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self._num_batches):
            batch: List[int] = []
            replace = len(self._available_labels) < self.num_identities
            chosen = self._rng.choice(
                self._available_labels, size=self.num_identities, replace=replace
            )
            for label in chosen:
                pool = self.index_by_label[int(label)]
                picks = self._rng.choice(
                    pool, size=self.num_instances, replace=len(pool) < self.num_instances
                )
                batch.extend(int(p) for p in picks)
            yield batch


def build_dataloaders(
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    identity_to_index: Optional[Dict[str, int]] = None,
    *,
    preprocess: Optional[PreprocessConfig] = None,
    augmentation: Optional[AugmentationConfig] = None,
    batch_size: int = 32,
    num_instances: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    use_pk_sampler: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, int]]:
    """Build train (+ optional val) loaders sharing one identity mapping."""
    identity_to_index = identity_to_index or build_identity_mapping(
        list(train_records) + list(val_records)
    )
    preprocess = preprocess or PreprocessConfig()

    train_dataset = ReIDDataset(
        train_records,
        identity_to_index,
        preprocess=preprocess,
        augmentation=augmentation,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    if use_pk_sampler and len(train_dataset) >= num_instances:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PKSampler(
                train_dataset.labels,
                batch_size=batch_size,
                num_instances=num_instances,
                seed=seed,
            ),
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            generator=generator,
        )

    val_loader: Optional[DataLoader] = None
    if val_records:
        val_loader = DataLoader(
            ReIDDataset(
                val_records,
                identity_to_index,
                preprocess=preprocess,
                augmentation=AugmentationConfig.disabled(),
                seed=seed,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, identity_to_index
