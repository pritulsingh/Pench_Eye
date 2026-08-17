"""
Losses for metric learning.

`total = arcface_ce + triplet_weight * batch_hard_triplet`

Batch-hard mining (Hermans et al. 2017) picks, per anchor, the hardest positive
and hardest negative *within the batch* — which is why the loader uses a
P×K sampler. With random batches many anchors have no positive and the triplet
term degenerates to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_euclidean(embeddings: torch.Tensor) -> torch.Tensor:
    """Pairwise distances for L2-normalised embeddings."""
    distances = torch.cdist(embeddings, embeddings, p=2)
    return distances.clamp(min=0)


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss with an optional soft-margin formulation."""

    def __init__(self, margin: float = 0.3, soft_margin: bool = False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.size(0) < 2:
            return embeddings.sum() * 0.0

        distances = pairwise_euclidean(embeddings)
        same = labels.view(-1, 1).eq(labels.view(1, -1))
        eye = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
        positive_mask = same & ~eye
        negative_mask = ~same

        # Anchors with no positive or no negative in-batch cannot form a triplet.
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0

        hardest_positive = (distances.masked_fill(~positive_mask, float("-inf")))[valid].max(dim=1).values
        hardest_negative = (distances.masked_fill(~negative_mask, float("inf")))[valid].min(dim=1).values

        if self.soft_margin:
            return F.softplus(hardest_positive - hardest_negative).mean()
        return F.relu(hardest_positive - hardest_negative + self.margin).mean()


@dataclass
class LossOutput:
    total: torch.Tensor
    components: Dict[str, float]


class ReIDLoss(nn.Module):
    """Combines the ArcFace cross-entropy term with the triplet term."""

    def __init__(
        self,
        triplet_weight: float = 0.0,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.1,
        soft_margin_triplet: bool = False,
    ):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = (
            BatchHardTripletLoss(margin=triplet_margin, soft_margin=soft_margin_triplet)
            if triplet_weight > 0
            else None
        )

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> LossOutput:
        ce = self.cross_entropy(logits, labels)
        components = {"arcface": float(ce.detach())}
        total = ce

        if self.triplet is not None:
            triplet = self.triplet(embeddings, labels)
            components["triplet"] = float(triplet.detach())
            total = total + self.triplet_weight * triplet

        components["total"] = float(total.detach())
        return LossOutput(total=total, components=components)
