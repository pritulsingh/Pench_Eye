"""
Re-ID model: backbone → global pooling → BNNeck → 512-d L2-normalised embedding.

The 512-d normalised vector is the contract the application already relies on
(`Embedding.embedding` is `vector(512)`, similarity is cosine). ArcFace logits
are a *training-time* head only and are never persisted.

BNNeck (BatchNorm between embedding and classifier) is standard practice in
person Re-ID: the classifier trains on the batch-normed feature while retrieval
uses the pre-BN feature, which measurably improves cosine retrieval.

`tiny` backbone exists so tests and smoke runs never download ImageNet weights.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BACKBONES = ("resnet50", "resnet34", "resnet18", "osnet_x1_0", "tiny")
DEFAULT_EMBEDDING_DIM = 512


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    pretrained: bool = True
    dropout: float = 0.0
    use_bnneck: bool = True
    architecture: str = "reid-embedder-v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ModelConfig":
        if not data:
            return cls()
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)  # type: ignore[arg-type]


class _TinyBackbone(nn.Module):
    """Small CNN for tests/smoke runs — no pretrained download."""

    out_features = 64

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Choose from {SUPPORTED_BACKBONES}.")

    if name == "tiny":
        backbone = _TinyBackbone()
        return backbone, backbone.out_features

    if name.startswith("resnet"):
        import torchvision.models as models

        weights = None
        if pretrained:
            try:
                weights_enum = {
                    "resnet18": models.ResNet18_Weights,
                    "resnet34": models.ResNet34_Weights,
                    "resnet50": models.ResNet50_Weights,
                }[name]
                weights = weights_enum.DEFAULT
            except Exception:  # offline or torchvision without weights API
                weights = None

        builder = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}[name]
        try:
            model = builder(weights=weights)
        except Exception:
            model = builder(weights=None)
        feature_dim = model.fc.in_features
        # Drop avgpool + fc; pooling is handled by the head.
        model = nn.Sequential(*list(model.children())[:-2])
        return model, feature_dim

    # osnet_x1_0 — optional, only if torchreid is installed.
    try:
        import torchreid  # type: ignore

        model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1, pretrained=pretrained, loss="triplet"
        )
        feature_dim = int(getattr(model, "feature_dim", 512))
        if hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        return model, feature_dim
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backbone 'osnet_x1_0' requires torchreid (`pip install torchreid`). "
            "Use 'resnet50' for the supported default."
        ) from exc


class TigerReIDNet(nn.Module):
    """Produces a 512-d L2-normalised embedding for a batch of preprocessed crops."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone, feature_dim = _build_backbone(self.config.backbone, self.config.pretrained)
        self.backbone_feature_dim = feature_dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(feature_dim, self.config.embedding_dim, bias=False)
        self.bnneck = (
            nn.BatchNorm1d(self.config.embedding_dim) if self.config.use_bnneck else nn.Identity()
        )
        if isinstance(self.bnneck, nn.BatchNorm1d):
            self.bnneck.bias.requires_grad_(False)
        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out")

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = self.pool(feats).flatten(1)
        elif feats.dim() != 2:
            feats = feats.flatten(1)
        return feats

    def forward(self, x: torch.Tensor, return_logits_feature: bool = False):
        """
        Returns the L2-normalised embedding. With `return_logits_feature=True`
        also returns the post-BNNeck feature that the ArcFace head consumes.
        """
        feats = self._features(x)
        raw = self.embedding(self.dropout(feats))
        normalized = F.normalize(raw, p=2, dim=1)
        if return_logits_feature:
            return normalized, self.bnneck(raw)
        return normalized

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)


class ArcFaceHead(nn.Module):
    """
    Additive angular margin head (ArcFace, Deng et al. 2019).

    Adds margin `m` to the angle between a feature and its class centre, then
    scales by `s`. Produces tighter identity clusters than plain softmax, which
    is what makes cosine thresholds meaningful at inference time.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.30):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)
        self._mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-9))
        phi = cosine * self._cos_m - sine * self._sin_m
        # Keep the function monotonic past the margin boundary.
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)


def build_model(config: Optional[ModelConfig] = None) -> TigerReIDNet:
    return TigerReIDNet(config)
