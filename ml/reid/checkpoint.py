"""
Checkpoint format for tiger Re-ID.

A checkpoint is self-describing: it carries the architecture, preprocessing and
identity mapping needed to rebuild the exact model that produced a given
embedding. That matters because embeddings persisted in the database are only
comparable to embeddings from the same model *and* the same preprocessing.

Layout written by `ml/reid/train.py`:

    ml/weights/tiger_reid/
    ├── best.pt                # highest validation Rank-1
    ├── latest.pt              # most recent epoch (for resume)
    ├── config.json            # full run configuration
    ├── class_mapping.json     # identity ↔ class index
    └── training_history.json  # per-epoch metrics

`load_checkpoint` also accepts a bare `state_dict` file so older ad-hoc weights
still load, though without metadata the caller must supply the config.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ml.reid.model import ModelConfig, TigerReIDNet, build_model
from ml.reid.preprocessing import PreprocessConfig

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 2
BEST_CHECKPOINT_NAME = "best.pt"
LATEST_CHECKPOINT_NAME = "latest.pt"
CONFIG_NAME = "config.json"
CLASS_MAPPING_NAME = "class_mapping.json"
HISTORY_NAME = "training_history.json"

# Search order used when only a directory (or nothing) is given.
DEFAULT_CHECKPOINT_CANDIDATES = (
    "ml/weights/tiger_reid/best.pt",
    "ml/weights/tiger_reid/latest.pt",
    "ml/weights/tiger_reid.pt",
)


@dataclass
class CheckpointPayload:
    """Everything needed to reconstruct a model and interpret its embeddings."""

    model_state: Dict[str, Any]
    model_config: ModelConfig
    preprocess_config: PreprocessConfig
    identity_to_index: Dict[str, int]
    epoch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    train_config: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "tiger-reid-untrained"
    format_version: int = CHECKPOINT_FORMAT_VERSION

    @property
    def index_to_identity(self) -> Dict[int, str]:
        return {v: k for k, v in self.identity_to_index.items()}


def make_model_version(
    backbone: str, embedding_dim: int, epoch: int, rank1: Optional[float] = None
) -> str:
    """Human-readable version string surfaced by the API as the active model."""
    parts = [f"tiger-reid-{backbone}-{embedding_dim}d", f"ep{epoch}"]
    if rank1 is not None:
        parts.append(f"r1-{rank1:.3f}")
    return "-".join(parts)


def save_checkpoint(
    path: str | Path,
    *,
    model: TigerReIDNet,
    identity_to_index: Dict[str, int],
    preprocess_config: PreprocessConfig,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    arcface_state: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = metrics or {}
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": model.config.architecture,
        "model_config": model.config.to_dict(),
        "preprocess_config": preprocess_config.to_dict(),
        "embedding_dim": model.embedding_dim,
        "model_state_dict": model.state_dict(),
        "identity_to_index": identity_to_index,
        "epoch": epoch,
        "metrics": metrics,
        "train_config": train_config or {},
        "model_version": model_version
        or make_model_version(
            model.config.backbone, model.embedding_dim, epoch, metrics.get("val_rank1")
        ),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state_dict"] = scaler.state_dict()
    if arcface_state is not None:
        payload["arcface_state_dict"] = arcface_state

    torch.save(payload, path)
    return path


def load_raw_checkpoint(path: str | Path) -> Dict[str, Any]:
    """`torch.load` with `weights_only=False` (our payloads contain metadata)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location="cpu")


def load_checkpoint(
    path: str | Path,
    *,
    fallback_model_config: Optional[ModelConfig] = None,
) -> CheckpointPayload:
    raw = load_raw_checkpoint(path)

    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        # Bare state_dict from an external/legacy source.
        if fallback_model_config is None:
            raise ValueError(
                f"{path} looks like a bare state_dict; supply fallback_model_config "
                "so the architecture can be reconstructed."
            )
        state = raw if isinstance(raw, dict) else {}
        return CheckpointPayload(
            model_state=state,
            model_config=fallback_model_config,
            preprocess_config=PreprocessConfig(),
            identity_to_index={},
            model_version="tiger-reid-external-state-dict",
            format_version=1,
        )

    model_config = ModelConfig.from_dict(raw.get("model_config"))
    if fallback_model_config is not None and not raw.get("model_config"):
        model_config = fallback_model_config

    return CheckpointPayload(
        model_state=raw["model_state_dict"],
        model_config=model_config,
        preprocess_config=PreprocessConfig.from_dict(raw.get("preprocess_config")),
        identity_to_index=dict(raw.get("identity_to_index") or {}),
        epoch=int(raw.get("epoch", 0)),
        metrics=dict(raw.get("metrics") or {}),
        train_config=dict(raw.get("train_config") or {}),
        model_version=str(raw.get("model_version", "tiger-reid-unknown")),
        format_version=int(raw.get("format_version", 1)),
    )


def load_model_for_inference(
    path: str | Path,
    device: str = "cpu",
    *,
    fallback_model_config: Optional[ModelConfig] = None,
) -> Tuple[TigerReIDNet, CheckpointPayload]:
    """Rebuild the model from a checkpoint and put it in eval mode."""
    payload = load_checkpoint(path, fallback_model_config=fallback_model_config)
    # Never fetch ImageNet weights when restoring — the checkpoint supersedes them.
    config = ModelConfig.from_dict({**payload.model_config.to_dict(), "pretrained": False})
    model = build_model(config)

    missing, unexpected = model.load_state_dict(payload.model_state, strict=False)
    if missing:
        logger.warning("Checkpoint %s is missing %d parameter(s): %s", path, len(missing), list(missing)[:5])
    if unexpected:
        logger.warning(
            "Checkpoint %s has %d unexpected parameter(s): %s", path, len(unexpected), list(unexpected)[:5]
        )

    model.to(device).eval()
    return model, payload


def resolve_checkpoint_path(
    explicit: Optional[str | Path] = None,
    *,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolve a checkpoint path.

    Accepts a file, or a run directory (preferring `best.pt` over `latest.pt`).
    With nothing supplied, probes `DEFAULT_CHECKPOINT_CANDIDATES`.
    """
    root = project_root or Path(__file__).resolve().parents[2]

    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        if candidate.is_dir():
            for name in (BEST_CHECKPOINT_NAME, LATEST_CHECKPOINT_NAME):
                if (candidate / name).is_file():
                    return candidate / name
            return None
        return candidate if candidate.is_file() else None

    for relative in DEFAULT_CHECKPOINT_CANDIDATES:
        candidate = (root / relative).resolve()
        if candidate.is_file():
            return candidate
    return None


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
