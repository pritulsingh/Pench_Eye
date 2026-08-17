"""
Train a tiger individual Re-ID model.

    python -m ml.reid.train --data data/reid --output ml/weights/tiger_reid \
        --epochs 50 --batch-size 32 --embedding-dim 512 --device cuda

Objective: ArcFace cross-entropy, optionally plus batch-hard triplet loss.
Model selection: highest validation Rank-1 (a retrieval metric), not validation
loss — loss on an ArcFace head is a poor proxy for retrieval quality.

Running this script without labelled tiger images produces a checkpoint that
predicts nothing useful. It validates that the pipeline executes; it does not
create a working identifier.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("ml.reid.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tiger Re-ID embedding model (ArcFace + optional triplet loss).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data", required=True, help="Dataset root directory or CSV annotation file.")
    data.add_argument("--output", default="ml/weights/tiger_reid", help="Checkpoint output directory.")
    data.add_argument("--experiment-name", default=None, help="Label recorded in the run config.")
    data.add_argument("--val-fraction", type=float, default=0.2)
    data.add_argument("--test-fraction", type=float, default=0.1)
    data.add_argument("--min-images-per-identity", type=int, default=2)
    data.add_argument("--no-verify-images", action="store_true", help="Skip decode check (faster).")
    data.add_argument(
        "--ignore-existing-splits",
        action="store_true",
        help="Re-split even if the dataset already declares train/val/test.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--backbone", default="resnet50", help="resnet18|resnet34|resnet50|osnet_x1_0|tiny")
    model.add_argument("--embedding-dim", type=int, default=512)
    model.add_argument("--no-pretrained", action="store_true", help="Random init instead of ImageNet.")
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--image-size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    model.add_argument("--no-clahe", action="store_true", help="Disable CLAHE preprocessing.")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=50)
    optim.add_argument("--batch-size", type=int, default=32)
    optim.add_argument("--num-instances", type=int, default=4, help="Images per identity per batch (P×K).")
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--weight-decay", type=float, default=5e-4)
    optim.add_argument("--scheduler", default="cosine", choices=("cosine", "step", "none"))
    optim.add_argument("--step-size", type=int, default=20)
    optim.add_argument("--gamma", type=float, default=0.1)
    optim.add_argument("--warmup-epochs", type=int, default=3)
    optim.add_argument("--arcface-scale", type=float, default=30.0)
    optim.add_argument("--arcface-margin", type=float, default=0.30)
    optim.add_argument("--triplet-weight", type=float, default=1.0, help="0 disables the triplet term.")
    optim.add_argument("--triplet-margin", type=float, default=0.3)
    optim.add_argument("--label-smoothing", type=float, default=0.1)
    optim.add_argument("--no-augmentation", action="store_true")
    optim.add_argument(
        "--horizontal-flip",
        action="store_true",
        help="Mirror crops. Off by default: left/right flanks are different patterns.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto", help="auto|cpu|cuda")
    runtime.add_argument("--num-workers", type=int, default=0)
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only).")
    runtime.add_argument("--no-amp", action="store_true", help="Force AMP off even on CUDA.")
    runtime.add_argument("--checkpoint-every", type=int, default=1)
    runtime.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    runtime.add_argument("--early-stopping-patience", type=int, default=0, help="0 disables.")
    runtime.add_argument("--eval-every", type=int, default=1)
    runtime.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch (smoke tests).")
    runtime.add_argument("--log-level", default="INFO")
    return parser


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # cuDNN conv has no deterministic path here


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_components: Dict[str, float]
    val_loss: Optional[float] = None
    val_rank1: Optional[float] = None
    val_rank5: Optional[float] = None
    val_map: Optional[float] = None
    seconds: float = 0.0
    lr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_components": self.train_components,
            "val_loss": self.val_loss,
            "val_rank1": self.val_rank1,
            "val_rank5": self.val_rank5,
            "val_map": self.val_map,
            "seconds": round(self.seconds, 2),
            "lr": self.lr,
        }


def _make_scheduler(args, optimizer, steps_per_epoch: int):
    import torch

    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    total = max(1, args.epochs * max(1, steps_per_epoch))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)


def _warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return float(epoch + 1) / float(warmup_epochs)


def evaluate_split(model, loader, device: str, criterion=None, arcface=None):
    """Embed a split and compute leave-one-out Re-ID metrics."""
    import torch

    from ml.reid.metrics import evaluate_reid

    model.eval()
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    indices: List[int] = []
    total_loss, batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            images, batch_labels, batch_indices = batch[0], batch[1], batch[2]
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            emb, bn_feature = model(images, return_logits_feature=True)
            if criterion is not None and arcface is not None:
                logits = arcface(bn_feature, batch_labels)
                total_loss += float(criterion(logits, emb, batch_labels).total)
                batches += 1

            embeddings.append(emb.detach().cpu().numpy())
            labels.extend(int(v) for v in batch_labels.detach().cpu().numpy())
            indices.extend(int(v) for v in batch_indices.detach().cpu().numpy())

    if not embeddings:
        return None, evaluate_reid(np.zeros((0, 1)), [])

    stacked = np.vstack(embeddings)
    records = getattr(loader.dataset, "records", [])
    sequence_ids = [records[i].group_key() for i in indices] if records else None

    metrics = evaluate_reid(stacked, labels, image_ids=indices, sequence_ids=sequence_ids)
    val_loss = (total_loss / batches) if batches else None
    return val_loss, metrics


def train(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from ml.reid.augmentation import AugmentationConfig
    from ml.reid.checkpoint import (
        BEST_CHECKPOINT_NAME,
        CLASS_MAPPING_NAME,
        CONFIG_NAME,
        HISTORY_NAME,
        LATEST_CHECKPOINT_NAME,
        load_raw_checkpoint,
        make_model_version,
        save_checkpoint,
        write_json,
    )
    from ml.reid.dataset import build_dataloaders, split_records
    from ml.reid.dataset.discovery import build_identity_mapping, load_dataset
    from ml.reid.losses import ReIDLoss
    from ml.reid.model import ArcFaceHead, ModelConfig, build_model
    from ml.reid.preprocessing import PreprocessConfig

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    records, summary = load_dataset(
        args.data,
        min_images_per_identity=args.min_images_per_identity,
        verify_images=not args.no_verify_images,
    )
    logger.info("\n%s", summary.format())
    if not records:
        raise SystemExit(
            "No usable images found. Expected data/reid/{split}/{IDENTITY}/*.jpg or a CSV "
            "with image_path,identity_id columns. See docs/reid_training.md."
        )

    splits = split_records(
        records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        respect_existing_splits=not args.ignore_existing_splits,
    )
    logger.info("\n%s", splits.format())
    if not splits.train:
        raise SystemExit("Split produced an empty training set; check dataset size and fractions.")

    identity_to_index = build_identity_mapping(records)
    num_classes = len(identity_to_index)
    if num_classes < 2:
        raise SystemExit(
            f"Re-ID needs at least 2 identities; found {num_classes}. "
            "A single-identity dataset cannot teach the model to discriminate."
        )

    preprocess = PreprocessConfig(
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        use_clahe=not args.no_clahe,
    )
    augmentation = (
        AugmentationConfig.disabled()
        if args.no_augmentation
        else AugmentationConfig(horizontal_flip=args.horizontal_flip)
    )

    train_loader, val_loader, identity_to_index = build_dataloaders(
        splits.train,
        splits.val,
        identity_to_index,
        preprocess=preprocess,
        augmentation=augmentation,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        num_workers=args.num_workers,
        seed=args.seed,
        use_pk_sampler=args.triplet_weight > 0,
    )

    # ── Model / optimiser ─────────────────────────────────────────────────
    model_config = ModelConfig(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    )
    model = build_model(model_config).to(device)
    arcface = ArcFaceHead(
        args.embedding_dim, num_classes, scale=args.arcface_scale, margin=args.arcface_margin
    ).to(device)
    criterion = ReIDLoss(
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}, {"params": arcface.parameters()}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = _make_scheduler(args, optimizer, steps_per_epoch)

    use_amp = device == "cuda" and args.amp and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else None

    run_config: Dict[str, Any] = {
        "experiment_name": args.experiment_name or output_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
        "device": device,
        "amp": use_amp,
        "num_classes": num_classes,
        "model_config": model_config.to_dict(),
        "preprocess_config": preprocess.to_dict(),
        "augmentation_config": augmentation.to_dict(),
        "dataset": {
            "total_images": summary.total_images,
            "total_identities": summary.total_identities,
            "total_sequences": summary.total_sequences,
            "split_counts": splits.counts(),
            "split_identity_counts": splits.identity_counts(),
            "train_only_identities": splits.train_only_identities,
            "unreadable_images": len(summary.unreadable),
            "dropped_identities": summary.dropped_identities,
        },
    }
    write_json(output_dir / CONFIG_NAME, run_config)
    write_json(
        output_dir / CLASS_MAPPING_NAME,
        {
            "identity_to_index": identity_to_index,
            "index_to_identity": {str(v): k for k, v in identity_to_index.items()},
        },
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_rank1 = -1.0
    history: List[Dict[str, Any]] = []

    if args.resume:
        raw = load_raw_checkpoint(args.resume)
        model.load_state_dict(raw["model_state_dict"], strict=False)
        if raw.get("arcface_state_dict"):
            try:
                arcface.load_state_dict(raw["arcface_state_dict"])
            except Exception as exc:
                logger.warning("Could not restore ArcFace head (%s); reinitialised.", exc)
        if raw.get("optimizer_state_dict"):
            try:
                optimizer.load_state_dict(raw["optimizer_state_dict"])
            except Exception as exc:
                logger.warning("Could not restore optimizer state (%s).", exc)
        if scheduler is not None and raw.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(raw["scheduler_state_dict"])
            except Exception as exc:
                logger.warning("Could not restore scheduler state (%s).", exc)
        if scaler is not None and raw.get("scaler_state_dict"):
            try:
                scaler.load_state_dict(raw["scaler_state_dict"])
            except Exception as exc:
                logger.warning("Could not restore AMP scaler state (%s).", exc)
        start_epoch = int(raw.get("epoch", 0))
        best_rank1 = float((raw.get("metrics") or {}).get("val_rank1", -1.0))
        history_path = output_dir / HISTORY_NAME
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
        logger.info("Resumed from %s at epoch %d (best Rank-1 %.4f).", args.resume, start_epoch, best_rank1)

    logger.info(
        "Training %s | %d identities | %d train / %d val images | device=%s | AMP=%s",
        args.backbone,
        num_classes,
        len(splits.train),
        len(splits.val),
        device,
        use_amp,
    )

    epochs_without_improvement = 0
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        arcface.train()

        factor = _warmup_factor(epoch, args.warmup_epochs)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor

        running: Dict[str, float] = {}
        seen_batches = 0

        for step, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    embeddings, bn_feature = model(images, return_logits_feature=True)
                    logits = arcface(bn_feature, labels)
                    loss_out = criterion(logits, embeddings, labels)
                scaler.scale(loss_out.total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                embeddings, bn_feature = model(images, return_logits_feature=True)
                logits = arcface(bn_feature, labels)
                loss_out = criterion(logits, embeddings, labels)
                loss_out.total.backward()
                optimizer.step()

            if scheduler is not None and args.scheduler == "cosine" and factor >= 1.0:
                scheduler.step()

            for key, value in loss_out.components.items():
                running[key] = running.get(key, 0.0) + value
            seen_batches += 1

            if step % 20 == 0:
                logger.info(
                    "  epoch %d step %d/%d  loss=%.4f  %s",
                    epoch + 1,
                    step,
                    len(train_loader),
                    loss_out.components["total"],
                    " ".join(f"{k}={v:.4f}" for k, v in loss_out.components.items() if k != "total"),
                )

        if scheduler is not None and args.scheduler == "step":
            scheduler.step()

        components = {k: v / max(1, seen_batches) for k, v in running.items()}
        metrics = EpochMetrics(
            epoch=epoch + 1,
            train_loss=components.get("total", 0.0),
            train_components=components,
            lr=optimizer.param_groups[0]["lr"],
        )

        should_eval = val_loader is not None and ((epoch + 1) % max(1, args.eval_every) == 0)
        if should_eval:
            val_loss, val_metrics = evaluate_split(model, val_loader, device, criterion, arcface)
            metrics.val_loss = val_loss
            metrics.val_rank1 = val_metrics.rank1
            metrics.val_rank5 = val_metrics.rank5
            metrics.val_map = val_metrics.mean_ap
            if val_metrics.num_queries == 0:
                logger.warning(
                    "Validation produced 0 valid queries — each identity needs >=2 images "
                    "in separate capture sequences. Rank-1 is not meaningful."
                )

        metrics.seconds = time.time() - epoch_start
        history.append(metrics.to_dict())

        logger.info(
            "epoch %d/%d  train_loss=%.4f%s  (%.1fs)",
            epoch + 1,
            args.epochs,
            metrics.train_loss,
            ""
            if metrics.val_rank1 is None
            else f"  val_loss={metrics.val_loss:.4f}  Rank-1={metrics.val_rank1:.4f}  mAP={metrics.val_map:.4f}",
            metrics.seconds,
        )

        checkpoint_metrics = {
            "train_loss": metrics.train_loss,
            **({"val_loss": metrics.val_loss} if metrics.val_loss is not None else {}),
            **({"val_rank1": metrics.val_rank1} if metrics.val_rank1 is not None else {}),
            **({"val_rank5": metrics.val_rank5} if metrics.val_rank5 is not None else {}),
            **({"val_map": metrics.val_map} if metrics.val_map is not None else {}),
        }

        if (epoch + 1) % max(1, args.checkpoint_every) == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                output_dir / LATEST_CHECKPOINT_NAME,
                model=model,
                identity_to_index=identity_to_index,
                preprocess_config=preprocess,
                epoch=epoch + 1,
                metrics=checkpoint_metrics,
                train_config=run_config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                arcface_state=arcface.state_dict(),
            )

        # Best = highest validation Rank-1; with no val split, fall back to the
        # last epoch so `best.pt` always exists.
        current = metrics.val_rank1 if metrics.val_rank1 is not None else -1.0
        improved = current > best_rank1 or (metrics.val_rank1 is None and epoch + 1 == args.epochs)
        if improved:
            best_rank1 = max(best_rank1, current)
            save_checkpoint(
                output_dir / BEST_CHECKPOINT_NAME,
                model=model,
                identity_to_index=identity_to_index,
                preprocess_config=preprocess,
                epoch=epoch + 1,
                metrics=checkpoint_metrics,
                train_config=run_config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                arcface_state=arcface.state_dict(),
                model_version=make_model_version(
                    args.backbone, args.embedding_dim, epoch + 1, metrics.val_rank1
                ),
            )
            epochs_without_improvement = 0
            logger.info("  ↳ new best checkpoint (Rank-1 %.4f)", max(best_rank1, 0.0))
        else:
            epochs_without_improvement += 1

        write_json(
            output_dir / HISTORY_NAME,
            {"best_val_rank1": best_rank1 if best_rank1 >= 0 else None, "epochs": history},
        )

        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            logger.info(
                "Early stopping: no validation improvement for %d epoch(s).",
                epochs_without_improvement,
            )
            break

    # ── Held-out test split ───────────────────────────────────────────────
    test_metrics = None
    if splits.test:
        from torch.utils.data import DataLoader

        from ml.reid.augmentation import AugmentationConfig as AugCfg
        from ml.reid.dataset import ReIDDataset

        test_loader = DataLoader(
            ReIDDataset(
                splits.test,
                identity_to_index,
                preprocess=preprocess,
                augmentation=AugCfg.disabled(),
                seed=args.seed,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        _, test_metrics = evaluate_split(model, test_loader, device)
        logger.info("\nHeld-out test split:\n%s", test_metrics.format())

    result = {
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / BEST_CHECKPOINT_NAME),
        "latest_checkpoint": str(output_dir / LATEST_CHECKPOINT_NAME),
        "best_val_rank1": best_rank1 if best_rank1 >= 0 else None,
        "epochs_completed": len(history),
        "num_identities": num_classes,
        "test_metrics": test_metrics.to_dict() if test_metrics else None,
    }
    write_json(
        output_dir / HISTORY_NAME,
        {
            "best_val_rank1": best_rank1 if best_rank1 >= 0 else None,
            "epochs": history,
            "test_metrics": result["test_metrics"],
        },
    )

    logger.info("\nTraining finished. Best checkpoint: %s", result["best_checkpoint"])
    if best_rank1 < 0:
        logger.warning(
            "No validation Rank-1 was measured, so this checkpoint is UNVALIDATED. "
            "Run `python -m ml.reid.evaluate` on a held-out split before trusting it."
        )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
