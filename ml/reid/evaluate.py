"""
Evaluate a trained tiger Re-ID checkpoint.

    python -m ml.reid.evaluate --checkpoint ml/weights/tiger_reid/best.pt \
        --data data/reid --split test

Reports Rank-1/5/10, mAP, same/different-identity cosine statistics and
(optionally) a verification ROC with TAR/FAR. Query and gallery are disjoint:
one image per identity is held out as the query, preferring an unseen capture
sequence, and the query is never present in its own gallery.

If evaluation cannot be performed (too few images per identity) that is reported
as such rather than silently producing a number.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("ml.reid.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a tiger Re-ID checkpoint with Rank-k / mAP metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt / latest.pt.")
    parser.add_argument("--data", required=True, help="Dataset root or CSV annotation file.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test", "all"),
        help="Which split to evaluate.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--min-images-per-identity", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Write metrics JSON here.")
    parser.add_argument("--roc", action="store_true", help="Include the verification ROC curve.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def select_records(args, records, splits):
    if args.split == "train":
        return splits.train
    if args.split == "val":
        return splits.val
    if args.split == "test":
        return splits.test
    return list(records)


def embed_records(model, records, preprocess, device: str, batch_size: int, num_workers: int, seed: int):
    """Run the model over records and return (embeddings, labels, sequences, paths)."""
    import torch
    from torch.utils.data import DataLoader

    from ml.reid.augmentation import AugmentationConfig
    from ml.reid.dataset import ReIDDataset
    from ml.reid.dataset.discovery import build_identity_mapping

    identity_to_index = build_identity_mapping(records)
    dataset = ReIDDataset(
        records,
        identity_to_index,
        preprocess=preprocess,
        augmentation=AugmentationConfig.disabled(),
        seed=seed,
        return_quality=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    sequences: List[str] = []
    paths: List[str] = []
    qualities: List[float] = []

    model.eval()
    with torch.no_grad():
        for images, _, indices, quality in loader:
            emb = model(images.to(device))
            embeddings.append(emb.cpu().numpy())
            for idx in indices.cpu().numpy():
                record = dataset.records[int(idx)]
                labels.append(record.identity)
                sequences.append(record.group_key())
                paths.append(str(record.path))
            qualities.extend(float(q) for q in quality)

    if not embeddings:
        return np.zeros((0, 1)), [], [], [], []
    return np.vstack(embeddings), labels, sequences, paths, qualities


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    from ml.reid.checkpoint import load_model_for_inference, write_json
    from ml.reid.dataset import split_records
    from ml.reid.dataset.discovery import load_dataset
    from ml.reid.metrics import (
        compute_roc,
        evaluate_query_gallery,
        evaluate_reid,
        split_query_gallery,
    )
    from ml.reid.train import resolve_device

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    device = resolve_device(args.device)

    model, payload = load_model_for_inference(args.checkpoint, device=device)
    logger.info(
        "Loaded %s | backbone=%s | embedding_dim=%d | trained epochs=%d",
        payload.model_version,
        payload.model_config.backbone,
        payload.model_config.embedding_dim,
        payload.epoch,
    )

    records, summary = load_dataset(args.data, min_images_per_identity=args.min_images_per_identity)
    logger.info("\n%s", summary.format())
    splits = split_records(
        records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    selected = select_records(args, records, splits)
    if not selected:
        raise SystemExit(
            f"Split '{args.split}' is empty. Adjust --val-fraction/--test-fraction or use --split all."
        )

    embeddings, labels, sequences, paths, qualities = embed_records(
        model,
        selected,
        payload.preprocess_config,
        device,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    logger.info("Embedded %d image(s) from split '%s'.", len(labels), args.split)

    loo = evaluate_reid(
        embeddings, labels, image_ids=list(range(len(labels))), sequence_ids=sequences
    )
    logger.info("\nLeave-one-out protocol\n%s", loo.format())

    query_idx, gallery_idx = split_query_gallery(labels, sequence_ids=sequences, seed=args.seed)
    qg = None
    if query_idx and gallery_idx:
        qg = evaluate_query_gallery(
            embeddings[query_idx],
            [labels[i] for i in query_idx],
            embeddings[gallery_idx],
            [labels[i] for i in gallery_idx],
        )
        logger.info("\nQuery/gallery protocol\n%s", qg.format())
    else:
        logger.warning(
            "Query/gallery split not possible — identities need >=2 images each. "
            "Only the leave-one-out protocol was computed."
        )

    result: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model_version": payload.model_version,
        "preprocessing_version": payload.preprocess_config.version,
        "backbone": payload.model_config.backbone,
        "embedding_dim": payload.model_config.embedding_dim,
        "trained_epochs": payload.epoch,
        "split": args.split,
        "num_images": len(labels),
        "num_identities": len(set(labels)),
        "leave_one_out": loo.to_dict(),
        "query_gallery": qg.to_dict() if qq_ok(qg) else None,
        "mean_image_quality": float(np.mean(qualities)) if qualities else None,
        "evaluable": loo.num_queries > 0,
    }
    if args.roc:
        result["roc"] = compute_roc(embeddings, labels)

    if not result["evaluable"]:
        logger.warning(
            "This split produced 0 valid queries, so Rank-k and mAP are NOT measured. "
            "The checkpoint remains unvalidated."
        )

    if args.output:
        write_json(args.output, result)
        logger.info("Wrote metrics to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))
    return result


def qq_ok(metrics) -> bool:
    return metrics is not None and metrics.num_queries > 0


def main(argv: Optional[List[str]] = None) -> int:
    evaluate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
