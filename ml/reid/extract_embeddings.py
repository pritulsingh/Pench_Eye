"""
Batch-extract embeddings from a trained checkpoint.

    python -m ml.reid.extract_embeddings \
        --checkpoint ml/weights/tiger_reid/best.pt \
        --input data/reid --output embeddings.parquet

Each row carries the image path, identity, 512-d embedding, model version and
preprocessing version. The version columns matter: embeddings from different
models or different preprocessing are not comparable, and mixing them silently
corrupts similarity search.

Parquet is used when pyarrow/pandas are available; otherwise the writer falls
back to JSONL or NPZ based on the output suffix.
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

logger = logging.getLogger("ml.reid.extract_embeddings")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract 512-d Re-ID embeddings for a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Dataset root, CSV annotations, or one image.")
    parser.add_argument("--output", required=True, help="Output .parquet / .jsonl / .npz path.")
    parser.add_argument("--split", default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--min-images-per-identity", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-quality", action="store_true", help="Add a sharpness column.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def write_table(rows: List[Dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    if suffix == ".parquet":
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(output, index=False)
            return output
        except ImportError:
            fallback = output.with_suffix(".jsonl")
            logger.warning(
                "pandas/pyarrow unavailable for Parquet output; writing JSONL to %s instead.", fallback
            )
            output = fallback
            suffix = ".jsonl"

    if suffix == ".npz":
        np.savez_compressed(
            output,
            embeddings=np.array([r["embedding"] for r in rows], dtype=np.float32),
            image_paths=np.array([r["image_path"] for r in rows]),
            identities=np.array([r["identity"] for r in rows]),
            model_version=np.array([rows[0]["model_version"]] if rows else []),
            preprocessing_version=np.array([rows[0]["preprocessing_version"]] if rows else []),
        )
        return output

    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return output


def extract(args: argparse.Namespace) -> Dict[str, Any]:
    from ml.reid.checkpoint import load_model_for_inference
    from ml.reid.dataset import split_records
    from ml.reid.dataset.discovery import ImageRecord, infer_flank, infer_sequence_id, load_dataset
    from ml.reid.evaluate import embed_records, select_records
    from ml.reid.train import resolve_device

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    device = resolve_device(args.device)
    model, payload = load_model_for_inference(args.checkpoint, device=device)
    logger.info("Loaded %s (embedding_dim=%d)", payload.model_version, payload.model_config.embedding_dim)

    input_path = Path(args.input).expanduser()
    if input_path.is_file() and input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        selected = [
            ImageRecord(
                path=input_path,
                identity=input_path.parent.name or "UNKNOWN",
                sequence_id=infer_sequence_id(input_path),
                flank=infer_flank(input_path),
            )
        ]
    else:
        records, summary = load_dataset(
            input_path, min_images_per_identity=args.min_images_per_identity
        )
        logger.info("\n%s", summary.format())
        splits = split_records(
            records, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
        )
        selected = select_records(args, records, splits)

    if not selected:
        raise SystemExit("No images to embed.")

    embeddings, labels, sequences, paths, qualities = embed_records(
        model,
        selected,
        payload.preprocess_config,
        device,
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    flank_by_path = {str(r.path): (r.flank or "unknown") for r in selected}
    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(paths):
        row: Dict[str, Any] = {
            "image_path": path,
            "identity": labels[i],
            "sequence_id": sequences[i],
            "flank": flank_by_path.get(path, "unknown"),
            "embedding": [float(v) for v in embeddings[i]],
            "embedding_dim": int(embeddings.shape[1]),
            "model_version": payload.model_version,
            "preprocessing_version": payload.preprocess_config.version,
        }
        if args.include_quality:
            row["quality_score"] = float(qualities[i]) if i < len(qualities) else None
        rows.append(row)

    written = write_table(rows, args.output)
    norms = np.linalg.norm(embeddings, axis=1) if len(embeddings) else np.zeros(1)
    logger.info(
        "Wrote %d embedding(s) to %s | dim=%d | L2 norm mean=%.6f",
        len(rows),
        written,
        embeddings.shape[1] if len(embeddings) else 0,
        float(norms.mean()),
    )
    return {
        "output": str(written),
        "count": len(rows),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
        "model_version": payload.model_version,
        "preprocessing_version": payload.preprocess_config.version,
    }


def main(argv: Optional[List[str]] = None) -> int:
    extract(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
