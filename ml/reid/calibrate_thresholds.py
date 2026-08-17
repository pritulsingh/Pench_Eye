"""
Calibrate identity-decision thresholds from measured similarity distributions.

    python -m ml.reid.calibrate_thresholds \
        --checkpoint ml/weights/tiger_reid/best.pt \
        --data data/reid --split val --output ml/weights/tiger_reid/thresholds.json

The three thresholds the application uses are chosen from data, not guessed:

* `auto_match_threshold`   — lowest score whose false-accept rate is at or below
  `--target-far` (default 0.01). Above this, two crops are merged into one
  identity automatically, so a false accept means two different tigers are
  recorded as the same animal. That is the most damaging error this system can
  make, hence the FAR-first criterion.
* `review_threshold`       — the score that captures `--review-recall` of true
  same-identity pairs. Between the two thresholds a human decides.
* `new_individual_threshold` — a low percentile of the same-identity
  distribution. Below it, a genuine match is very unlikely, so a new individual
  is the better hypothesis.

The defaults currently shipped in `.env.example` (0.90 / 0.75 / 0.60) are
placeholders and are *not* evidence-based. Replace them with this tool's output.
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

logger = logging.getLogger("ml.reid.calibrate_thresholds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend identity thresholds from measured cosine similarity distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--min-images-per-identity", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="Max acceptable false-accept rate for auto-match (merging two tigers).",
    )
    parser.add_argument(
        "--review-recall",
        type=float,
        default=0.95,
        help="Fraction of true same-identity pairs that must reach human review.",
    )
    parser.add_argument(
        "--new-individual-percentile",
        type=float,
        default=2.0,
        help="Percentile of the same-identity distribution below which a new individual is assumed.",
    )
    parser.add_argument("--output", default=None, help="Write recommendations JSON here.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def pair_scores(embeddings: np.ndarray, labels: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """All distinct pairs split into same-identity and different-identity scores."""
    from ml.reid.metrics import l2_normalize

    normalized = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    if n < 2:
        return np.zeros(0), np.zeros(0)

    sim = normalized @ normalized.T
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    same = labels_arr[:, None] == labels_arr[None, :]
    return sim[same & upper], sim[~same & upper]


def threshold_table(same: np.ndarray, different: np.ndarray) -> List[Dict[str, float]]:
    """Precision / recall / FAR sweep over candidate thresholds."""
    table: List[Dict[str, float]] = []
    for threshold in np.round(np.arange(0.0, 1.0, 0.01), 2):
        true_positive = int((same >= threshold).sum())
        false_positive = int((different >= threshold).sum())
        false_negative = int((same < threshold).sum())
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        table.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tar": float((same >= threshold).mean()) if same.size else 0.0,
                "far": float((different >= threshold).mean()) if different.size else 0.0,
                "true_positive": true_positive,
                "false_positive": false_positive,
            }
        )
    return table


def recommend(
    same: np.ndarray,
    different: np.ndarray,
    *,
    target_far: float,
    review_recall: float,
    new_individual_percentile: float,
) -> Dict[str, Any]:
    if same.size == 0 or different.size == 0:
        return {
            "calibrated": False,
            "reason": (
                "Need both same-identity and different-identity pairs. Provide a split with "
                ">=2 identities and >=2 images per identity."
            ),
        }

    table = threshold_table(same, different)

    # Auto-match: cheapest threshold that meets the FAR budget, then maximise TAR.
    acceptable = [row for row in table if row["far"] <= target_far]
    if acceptable:
        auto_match = max(acceptable, key=lambda r: (r["tar"], -r["threshold"]))["threshold"]
    else:
        auto_match = float(np.percentile(different, 99.9))
        logger.warning(
            "No threshold reaches FAR <= %.4f; falling back to the 99.9th percentile of "
            "different-identity scores (%.4f). The model separates identities poorly.",
            target_far,
            auto_match,
        )

    review = float(np.percentile(same, max(0.0, (1.0 - review_recall) * 100.0)))
    new_individual = float(np.percentile(same, new_individual_percentile))

    # Enforce ordering the decision engine assumes.
    review = min(review, auto_match - 0.01)
    new_individual = min(new_individual, review - 0.01)
    auto_match = float(np.clip(auto_match, 0.05, 0.999))
    review = float(np.clip(review, 0.02, auto_match - 0.01))
    new_individual = float(np.clip(new_individual, 0.0, review - 0.01))

    def row_at(threshold: float) -> Dict[str, float]:
        return min(table, key=lambda r: abs(r["threshold"] - threshold))

    best_f1 = max(table, key=lambda r: r["f1"])
    return {
        "calibrated": True,
        "auto_match_threshold": round(auto_match, 4),
        "review_threshold": round(review, 4),
        "new_individual_threshold": round(new_individual, 4),
        "criteria": {
            "target_far": target_far,
            "review_recall": review_recall,
            "new_individual_percentile": new_individual_percentile,
        },
        "operating_points": {
            "auto_match": row_at(auto_match),
            "review": row_at(review),
            "best_f1": best_f1,
        },
        "distributions": {
            "same_identity": {
                "count": int(same.size),
                "mean": float(same.mean()),
                "std": float(same.std()),
                "percentiles": {f"p{q}": float(np.percentile(same, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            },
            "different_identity": {
                "count": int(different.size),
                "mean": float(different.mean()),
                "std": float(different.std()),
                "percentiles": {
                    f"p{q}": float(np.percentile(different, q)) for q in (1, 5, 25, 50, 75, 95, 99)
                },
            },
            "separation": float(same.mean() - different.mean()),
        },
        "threshold_table": table,
    }


def calibrate(args: argparse.Namespace) -> Dict[str, Any]:
    from ml.reid.checkpoint import load_model_for_inference, write_json
    from ml.reid.dataset import split_records
    from ml.reid.dataset.discovery import load_dataset
    from ml.reid.evaluate import embed_records, select_records
    from ml.reid.train import resolve_device

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    device = resolve_device(args.device)
    model, payload = load_model_for_inference(args.checkpoint, device=device)

    records, summary = load_dataset(args.data, min_images_per_identity=args.min_images_per_identity)
    logger.info("\n%s", summary.format())
    splits = split_records(
        records, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )
    selected = select_records(args, records, splits)
    if not selected:
        raise SystemExit(f"Split '{args.split}' is empty; nothing to calibrate on.")

    embeddings, labels, _, _, _ = embed_records(
        model,
        selected,
        payload.preprocess_config,
        device,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    same, different = pair_scores(embeddings, labels)
    logger.info(
        "Split '%s': %d images, %d identities, %d same-identity pairs, %d different-identity pairs.",
        args.split,
        len(labels),
        len(set(labels)),
        same.size,
        different.size,
    )

    result = recommend(
        same,
        different,
        target_far=args.target_far,
        review_recall=args.review_recall,
        new_individual_percentile=args.new_individual_percentile,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "model_version": payload.model_version,
            "preprocessing_version": payload.preprocess_config.version,
            "split": args.split,
            "num_images": len(labels),
            "num_identities": len(set(labels)),
        }
    )

    if result.get("calibrated"):
        logger.info(
            "\nRecommended thresholds (from measured distributions)\n"
            "  AUTO_MATCH_THRESHOLD      = %.4f   (FAR %.4f, TAR %.4f)\n"
            "  REVIEW_THRESHOLD          = %.4f\n"
            "  NEW_INDIVIDUAL_THRESHOLD  = %.4f\n"
            "  same/different separation = %.4f",
            result["auto_match_threshold"],
            result["operating_points"]["auto_match"]["far"],
            result["operating_points"]["auto_match"]["tar"],
            result["review_threshold"],
            result["new_individual_threshold"],
            result["distributions"]["separation"],
        )
        if result["distributions"]["separation"] < 0.05:
            logger.warning(
                "Same- and different-identity distributions barely separate (%.4f). "
                "These thresholds cannot make the model reliable — it needs more/better training data.",
                result["distributions"]["separation"],
            )
    else:
        logger.warning("Calibration not possible: %s", result.get("reason"))

    if args.output:
        write_json(args.output, result)
        logger.info("Wrote calibration to %s", args.output)
    else:
        compact = {k: v for k, v in result.items() if k != "threshold_table"}
        print(json.dumps(compact, indent=2, default=str))
    return result


def main(argv: Optional[List[str]] = None) -> int:
    calibrate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
