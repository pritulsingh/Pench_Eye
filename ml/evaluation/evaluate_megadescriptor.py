from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ml.megadescriptor import MegaDescriptor
from ml.reid.metrics import evaluate_reid

from .amur_dataset import inspect_records, load_records


DEFAULT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def _verification_pairs(embeddings: np.ndarray, labels: list[str], sequences: list[str | None]):
    similarities = embeddings @ embeddings.T
    same_scores, different_scores = [], []
    has_sequences = all(sequences)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if has_sequences and sequences[i] == sequences[j]:
                continue
            target = same_scores if labels[i] == labels[j] else different_scores
            target.append(float(similarities[i, j]))
    return np.asarray(same_scores), np.asarray(different_scores), has_sequences


def calibrate_thresholds(same: np.ndarray, different: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
    rows = []
    for threshold in thresholds:
        tp = int((same >= threshold).sum())
        fn = int((same < threshold).sum())
        fp = int((different >= threshold).sum())
        tn = int((different < threshold).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "false_match_rate": fp / (fp + tn) if fp + tn else 0.0,
                "false_non_match_rate": fn / (fn + tp) if fn + tp else 0.0,
            }
        )
    return rows


def _distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {key: None for key in ("min", "mean", "median", "max")}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def evaluate(dataset: str | Path, output: str | Path, model_name: str) -> dict:
    records, warnings = load_records(dataset)
    inspection = inspect_records(records)
    usable = [r for r in inspection["usable_records"] if r.individual_id]
    if len({r.individual_id for r in usable}) < 2:
        raise ValueError("At least two labelled individuals are required for Re-ID evaluation.")
    model = MegaDescriptor(model_name=model_name)
    embeddings = np.asarray([model.get_embedding(r.path) for r in usable], dtype=np.float32)
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    labels = [r.individual_id for r in usable]
    sequences = [r.sequence_id for r in usable]
    metrics = evaluate_reid(
        embeddings,
        labels,
        image_ids=[str(r.path) for r in usable],
        sequence_ids=sequences if all(sequences) else None,
        ranks=(1, 5),
        allow_same_sequence_fallback=False,
    )
    same, different, sequence_filtered = _verification_pairs(embeddings, labels, sequences)
    threshold_rows = calibrate_thresholds(same, different)
    recommended = max(threshold_rows, key=lambda row: (row["f1"], -row["false_match_rate"])) if same.size and different.size else None
    report = {
        "dataset": str(Path(dataset).expanduser().resolve()),
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
        "metrics": metrics.to_dict(),
        "same_tiger_similarity": _distribution(same),
        "different_tiger_similarity": _distribution(different),
        "thresholds": threshold_rows,
        "recommended_threshold_by_pairwise_f1": recommended,
        "sequence_pairs_excluded": sequence_filtered,
        "warnings": warnings,
        "conclusion": "Baseline metrics generated. Domain experts must define acceptable operating targets before deciding on fine-tuning.",
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (destination / "thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    try:
        import matplotlib.pyplot as plt

        plt.hist(same, bins=30, alpha=0.6, label="Same individual")
        plt.hist(different, bins=30, alpha=0.6, label="Different individuals")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Pair count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / "similarity_distributions.png", dpi=160)
        plt.close()
    except ImportError:
        report["warnings"].append("matplotlib unavailable; similarity plot was not generated.")
        (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pretrained MegaDescriptor on labelled Amur tigers")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="artifacts/amur_evaluation")
    parser.add_argument("--model-name", default="hf-hub:BVRA/MegaDescriptor-T-224")
    args = parser.parse_args()
    report = evaluate(args.dataset, args.output, args.model_name)
    metrics = report["metrics"]
    print(f"Top-1 / Recall@1: {metrics['rank1']:.4f}")
    print(f"Top-5 / Recall@5: {metrics['rank5']:.4f}")
    print(f"mAP: {metrics['mean_ap']:.4f}")
    print(f"Report: {Path(args.output) / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())