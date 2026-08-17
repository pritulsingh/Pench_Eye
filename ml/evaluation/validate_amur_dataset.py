from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .amur_dataset import inspect_records, load_records


def build_report(dataset: str | Path) -> dict:
    records, warnings = load_records(dataset)
    inspection = inspect_records(records)
    usable = inspection.pop("usable_records")
    individuals = Counter(r.individual_id for r in usable if r.individual_id)
    cameras = Counter(r.camera_id for r in usable if r.camera_id)
    counts = list(individuals.values())
    return {
        "dataset": str(Path(dataset).expanduser().resolve()),
        **inspection,
        "individuals": len(individuals),
        "cameras": len(cameras),
        "missing_individual_metadata": sum(not r.individual_id for r in usable),
        "missing_camera_metadata": sum(not r.camera_id for r in usable),
        "missing_timestamp_metadata": sum(not r.timestamp for r in usable),
        "missing_sequence_metadata": sum(not r.sequence_id for r in usable),
        "images_per_individual": {
            "min": min(counts) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "max": max(counts) if counts else None,
            "counts": dict(sorted(individuals.items())),
        },
        "images_per_camera": dict(sorted(cameras.items())),
        "warnings": warnings,
    }


def format_report(report: dict) -> str:
    stats = report["images_per_individual"]
    lines = [
        "Amur Tiger Dataset",
        "===================",
        f"Images:           {report['usable_images']} / {report['declared_images']} usable",
        f"Individuals:      {report['individuals']}",
        f"Cameras:          {report['cameras']}",
        f"Missing files:    {len(report['missing_files'])}",
        f"Corrupt images:   {len(report['corrupt_files'])}",
        f"Duplicate groups: {len(report['duplicate_groups'])}",
        "",
        "Images per individual",
        f"Min: {stats['min']}  Median: {stats['median']}  Max: {stats['max']}",
    ]
    if report["warnings"]:
        lines.extend(["", "Limitations", *[f"- {item}" for item in report["warnings"]]])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an Amur tiger dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    report = build_report(args.dataset)
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())