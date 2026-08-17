from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AmurRecord:
    path: Path
    individual_id: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None
    sequence_id: Optional[str] = None


def _metadata_path(dataset: Path) -> Optional[Path]:
    if dataset.is_file() and dataset.suffix.lower() == ".csv":
        return dataset
    candidate = dataset / "metadata.csv"
    return candidate if candidate.is_file() else None


def load_records(dataset: str | Path) -> tuple[list[AmurRecord], list[str]]:
    source = Path(dataset).expanduser().resolve()
    metadata = _metadata_path(source)
    warnings: list[str] = []
    if metadata:
        base = metadata.parent
        records: list[AmurRecord] = []
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if "image_path" not in fields:
                raise ValueError(f"{metadata} must contain an image_path column")
            identity_key = "individual_id" if "individual_id" in fields else "identity_id" if "identity_id" in fields else None
            if identity_key is None:
                warnings.append("No individual_id column: individual-level Re-ID metrics are unavailable.")
            if "sequence_id" not in fields:
                warnings.append("No sequence_id column: capture-event leakage cannot be fully prevented.")
            for row in reader:
                raw = (row.get("image_path") or "").strip()
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = base / path
                records.append(
                    AmurRecord(
                        path=path.resolve(),
                        individual_id=(row.get(identity_key) or "").strip() or None if identity_key else None,
                        camera_id=(row.get("camera_id") or "").strip() or None,
                        timestamp=(row.get("timestamp") or "").strip() or None,
                        sequence_id=(row.get("sequence_id") or "").strip() or None,
                    )
                )
        return records, warnings

    root = source / "images" if (source / "images").is_dir() else source
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {source}")
    records = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        relative = path.relative_to(root)
        identity = relative.parts[0] if len(relative.parts) > 1 else None
        records.append(AmurRecord(path=path.resolve(), individual_id=identity))
    warnings.extend(
        [
            "No metadata.csv found; identity labels are inferred only from identity subdirectories.",
            "No sequence metadata: capture-event leakage cannot be fully prevented.",
        ]
    )
    return records, warnings


def inspect_records(records: Iterable[AmurRecord]) -> dict:
    records = list(records)
    missing: list[str] = []
    corrupt: list[str] = []
    hashes: dict[str, list[str]] = {}
    usable: list[AmurRecord] = []
    for record in records:
        if not record.path.is_file():
            missing.append(str(record.path))
            continue
        try:
            digest = hashlib.sha256(record.path.read_bytes()).hexdigest()
            with Image.open(record.path) as image:
                image.verify()
        except Exception:
            corrupt.append(str(record.path))
            continue
        hashes.setdefault(digest, []).append(str(record.path))
        usable.append(record)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "declared_images": len(records),
        "usable_images": len(usable),
        "missing_files": missing,
        "corrupt_files": corrupt,
        "duplicate_groups": duplicate_groups,
        "usable_records": usable,
    }