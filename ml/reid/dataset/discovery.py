"""
Dataset discovery for tiger Re-ID.

Two input formats are supported:

1. Directory layout (preferred)

       data/reid/
       ├── train/TIGER_001/img_001.jpg
       ├── val/TIGER_001/img_004.jpg
       └── test/TIGER_002/img_009.jpg

   A split directory level is optional. When absent, all identities are read
   from the root and split later by `ml.reid.dataset.splitting`.

2. Flat CSV annotations

       image_path,identity_id,split,sequence_id,flank
       data/img001.jpg,TIGER_001,train,SEQ_A,left

   Only `image_path` and `identity_id` are required.

Capture sequences matter: consecutive frames from one camera burst are near
duplicates, so letting them straddle train and evaluation inflates every metric.
When no `sequence_id` is supplied it is inferred from the filename (see
`infer_sequence_id`), which keeps bursts named `SEQ12_0001.jpg`,
`CAM003_20240117_0002.jpg` etc. together.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")

# Trailing frame counter, e.g. IMG_0007 / seq3-04 / burst_012
_FRAME_SUFFIX = re.compile(r"[_\-.]?(\d{1,6})$")


@dataclass
class ImageRecord:
    """One identity-labelled image on disk."""

    path: Path
    identity: str
    split: Optional[str] = None
    sequence_id: Optional[str] = None
    flank: Optional[str] = None

    def group_key(self) -> str:
        """Key used to keep near-duplicate frames in the same split."""
        return f"{self.identity}::{self.sequence_id or self.path.stem}"


@dataclass
class IdentityStats:
    identity: str
    image_count: int
    sequence_count: int
    splits: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetSummary:
    root: Optional[Path]
    total_images: int
    total_identities: int
    total_sequences: int
    identities: List[IdentityStats]
    split_counts: Dict[str, int]
    unreadable: List[Path]
    dropped_identities: List[Tuple[str, int]]

    def format(self) -> str:
        lines = [
            "Dataset summary",
            f"  root            : {self.root or '(csv annotations)'}",
            f"  images          : {self.total_images}",
            f"  identities      : {self.total_identities}",
            f"  sequences       : {self.total_sequences}",
        ]
        if self.split_counts:
            splits = ", ".join(f"{k}={v}" for k, v in sorted(self.split_counts.items()))
            lines.append(f"  split counts    : {splits}")
        if self.total_identities:
            counts = [i.image_count for i in self.identities]
            lines.append(
                f"  images/identity : min={min(counts)} max={max(counts)} "
                f"mean={sum(counts) / len(counts):.1f}"
            )
        if self.dropped_identities:
            dropped = ", ".join(f"{n}({c})" for n, c in self.dropped_identities)
            lines.append(f"  dropped (too few images): {dropped}")
        if self.unreadable:
            lines.append(f"  unreadable files: {len(self.unreadable)}")
            for p in self.unreadable[:5]:
                lines.append(f"      {p}")
            if len(self.unreadable) > 5:
                lines.append(f"      … {len(self.unreadable) - 5} more")
        return "\n".join(lines)


def infer_sequence_id(path: Path) -> str:
    """
    Group near-duplicate frames by stripping a trailing frame counter.

    `CAM003_20240117_0002.jpg` and `CAM003_20240117_0003.jpg` both map to
    `CAM003_20240117`. Files with no counter fall back to their own stem, so
    unrelated images are never merged.
    """
    stem = path.stem
    match = _FRAME_SUFFIX.search(stem)
    if match and match.start() > 0:
        return stem[: match.start()]
    return stem


def infer_flank(path: Path) -> Optional[str]:
    """Read a left/right flank hint out of the filename, if present."""
    lowered = path.stem.lower()
    if "left" in lowered or lowered.endswith("_l"):
        return "left"
    if "right" in lowered or lowered.endswith("_r"):
        return "right"
    return None


def is_readable_image(path: Path) -> bool:
    """Verify an image decodes, so corrupt files fail at discovery not mid-epoch."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _iter_identity_dirs(root: Path) -> Iterable[Tuple[Path, Optional[str]]]:
    """Yield (identity_dir, split) pairs, tolerating a missing split level."""
    split_dirs = [root / s for s in SPLIT_NAMES if (root / s).is_dir()]
    if split_dirs:
        for split_dir in split_dirs:
            for identity_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                yield identity_dir, split_dir.name
        return
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        yield identity_dir, None


def discover_dataset(
    root: str | Path,
    *,
    min_images_per_identity: int = 2,
    verify_images: bool = True,
) -> Tuple[List[ImageRecord], DatasetSummary]:
    """
    Walk a directory-layout dataset.

    Identities with fewer than `min_images_per_identity` usable images are
    dropped: a single image cannot appear in both a gallery and a query, so such
    identities silently break Rank-k evaluation.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    by_identity: Dict[str, List[ImageRecord]] = {}
    unreadable: List[Path] = []

    for identity_dir, split in _iter_identity_dirs(root):
        identity = identity_dir.name
        for image_path in sorted(identity_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if verify_images and not is_readable_image(image_path):
                unreadable.append(image_path)
                continue
            by_identity.setdefault(identity, []).append(
                ImageRecord(
                    path=image_path,
                    identity=identity,
                    split=split,
                    sequence_id=infer_sequence_id(image_path),
                    flank=infer_flank(image_path),
                )
            )

    records, dropped = _filter_identities(by_identity, min_images_per_identity)
    return records, summarize(records, root=root, unreadable=unreadable, dropped=dropped)


def load_csv_annotations(
    csv_path: str | Path,
    *,
    root: Optional[str | Path] = None,
    min_images_per_identity: int = 2,
    verify_images: bool = True,
) -> Tuple[List[ImageRecord], DatasetSummary]:
    """Load a flat CSV annotation file. Relative paths resolve against `root`."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")
    base = Path(root).expanduser() if root else csv_path.parent

    by_identity: Dict[str, List[ImageRecord]] = {}
    unreadable: List[Path] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"image_path", "identity_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing)}")

        for row in reader:
            raw_path = (row.get("image_path") or "").strip()
            identity = (row.get("identity_id") or "").strip()
            if not raw_path or not identity:
                continue
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = base / image_path
            if not image_path.is_file():
                unreadable.append(image_path)
                continue
            if verify_images and not is_readable_image(image_path):
                unreadable.append(image_path)
                continue

            split = (row.get("split") or "").strip().lower() or None
            sequence = (row.get("sequence_id") or "").strip() or infer_sequence_id(image_path)
            flank = (row.get("flank") or "").strip().lower() or infer_flank(image_path)
            by_identity.setdefault(identity, []).append(
                ImageRecord(
                    path=image_path,
                    identity=identity,
                    split=split,
                    sequence_id=sequence,
                    flank=flank,
                )
            )

    records, dropped = _filter_identities(by_identity, min_images_per_identity)
    return records, summarize(records, root=base, unreadable=unreadable, dropped=dropped)


def load_dataset(
    source: str | Path,
    *,
    min_images_per_identity: int = 2,
    verify_images: bool = True,
) -> Tuple[List[ImageRecord], DatasetSummary]:
    """Dispatch to CSV or directory loading based on the path type."""
    path = Path(source).expanduser()
    if path.is_file() and path.suffix.lower() == ".csv":
        return load_csv_annotations(
            path,
            min_images_per_identity=min_images_per_identity,
            verify_images=verify_images,
        )
    return discover_dataset(
        path,
        min_images_per_identity=min_images_per_identity,
        verify_images=verify_images,
    )


def _filter_identities(
    by_identity: Dict[str, List[ImageRecord]], minimum: int
) -> Tuple[List[ImageRecord], List[Tuple[str, int]]]:
    records: List[ImageRecord] = []
    dropped: List[Tuple[str, int]] = []
    for identity in sorted(by_identity):
        items = by_identity[identity]
        if len(items) < minimum:
            dropped.append((identity, len(items)))
            logger.warning(
                "Identity %s has only %d image(s); minimum is %d — dropped.",
                identity,
                len(items),
                minimum,
            )
            continue
        records.extend(items)
    return records, dropped


def build_identity_mapping(records: Sequence[ImageRecord]) -> Dict[str, int]:
    """Stable identity → contiguous class index map (sorted, so reproducible)."""
    return {identity: idx for idx, identity in enumerate(sorted({r.identity for r in records}))}


def summarize(
    records: Sequence[ImageRecord],
    *,
    root: Optional[Path] = None,
    unreadable: Optional[Sequence[Path]] = None,
    dropped: Optional[Sequence[Tuple[str, int]]] = None,
) -> DatasetSummary:
    per_identity: Dict[str, List[ImageRecord]] = {}
    for record in records:
        per_identity.setdefault(record.identity, []).append(record)

    stats: List[IdentityStats] = []
    split_counts: Dict[str, int] = {}
    sequences: set[str] = set()

    for identity in sorted(per_identity):
        items = per_identity[identity]
        identity_splits: Dict[str, int] = {}
        for item in items:
            sequences.add(item.group_key())
            key = item.split or "unassigned"
            identity_splits[key] = identity_splits.get(key, 0) + 1
            split_counts[key] = split_counts.get(key, 0) + 1
        stats.append(
            IdentityStats(
                identity=identity,
                image_count=len(items),
                sequence_count=len({i.group_key() for i in items}),
                splits=identity_splits,
            )
        )

    return DatasetSummary(
        root=root,
        total_images=len(records),
        total_identities=len(per_identity),
        total_sequences=len(sequences),
        identities=stats,
        split_counts=split_counts,
        unreadable=list(unreadable or []),
        dropped_identities=list(dropped or []),
    )
