"""
Split records into train / val / test without leaking capture sequences.

Two rules drive this module:

1. **Sequence-level splitting.** Frames from one camera burst are near
   duplicates. If one lands in train and its neighbour in val, validation
   Rank-1 measures memorisation, not recognition. Splitting happens on
   `ImageRecord.group_key()` (identity + sequence), never on single images.

2. **Closed-set identities.** Every identity should appear in all three splits
   so that a query always has a gallery mate. Identities that cannot support
   this (too few sequences) are kept in train only and reported, rather than
   producing an unanswerable query.

Explicit splits already present in the data are respected as-is.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ml.reid.dataset.discovery import ImageRecord

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: List[ImageRecord]
    val: List[ImageRecord]
    test: List[ImageRecord]
    # Identities that could not be represented in val/test.
    train_only_identities: List[str]
    used_explicit_splits: bool

    def counts(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def identity_counts(self) -> Dict[str, int]:
        return {
            "train": len({r.identity for r in self.train}),
            "val": len({r.identity for r in self.val}),
            "test": len({r.identity for r in self.test}),
        }

    def format(self) -> str:
        c, ic = self.counts(), self.identity_counts()
        lines = [
            "Split summary" + ("  (explicit splits from dataset)" if self.used_explicit_splits else ""),
            f"  train : {c['train']:>6} images / {ic['train']} identities",
            f"  val   : {c['val']:>6} images / {ic['val']} identities",
            f"  test  : {c['test']:>6} images / {ic['test']} identities",
        ]
        if self.train_only_identities:
            shown = ", ".join(self.train_only_identities[:8])
            more = "" if len(self.train_only_identities) <= 8 else f" … +{len(self.train_only_identities) - 8}"
            lines.append(f"  train-only identities (too few sequences): {shown}{more}")
        return "\n".join(lines)


def verify_no_sequence_leakage(result: SplitResult) -> List[str]:
    """Return sequence keys present in more than one split (should be empty)."""
    groups: Dict[str, set[str]] = {}
    for split_name, records in (
        ("train", result.train),
        ("val", result.val),
        ("test", result.test),
    ):
        for record in records:
            groups.setdefault(record.group_key(), set()).add(split_name)
    return sorted(key for key, splits in groups.items() if len(splits) > 1)


def split_records(
    records: Sequence[ImageRecord],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
    respect_existing_splits: bool = True,
) -> SplitResult:
    """
    Produce a reproducible, sequence-safe split.

    `seed` fully determines the outcome for a given record set.
    """
    if not records:
        return SplitResult([], [], [], [], False)

    if respect_existing_splits and any(r.split for r in records):
        return _use_explicit_splits(records)

    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must each be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must leave data for training.")

    by_identity: Dict[str, Dict[str, List[ImageRecord]]] = {}
    for record in records:
        by_identity.setdefault(record.identity, {}).setdefault(record.group_key(), []).append(record)

    train: List[ImageRecord] = []
    val: List[ImageRecord] = []
    test: List[ImageRecord] = []
    train_only: List[str] = []

    for identity in sorted(by_identity):
        groups = by_identity[identity]
        keys = sorted(groups)
        # Per-identity seed keeps each identity's split stable even if other
        # identities are added or removed later.
        random.Random(f"{seed}:{identity}").shuffle(keys)

        if len(keys) < 2:
            train_only.append(identity)
            for key in keys:
                train.extend(groups[key])
            continue

        n_test = int(round(len(keys) * test_fraction)) if test_fraction else 0
        n_val = int(round(len(keys) * val_fraction)) if val_fraction else 0
        # Guarantee at least one evaluation sequence, and at least one for train.
        if test_fraction and n_test == 0:
            n_test = 1
        if val_fraction and n_val == 0:
            n_val = 1
        while len(keys) - n_test - n_val < 1 and (n_test + n_val) > 1:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

        test_keys = keys[:n_test]
        val_keys = keys[n_test : n_test + n_val]
        train_keys = keys[n_test + n_val :]

        for key in train_keys:
            train.extend(groups[key])
        for key in val_keys:
            val.extend(groups[key])
        for key in test_keys:
            test.extend(groups[key])

        if not val_keys and not test_keys:
            train_only.append(identity)

    result = SplitResult(train, val, test, train_only, used_explicit_splits=False)
    leaks = verify_no_sequence_leakage(result)
    if leaks:  # defensive: indicates a bug in grouping, not in user data
        raise RuntimeError(f"Sequence leakage detected across splits: {leaks[:5]}")
    return result


def _use_explicit_splits(records: Sequence[ImageRecord]) -> SplitResult:
    buckets: Dict[str, List[ImageRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        key = (record.split or "train").lower()
        buckets.setdefault(key if key in buckets else "train", []).append(record)

    result = SplitResult(
        buckets["train"],
        buckets["val"],
        buckets["test"],
        train_only_identities=sorted(
            {r.identity for r in buckets["train"]}
            - {r.identity for r in buckets["val"]}
            - {r.identity for r in buckets["test"]}
        ),
        used_explicit_splits=True,
    )
    leaks = verify_no_sequence_leakage(result)
    if leaks:
        logger.warning(
            "Dataset-provided splits share %d capture sequence(s) across splits; "
            "evaluation metrics will be optimistic. Example: %s",
            len(leaks),
            leaks[:3],
        )
    return result
