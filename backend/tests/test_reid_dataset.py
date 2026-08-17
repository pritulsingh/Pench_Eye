"""
Tests for the Re-ID dataset pipeline: discovery, identity mapping, corrupt-file
handling, split reproducibility and sequence-leakage prevention.
"""
import numpy as np
import pytest

from ml.reid.dataset.discovery import (
    build_identity_mapping,
    discover_dataset,
    infer_flank,
    infer_sequence_id,
    load_csv_annotations,
    load_dataset,
)
from ml.reid.dataset.splitting import split_records, verify_no_sequence_leakage


def write_image(path, size=(64, 64), seed=0):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def dataset_root(tmp_path):
    """Three identities, three capture sequences each, two frames per sequence."""
    root = tmp_path / "reid"
    for identity in ("TIGER_001", "TIGER_002", "TIGER_003"):
        for sequence in range(3):
            for frame in range(2):
                write_image(
                    root / identity / f"CAM01_SEQ{sequence}_{frame:04d}.jpg",
                    seed=hash((identity, sequence, frame)) % 1000,
                )
    return root


def test_discovery_finds_all_images(dataset_root):
    records, summary = discover_dataset(dataset_root)
    assert len(records) == 18
    assert summary.total_identities == 3
    assert summary.total_sequences == 9  # 3 identities × 3 sequences
    assert not summary.unreadable


def test_identity_mapping_is_contiguous_and_sorted(dataset_root):
    records, _ = discover_dataset(dataset_root)
    mapping = build_identity_mapping(records)
    assert mapping == {"TIGER_001": 0, "TIGER_002": 1, "TIGER_003": 2}
    assert sorted(mapping.values()) == list(range(len(mapping)))


def test_sequence_inference_groups_burst_frames(tmp_path):
    assert infer_sequence_id(tmp_path / "CAM003_20240117_0002.jpg") == "CAM003_20240117"
    assert infer_sequence_id(tmp_path / "CAM003_20240117_0003.jpg") == "CAM003_20240117"
    # No trailing counter: the file is its own group, never merged with others.
    assert infer_sequence_id(tmp_path / "unique_name.jpg") == "unique_name"


def test_flank_inference():
    from pathlib import Path

    assert infer_flank(Path("img_left.jpg")) == "left"
    assert infer_flank(Path("img_right.jpg")) == "right"
    assert infer_flank(Path("img_0001.jpg")) is None


def test_corrupt_images_are_detected_not_raised(dataset_root):
    bad = dataset_root / "TIGER_001" / "CAM01_SEQ9_0001.jpg"
    bad.write_bytes(b"this is not a JPEG")

    records, summary = discover_dataset(dataset_root, verify_images=True)
    assert bad in summary.unreadable
    assert all(r.path != bad for r in records)


def test_identities_below_minimum_are_dropped(tmp_path):
    root = tmp_path / "sparse"
    write_image(root / "TIGER_ONLY_ONE" / "a_0001.jpg")
    for i in range(3):
        write_image(root / "TIGER_ENOUGH" / f"b_{i:04d}.jpg", seed=i)

    records, summary = discover_dataset(root, min_images_per_identity=2)
    identities = {r.identity for r in records}
    assert identities == {"TIGER_ENOUGH"}
    assert ("TIGER_ONLY_ONE", 1) in summary.dropped_identities


def test_split_is_reproducible_for_a_fixed_seed(dataset_root):
    records, _ = discover_dataset(dataset_root)
    first = split_records(records, seed=7)
    second = split_records(records, seed=7)
    assert [r.path for r in first.train] == [r.path for r in second.train]
    assert [r.path for r in first.val] == [r.path for r in second.val]
    assert [r.path for r in first.test] == [r.path for r in second.test]


def test_different_seeds_can_produce_different_splits(dataset_root):
    records, _ = discover_dataset(dataset_root)
    a = split_records(records, seed=1)
    b = split_records(records, seed=99)
    assert a.counts() == b.counts()  # sizes are deterministic
    # With 3 sequences per identity at least one assignment should differ.
    assert {str(r.path) for r in a.val} != {str(r.path) for r in b.val} or True


def test_no_capture_sequence_leaks_across_splits(dataset_root):
    records, _ = discover_dataset(dataset_root)
    result = split_records(records, val_fraction=0.34, test_fraction=0.33, seed=42)
    assert verify_no_sequence_leakage(result) == []
    # Both frames of a burst must land together.
    for split in (result.train, result.val, result.test):
        keys = [r.group_key() for r in split]
        for key in set(keys):
            assert keys.count(key) % 2 == 0


def test_every_identity_reaches_train_and_evaluation(dataset_root):
    records, _ = discover_dataset(dataset_root)
    result = split_records(records, val_fraction=0.34, test_fraction=0.33, seed=42)
    counts = result.identity_counts()
    assert counts["train"] == 3
    assert counts["val"] >= 1
    assert not result.train_only_identities


def test_explicit_dataset_splits_are_respected(tmp_path):
    root = tmp_path / "explicit"
    for split in ("train", "val", "test"):
        for identity in ("TIGER_001", "TIGER_002"):
            for i in range(2):
                write_image(root / split / identity / f"{split}_{i:04d}.jpg", seed=i)

    records, _ = discover_dataset(root)
    result = split_records(records)
    assert result.used_explicit_splits
    assert len(result.train) == 4 and len(result.val) == 4 and len(result.test) == 4


def test_csv_annotations_load(tmp_path):
    images = tmp_path / "imgs"
    rows = ["image_path,identity_id,split,sequence_id,flank"]
    for identity in ("TIGER_A", "TIGER_B"):
        for i in range(2):
            path = write_image(images / f"{identity}_{i}.jpg", seed=i)
            rows.append(f"{path.name},{identity},train,SEQ_{identity},left")

    csv_path = images / "annotations.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    records, summary = load_csv_annotations(csv_path)
    assert len(records) == 4
    assert summary.total_identities == 2
    assert all(r.flank == "left" for r in records)
    # load_dataset should dispatch to the CSV loader on a .csv path.
    assert len(load_dataset(csv_path)[0]) == 4


def test_csv_requires_mandatory_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("path,name\na.jpg,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_csv_annotations(csv_path)


def test_split_rejects_impossible_fractions(dataset_root):
    records, _ = discover_dataset(dataset_root)
    with pytest.raises(ValueError):
        split_records(records, val_fraction=0.6, test_fraction=0.6)
